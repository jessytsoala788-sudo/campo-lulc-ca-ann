"""
==============================================================================
 PIPELINE CA-ANN POUR LA PREDICTION LULC (Python) 
------------------------------------------------------------------------------
 Auteure : Jessy Houston Tsoala Tchoffo
 Contexte : Memoire M2 Geomatique et Changement Climatique - UNIKIN
 Zone d'etude : Commune de Campo, Sud Cameroun

 ----------------------------------------------------------------------------
   PHASE A -- VALIDATION 
     - Calibration Markov + MLP sur 2000 -> 2015 
     - Simulation de 2025 a partir de l'etat 2015 observe
     - Comparaison de cette simulation a la vraie carte 2025
       -> Kappa / Pontius 

   PHASE B -- PROJECTION FINALE (2035, 2045)
     - Recalibration Markov + MLP sur 2015 -> 2025 (periode la plus recente
       et la plus pertinente pour projeter le futur)
     - Projection vers 2035 et 2045 a partir de l'etat 2025 observe
     - Cette phase ne sert PAS a la validation : ses indicateurs de qualite
       sont ceux obtenus en Phase A 
==============================================================================
"""

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import json
import os

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

CONFIG = {
    # --- Cartes LULC classifiees (rasters entiers, valeurs = codes de classe) ---
    "lulc_2000": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\LULC_2000_alignement.tif",
    "lulc_2015": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\LULC_2015_alignement.tif",
    "lulc_2025": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\LULC_2025_alignement.tif",

    # --- Variables explicatives (rasters continus, deja alignes sur la meme grille) ---
    "variables": {
        "pente": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\pente_alignement.tif",
        "aspect": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\aspect_alignement.tif",
        "dem": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\dem_alignement.tif",
        "dist_rivieres": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\dis_riv_alignement.tif",
        "dist_routes": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\dis_route_alignement.tif",
        "dist_parc": "D:\\Master GCC\\Memoire MGCC\\Mémoire 2 vrai\\Resultats\\R_Classification\\LULC_TIF\\dist_parc_alignement.tif",
    },

    # --- Dictionnaire des classes LULC (code entier -> nom) ---
    "classes": {
        1: "Foret dense",
        2: "Foret degradee",
        3: "Agriculture",
        4: "Mangrove",
        5: "Eau",
        6: "Sol nu/Bati",
    },

    # --- Annees ---
    "annee_validation_debut": 2000,   
    "annee_validation_fin": 2015,     
    "annee_test": 2025,               
    "annee_base": 2015,              
    "annee_calibration": 2025,        
    "annees_cibles": [2035, 2045],

    # --- Parametres CA-ANN ---
    "taille_voisinage": 3,
    "mlp_hidden_layers": (64, 32),  
                                    
    "mlp_max_iter": 300,
    "mlp_random_state": 42,
    "fraction_echantillon_entrainement": 0.15,
    "valeur_nodata_entiere": 32767,

    # --- Sortie ---
    "dossier_sortie": "/mnt/user-data/outputs/resultats_python",
}

os.makedirs(CONFIG["dossier_sortie"], exist_ok=True)


# ==============================================================================
# 2. LECTURE ET VERIFICATION DES RASTERS 
# ==============================================================================

def lire_raster(chemin):
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}\n"
            f"-> Verifie le chemin dans CONFIG ou uploade le fichier."
        )
    with rasterio.open(chemin) as src:
        arr = src.read(1)
        profile = src.profile
        nodata = src.nodata
    return arr, profile, nodata


def verifier_alignement(profiles):
    ref = profiles[0]
    for i, p in enumerate(profiles[1:], start=1):
        if (p["width"], p["height"]) != (ref["width"], ref["height"]):
            raise ValueError(
                f"Desalignement detecte (raster #{i}): "
                f"dimensions {p['width']}x{p['height']} != {ref['width']}x{ref['height']}. "
                f"Realigne tous les rasters (Warp/resample) avant de continuer."
            )
        if p["crs"] != ref["crs"]:
            raise ValueError(f"CRS different pour le raster #{i} : {p['crs']} != {ref['crs']}")
        if p["transform"] != ref["transform"]:
            raise ValueError(f"Transform (emprise/resolution) different pour le raster #{i}.")
    print("Alignement verifie : tous les rasters partagent la meme grille.")


# ==============================================================================
# 3. MATRICE DE TRANSITION DE MARKOV 
# ==============================================================================

def matrice_transition_markov(lulc_t1, lulc_t2, classes, masque_valide):
    codes = sorted(classes.keys())
    n = len(codes)
    comptage = np.zeros((n, n), dtype=np.int64)

    t1 = lulc_t1[masque_valide]
    t2 = lulc_t2[masque_valide]

    idx = {c: i for i, c in enumerate(codes)}
    for c_from, c_to in zip(t1, t2):
        if c_from in idx and c_to in idx:
            comptage[idx[c_from], idx[c_to]] += 1

    totaux = comptage.sum(axis=1, keepdims=True)
    totaux[totaux == 0] = 1
    proba = comptage / totaux

    return {"codes": codes, "comptage": comptage, "probabilite": proba}


def projeter_quotas_markov(comptage, codes, n_pas_calibration, n_pas_projection):
    """
    METHODE PRINCIPALE 

    Projette les quotas de pixels a partir de la seule matrice de Markov,
    elevee a la puissance fractionnaire correspondant a la duree de
    projection relative a la duree de calibration. C'est la methode
    standard des modeles CA-Markov (Eastman, TerrSet ; Muhammad et al., 2022 ;
    Tiye et al., 2025) : elle ne combine pas deux mecanismes d'extrapolation
    independants et evite ainsi le double comptage signale par le Prof Michel.

    Principe : P_calib est la matrice de probabilites de transition observee
    sur la periode de calibration (duree n_pas_calibration). On calcule
    P_projection = P_calib ^ (n_pas_projection / n_pas_calibration) par
    decomposition spectrale (puissance matricielle fractionnaire), puis on
    l'applique a la distribution de pixels a l'annee de depart de la
    projection pour obtenir les quotas cibles.
    """
    from scipy.linalg import fractional_matrix_power

    totaux = comptage.sum(axis=1, keepdims=True)
    totaux_safe = totaux.copy()
    totaux_safe[totaux_safe == 0] = 1
    P_calib = comptage / totaux_safe

    exposant = n_pas_projection / n_pas_calibration
    P_proj = fractional_matrix_power(P_calib, exposant)
    P_proj = np.real(P_proj)  # elimine les residus imaginaires numeriques negligeables
    P_proj = np.clip(P_proj, 0, 1)
    P_proj = P_proj / P_proj.sum(axis=1, keepdims=True)  # renormalisation (chaque ligne doit sommer a 1)

    # distribution de depart = comptage total par classe a la fin de la periode de calibration
    distribution_depart = comptage.sum(axis=1)  # nb de pixels par classe (etat de depart)

    quotas_matrice = (distribution_depart[:, None] * P_proj).astype(np.int64)
    return quotas_matrice


def projeter_quotas_puyravaud_sensibilite(comptage, codes, n_pas_calibration, n_pas_projection):
    """
    TEST DE SENSIBILITE UNIQUEMENT  Reproduit l'ancienne
    logique (mise a l'echelle lineaire des comptages Markov par un facteur
    temporel, dans l'esprit du taux de Puyravaud applique aux superficies
    totales par classe). 
    """
    facteur = n_pas_projection / n_pas_calibration
    quotas_futurs = np.round(comptage * facteur).astype(np.int64)
    return quotas_futurs


def comparer_methodes_quotas(quotas_markov, quotas_puyravaud, codes, classes):
    """
    Compare les quotas nets par classe obtenus par les deux methodes, pour
    documenter l'ecart dans l'article 
    """
    print("\nComparaison quotas nets par classe -- Markov (methode principale) vs "
          "Puyravaud lineaire (sensibilite) :")
    ecarts = {}
    for j, code in enumerate(codes):
        net_markov = int(np.sum(quotas_markov[:, j]) - quotas_markov[codes.index(code), j])
        net_puyravaud = int(np.sum(quotas_puyravaud[:, j]) - quotas_puyravaud[codes.index(code), j])
        ecart_pct = 100 * (net_puyravaud - net_markov) / net_markov if net_markov != 0 else float("nan")
        ecarts[classes[code]] = {
            "quota_markov": net_markov,
            "quota_puyravaud_sensibilite": net_puyravaud,
            "ecart_pct": ecart_pct,
        }
        print(f"  {classes[code]:20s} : Markov = {net_markov:+8d} px | "
              f"Puyravaud (sensibilite) = {net_puyravaud:+8d} px | ecart = {ecart_pct:+.1f}%")
    return ecarts


# ==============================================================================
# 4. VARIABLES EXPLICATIVES + TERME DE VOISINAGE 
# ==============================================================================

def densite_voisinage(lulc, code_classe, taille_fenetre):
    masque_classe = (lulc == code_classe).astype(np.float32)
    return uniform_filter(masque_classe, size=taille_fenetre, mode="nearest")


def construire_pile_variables(variables_arrays, lulc_actuel, classes, taille_fenetre):
    piles = list(variables_arrays.values())
    noms = list(variables_arrays.keys())
    for code, nom_classe in classes.items():
        piles.append(densite_voisinage(lulc_actuel, code, taille_fenetre))
        noms.append(f"voisinage_{nom_classe}")
    stack = np.stack(piles, axis=-1)
    return stack, noms


# ==============================================================================
# 5. MODELE DE POTENTIEL DE TRANSITION (ANN) 
# ==============================================================================

def entrainer_potentiel_transition(stack_features, lulc_t1, lulc_t2, masque_valide,
                                    fraction_echantillon, hidden_layers, max_iter, random_state,
                                    n_max_par_classe=45000):
    H, W, n_features = stack_features.shape
    X_full = stack_features.reshape(-1, n_features)
    y_full = lulc_t2.reshape(-1)
    valide_full = masque_valide.reshape(-1)

    X_valide = X_full[valide_full]
    y_valide = y_full[valide_full]

    rng = np.random.default_rng(random_state)
    classes_uniques = np.unique(y_valide)

    indices_par_classe = []
    for c in classes_uniques:
        idx_c = np.where(y_valide == c)[0]
        n_disponible = len(idx_c)
        remplace = n_disponible < n_max_par_classe
        idx_choisi = rng.choice(idx_c, size=n_max_par_classe, replace=remplace)
        indices_par_classe.append(idx_choisi)
        print(f"  Classe {c} : {n_disponible} pixels disponibles -> "
              f"{n_max_par_classe} echantillonnes ({'sur-echantillonnage' if remplace else 'sous-echantillonnage'})")

    idx_echantillon = np.concatenate(indices_par_classe)
    rng.shuffle(idx_echantillon)

    X_train = X_valide[idx_echantillon]
    y_train = y_valide[idx_echantillon]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Architecture et hyperparametres explicites  :
    #   - 2 couches cachees : 64 puis 32 neurones (config["mlp_hidden_layers"]),
    #     architecture retenue apres comparaison de 4 architectures candidates
    #     (analyse de sensibilite, section resultats)
    #   - fonction d'activation : ReLU (parametre 'relu', par defaut MLPClassifier)
    #   - solveur : Adam (parametre 'adam', par defaut MLPClassifier)
    #   - taux d'apprentissage initial : 0.001 (defaut Adam, non modifie)
    #   - nombre max d'iterations : config["mlp_max_iter"] (300)
    #   - critere d'arret : early stopping sur 15% des donnees d'entrainement
    #     mises de cote comme validation interne (validation_fraction=0.15),
    #     arret si le score de validation ne s'ameliore plus pendant
    #     n_iter_no_change iterations consecutives (defaut = 10)
    #   - normalisation des variables d'entree : centrage-reduction (StandardScaler)
    #   - graine aleatoire : config["mlp_random_state"] (42), fixee pour la
    #     reproductibilite du tirage des poids initiaux et de l'echantillonnage
    #   - partition entrainement/test : echantillon equilibre par classe
    #     (n_max_par_classe), scinde en 85% entrainement / 15% validation
    #     interne au MLP (early_stopping + validation_fraction=0.15)
    activation = "relu"
    solveur = "adam"
    taux_apprentissage_initial = 0.001
    n_iter_no_change = 10

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solveur,
        learning_rate_init=taux_apprentissage_initial,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=n_iter_no_change,
    )
    mlp.fit(X_train_scaled, y_train)

    print("\nArchitecture du MLP (a reporter dans l'article, section methodologie) :")
    print(f"  Couches cachees          : {hidden_layers}")
    print(f"  Fonction d'activation    : {activation}")
    print(f"  Solveur                  : {solveur}")
    print(f"  Taux d'apprentissage init: {taux_apprentissage_initial}")
    print(f"  Iterations max           : {max_iter}")
    print(f"  Iterations reellement effectuees : {mlp.n_iter_}")
    print(f"  Critere d'arret          : early stopping (validation_fraction=0.15, "
          f"n_iter_no_change={n_iter_no_change})")
    print(f"  Normalisation entrees    : StandardScaler (centrage-reduction)")
    print(f"  Graine aleatoire         : {random_state}")

    score_train = mlp.score(X_train_scaled, y_train)
    print(f"Score d'entrainement du MLP : {score_train:.4f}")
    print(f"  (echantillon equilibre : {len(idx_echantillon)} pixels, "
          f"{n_max_par_classe} par classe, sur {X_valide.shape[0]} pixels valides au total)")

    return mlp, scaler


def predire_probabilites(mlp, scaler, stack_features, masque_valide):
    H, W, n_features = stack_features.shape
    X_full = stack_features.reshape(-1, n_features)

    proba_full = np.zeros((X_full.shape[0], len(mlp.classes_)), dtype=np.float32)
    valide_flat = masque_valide.reshape(-1)

    X_valide_scaled = scaler.transform(X_full[valide_flat])
    proba_full[valide_flat] = mlp.predict_proba(X_valide_scaled)

    proba_map = proba_full.reshape(H, W, len(mlp.classes_))
    return proba_map, mlp.classes_


# ==============================================================================
# 6. ALLOCATION SPATIALE 
# ==============================================================================

def allouer_ca(lulc_actuel, proba_map, classes_mlp, quotas_cible, masque_valide):
    H, W = lulc_actuel.shape
    n_classes = len(classes_mlp)
    idx_classe_mlp = {c: i for i, c in enumerate(classes_mlp)}

    lulc_flat = lulc_actuel.reshape(-1).copy()
    masque_flat = masque_valide.reshape(-1)
    proba_flat = proba_map.reshape(-1, n_classes)

    idx_courant = np.full(lulc_flat.shape, -1, dtype=np.int64)
    for code, i in idx_classe_mlp.items():
        idx_courant[lulc_flat == code] = i

    pixels_valides = np.where(masque_flat & (idx_courant >= 0))[0]
    if len(pixels_valides) == 0:
        return lulc_actuel.copy()

    proba_pv = proba_flat[pixels_valides]
    idx_courant_pv = idx_courant[pixels_valides]
    proba_courant = proba_pv[np.arange(len(pixels_valides)), idx_courant_pv]

    proba_alt = proba_pv.copy()
    proba_alt[np.arange(len(pixels_valides)), idx_courant_pv] = -1.0
    meilleure_alt_idx = np.argmax(proba_alt, axis=1)
    meilleure_alt_proba = proba_alt[np.arange(len(pixels_valides)), meilleure_alt_idx]

    marge = meilleure_alt_proba - proba_courant
    candidats_mask = marge > 0

    pix_candidats = pixels_valides[candidats_mask]
    codes_array = np.array(classes_mlp)
    classes_candidats = codes_array[meilleure_alt_idx[candidats_mask]]
    marge_candidats = marge[candidats_mask]

    ordre = np.argsort(-marge_candidats)
    quotas_restants = {c: max(int(q), 0) for c, q in quotas_cible.items()}

    for k in ordre:
        cible = classes_candidats[k]
        if quotas_restants.get(cible, 0) <= 0:
            continue
        pix = pix_candidats[k]
        lulc_flat[pix] = cible
        quotas_restants[cible] -= 1

    return lulc_flat.reshape(H, W)


# ==============================================================================
# 7. VALIDATION : KAPPA + INDICES DE PONTIUS 
# ==============================================================================

def calculer_kappa(reference, simule, masque_valide, codes):
    ref = reference[masque_valide]
    sim = simule[masque_valide]
    n = len(ref)

    n_classes = len(codes)
    idx = {c: i for i, c in enumerate(codes)}

    matrice_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for r, s in zip(ref, sim):
        if r in idx and s in idx:
            matrice_confusion[idx[r], idx[s]] += 1

    po = np.trace(matrice_confusion) / n
    marge_ligne = matrice_confusion.sum(axis=1) / n
    marge_col = matrice_confusion.sum(axis=0) / n
    pe = np.sum(marge_ligne * marge_col)
    kappa_general = (po - pe) / (1 - pe) if (1 - pe) != 0 else np.nan

    pmax = np.sum(np.minimum(marge_ligne, marge_col))
    khisto = (pmax - pe) / (1 - pe) if (1 - pe) != 0 else np.nan
    klocation = (po - pe) / (pmax - pe) if (pmax - pe) != 0 else np.nan

    return {
        "matrice_confusion": matrice_confusion.tolist(),
        "kappa_general": float(kappa_general),
        "kappa_histogramme": float(khisto),
        "kappa_lieu": float(klocation),
        "pourcentage_accord": float(po * 100),
    }


def calculer_pontius(reference, simule, actuel, masque_valide):
    ref = reference[masque_valide]
    sim = simule[masque_valide]
    act = actuel[masque_valide]

    changement_reel = ref != act
    changement_simule = sim != act

    hits = np.sum(changement_reel & changement_simule & (ref == sim))
    misses = np.sum(changement_reel & ~changement_simule)
    fausses_alertes = np.sum(~changement_reel & changement_simule)
    erreurs_localisation = np.sum(changement_reel & changement_simule & (ref != sim))

    denom = hits + misses + fausses_alertes + erreurs_localisation
    fom = hits / denom if denom > 0 else np.nan

    return {
        "hits": int(hits),
        "misses": int(misses),
        "fausses_alertes": int(fausses_alertes),
        "erreurs_localisation": int(erreurs_localisation),
        "figure_of_merit": float(fom),
    }


# ==============================================================================
# 8. EXPORT RASTER 
# ==============================================================================

def exporter_raster(array, profile, chemin_sortie):
    profile_sortie = profile.copy()
    profile_sortie.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(chemin_sortie, "w", **profile_sortie) as dst:
        dst.write(array.astype(rasterio.uint8), 1)
    print(f"Raster exporte : {chemin_sortie}")


# ==============================================================================
# 9. DIAGNOSTIC DE MULTICOLINEARITE : CORRELATION + VIF 
# ==============================================================================

def diagnostiquer_multicolinearite(variables_arrays, masque_valide):
    """
    Calcule la matrice de correlation de Pearson entre les variables
    explicatives brutes (avant ajout des termes de voisinage CA, qui sont
    specifiques a chaque classe et n'ont pas a entrer dans ce diagnostic),
    ainsi que le VIF (Variance Inflation Factor) de chacune.

    VIF_i = 1 / (1 - R2_i), ou R2_i est le R2 de la regression lineaire de
    la variable i sur toutes les autres variables. Regle usuelle :
      VIF < 5   : pas de probleme de colinearite
      5 <= VIF < 10 : colinearite moderee, a surveiller
      VIF >= 10 : colinearite forte, la variable est redondante avec les autres

    Implementation en numpy pur (pas de dependance a statsmodels).
    """
    noms = list(variables_arrays.keys())
    X = np.stack([variables_arrays[n][masque_valide] for n in noms], axis=1).astype(np.float64)

    # --- Matrice de correlation de Pearson ---
    correlation = np.corrcoef(X, rowvar=False)

    # --- VIF par regression lineaire (avec constante) ---
    n_var = X.shape[1]
    vif = {}
    for i in range(n_var):
        y = X[:, i]
        X_autres = np.delete(X, i, axis=1)
        X_design = np.column_stack([np.ones(X_autres.shape[0]), X_autres])
        coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        y_pred = X_design @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif[noms[i]] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

    print("\nMatrice de correlation de Pearson (variables explicatives) :")
    print("           " + "  ".join(f"{n[:8]:>8s}" for n in noms))
    for i, n in enumerate(noms):
        print(f"{n[:10]:10s} " + "  ".join(f"{correlation[i,j]:8.3f}" for j in range(n_var)))

    print("\nVIF (Variance Inflation Factor) par variable :")
    for n in noms:
        alerte = "  <-- colinearite forte" if vif[n] >= 10 else ("  <-- a surveiller" if vif[n] >= 5 else "")
        print(f"  {n:15s} : {vif[n]:.2f}{alerte}")

    return {
        "noms_variables": noms,
        "matrice_correlation": correlation.tolist(),
        "vif": {n: float(v) for n, v in vif.items()},
    }


# ==============================================================================
# 10. CONSTRUCTION DU MASQUE VALIDE 
# ==============================================================================

def construire_masque_valide(lulc_arrays, nodata_list, variables_arrays, classes, config):
    """
    lulc_arrays : liste de rasters LULC (ex: [lulc_2000, lulc_2015, lulc_2025])
    nodata_list : liste des valeurs nodata declarees correspondantes (ou None)
    """
    masque_valide = np.ones_like(lulc_arrays[0], dtype=bool)
    codes_valides_lulc = set(classes.keys())

    for arr, nodata in zip(lulc_arrays, nodata_list):
        masque_valide &= np.isin(arr, list(codes_valides_lulc))
        if nodata is not None:
            masque_valide &= (arr != nodata)

    for nom, arr in variables_arrays.items():
        masque_valide &= ~np.isnan(arr)
        masque_valide &= (np.abs(arr) < 1e29)
        masque_valide &= (arr != config.get("valeur_nodata_entiere", 32767))

    return masque_valide


# ==============================================================================
# 11. ANALYSE DE SENSIBILITE 
# ==============================================================================

def executer_validation_phase_a(lulc_2000, lulc_2015, lulc_2025, variables_arrays,
                                 masque_valide, classes, config, hidden_layers=None,
                                 random_state=None, silencieux=True):
    """
    Reproduit exactement la Phase A (calibration 2000->2015, simulation de 2025,
    comparaison au 2025 reel), mais parametrable en variables et en architecture
    MLP, pour servir de brique de base a l'analyse de sensibilite. Retourne
    uniquement les indicateurs de synthese (Kappa general, FoM) pour rester
    leger en sortie.
    """
    import io, contextlib

    hidden_layers = hidden_layers or config["mlp_hidden_layers"]
    random_state = random_state if random_state is not None else config["mlp_random_state"]

    def _run():
        markov_calib = matrice_transition_markov(lulc_2000, lulc_2015, classes, masque_valide)
        n_pas_calibration_val = config["annee_validation_fin"] - config["annee_validation_debut"]
        n_pas_test = config["annee_test"] - config["annee_validation_fin"]

        stack_2000, _ = construire_pile_variables(variables_arrays, lulc_2000, classes, config["taille_voisinage"])
        stack_2015, _ = construire_pile_variables(variables_arrays, lulc_2015, classes, config["taille_voisinage"])

        mlp_val, scaler_val = entrainer_potentiel_transition(
            stack_2000, lulc_2000, lulc_2015, masque_valide,
            config["fraction_echantillon_entrainement"],
            hidden_layers, config["mlp_max_iter"], random_state,
        )

        quotas_matrice_val = projeter_quotas_markov(markov_calib["comptage"], markov_calib["codes"],
                                                     n_pas_calibration_val, n_pas_test)
        quotas_test = {}
        for j, code_cible in enumerate(markov_calib["codes"]):
            quotas_test[code_cible] = int(np.sum(quotas_matrice_val[:, j])
                                           - quotas_matrice_val[markov_calib["codes"].index(code_cible), j])

        proba_map_val, classes_mlp_val = predire_probabilites(mlp_val, scaler_val, stack_2015, masque_valide)
        lulc_2025_simule = allouer_ca(lulc_2015, proba_map_val, classes_mlp_val, quotas_test, masque_valide)

        kappa_val = calculer_kappa(lulc_2025, lulc_2025_simule, masque_valide, markov_calib["codes"])
        pontius_val = calculer_pontius(lulc_2025, lulc_2025_simule, lulc_2015, masque_valide)
        return kappa_val["kappa_general"], pontius_val["figure_of_merit"]

    if silencieux:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            kappa_g, fom = _run()
    else:
        kappa_g, fom = _run()

    return {"kappa_general": kappa_g, "figure_of_merit": fom}


def analyse_sensibilite_variables(lulc_2000, lulc_2015, lulc_2025, variables_arrays,
                                   masque_valide, classes, config):
    """
    Retire une variable explicative a la fois (les autres restant inchangees),
    refait la validation Phase A, et mesure la chute de performance par
    rapport au modele complet. Une chute importante = variable influente.
    """
    print("\n" + "=" * 70)
    print("ANALYSE DE SENSIBILITE -- VARIABLES EXPLICATIVES")
    print("=" * 70)

    reference = executer_validation_phase_a(lulc_2000, lulc_2015, lulc_2025, variables_arrays,
                                             masque_valide, classes, config)
    print(f"Modele complet (6 variables)     : Kappa = {reference['kappa_general']:.4f}  "
          f"FoM = {reference['figure_of_merit']:.4f}")

    resultats = {"modele_complet": reference}
    for nom_exclu in variables_arrays.keys():
        variables_reduites = {k: v for k, v in variables_arrays.items() if k != nom_exclu}
        res = executer_validation_phase_a(lulc_2000, lulc_2015, lulc_2025, variables_reduites,
                                           masque_valide, classes, config)
        delta_kappa = res["kappa_general"] - reference["kappa_general"]
        delta_fom = res["figure_of_merit"] - reference["figure_of_merit"]
        print(f"Sans '{nom_exclu:15s}'          : Kappa = {res['kappa_general']:.4f} ({delta_kappa:+.4f})  "
              f"FoM = {res['figure_of_merit']:.4f} ({delta_fom:+.4f})")
        resultats[f"sans_{nom_exclu}"] = {**res, "delta_kappa": delta_kappa, "delta_fom": delta_fom}

    return resultats


def analyse_sensibilite_mlp(lulc_2000, lulc_2015, lulc_2025, variables_arrays,
                             masque_valide, classes, config):
    """
    Teste plusieurs architectures MLP et plusieurs graines aleatoires, pour
    verifier que les resultats ne dependent pas d'un choix arbitraire de
    parametres.
    """
    print("\n" + "=" * 70)
    print("ANALYSE DE SENSIBILITE -- ARCHITECTURE ET GRAINE ALEATOIRE DU MLP")
    print("=" * 70)

    architectures = [(16,), (32, 16), (64, 32), (32, 16, 8)]
    resultats = {"architectures": {}, "graines": {}}

    print("\n-- Variation de l'architecture (graine fixe = config par defaut) --")
    for arch in architectures:
        res = executer_validation_phase_a(lulc_2000, lulc_2015, lulc_2025, variables_arrays,
                                           masque_valide, classes, config, hidden_layers=arch)
        print(f"  Architecture {str(arch):15s} : Kappa = {res['kappa_general']:.4f}  "
              f"FoM = {res['figure_of_merit']:.4f}")
        resultats["architectures"][str(arch)] = res

    print("\n-- Variation de la graine aleatoire (architecture fixe = config par defaut) --")
    graines = [0, 1, 42, 123, 2026]
    for graine in graines:
        res = executer_validation_phase_a(lulc_2000, lulc_2015, lulc_2025, variables_arrays,
                                           masque_valide, classes, config, random_state=graine)
        print(f"  Graine {graine:6d}                : Kappa = {res['kappa_general']:.4f}  "
              f"FoM = {res['figure_of_merit']:.4f}")
        resultats["graines"][str(graine)] = res

    kappas_arch = [v["kappa_general"] for v in resultats["architectures"].values()]
    foms_arch = [v["figure_of_merit"] for v in resultats["architectures"].values()]
    kappas_graine = [v["kappa_general"] for v in resultats["graines"].values()]
    foms_graine = [v["figure_of_merit"] for v in resultats["graines"].values()]

    print(f"\nEtendue Kappa selon l'architecture : [{min(kappas_arch):.4f} ; {max(kappas_arch):.4f}]")
    print(f"Etendue FoM selon l'architecture   : [{min(foms_arch):.4f} ; {max(foms_arch):.4f}]")
    print(f"Etendue Kappa selon la graine       : [{min(kappas_graine):.4f} ; {max(kappas_graine):.4f}]")
    print(f"Etendue FoM selon la graine         : [{min(foms_graine):.4f} ; {max(foms_graine):.4f}]")

    return resultats


# ==============================================================================
# 10. PIPELINE PRINCIPAL -- RESTRUCTURE EN DEUX PHASES
# ==============================================================================

def executer_pipeline(config):
    print("=" * 70)
    print("PIPELINE CA-ANN -- PREDICTION LULC (version corrigee, sans circularite)")
    print("=" * 70)

    os.makedirs(config["dossier_sortie"], exist_ok=True)

    # --- Lecture de toutes les cartes LULC necessaires (2000, 2015, 2025) ---
    lulc_2000, profile_2000, nodata_2000 = lire_raster(config["lulc_2000"])
    lulc_2015, profile_2015, nodata_2015 = lire_raster(config["lulc_2015"])
    lulc_2025, profile_2025, nodata_2025 = lire_raster(config["lulc_2025"])

    variables_arrays = {}
    profiles_variables = []
    for nom, chemin in config["variables"].items():
        arr, prof, _ = lire_raster(chemin)
        variables_arrays[nom] = arr.astype(np.float32)
        profiles_variables.append(prof)

    verifier_alignement([profile_2000, profile_2015, profile_2025] + profiles_variables)

    masque_valide = construire_masque_valide(
        [lulc_2000, lulc_2015, lulc_2025],
        [nodata_2000, nodata_2015, nodata_2025],
        variables_arrays, config["classes"], config,
    )
    print(f"Pixels valides (communs aux 3 dates) : {masque_valide.sum()} / {masque_valide.size}")

    resultats = {}

    # --- Diagnostic de multicolinearite  : a faire une seule
    # fois, sur les variables brutes, avant toute phase de calibration ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC DE MULTICOLINEARITE (variables explicatives brutes)")
    print("=" * 70)
    diagnostic_colinearite = diagnostiquer_multicolinearite(variables_arrays, masque_valide)
    resultats["diagnostic_multicolinearite"] = diagnostic_colinearite

    # ==========================================================================
    # PHASE A  : calibration sur 2000->2015,
    # simulation de 2025 , comparaison au 2025 reel
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE A -- VALIDATION HORS ECHANTILLON (calibration 2000-2015)")
    print("=" * 70)

    markov_calib = matrice_transition_markov(lulc_2000, lulc_2015, config["classes"], masque_valide)
    print("\nMatrice de transition Markov (probabilites, 2000->2015) :")
    for i, c in enumerate(markov_calib["codes"]):
        print(f"  {config['classes'][c]:20s} : {np.round(markov_calib['probabilite'][i], 4)}")

    n_pas_calibration_val = config["annee_validation_fin"] - config["annee_validation_debut"]  # 15 ans
    n_pas_test = config["annee_test"] - config["annee_validation_fin"]                          # 10 ans

    stack_2000, _ = construire_pile_variables(
        variables_arrays, lulc_2000, config["classes"], config["taille_voisinage"]
    )
    stack_2015, _ = construire_pile_variables(
        variables_arrays, lulc_2015, config["classes"], config["taille_voisinage"]
    )

    # Le MLP apprend ici la transition 2000 -> 2015 UNIQUEMENT 
    mlp_val, scaler_val = entrainer_potentiel_transition(
        stack_2000, lulc_2000, lulc_2015, masque_valide,
        config["fraction_echantillon_entrainement"],
        config["mlp_hidden_layers"], config["mlp_max_iter"], config["mlp_random_state"],
    )

    quotas_matrice_val = projeter_quotas_markov(markov_calib["comptage"], markov_calib["codes"],
                                                 n_pas_calibration_val, n_pas_test)
    quotas_matrice_val_sensibilite = projeter_quotas_puyravaud_sensibilite(
        markov_calib["comptage"], markov_calib["codes"], n_pas_calibration_val, n_pas_test)
    comparer_methodes_quotas(quotas_matrice_val, quotas_matrice_val_sensibilite,
                              markov_calib["codes"], config["classes"])
    quotas_test = {}
    for j, code_cible in enumerate(markov_calib["codes"]):
        quotas_test[code_cible] = int(np.sum(quotas_matrice_val[:, j])
                                       - quotas_matrice_val[markov_calib["codes"].index(code_cible), j])

    # On applique le modele (calibre 2000->2015) sur l'etat 2015 pour simuler 2025
    proba_map_val, classes_mlp_val = predire_probabilites(mlp_val, scaler_val, stack_2015, masque_valide)
    lulc_2025_simule_hors_echantillon = allouer_ca(
        lulc_2015, proba_map_val, classes_mlp_val, quotas_test, masque_valide
    )

    # Comparaison a la VRAIE carte 2025, jamais vue par mlp_val ni markov_calib
    kappa_val = calculer_kappa(lulc_2025, lulc_2025_simule_hors_echantillon, masque_valide, markov_calib["codes"])
    pontius_val = calculer_pontius(lulc_2025, lulc_2025_simule_hors_echantillon, lulc_2015, masque_valide)

    print("\nResultats de VALIDATION HONNETE (2025 simule depuis un modele calibre 2000-2015, vs 2025 reel) :")
    print(f"  Kappa general      : {kappa_val['kappa_general']:.4f}")
    print(f"  Kappa histogramme  : {kappa_val['kappa_histogramme']:.4f}")
    print(f"  Kappa lieu         : {kappa_val['kappa_lieu']:.4f}")
    print(f"  % accord global    : {kappa_val['pourcentage_accord']:.2f}%")
    print(f"  Figure of Merit    : {pontius_val['figure_of_merit']:.4f}")
    print("  NOTE : ces indicateurs sont ceux a rapporter dans l'article comme")
    print("  mesure de la performance reelle du modele (validation hors echantillon).")

    resultats["validation_hors_echantillon"] = {
        "periode_calibration": f"{config['annee_validation_debut']}-{config['annee_validation_fin']}",
        "periode_test": f"{config['annee_validation_fin']}-{config['annee_test']}",
        "kappa": kappa_val,
        "pontius": pontius_val,
    }

    chemin_sortie_val = os.path.join(config["dossier_sortie"], "LULC_2025_simule_validation.tif")
    exporter_raster(lulc_2025_simule_hors_echantillon, profile_2025, chemin_sortie_val)

    # --- Analyse de sensibilite (commentaire 94-c) ---
    sensibilite_variables = analyse_sensibilite_variables(
        lulc_2000, lulc_2015, lulc_2025, variables_arrays, masque_valide, config["classes"], config
    )
    sensibilite_mlp = analyse_sensibilite_mlp(
        lulc_2000, lulc_2015, lulc_2025, variables_arrays, masque_valide, config["classes"], config
    )
    resultats["sensibilite_variables"] = sensibilite_variables
    resultats["sensibilite_mlp"] = sensibilite_mlp

    # ==========================================================================
    # PHASE B -- PROJECTION FINALE : recalibration sur 2015->2025 (periode la
    # plus recente), projection vers 2035 et 2045 a partir de l'etat 2025 reel.
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE B -- PROJECTION FINALE (recalibration 2015-2025)")
    print("=" * 70)

    markov_final = matrice_transition_markov(lulc_2015, lulc_2025, config["classes"], masque_valide)
    print("\nMatrice de transition Markov (probabilites, 2015->2025) :")
    for i, c in enumerate(markov_final["codes"]):
        print(f"  {config['classes'][c]:20s} : {np.round(markov_final['probabilite'][i], 4)}")

    n_pas_calibration_final = config["annee_calibration"] - config["annee_base"]  # 10 ans

    stack_2025, _ = construire_pile_variables(
        variables_arrays, lulc_2025, config["classes"], config["taille_voisinage"]
    )

    mlp_final, scaler_final = entrainer_potentiel_transition(
        stack_2015, lulc_2015, lulc_2025, masque_valide,
        config["fraction_echantillon_entrainement"],
        config["mlp_hidden_layers"], config["mlp_max_iter"], config["mlp_random_state"],
    )

    resultats["markov_projection"] = {
        "codes": markov_final["codes"],
        "probabilite": markov_final["probabilite"].tolist(),
        "comptage": markov_final["comptage"].tolist(),
    }

    for annee_cible in config["annees_cibles"]:
        n_pas_projection = annee_cible - config["annee_calibration"]
        print(f"\n--- Projection {annee_cible} (horizon +{n_pas_projection} ans depuis 2025) ---")

        quotas_matrice = projeter_quotas_markov(markov_final["comptage"], markov_final["codes"],
                                                 n_pas_calibration_final, n_pas_projection)
        quotas_matrice_sensibilite = projeter_quotas_puyravaud_sensibilite(
            markov_final["comptage"], markov_final["codes"], n_pas_calibration_final, n_pas_projection)
        ecarts_sensibilite = comparer_methodes_quotas(quotas_matrice, quotas_matrice_sensibilite,
                                                       markov_final["codes"], config["classes"])
        resultats[f"sensibilite_quotas_{annee_cible}"] = ecarts_sensibilite
        quotas_cible = {}
        for j, code_cible in enumerate(markov_final["codes"]):
            quotas_cible[code_cible] = int(np.sum(quotas_matrice[:, j])
                                            - quotas_matrice[markov_final["codes"].index(code_cible), j])

        proba_map, classes_mlp = predire_probabilites(mlp_final, scaler_final, stack_2025, masque_valide)
        lulc_simule = allouer_ca(lulc_2025, proba_map, classes_mlp, quotas_cible, masque_valide)

        chemin_sortie = os.path.join(config["dossier_sortie"], f"LULC_simule_{annee_cible}.tif")
        exporter_raster(lulc_simule, profile_2025, chemin_sortie)

        resultats[f"quotas_{annee_cible}"] = quotas_cible

    print("\nRAPPEL A INSERER DANS L'ARTICLE :")
    print("Les cartes 2035/2045 sont produites par un modele recalibre sur 2015-2025")
    print("(periode la plus recente). Leur fiabilite attendue est celle mesuree en")
    print("Phase A (validation hors echantillon 2000-2015 -> 2025), et non un chiffre")
    print("recalcule en circuit ferme sur 2015-2025.")

    chemin_rapport = os.path.join(config["dossier_sortie"], "rapport_validation.json")
    with open(chemin_rapport, "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    print(f"\nRapport complet sauvegarde : {chemin_rapport}")

    return resultats


if __name__ == "__main__":
    executer_pipeline(CONFIG)