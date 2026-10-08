"""
==============================================================================
 PIPELINE CA-ANN POUR LA PREDICTION LULC -- VERSION 2 (CORRIGEE)
------------------------------------------------------------------------------
 Auteure : Jessy Houston Tsoala Tchoffo -- Commune de Campo, Sud Cameroun

 CORRECTIONS PAR RAPPORT A LA VERSION 1 (reponse aux relecteurs)
 ----------------------------------------------------------------
 [C1] QUOTAS DE MARKOV (erreur principale de la v1)
      v1 : les quotas etaient les GAINS BRUTS de chaque classe, calcules sur la
           distribution du DEBUT de la periode de calibration (2000 ou 2015),
           et seuls les gains etaient controles lors de l'allocation.
           -> superficies simulees incoherentes avec la matrice de Markov
              (ex. 2035 : foret dense 209 529 ha au lieu de ~228 300 ha).
      v2 : quotas PAR TRANSITION n_ij = N_i(t0) x P^k_ij, ou N_i(t0) est l'etat
           de la carte DE DEPART de la simulation (2015 en validation, 2025 en
           projection) et P^k la matrice normalisee au pas de simulation.
           L'allocation respecte chaque transition i -> j : les superficies
           simulees reproduisent exactement la chaine de Markov.
 [C2] Normalisation de la matrice 2000-2015 (15 ans) au pas de 10 ans par
      puissance matricielle fractionnaire P^(10/15) (deja presente en v1,
      desormais documentee et exportee).
 [C3] Simulation 2045 en deux pas de 10 ans (2025 -> 2035 -> 2045), avec mise a
      jour du voisinage a partir de la carte simulee 2035 (composante CA).
 [C4] Exposition (aspect, variable circulaire) remplacee par cos(aspect)
      (northness) et sin(aspect) (eastness).
 [C5] Distance au parc signee (negative a l'interieur) si un masque du parc
      est fourni (CONFIG["masque_parc"]).
 [C6] Choix de l'architecture sur la validation INTERNE du MLP (donnees
      2000->2015 uniquement), puis evaluation UNIQUE sur 2025. Les indicateurs
      de validation sont rapportes en moyenne +/- ecart-type sur plusieurs
      graines aleatoires.
 [C7] Sur-echantillonnage plafonne (facteur de duplication maximal).
 [C8] Exports : quotas utilises, superficies par classe, decomposition
      succes / omissions / fausses alarmes, modele nul, courbes ROC.
==============================================================================
"""

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
from scipy.linalg import fractional_matrix_power
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, csv, io, contextlib

D = os.path.join("data", "")   # dossier des rasters alignes (voir README)

CONFIG = {
    "lulc_2000": D + "LULC_2000_alignement.tif",
    "lulc_2015": D + "LULC_2015_alignement.tif",
    "lulc_2025": D + "LULC_2025_alignement.tif",
    "variables": {
        "pente": D + "pente_alignement.tif",
        "aspect": D + "aspect_alignement.tif",          # transformee en cos/sin [C4]
        "dem": D + "dem_alignement.tif",
        "dist_rivieres": D + "dis_riv_alignement.tif",
        "dist_routes": D + "dis_route_alignement.tif",
        "dist_parc": D + "dist_parc_alignement.tif",
        # Variables optionnelles recommandees par le relecteur (decommenter si disponibles) :
        # "dist_villages": D + "dist_villages_alignement.tif",
        # "concession": D + "concession_camvert_alignement.tif",   # 1 = dans la concession, 0 = hors
    },
    # [C5] Raster 1 = interieur du parc, 0 = exterieur (meme grille). None si indisponible.
    "masque_parc": None,   # ex. D + "parc_masque_alignement.tif"

    "classes": {1: "Foret dense", 2: "Foret degradee", 3: "Agriculture",
                4: "Mangrove", 5: "Eau", 6: "Sol nu/Bati"},

    "annee_validation_debut": 2000, "annee_validation_fin": 2015, "annee_test": 2025,
    "annee_base": 2015, "annee_calibration": 2025, "annees_cibles": [2035, 2045],
    "pas_simulation": 10,                    # [C3] pas de 10 ans

    "taille_voisinage": 3,                   # fenetre 3x3
    "architectures": [(16,), (32, 16), (64, 32), (32, 16, 8)],
    "graines": [0, 1, 42, 123, 2026],        # [C6]
    "mlp_max_iter": 300,
    "n_max_par_classe": 45000,
    "facteur_duplication_max": 10,           # [C7]
    "valeur_nodata_entiere": 32767,
    "transitions_roc": [(1, 2), (1, 3), (1, None)],
    "executer_sensibilite_variables": True,

    "dossier_sortie": "resultats",
}


# ============================================================================== lecture
def lire_raster(chemin):
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    with rasterio.open(chemin) as src:
        return src.read(1), src.profile, src.nodata


def verifier_alignement(profiles):
    ref = profiles[0]
    for i, p in enumerate(profiles[1:], start=1):
        if (p["width"], p["height"]) != (ref["width"], ref["height"]) or p["crs"] != ref["crs"] \
                or p["transform"] != ref["transform"]:
            raise ValueError(f"Raster #{i} non aligne sur la grille de reference.")
    print("Alignement verifie.")


def preparer_variables(variables_brutes, config):
    """[C4] aspect -> northness/eastness ; [C5] distance au parc signee."""
    out = {}
    for nom, arr in variables_brutes.items():
        if nom == "aspect":
            a = arr.astype(np.float32)
            plat = (a < 0)                       # -1 = zone plate dans ArcGIS
            rad = np.deg2rad(np.where(plat, 0, a))
            out["aspect_cos_northness"] = np.where(plat, 0, np.cos(rad)).astype(np.float32)
            out["aspect_sin_eastness"] = np.where(plat, 0, np.sin(rad)).astype(np.float32)
        else:
            out[nom] = arr.astype(np.float32)
    if config.get("masque_parc") and "dist_parc" in out:
        masque, _, _ = lire_raster(config["masque_parc"])
        out["dist_parc"] = np.where(masque == 1, -out["dist_parc"], out["dist_parc"]).astype(np.float32)
        print("Distance au parc signee (negative a l'interieur).")
    elif "dist_parc" in out:
        print("ATTENTION : masque du parc non fourni, distance au parc NON signee.")
    return out


def construire_masque_valide(lulc_arrays, nodata_list, variables_arrays, classes, config):
    m = np.ones_like(lulc_arrays[0], dtype=bool)
    for arr, nd in zip(lulc_arrays, nodata_list):
        m &= np.isin(arr, list(classes.keys()))
        if nd is not None:
            m &= (arr != nd)
    for arr in variables_arrays.values():
        m &= ~np.isnan(arr) & (np.abs(arr) < 1e29) & (arr != config["valeur_nodata_entiere"])
    return m


# ============================================================================== Markov
def matrice_comptage(lulc_t1, lulc_t2, codes, masque):
    idx = {c: i for i, c in enumerate(codes)}
    a = np.vectorize(idx.get)(lulc_t1[masque]); b = np.vectorize(idx.get)(lulc_t2[masque])
    n = len(codes)
    return np.bincount(a * n + b, minlength=n * n).reshape(n, n).astype(np.int64)


def matrice_probabilites(comptage, duree_calibration, pas):
    """[C2] P normalisee au pas de simulation : P^(pas / duree_calibration)."""
    tot = comptage.sum(axis=1, keepdims=True).astype(float)
    P = np.divide(comptage, tot, out=np.zeros_like(comptage, dtype=float), where=tot > 0)
    vide = tot.ravel() == 0
    P[vide, :] = 0; P[vide, vide] = 1                    # classe absente : persistance
    if duree_calibration != pas:
        P = np.real(fractional_matrix_power(P, pas / duree_calibration))
        P = np.clip(P, 0, None)
        s = P.sum(axis=1, keepdims=True)
        P = np.divide(P, s, out=np.eye(len(P)), where=s > 0)
    return P


def quotas_transitions(etat_depart, P, codes, masque):
    """[C1] n_ij = N_i(t0) x P_ij, arrondi en conservant exactement N_i(t0) par ligne."""
    N = np.array([np.sum((etat_depart == c) & masque) for c in codes], dtype=np.int64)
    brut = N[:, None] * P
    q = np.floor(brut).astype(np.int64)
    for i in range(len(codes)):                       # methode du plus fort reste
        manque = N[i] - q[i].sum()
        if manque > 0:
            q[i, np.argsort(-(brut[i] - q[i]))[:manque]] += 1
    return q, N


# ============================================================================== variables + MLP
def densite_voisinage(lulc, code, taille):
    return uniform_filter((lulc == code).astype(np.float32), size=taille, mode="nearest")


def construire_pile(variables_arrays, lulc, classes, taille):
    piles = list(variables_arrays.values()) + [densite_voisinage(lulc, c, taille) for c in classes]
    return np.stack(piles, axis=-1)


def entrainer_mlp(stack, lulc_t2, masque, hidden, graine, config, silencieux=False):
    """Cible du MLP = classe a t2 ; variables = etat a t1 (variables + voisinage a t1)."""
    nf = stack.shape[-1]
    X = stack.reshape(-1, nf)[masque.ravel()]
    y = lulc_t2.ravel()[masque.ravel()]
    rng = np.random.default_rng(graine)
    idx = []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        n = min(config["n_max_par_classe"], len(ic) * config["facteur_duplication_max"])
        idx.append(rng.choice(ic, size=n, replace=n > len(ic)))
        if not silencieux:
            print(f"  Classe {c} : {len(ic)} pixels -> {n} echantillonnes "
                  f"({'duplication x%.1f' % (n / len(ic)) if n > len(ic) else 'sous-echantillonnage'})")
    idx = np.concatenate(idx); rng.shuffle(idx)
    scaler = StandardScaler().fit(X[idx])
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                        learning_rate_init=0.001, max_iter=config["mlp_max_iter"],
                        random_state=graine, early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=10)
    mlp.fit(scaler.transform(X[idx]), y[idx])
    return mlp, scaler


def predire_probabilites(mlp, scaler, stack, masque):
    H, W, nf = stack.shape
    proba = np.zeros((H * W, len(mlp.classes_)), dtype=np.float32)
    v = masque.ravel()
    proba[v] = mlp.predict_proba(scaler.transform(stack.reshape(-1, nf)[v]))
    return proba.reshape(H, W, -1), list(mlp.classes_)


# ============================================================================== allocation [C1]
def allouer_par_transition(lulc_t0, proba_map, classes_mlp, quotas, codes, masque, max_tours=6):
    """
    Alloue exactement n_ij pixels de la classe i vers la classe j (i != j).
    Pour chaque transition, les pixels candidats de la classe i sont classes par
    probabilite MLP de la classe j ; les conflits (un pixel candidat a plusieurs
    transitions) sont resolus globalement par ordre decroissant de probabilite.
    """
    H, W = lulc_t0.shape
    flat = lulc_t0.ravel().copy()
    mflat = masque.ravel()
    pflat = proba_map.reshape(-1, proba_map.shape[-1])
    col = {c: classes_mlp.index(c) for c in codes if c in classes_mlp}
    restant = quotas.copy(); np.fill_diagonal(restant, 0)
    deja = np.zeros(flat.size, dtype=bool)

    for tour in range(max_tours):
        cand_pix, cand_j, cand_s = [], [], []
        for i, ci in enumerate(codes):
            pix_i = np.where(mflat & (lulc_t0.ravel() == ci) & ~deja)[0]
            if pix_i.size == 0:
                continue
            for j, cj in enumerate(codes):
                n = restant[i, j]
                if i == j or n <= 0 or cj not in col:
                    continue
                s = pflat[pix_i, col[cj]]
                k = min(pix_i.size, int(n * (3 * (tour + 1))))
                top = np.argpartition(-s, k - 1)[:k] if k < pix_i.size else np.arange(pix_i.size)
                cand_pix.append(pix_i[top]); cand_j.append(np.full(top.size, j)); cand_s.append(s[top])
        if not cand_pix:
            break
        P_ = np.concatenate(cand_pix); J_ = np.concatenate(cand_j); S_ = np.concatenate(cand_s)
        ordre = np.argsort(-S_, kind="stable")
        codes_arr = np.array(codes)
        for k in ordre:
            p = P_[k]
            if deja[p]:
                continue
            i = codes.index(flat[p]); j = J_[k]
            if restant[i, j] > 0:
                flat[p] = codes_arr[j]; deja[p] = True; restant[i, j] -= 1
        if restant.sum() == 0:
            break
    if restant.sum() > 0:
        print(f"  ATTENTION : {int(restant.sum())} pixels de quotas non alloues (candidats insuffisants).")
    return flat.reshape(H, W), restant


# ============================================================================== indicateurs
def matrice_confusion(ref, sim, masque, codes):
    return matrice_comptage(ref, sim, codes, masque)


def indicateurs(ref, sim, act, masque, codes):
    M = matrice_confusion(ref, sim, masque, codes).astype(float); n = M.sum()
    po = np.trace(M) / n; r = M.sum(1) / n; c = M.sum(0) / n; pe = np.sum(r * c)
    pmax = np.sum(np.minimum(r, c))
    a, s, t = ref[masque], sim[masque], act[masque]
    ch_r, ch_s = a != t, s != t
    hits = int(np.sum(ch_r & ch_s & (a == s)))
    wrong = int(np.sum(ch_r & ch_s & (a != s)))
    misses = int(np.sum(ch_r & ~ch_s)); fa = int(np.sum(~ch_r & ch_s))
    d = hits + wrong + misses + fa
    q = 0.5 * np.sum(np.abs(r - c))                       # Pontius & Millones (2011)
    return {"precision_globale": 100 * po,
            "kappa_general": (po - pe) / (1 - pe),
            "kappa_histogramme": (pmax - pe) / (1 - pe),
            "kappa_lieu": (po - pe) / (pmax - pe) if pmax != pe else float("nan"),
            "desaccord_quantite": 100 * q, "desaccord_allocation": 100 * (1 - po - q),
            "succes": hits, "mauvaise_classe": wrong, "omissions": misses, "fausses_alarmes": fa,
            "figure_of_merit": hits / d if d else float("nan")}


def superficies(lulc, masque, codes):
    return {c: float(np.sum((lulc == c) & masque) * 0.09) for c in codes}


def evaluer_roc(proba_map, classes_mlp, lulc_debut, lulc_fin, masque, transitions, classes, chemin):
    res = {}; fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for dep, arr in transitions:
        m = masque & (lulc_debut == dep)
        if arr is None:
            nom = f"{classes[dep]} -> toute autre classe"
            y = (lulc_fin[m] != dep).astype(np.int8); s = 1 - proba_map[..., classes_mlp.index(dep)][m]
        else:
            nom = f"{classes[dep]} -> {classes[arr]}"
            y = (lulc_fin[m] == arr).astype(np.int8); s = proba_map[..., classes_mlp.index(arr)][m]
        if 0 < y.sum() < y.size:
            auc = float(roc_auc_score(y, s)); f, t, _ = roc_curve(y, s)
            ax.plot(f, t, lw=2, label=f"{nom} (AUC = {auc:.3f})")
            res[nom] = {"auc": auc, "pixels_changes": int(y.sum()), "pixels_testes": int(y.size)}
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatoire (AUC = 0,5)")
    ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(chemin, dpi=300); plt.close(fig)
    return res


def exporter_raster(arr, profile, chemin):
    p = profile.copy(); p.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(chemin, "w", **p) as dst:
        dst.write(arr.astype(rasterio.uint8), 1)


def exporter_csv(chemin, entetes, lignes):
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";"); w.writerow(entetes); w.writerows(lignes)


def diagnostiquer_multicolinearite(variables_arrays, masque):
    noms = list(variables_arrays)
    X = np.stack([variables_arrays[n][masque] for n in noms], 1).astype(np.float64)
    R = np.corrcoef(X, rowvar=False); vif = {}
    for i, n in enumerate(noms):
        Xd = np.column_stack([np.ones(len(X)), np.delete(X, i, 1)])
        b = np.linalg.lstsq(Xd, X[:, i], rcond=None)[0]
        r2 = 1 - np.sum((X[:, i] - Xd @ b) ** 2) / np.sum((X[:, i] - X[:, i].mean()) ** 2)
        vif[n] = 1 / (1 - r2)
    print("\nVIF :", {k: round(v, 2) for k, v in vif.items()})
    print("Paires |r| > 0,7 :", [(noms[i], noms[j], round(R[i, j], 3)) for i in range(len(noms))
                                 for j in range(i + 1, len(noms)) if abs(R[i, j]) > 0.7])
    return {"noms": noms, "correlation": R.tolist(), "vif": vif}


# ============================================================================== Phase A
def simuler_validation(lulc_2000, lulc_2015, lulc_2025, variables, masque, config, hidden, graine,
                       silencieux=True):
    codes = sorted(config["classes"])
    duree = config["annee_validation_fin"] - config["annee_validation_debut"]
    pas = config["annee_test"] - config["annee_validation_fin"]
    P = matrice_probabilites(matrice_comptage(lulc_2000, lulc_2015, codes, masque), duree, pas)
    q, _ = quotas_transitions(lulc_2015, P, codes, masque)            # [C1] etat de depart = 2015
    s00 = construire_pile(variables, lulc_2000, codes, config["taille_voisinage"])
    s15 = construire_pile(variables, lulc_2015, codes, config["taille_voisinage"])
    mlp, sc = entrainer_mlp(s00, lulc_2015, masque, hidden, graine, config, silencieux)
    proba, cm = predire_probabilites(mlp, sc, s15, masque)
    sim, _ = allouer_par_transition(lulc_2015, proba, cm, q, codes, masque)
    return sim, proba, cm, q, P, mlp


def executer_pipeline(config):
    os.makedirs(config["dossier_sortie"], exist_ok=True); out = config["dossier_sortie"]
    codes = sorted(config["classes"]); noms = [config["classes"][c] for c in codes]
    l00, p00, n00 = lire_raster(config["lulc_2000"])
    l15, p15, n15 = lire_raster(config["lulc_2015"])
    l25, p25, n25 = lire_raster(config["lulc_2025"])
    brutes, profs = {}, []
    for n, ch in config["variables"].items():
        a, p, _ = lire_raster(ch); brutes[n] = a; profs.append(p)
    verifier_alignement([p00, p15, p25] + profs)
    variables = preparer_variables(brutes, config)
    masque = construire_masque_valide([l00, l15, l25], [n00, n15, n25], variables, config["classes"], config)
    print(f"Pixels valides : {masque.sum()}")
    R = {"multicolinearite": diagnostiquer_multicolinearite(variables, masque)}

    # ---- [C6] choix de l'architecture sur la validation INTERNE (2000 -> 2015 uniquement)
    print("\nChoix de l'architecture (score de validation interne du MLP, donnees 2000->2015) :")
    s00 = construire_pile(variables, l00, codes, config["taille_voisinage"])
    scores = {}
    for h in config["architectures"]:
        sc_g = []
        for g in config["graines"]:
            with contextlib.redirect_stdout(io.StringIO()):
                mlp, _ = entrainer_mlp(s00, l15, masque, h, g, config, silencieux=True)
            sc_g.append(mlp.best_validation_score_)
        scores[str(h)] = (float(np.mean(sc_g)), float(np.std(sc_g)))
        print(f"  {str(h):12s} : {scores[str(h)][0]:.4f} +/- {scores[str(h)][1]:.4f}")
    h_ret = max(config["architectures"], key=lambda h: scores[str(h)][0])
    print(f"Architecture retenue : {h_ret}")
    R["choix_architecture"] = {"scores_validation_interne": scores, "retenue": str(h_ret)}

    # ---- Phase A : validation hors echantillon, moyenne +/- ecart-type sur les graines
    print("\nPHASE A -- validation 2015 -> 2025 (calibration 2000-2015)")
    res_g = []
    for g in config["graines"]:
        sim, proba, cm, q, P, _ = simuler_validation(l00, l15, l25, variables, masque, config, h_ret, g)
        res_g.append(indicateurs(l25, sim, l15, masque, codes))
        if g == config["graines"][0]:
            sim_ref, proba_ref, cm_ref, q_ref, P_ref = sim, proba, cm, q, P
    cles = res_g[0].keys()
    R["validation"] = {k: {"moyenne": float(np.mean([r[k] for r in res_g])),
                           "ecart_type": float(np.std([r[k] for r in res_g]))} for k in cles}
    R["modele_nul_persistance"] = indicateurs(l25, l15, l15, masque, codes)
    for k in ["precision_globale", "kappa_general", "kappa_lieu", "figure_of_merit"]:
        print(f"  {k:20s} : {R['validation'][k]['moyenne']:.4f} +/- {R['validation'][k]['ecart_type']:.4f}"
              f"   (modele nul : {R['modele_nul_persistance'][k]:.4f})")
    exporter_csv(os.path.join(out, "quotas_validation_2015_2025_pixels.csv"), ["de \\ vers"] + noms,
                 [[noms[i]] + list(q_ref[i]) for i in range(len(codes))])
    exporter_csv(os.path.join(out, "matrice_P_2000_2015_normalisee_10ans.csv"), ["de \\ vers"] + noms,
                 [[noms[i]] + [round(v, 6) for v in P_ref[i]] for i in range(len(codes))])
    s_obs, s_sim = superficies(l25, masque, codes), superficies(sim_ref, masque, codes)
    exporter_csv(os.path.join(out, "superficies_validation_2025.csv"),
                 ["classe", "observe_ha", "simule_ha", "ecart_ha", "ecart_%"],
                 [[config["classes"][c], round(s_obs[c], 1), round(s_sim[c], 1),
                   round(s_sim[c] - s_obs[c], 1), round(100 * (s_sim[c] - s_obs[c]) / s_obs[c], 1)]
                  for c in codes])
    R["roc"] = evaluer_roc(proba_ref, cm_ref, l15, l25, masque, config["transitions_roc"],
                           config["classes"], os.path.join(out, "courbes_ROC_validation.png"))
    exporter_raster(sim_ref, p25, os.path.join(out, "LULC_2025_simule_validation.tif"))

    # ---- sensibilite aux variables (comparee au bruit des graines)
    if config["executer_sensibilite_variables"]:
        print("\nSensibilite aux variables (moyenne sur les graines) :")
        sens = {}
        for exclu in list(variables):
            v2 = {k: v for k, v in variables.items() if k != exclu}
            kap = []
            for g in config["graines"]:
                sim, *_ = simuler_validation(l00, l15, l25, v2, masque, config, h_ret, g)
                kap.append((indicateurs(l25, sim, l15, masque, codes)["kappa_general"],
                            indicateurs(l25, sim, l15, masque, codes)["figure_of_merit"]))
            kap = np.array(kap)
            sens[exclu] = {"kappa_moy": float(kap[:, 0].mean()), "kappa_et": float(kap[:, 0].std()),
                           "fom_moy": float(kap[:, 1].mean()), "fom_et": float(kap[:, 1].std())}
            print(f"  sans {exclu:22s}: Kappa {sens[exclu]['kappa_moy']:.4f} +/- {sens[exclu]['kappa_et']:.4f}"
                  f" | FoM {sens[exclu]['fom_moy']:.4f} +/- {sens[exclu]['fom_et']:.4f}")
        R["sensibilite_variables"] = sens

    # ---- Phase B : projection 2035 puis 2045 par pas de 10 ans [C3]
    print("\nPHASE B -- projections (calibration 2015-2025)")
    duree = config["annee_calibration"] - config["annee_base"]
    P_B = matrice_probabilites(matrice_comptage(l15, l25, codes, masque), duree, config["pas_simulation"])
    s15 = construire_pile(variables, l15, codes, config["taille_voisinage"])
    mlp_B, sc_B = entrainer_mlp(s15, l25, masque, h_ret, config["graines"][0], config)
    etat = l25; lignes_sup = [["2025"] + [round(superficies(l25, masque, codes)[c], 1) for c in codes]]
    R["projections"] = {}
    for annee in config["annees_cibles"]:
        q, N = quotas_transitions(etat, P_B, codes, masque)
        stack = construire_pile(variables, etat, codes, config["taille_voisinage"])
        proba, cm = predire_probabilites(mlp_B, sc_B, stack, masque)
        sim, reste = allouer_par_transition(etat, proba, cm, q, codes, masque)
        exporter_raster(sim, p25, os.path.join(out, f"LULC_simule_{annee}.tif"))
        exporter_csv(os.path.join(out, f"quotas_{annee}_pixels.csv"), ["de \\ vers"] + noms,
                     [[noms[i]] + list(q[i]) for i in range(len(codes))])
        sup = superficies(sim, masque, codes)
        lignes_sup.append([str(annee)] + [round(sup[c], 1) for c in codes])
        R["projections"][annee] = {"superficies_ha": {config["classes"][c]: sup[c] for c in codes},
                                   "quotas_non_alloues": int(reste.sum())}
        print(f"  {annee} : " + ", ".join(f"{config['classes'][c]} {sup[c]:,.0f} ha" for c in codes))
        etat = sim
    exporter_csv(os.path.join(out, "superficies_projections.csv"), ["annee"] + noms, lignes_sup)
    exporter_csv(os.path.join(out, "matrice_P_2015_2025.csv"), ["de \\ vers"] + noms,
                 [[noms[i]] + [round(v, 6) for v in P_B[i]] for i in range(len(codes))])

    with open(os.path.join(out, "rapport_validation_v2.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nTermine. Resultats dans : {out}")
    return R


if __name__ == "__main__":
    executer_pipeline(CONFIG)
