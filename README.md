# Dynamique et modélisation prédictive de l'occupation des terres dans la commune de Campo (Sud-Cameroun)

Code associé à l'article *« Modélisation prédictive des changements d'occupation et d'utilisation des terres dans la commune de Campo (Sud-Cameroun) »* (Tsoala Tchoffo J.H. et al.).

Le dépôt contient :

- les scripts **Google Earth Engine** de classification des cartes d'occupation du sol (2000, 2015, 2025) ;
- le script **Google Earth Engine** de contrôle indépendant des cartes par les produits JRC-TMF et Global Forest Change ;
- le pipeline **Python CA-ANN** de validation et de projection (2035, 2045).

## Structure

```
campo-lulc-ca-ann/
├── gee/
│   ├── 01_classification_LULC_2000.js   Landsat 5 TM + Landsat 7 ETM+
│   ├── 02_classification_LULC_2015.js   Landsat 8 OLI
│   ├── 03_classification_LULC_2025.js   Landsat 8 OLI + Landsat 9 OLI-2
│   └── 04_controle_JRC_TMF_Hansen.js    contrôle indépendant des cartes
├── python/
│   └── pipeline_CA_ANN.py               validation (2015 -> 2025) et projections (2035, 2045)
├── requirements.txt
├── LICENSE
├── .zenodo.json                        métadonnées pour l'archive Zenodo
├── CHANGELOG.md
└── README.md
```

## 1. Classification (Google Earth Engine)

Chaque script construit un composite médian Landsat (réflectance de surface, Collection 2, niveau 2), masque les nuages et ombres avec la bande `QA_PIXEL`, calcule le NDVI, le NDWI, le NDBI et le MNDWI, puis classe l'image par Random Forest (100 arbres) à partir de 7 variables : NIR, SWIR1, SWIR2, NDVI, NDWI, NDBI et MNDWI.

| Date | Fenêtre principale | Fenêtre de secours | Capteurs |
|---|---|---|---|
| 2000 | 01/01/1999 – 31/12/2001 | 01/01/1998 – 01/04/2003 | Landsat 5 TM, Landsat 7 ETM+ |
| 2015 | 01/01/2014 – 31/12/2015 | 01/01/2013 – 31/12/2017 | Landsat 8 OLI |
| 2025 | 01/01/2025 – 31/12/2025 | 01/01/2023 – 30/06/2026 | Landsat 8 OLI, Landsat 9 OLI-2 |

**Avant l'exécution**, importer dans l'éditeur GEE :
- le contour de la commune, nommé `aoi` ;
- les points d'entraînement de chaque classe, avec une propriété `class` de 1 à 6.

Codes des classes : 1 forêt dense, 2 forêt dégradée, 3 agriculture, 4 mangrove, 5 eau, 6 sol nu/bâti.

Les points photo-interprétés sont répartis aléatoirement en 80 % pour l'entraînement et 20 % pour la validation (graine 42). Les cartes sont exportées à 30 m en UTM zone 32N (EPSG:32632).

## 2. Contrôle indépendant (Google Earth Engine)

`04_controle_JRC_TMF_Hansen.js` croise les cartes classifiées avec :
- **JRC Tropical Moist Forest** (Vancutsem et al., 2021) : état de la forêt (intacte, dégradée, déforestée, en régénération) en 2000, 2015 et 2023 ;
- **Global Forest Change** (Hansen et al., 2013) : perte annuelle de couvert arboré.

Avant de l'exécuter, renommer les imports `aoi`, `LULC_2000`, `LULC_2015` et `LULC_2025`.

## 3. Modélisation CA-ANN (Python)

### Installation

```bash
pip install -r requirements.txt
```

### Données attendues

Placer dans un dossier `data/` les rasters **alignés sur la même grille** (30 m, EPSG:32632) :

| Fichier | Contenu |
|---|---|
| `LULC_2000_alignement.tif`, `LULC_2015_alignement.tif`, `LULC_2025_alignement.tif` | cartes classifiées |
| `pente_alignement.tif`, `aspect_alignement.tif`, `dem_alignement.tif` | dérivés du MNT SRTM 30 m |
| `dis_riv_alignement.tif` | distance aux cours d'eau (Atlas forestier interactif du Cameroun, MINFOF et WRI, 2020) |
| `dis_route_alignement.tif` | distance aux routes (Atlas forestier interactif du Cameroun, MINFOF et WRI, 2020) |
| `dist_parc_alignement.tif` | distance à la limite du Parc National de Campo Ma'an |

Les chemins se modifient dans le dictionnaire `CONFIG`, en tête du script.

### Exécution

```bash
python python/pipeline_CA_ANN.py
```

### Étapes du pipeline

1. **Multicolinéarité** : corrélation de Pearson et facteur d'inflation de la variance (VIF). L'exposition est décomposée en cos(exposition) et sin(exposition).
2. **Choix de l'architecture du MLP** parmi (16), (32, 16), (64, 32) et (32, 16, 8), sur la validation interne des seules données 2000-2015.
3. **Validation (phase A)** : calibration sur 2000-2015, simulation de 2025 à partir de 2015, comparaison avec la carte observée de 2025. Les résultats sont donnés en moyenne ± écart-type sur 5 graines aléatoires et comparés à un modèle nul de persistance.
   - La matrice 2000-2015 (15 ans) est ramenée à un pas de 10 ans par puissance matricielle fractionnaire.
   - Les quotas sont calculés **par transition** à partir de l'état de départ : nombre de pixels de la classe i × probabilité de passer de i à j.
4. **Analyse de sensibilité** : retrait de chaque variable à tour de rôle, sur les 5 graines.
5. **Projections (phase B)** : recalibration sur 2015-2025, puis simulation de 2035 et de 2045 en deux pas de 10 ans, le voisinage étant recalculé à chaque pas.

### Fichiers produits (dossier `resultats/`)

| Fichier | Contenu |
|---|---|
| `rapport_validation_v2.json` | multicolinéarité, choix de l'architecture, indicateurs de validation, modèle nul, AUC, sensibilité, projections |
| `superficies_validation_2025.csv` | superficies observées et simulées en 2025 |
| `superficies_projections.csv` | superficies 2025, 2035 et 2045 |
| `matrice_P_2000_2015_normalisee_10ans.csv`, `matrice_P_2015_2025.csv` | matrices de transition utilisées |
| `quotas_validation_2015_2025_pixels.csv`, `quotas_2035_pixels.csv`, `quotas_2045_pixels.csv` | quotas par transition |
| `courbes_ROC_validation.png` | courbes ROC du potentiel de transition |
| `LULC_2025_simule_validation.tif`, `LULC_simule_2035.tif`, `LULC_simule_2045.tif` | cartes simulées |

## Données

Les rasters d'entrée (cartes LULC classifiées et variables explicatives) sont trop volumineux pour GitHub. Ils sont disponibles sur demande auprès de l'auteure ou via ce dossier : https://drive.google.com/drive/folders/1jsOv_me9SG2ZKRyqX9koboM0IGldXNAy?usp=drive_link

## Versions

- **v2.0.0** : version révisée après évaluation de l'article (voir `CHANGELOG.md`). C'est la version à utiliser.
- **v1.0.0** : version de la première soumission, archivée sur Zenodo (DOI : 10.5281/zenodo.22765891). Elle contient une erreur dans le calcul des quotas de Markov, corrigée en v2.0.0.

## Citation

Si vous utilisez ce code, merci de citer :

> Tsoala Tchoffo, J.H., Opelele, M.G., Avoto, E.Y.E., Ondon, N.C.B., Tshimanga, M.R., Bolaluambe, P.C., Lele, N.B., Lutete, L.E. et Semeki, J. *Modélisation prédictive des changements d'occupation et d'utilisation des terres dans la commune de Campo (Sud-Cameroun) à l'aide de Google Earth Engine et du modèle automates cellulaires–réseaux de neurones artificiels (CA-ANN)*.

ainsi que l'archive Zenodo de la version utilisée.

## Auteurs

Tsoala Tchoffo Jessy Houston, Opelele Michel Gustave, Avoto Essi Yann Edwin, Ondon Nkoua Cedrick Belmich, Tshimanga Muamba Raphael, Bolaluambe Papy Claude, Lele Nyami Bonaventure, Lutete Landu Eric, Semeki Ngabinzeke Jean.

## Licence

Code distribué sous licence MIT (voir le fichier `LICENSE`).

## Contact

Jessy Houston Tsoala Tchoffo — jessytsoala788@gmail.com
