
# Modélisation prédictive des changements d'occupation et d'utilisation des terres — Commune de Campo (Sud-Cameroun)

Ce dépôt contient le code utilisé pour l'analyse et la prédiction des changements d'occupation et d'utilisation des terres (LULC) dans la commune de Campo (Sud-Cameroun) entre 2000 et 2045, développé dans le cadre d'un mémoire de Master en Géomatique et Changement Climatique.

Deux volets méthodologiques sont couverts :
1. **Classification et analyse diachronique** (Google Earth Engine) : classification supervisée Random Forest des images Landsat pour 2000, 2015 et 2025, et calcul des matrices de transition.
2. **Modélisation prédictive CA-ANN** (Python) : couplage automates cellulaires–réseau de neurones artificiels pour projeter l'occupation du sol à l'horizon 2035 et 2045, avec une validation hors échantillon rigoureuse.

## Structure du dépôt

```
├── gee/
│   └── classification_LULC.js       # Script Google Earth Engine (classification Random Forest, 2000/2015/2025)
├── python/
│   └── pipeline_CA_ANN.py           # Pipeline complet CA-ANN (voir description ci-dessous)
├── data/                            # Rasters d'entrée (voir note sur les données volumineuses)
├── outputs/                         # Cartes simulées produites (2025 validation, 2035, 2045)
├── requirements.txt                 # Dépendances Python
└── README.md
```

> **Note sur les données volumineuses** : les rasters LULC classifiés et les variables explicatives (~plusieurs dizaines de Mo chacun) ne sont pas hébergés directement sur GitHub. Ils sont disponibles sur demande auprès de l'auteure, ou via [lien Google Drive / autre hébergement à compléter].

## Données utilisées

- **Imagerie satellite* : Landsat 5 TM et 7 ETM+ (2000), Landsat 8 OLI (2015), Landsat 8/9 OLI (2025), Collection 2 Niveau 2 (réflectance de surface), via Google Earth Engine.
- **Classes d'occupation du sol (6)** : forêt dense, forêt dégradée, agriculture, mangrove, eau, sol nu/bâti.
- **Variables explicatives (6)** : pente, exposition, altitude (MNT/SRTM), distance aux rivières, distance aux routes, distance à la limite du Parc National de Campo Ma'an.

## Méthodologie du pipeline CA-ANN

Le script `pipeline_CA_ANN.py` est structuré en deux phases indépendantes, afin d'éviter toute circularité entre calibration et validation :

- **Phase A — Validation hors échantillon** : le modèle (matrice de Markov + perceptron multicouche) est calibré sur la période 2000–2015, puis appliqué pour simuler l'état 2025 à partir de l'état 2015 observé. Cette simulation est comparée à la carte 2025 réellement classifiée, données jamais utilisées pendant la calibration. C'est cette phase qui fournit les indicateurs de performance réels du modèle (Kappa, Figure of Merit de Pontius et Millones).
- **Phase B — Projection finale** : le modèle est recalibré sur la période la plus récente (2015–2025) pour produire les cartes simulées de 2035 et 2045.

Le pipeline inclut également :
- un diagnostic de multicolinéarité (corrélation de Pearson, VIF) entre les variables explicatives ;
- une comparaison entre deux méthodes de projection des quotas de superficie (matrice de Markov élevée à une puissance fractionnaire, retenue comme méthode principale, vs. extrapolation linéaire à la Puyravaud, utilisée uniquement comme test de sensibilité) ;
- une analyse de sensibilité aux variables explicatives et à l'architecture du réseau de neurones ;
- un échantillonnage stratifié équilibré par classe pour limiter le biais d'apprentissage lié au déséquilibre entre classes majoritaires et minoritaires.

## Prérequis techniques

```bash
pip install numpy rasterio scikit-learn scipy
```

Voir `requirements.txt` pour les versions exactes testées.

## Utilisation

1. Adapter les chemins d'accès aux rasters dans le dictionnaire `CONFIG` en tête du script (`lulc_2000`, `lulc_2015`, `lulc_2025`, et le dictionnaire `variables`).
2. Exécuter le pipeline complet :

```bash
python pipeline_CA_ANN.py
```

3. Les résultats (cartes simulées, rapport de validation JSON) sont écrits dans le dossier défini par `CONFIG["dossier_sortie"]`.

## Résultats principaux

- Validation hors échantillon (2000-2015 → test sur 2025) : Kappa général = 0,433 ; Figure of Merit = 0,142.
- Les variables les plus influentes sur la performance du modèle sont le modèle numérique de terrain et la distance à la limite du Parc National de Campo Ma'an.
- Projection tendancielle 2045 : recul de la forêt dense à 71,2 % de la superficie communale (contre 84,1 % en 2025), quasi-triplement de la superficie agricole.

Le détail complet des résultats, la discussion de leurs limites (précision de localisation du changement, biais de quantité par classe, absence d'harmonisation radiométrique inter-capteurs) et les analyses de sensibilité sont présentés dans l'article associé à ce dépôt.

## Citation

Si vous utilisez ce code, merci de citer :

> Tsoala Tchoffo, J.H. et al. (2026). *Modélisation prédictive des changements d'occupation et d'utilisation des terres dans la commune de Campo (Sud-Cameroun) à l'aide de Google Earth Engine et du modèle automates cellulaires–réseaux de neurones artificiels (CA-ANN)*. [Détails de publication à compléter].
>
> Code disponible via ce dépôt, archivé sur Zenodo : [DOI à insérer après création de la release].

## Auteurs

Tsoala Tchoffo Jessy Houston, Avoto Essi Yann Edwin, Ondon Nkoua Cedrick Belmich, Tshimanga Muamba Raphael, Bolaluambe Papy Claude, Lutete Landu Eric, Semeki Jean, Opelele Michel Gustave — Université de Kinshasa, Faculté des Sciences Agronomiques et Environnement.

## Licence

Ce projet est distribué sous licence [MIT](https://opensource.org/licenses/MIT) — libre réutilisation avec attribution. *(À adapter selon tes préférences ; MIT est un choix courant et simple pour du code de recherche.)*

## Remerciements

Voir les remerciements complets dans l'article associé à ce dépôt.
