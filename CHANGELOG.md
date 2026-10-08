# Journal des modifications

## Version 2.0.0

Version révisée à la suite de l'évaluation de l'article.

### Corrigé
- **Quotas de Markov.** La version 1.0.0 utilisait comme quotas les gains bruts de chaque classe, calculés sur la distribution du début de la période de calibration, et ne contrôlait pas les pertes lors de l'allocation. Les superficies simulées ne correspondaient donc pas à la chaîne de Markov. Les quotas sont désormais calculés par transition (classe i vers classe j) à partir de l'état de départ de la simulation, et l'allocation respecte chaque transition.

### Modifié
- L'exposition est remplacée par cos(exposition) et sin(exposition), car c'est une variable circulaire.
- L'architecture du réseau de neurones est choisie sur la validation interne des données 2000-2015 ; la carte de 2025 ne sert plus qu'à la validation.
- Les indicateurs de validation et l'analyse de sensibilité sont donnés en moyenne ± écart-type sur 5 graines aléatoires.
- La duplication des classes rares est plafonnée à dix fois leur effectif.
- La projection 2045 est obtenue en deux pas de 10 ans (2025 → 2035 → 2045), avec mise à jour du voisinage.

### Ajouté
- Comparaison avec un modèle nul de persistance.
- Décomposition du désaccord en quantité et allocation, et en succès, omissions et fausses alarmes.
- Courbes ROC et AUC du potentiel de transition.
- Export des matrices de transition et des quotas utilisés (CSV).
- Scripts Google Earth Engine de classification (2000, 2015, 2025).
- Script Google Earth Engine de contrôle indépendant par les produits JRC-TMF et Global Forest Change.

## Version 1.0.0

Version initiale associée à la première soumission de l'article.
