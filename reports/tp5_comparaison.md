# Modèle manuel ou modèle AutoML ? (TP05)

| Critère | Manuel (RF 500/16) | AutoML (FLAML 90 s) |
|---|---:|---:|
| Famille de modèle | Random Forest | XGBoost |
| F1 sur le jeu de test | 0.7893 | 0.8075 |
| Temps de recherche machine | 1.46 s | 90.20 s |
| Temps humain (estimation) | 6 h | 1,5 h |
| Latence médiane (ms) | 185.18 | 16.11 |
| Latence p95 (ms) | 246.35 | 24.06 |
| Taille du fichier (Mo) | 12.86 | 0.06 |
| Explicabilité | Importance des variables, 500 arbres, profondeur 16 | Importance des variables, 49 arbres, max_leaves = 8 |

*Les latences correspondent à la moyenne de deux séries de 200 requêtes.*

## Décision

Nous choisissons le modèle AutoML basé sur XGBoost pour la production.
Il obtient un meilleur F1-score que le modèle manuel.
Sa taille est également beaucoup plus réduite.
Sa latence est nettement inférieure lors de nos tests locaux.
Malgré un temps de recherche plus long, il présente le meilleur compromis pour notre déploiement.
