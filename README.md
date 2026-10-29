# TP5 - AutoML et déploiement

## Présentation
Ce projet compare deux approches de détection de fraude :
- Un modèle manuel basé sur Random Forest.
- Un modèle AutoML optimisé avec FLAML et basé sur XGBoost.

## Dataset
Le projet utilise un jeu de données de transactions contenant 5 000 lignes, dont 568 fraudes.

## Résultats

| Critère | Random Forest | AutoML (XGBoost) |
|---|---:|---:|
| F1-score | 0.7893 | 0.8075 |
| Latence médiane | 185.18 ms | 16.11 ms |
| Taille du modèle | 12.86 Mo | 0.06 Mo |

## Structure
- `donnees.py` : préparation des données.
- `src/automl_deploy.py` : entraînement AutoML.
- `src/modele_manuel.py` : entraînement du modèle manuel.
- `src/promouvoir.py` : promotion des modèles.
- `src/client.py` : test des API de prédiction.
- `reports/tp5_comparaison.md` : comparaison détaillée.

## Installation
```bash
pip install -r requirements.txt
```

## Technologies
Python, FLAML, XGBoost, Scikit-learn et MLflow.