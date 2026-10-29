
import os
import time
import logging
import pickle

import pandas as pd
import mlflow
import mlflow.sklearn
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Configuration
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

logging.basicConfig(
    filename="reports/flaml_tp5.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp5-automl-deploiement")

# Chargement des données
df = pd.read_csv("data/transactions.csv")
X = df.drop(columns=["fraude"])
y = df["fraude"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Entraînement AutoML
automl = AutoML()
start = time.time()

automl.fit(
    X_train=X_train,
    y_train=y_train,
    task="classification",
    metric="f1",
    time_budget=90,
    seed=42,
    log_file_name="reports/flaml_search.log",
)

duration = time.time() - start

# Évaluation du modèle FLAML
pred_automl = automl.predict(X_test)
f1_automl = f1_score(y_test, pred_automl)

# Récupération de l'estimateur entraîné
estimator = automl.model.estimator
pred_estimator = estimator.predict(X_test)
f1_estimator = f1_score(y_test, pred_estimator)

print(f"Meilleur algorithme : {automl.best_estimator}")
print(f"F1 AutoML : {f1_automl:.4f}")
print(f"F1 estimateur : {f1_estimator:.4f}")
print(f"Durée de recherche : {duration:.2f} secondes")

# Sauvegarde locale
with open("models/model_automl.pkl", "wb") as f:
    pickle.dump(estimator, f)

# Journalisation MLflow
with mlflow.start_run(run_name="automl-90s") as run:
    mlflow.log_param("outil", "flaml")
    mlflow.log_param("budget_s", 90)
    mlflow.log_param("famille", automl.best_estimator)

    for key, value in automl.best_config.items():
        if isinstance(value, (str, int, float, bool)):
            mlflow.log_param(f"cfg_{key}", value)

    mlflow.log_metric("f1", float(f1_estimator))
    mlflow.log_metric("f1_automl", float(f1_automl))
    mlflow.log_metric("duree_recherche_s", float(duration))

    mlflow.log_artifact("reports/flaml_tp5.log", artifact_path="reports")
    mlflow.log_artifact("reports/flaml_search.log", artifact_path="reports")

    mlflow.sklearn.log_model(
        sk_model=estimator,
        name="model",
        serialization_format="cloudpickle",
        registered_model_name="fraude-automl",
    )

    print(f"Run MLflow : {run.info.run_id}")

print("Entraînement AutoML terminé.")
