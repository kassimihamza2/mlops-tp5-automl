
import os
import time
import pickle

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Création des dossiers
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Configuration MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp5-automl-deploiement")

# Chargement des données
df = pd.read_csv("data/transactions.csv")

X = df.drop(columns=["fraude"])
y = df["fraude"]

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Entraînement manuel
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)

start = time.time()
model.fit(X_train, y_train)
duration = time.time() - start

# Évaluation
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

print("Algorithme : RandomForestClassifier")
print(f"F1 manuel : {f1:.4f}")
print(f"Durée d'entraînement : {duration:.2f} secondes")

# Sauvegarde locale
with open("models/model_manuel.pkl", "wb") as f:
    pickle.dump(model, f)

# Enregistrement dans MLflow
with mlflow.start_run(run_name="random-forest-manuel") as run:
    mlflow.log_param("outil", "scikit-learn")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", 16)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("f1", float(f1))
    mlflow.log_metric("duree_entrainement_s", float(duration))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        serialization_format="cloudpickle",
        registered_model_name="fraude-manuel"
    )

    print(f"Run MLflow : {run.info.run_id}")

print("Entraînement manuel terminé.")
