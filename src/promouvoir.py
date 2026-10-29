
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

modeles = ["fraude-automl", "fraude-manuel"]

for nom in modeles:
    versions = client.search_model_versions(f"name='{nom}'")

    if not versions:
        print(f"Aucune version trouvée pour {nom}")
        continue

    derniere_version = max(versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(
        name=nom,
        alias="production",
        version=derniere_version.version
    )

    print(
        f"{nom} : version {derniere_version.version} "
        f"promue avec l'alias production"
    )

print("Promotion terminée.")
