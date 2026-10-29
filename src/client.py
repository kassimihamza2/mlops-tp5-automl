
import time
import statistics
import requests

# Données d'une transaction à tester
transaction = {
    "montant": 8500.0,
    "anciennete": 3,
    "freq_24h": 9,
    "nuit": 1
}

def tester_modele(port):
    url = f"http://127.0.0.1:{port}/invocations"
    payload = {"dataframe_records": [transaction]}

    print(f"\n--- Test du modèle sur le port {port} ---")

    # 1. Vérifier la première prédiction
    response = requests.post(url, json=payload, timeout=30)
    print("Code HTTP :", response.status_code)

    if response.status_code != 200:
        print("Erreur :", response.text)
        return

    print("Prédiction :", response.json())

    # 2. Échauffement : 10 requêtes non chronométrées
    for _ in range(10):
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

    # 3. Mesurer 200 requêtes
    temps = []

    for _ in range(200):
        debut = time.perf_counter()
        response = requests.post(url, json=payload, timeout=30)
        fin = time.perf_counter()

        response.raise_for_status()
        temps.append((fin - debut) * 1000)

    mediane = statistics.median(temps)
    p95 = sorted(temps)[int(0.95 * len(temps)) - 1]

    print(f"Médiane : {mediane:.2f} ms")
    print(f"P95 : {p95:.2f} ms")
    print("Nombre de requêtes mesurées :", len(temps))


tester_modele(8100)
tester_modele(8101)
