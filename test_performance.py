import requests
import time
import os
import sys

# --- Configuration ---
API_URL = "http://localhost:8000/segment/"
IMAGE_PATH = "test_images/sample_image.jpg"  # Créez un dossier test_images/ avec une image dedans
API_KEY_PATH = "internal_api_key.txt"


def test_single_request():
    """Envoie une seule requête à l'API et mesure le temps de réponse."""
    # --- Validation des prérequis ---
    if not os.path.exists(IMAGE_PATH):
        print(
            f"Erreur: L'image de test '{IMAGE_PATH}' n'a pas été trouvée. Assurez-vous de l'avoir créée."
        )
        sys.exit(1)

    if not os.path.exists(API_KEY_PATH):
        print(f"Erreur: Le fichier de clé API '{API_KEY_PATH}' n'a pas été trouvé.")
        sys.exit(1)

    with open(API_KEY_PATH, "r") as f:
        api_key = f.read().strip()

    headers = {"X-API-Key": api_key}
    print(f"Envoi d'une requête à {API_URL}...")

    with open(IMAGE_PATH, "rb") as image_file:
        files = {"file": (os.path.basename(IMAGE_PATH), image_file, "image/jpeg")}

        try:
            start_time = time.time()
            response = requests.post(
                API_URL, headers=headers, files=files, timeout=60
            )  # Ajout d'un timeout
            end_time = time.time()

            # Vérifier si la requête a réussi
            response.raise_for_status()

            print(f"Statut de la réponse: {response.status_code} (OK)")
            print(f"Temps de réponse (latence): {end_time - start_time:.4f} secondes")

        except requests.exceptions.RequestException as e:
            print(f"\nERREUR: La requête a échoué. {e}")
            print(
                "Vérifiez que vos conteneurs Docker sont bien en cours d'exécution avec 'docker-compose up'."
            )


if __name__ == "__main__":
    test_single_request()
