import requests
import time
import os
import sys

# --- Configuration ---
API_URL = "http://localhost:8000/predict"  # CORRECTION: L'endpoint est /predict
IMAGE_PATH = "test_images/sample_image.jpg"  # Créez un dossier test_images/ avec une image dedans


def test_single_request():
    """Envoie une seule requête à l'API et mesure le temps de réponse."""
    # --- Validation des prérequis ---
    if not os.path.exists(IMAGE_PATH):
        print(
            f"Erreur: L'image de test '{IMAGE_PATH}' n'a pas été trouvée. Assurez-vous de l'avoir créée."
        )
        sys.exit(1)

    print(f"Envoi d'une requête à {API_URL}...")

    with open(IMAGE_PATH, "rb") as image_file:
        files = {"image": (os.path.basename(IMAGE_PATH), image_file, "image/jpeg")} # CORRECTION: Le nom du champ est 'image'

        try:
            start_time = time.time()
            # L'authentification par clé API n'est pas utilisée, on la retire.
            response = requests.post(API_URL, files=files, timeout=60)
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
