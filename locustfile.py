from locust import HttpUser, task, between
import os

# --- Configuration ---
IMAGE_PATH = "test_images/sample_image.jpg"
API_KEY_PATH = "internal_api_key.txt"

# Charger la clé API et l'image une seule fois au démarrage
with open(API_KEY_PATH, "r") as f:
    API_KEY = f.read().strip()

with open(IMAGE_PATH, "rb") as f:
    IMAGE_BYTES = f.read()


class SegmentationUser(HttpUser):
    """
    Utilisateur virtuel qui envoie des images à l'API de segmentation.
    """

    # Temps d'attente entre 1 et 3 secondes entre chaque tâche
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    @task
    def segment_image(self):
        headers = {"X-API-Key": API_KEY}
        files = {"file": ("sample.jpg", IMAGE_BYTES, "image/jpeg")}

        self.client.post("/segment/", headers=headers, files=files, name="/segment/")
