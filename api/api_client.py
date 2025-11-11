import requests
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

def call_segmentation_api(image_path, api_url="http://127.0.0.1:5000/predict"):
    """
    Appelle l'API de segmentation avec une image et retourne la réponse.

    Args:
        image_path (str): Le chemin vers le fichier image à envoyer.
        api_url (str): L'URL de l'endpoint de prédiction de l'API.

    Returns:
        dict: La réponse JSON de l'API, ou None en cas d'erreur.
    """
    if not os.path.exists(image_path):
        print(f"Erreur : Le fichier image n'a pas été trouvé à l'adresse '{image_path}'")
        return None

    print(f"Envoi de l'image '{os.path.basename(image_path)}' à l'API à l'adresse {api_url}...")

    try:
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(api_url, files=files)
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)

        print("Réponse reçue avec succès !")
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'appel à l'API : {e}")
        return None

def display_results(original_image_path, api_response):
    """Affiche l'image originale et l'image de segmentation reçue de l'API."""
    original_image = Image.open(original_image_path)

    # Décode l'image de segmentation depuis la chaîne Base64
    img_data_str = api_response['segmented_image'].split(',')[1]
    img_data = base64.b64decode(img_data_str)
    segmented_image = Image.open(io.BytesIO(img_data))

    # Affiche les images côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Image Originale")
    axes[0].axis('off')

    axes[1].imshow(segmented_image)
    axes[1].set_title("Prédiction de Segmentation")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python api_client.py <chemin_vers_image>")
        sys.exit(1)

    image_file = sys.argv[1]
    
    # 1. Appeler l'API
    response_data = call_segmentation_api(image_file)

    if response_data:
        # 2. Afficher les statistiques
        print("\n--- Statistiques de Segmentation ---")
        for stat in sorted(response_data['statistics'], key=lambda x: x['percentage'], reverse=True):
            print(f"- {stat['class_name']:<15}: {stat['percentage']:.2f}%")

        # 3. Afficher les images
        print("\nAffichage des résultats visuels...")
        display_results(image_file, response_data)