from dotenv import load_dotenv
load_dotenv()  # Charge les variables depuis le fichier .env

# Le reste de votre code...
import os
import traceback
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import sys
import base64
import io

# Importez les fonctions de votre script de pipeline existant
# Assurez-vous que segmentation_pipeline.py est dans le même répertoire
# ou, mieux, que le package est installé en mode éditable (pip install -e .)
from emmanuel_segmentation_package.pipeline import (
    create_weighted_loss,
    create_segmentation_image,
    load_segmentation_model,
    download_file_from_url,
    preprocess_image,
    predict_segmentation,
    get_class_statistics,
    MODEL_URL,
    CONFIG_URL,
    CLASS_WEIGHTS_URL,
    MODEL_CACHE_DIR
)

# --- Initialisation de l'application Flask ---
app = Flask(__name__)

# --- Variables globales pour le modèle et la configuration ---
# Elles seront initialisées une seule fois au démarrage.
model = None
config = None

def initialize_model():
    """
    Télécharge les fichiers nécessaires et charge le modèle en mémoire.
    Cette fonction est appelée une seule fois avant le démarrage du serveur.
    """
    global model, config

    print("Initialisation du modèle pour l'API...")

    # Crée le répertoire de cache s'il n'existe pas
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Définir les chemins locaux pour les fichiers
    final_model_path = os.path.join(MODEL_CACHE_DIR, "final_optimized_model.keras")
    final_config_path = os.path.join(MODEL_CACHE_DIR, "class_mapping.json")
    class_weights_path = os.path.join(MODEL_CACHE_DIR, "class_weights.json")

    # Télécharger les fichiers s'ils n'existent pas
    if not all([
        download_file_from_url(MODEL_URL, final_model_path),
        download_file_from_url(CONFIG_URL, final_config_path),
        download_file_from_url(CLASS_WEIGHTS_URL, class_weights_path)
    ]):
        print("ERREUR: Échec du téléchargement des fichiers du modèle. L'API ne peut pas démarrer.", file=sys.stderr)
        sys.exit(1)

    # Charger le modèle et la configuration
    try:
        model, config = load_segmentation_model(final_model_path, final_config_path, class_weights_path)
        print("Modèle et configuration chargés avec succès. L'API est prête.")
    except Exception as e:
        print(f"ERREUR: Échec du chargement du modèle : {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


@app.route('/', methods=['GET'])
def index():
    """Route de base pour vérifier que l'API est en cours d'exécution."""
    return jsonify({"status": "API de segmentation en cours d'exécution", "message": "Envoyez une requête POST à /predict avec une image."})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour recevoir une image et retourner la prédiction de segmentation.
    """
    # Vérifier si un fichier image est présent dans la requête
    if 'image' not in request.files:
        return jsonify({"error": "Aucun fichier 'image' n'a été trouvé dans la requête"}), 400

    file = request.files['image']

    # Vérifier si le nom de fichier est vide
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    try:
        # Lire l'image directement depuis le flux de la requête
        image = Image.open(file.stream).convert('RGB')
        original_image_array = np.array(image)

        # Prétraiter l'image (réutiliser la fonction du pipeline)
        processed_image_batch, original_size, _ = preprocess_image(image_array=original_image_array, img_size=(224, 224))

        # Effectuer la prédiction
        prediction_mask = predict_segmentation(model, processed_image_batch, original_size, config)

        # Calculer les statistiques
        class_stats = get_class_statistics(prediction_mask, config)

        # Créer l'image de segmentation colorée
        segmented_image = create_segmentation_image(prediction_mask, config)

        # Encoder l'image en Base64 pour l'inclure dans le JSON
        buffered = io.BytesIO()
        segmented_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Préparer la réponse finale
        response_data = {
            "statistics": class_stats,
            "segmented_image": f"data:image/png;base64,{img_str}"
        }

        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Une erreur est survenue lors du traitement de l'image", "details": str(e)}), 500


if __name__ == '__main__':
    # Initialiser le modèle avant de démarrer le serveur
    initialize_model()

    # Démarrer le serveur Flask
    # host='0.0.0.0' rend l'API accessible depuis d'autres machines sur le réseau
    app.run(host='0.0.0.0', port=5000, debug=True)