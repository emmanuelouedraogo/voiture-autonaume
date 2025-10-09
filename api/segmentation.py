import numpy as np
import cv2
import tensorflow as tf
import os
import json
import requests
from tqdm import tqdm


def download_file(url: str, destination: str):
    """Télécharge un fichier depuis une URL vers une destination, avec une barre de progression."""
    if os.path.exists(destination):
        print(
            f"Le fichier {os.path.basename(destination)} existe déjà. Téléchargement ignoré."
        )
        return

    print(f"Téléchargement de {os.path.basename(destination)} depuis {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kio

        with open(destination, "wb") as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)

        if total_size != 0 and bar.n != total_size:
            print("ERREUR: Le téléchargement semble incomplet.")
        else:
            print("Téléchargement terminé.")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Échec du téléchargement de {url}: {e}") from e


# --- 0. Définition des chemins de manière dynamique ---
# Les URLs sont lues depuis les variables d'environnement de github.
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v.0.0.1/best_model_final.keras",
)
CLASS_MAPPING_URL = os.getenv(
    "CLASS_MAPPING_URL",
    "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v.0.0.1/class_mapping.json",
)

# Définir le répertoire des modèles dans le home de l'utilisateur pour éviter les problèmes de permission.
# os.path.expanduser("~") retourne '/home/appuser' dans le conteneur Docker.
# C'est un emplacement sûr pour écrire des fichiers.
HOME_DIR = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME_DIR, "models")

print(f"Le répertoire des modèles est configuré sur : {MODELS_DIR}")


def load_segmentation_model():
    """
    Charge le modèle Keras, le mapping de classes et prépare les fonctions optimisées.
    Cette fonction est appelée une seule fois au démarrage de l'API.
    """
    predict_fn = None
    color_map = None
    img_height, img_width = 256, 512  # Tailles par défaut

    try:
        # S'assurer que le dossier des modèles existe
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Définir les chemins de destination et télécharger les fichiers si nécessaire
        model_path = os.path.join(MODELS_DIR, "best_model_final.keras")
        class_mapping_path = os.path.join(MODELS_DIR, "class_mapping.json")
        download_file(MODEL_URL, model_path)
        download_file(CLASS_MAPPING_URL, class_mapping_path)
        # --- Charger le modèle Keras depuis un fichier unique ---

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier modèle non trouvé: {model_path}")

        # Charger le modèle Keras complet
        model = tf.keras.models.load_model(model_path)

        _, img_height, img_width, _ = model.input_shape
        print(f"Modèle chargé. Taille d'entrée attendue : ({img_height}, {img_width})")

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[1, img_height, img_width, 3], dtype=tf.float32)
            ]
        )
        def predict_function(tensor):
            return model(tensor, training=False)

        predict_fn = predict_function
        print("Fonction de prédiction compilée avec succès.")

        # Charger le fichier de mapping des classes (couleurs)
        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(
                f"Fichier de mapping non trouvé: {class_mapping_path}"
            )

        with open(class_mapping_path, "r") as f:
            raw_mapping = json.load(f)

        class_mapping_dict = {int(k): v for k, v in raw_mapping.items() if k.isdigit()}

        # Vérifier si le dictionnaire de mapping est vide après le filtrage
        if not class_mapping_dict:
            raise ValueError(
                "Le fichier de mapping des classes ne contient aucune clé numérique valide."
            )

        max_class_index = max(class_mapping_dict.keys())
        color_map = np.zeros((max_class_index + 1, 3), dtype=np.uint8)
        for class_index, color in class_mapping_dict.items():
            color_map[class_index] = color
        print("Table de correspondance des couleurs créée avec succès.")

    except Exception as e:
        print(f"ERREUR critique lors du chargement du modèle ou du mapping : {e}")
        # Relancer l'exception pour que le lifespan de FastAPI échoue et arrête l'application
        raise RuntimeError(f"Échec du chargement du modèle: {e}") from e

    return predict_fn, color_map


def segment_image(
    image_bytes: bytes, predict_fn: callable, color_map: np.ndarray
) -> np.ndarray:
    """
    Prend une image en bytes, la segmente avec le modèle Keras et retourne un masque coloré.
    """
    if predict_fn is None or color_map is None:
        raise RuntimeError(
            "Le modèle ou le mapping de classes n'a pas pu être chargé. Vérifiez les logs du serveur."
        )

    # --- 2. Prétraitement de l'image ---
    # Récupérer la taille d'entrée depuis la signature de la fonction de prédiction
    _, img_height, img_width, _ = predict_fn.input_signature[0].shape

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Utiliser une interpolation plus rapide pour le redimensionnement
    img_resized = cv2.resize(
        img_rgb, (img_width, img_height), interpolation=cv2.INTER_NEAREST
    )

    # Normaliser et ajouter la dimension du batch
    input_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Convertir en tenseur TensorFlow
    input_tensor = tf.constant(input_array, dtype=tf.float32)

    # --- 3. Prédiction (optimisée) ---
    # Utilise la fonction compilée pour une inférence plus rapide
    predicted_logits = predict_fn(input_tensor)

    # Convertir le tenseur de sortie en tableau NumPy et trouver la classe prédite
    prediction_map = np.argmax(predicted_logits[0].numpy(), axis=-1)

    # --- 4. Post-traitement (optimisé) ---
    # Utilise l'indexation avancée de NumPy pour appliquer les couleurs en une seule opération
    rgb_mask = color_map[prediction_map]

    # Convertir le masque de RGB à BGR pour l'encodage avec OpenCV
    bgr_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)

    return bgr_mask
