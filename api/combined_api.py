import io
import os
import json
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse

# --- 1. Logique de Segmentation (précédemment dans segmentation.py) ---

# Définition des chemins de manière dynamique
APP_DIR = Path(__file__).parent.parent
MODELS_DIR = APP_DIR / "models"

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
        model_path = MODELS_DIR / "best_model_final.keras"
        class_mapping_path = MODELS_DIR / "class_mapping.json"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier modèle non trouvé: {model_path}")

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

        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(
                f"Fichier de mapping non trouvé: {class_mapping_path}"
            )

        with open(class_mapping_path, "r") as f:
            raw_mapping = json.load(f)

        class_mapping_dict = {int(k): v for k, v in raw_mapping.items() if k.isdigit()}

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
        raise RuntimeError(f"Échec du chargement du modèle: {e}") from e

    return predict_fn, color_map


def segment_image(
    image_bytes: bytes, predict_fn: callable, color_map: np.ndarray
) -> np.ndarray:
    """
    Prend une image en bytes, la segmente et retourne un masque coloré.
    """
    if predict_fn is None or color_map is None:
        raise RuntimeError(
            "Le modèle ou le mapping de classes n'a pas pu être chargé. Vérifiez les logs du serveur."
        )

    _, img_height, img_width, _ = predict_fn.input_signature[0].shape

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(
        img_rgb, (img_width, img_height), interpolation=cv2.INTER_NEAREST
    )

    input_array = np.expand_dims(img_resized, axis=0) / 255.0
    input_tensor = tf.constant(input_array, dtype=tf.float32)

    predicted_logits = predict_fn(input_tensor)
    prediction_map = np.argmax(predicted_logits[0].numpy(), axis=-1)

    rgb_mask = color_map[prediction_map]
    bgr_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)

    return bgr_mask


# --- 2. Logique de l'API (précédemment dans main.py) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle ML au démarrage et le libère à l'arrêt."""
    print("Chargement du modèle de segmentation...")
    # Appelle la fonction locale load_segmentation_model
    app.state.predict_fn, app.state.color_map = load_segmentation_model()
    print("Modèle chargé et prêt à l'emploi.")
    yield
    print("Libération des ressources...")
    app.state.predict_fn = None
    app.state.color_map = None


app = FastAPI(
    title="API de Segmentation d'Image (Combinée)",
    description="Une API qui prend une image en entrée et retourne son masque de segmentation.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Général"])
def read_root():
    """Point de terminaison racine pour vérifier que l'API est en ligne."""
    return {"message": "Bienvenue sur l'API de segmentation combinée !"}


@app.post("/segment/", tags=["Segmentation"])
async def create_segmentation(
    file: UploadFile = File(...),
    predict_fn=Depends(lambda: app.state.predict_fn),
    color_map=Depends(lambda: app.state.color_map),
):
    """
    Prend une image en entrée, la segmente et retourne le masque de segmentation.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Le fichier envoyé n'est pas une image."
        )

    image_bytes = await file.read()

    try:
        # Appelle la fonction locale segment_image
        segmented_mask = segment_image(image_bytes, predict_fn, color_map)

        is_success, buffer = cv2.imencode(".png", segmented_mask)
        if not is_success:
            raise HTTPException(
                status_code=500,
                detail="Erreur lors de l'encodage de l'image de segmentation.",
            )

        image_stream = io.BytesIO(buffer)
        return StreamingResponse(image_stream, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue lors de la segmentation : {e}",
        )