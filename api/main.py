import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse
import cv2
import os

from api.segmentation import segment_image, load_segmentation_model


def get_internal_api_key():
    """
    Lit la clé d'API interne depuis un secret Docker.
    Pour le développement local, elle peut être définie comme variable d'environnement.
    """
    secret_path = "/run/secrets/internal_api_key"
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            return f.read().strip()
    # Fallback pour le développement local sans Docker
    return os.getenv("INTERNAL_API_KEY", None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle ML au démarrage et le libère à l'arrêt."""
    print("Chargement du modèle de segmentation...")
    app.state.predict_fn, app.state.color_map = load_segmentation_model()
    print("Modèle chargé et prêt à l'emploi.")
    yield
    # Code à exécuter à l'arrêt de l'application (nettoyage)
    print("Libération des ressources...")
    app.state.predict_fn = None
    app.state.color_map = None


app = FastAPI(
    title="API de Segmentation d'Image",
    description="Une API qui prend une image en entrée et retourne son masque de segmentation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Général"])
def read_root():
    """Point de terminaison racine pour vérifier que l'API est en ligne."""
    return {"message": "Bienvenue sur l'API de segmentation !"}


@app.get("/secure-data", tags=["Général"])
def read_secure_data(api_key: str = Depends(get_internal_api_key)):
    """Point de terminaison de test pour vérifier l'accès au secret."""
    if not api_key:
        raise HTTPException(status_code=403, detail="Clé d'API interne non configurée.")

    return {"message": "Accès autorisé !", "first_chars_of_key": f"{api_key[:4]}..."}


@app.post("/segment/", tags=["Segmentation"])
async def create_segmentation(
    file: UploadFile = File(...),
    predict_fn=Depends(lambda: app.state.predict_fn),
    color_map=Depends(lambda: app.state.color_map),
):
    """
    Prend une image en entrée, la segmente et retourne le masque de segmentation.
    """
    # Vérifier que le fichier est bien une image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Le fichier envoyé n'est pas une image."
        )

    # Lire le contenu du fichier en bytes
    image_bytes = await file.read()

    try:
        # Appeler votre fonction de segmentation avec les dépendances injectées
        segmented_mask = segment_image(image_bytes, predict_fn, color_map)

        # Encoder l'image résultante (le masque) en format PNG
        is_success, buffer = cv2.imencode(".png", segmented_mask)
        if not is_success:
            raise HTTPException(
                status_code=500,
                detail="Erreur lors de l'encodage de l'image de segmentation.",
            )

        # Créer un flux de bytes à partir du buffer pour la réponse
        image_stream = io.BytesIO(buffer)

        # Retourner l'image en tant que réponse HTTP
        return StreamingResponse(image_stream, media_type="image/png")

    except Exception as e:
        # Gérer les erreurs potentielles de la fonction de segmentation
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue lors de la segmentation : {e}",
        )
