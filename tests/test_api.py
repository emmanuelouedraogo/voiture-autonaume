import pytest
from fastapi.testclient import TestClient
import os
from PIL import Image
import io
import numpy as np
import cv2

# Ajouter le répertoire racine au path pour que les imports fonctionnent
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


client = TestClient(app)


def test_read_root():
    """Teste si le point de terminaison racine est accessible et retourne le bon message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de segmentation !"}


@pytest.fixture
def sample_image_bytes():
    """Crée une fausse image en bytes pour les tests."""
    img = Image.new("RGB", (100, 100), color="red")
    byte_arr = io.BytesIO()
    img.save(byte_arr, format="PNG")
    return byte_arr.getvalue()


def test_segment_image_endpoint(sample_image_bytes, monkeypatch):
    """Teste le point de terminaison de segmentation avec une image valide."""
    # 1. On simule (mock) la fonction de segmentation pour ne pas dépendre du vrai modèle ML.
    def mock_segment_image(image_bytes, predict_fn, color_map):
        # Retourne une fausse image de masque (un carré noir)
        return np.zeros((100, 100, 3), dtype=np.uint8)

    monkeypatch.setattr("api.main.segment_image", mock_segment_image)

    # 2. On simule l'état de l'application comme si le modèle était chargé.
    #    Cela résout l'erreur "AttributeError: 'State' object has no attribute 'predict_fn'".
    app.state.predict_fn = lambda: "mock_predict_function"
    app.state.color_map = np.array([[0, 0, 0]], dtype=np.uint8)

    try:
        response = client.post(
            "/segment/", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        # 3. Maintenant que tout est simulé, on s'attend à un succès (code 200).
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    finally:
        # 4. Nettoyer l'état de l'application après le test pour ne pas affecter d'autres tests.
        app.state.predict_fn = None
        app.state.color_map = None
