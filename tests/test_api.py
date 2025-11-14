import pytest
import os
from PIL import Image
import io
import numpy as np
import json

# Ajouter le répertoire racine au path pour que les imports fonctionnent
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.run_api import app as flask_app


@pytest.fixture
def client():
    """Crée un client de test Flask."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


def test_index_route(client):
    """Teste la route racine '/'."""
    response = client.get('/')
    assert response.status_code == 200
    json_data = response.get_json()
    assert "API de segmentation en cours d'exécution" in json_data['status']


def test_predict_no_file(client, monkeypatch):
    """Teste l'endpoint /predict sans envoyer de fichier."""
    # Simuler que le modèle est chargé pour passer la première vérification (503)
    monkeypatch.setattr("api.run_api.model", "mock_model")
    monkeypatch.setattr("api.run_api.config", {"mock": "config"})

    response = client.post('/predict')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Aucun fichier 'image' n'a été trouvé" in json_data['error']


def test_predict_model_not_loaded(client):
    """
    Teste que l'endpoint /predict renvoie 503 si le modèle n'est pas chargé.
    C'est le comportement attendu dans un environnement de test propre.
    """
    # Créer une fausse image pour l'envoi
    fake_image = io.BytesIO()
    Image.new('RGB', (60, 30), color = 'red').save(fake_image, 'PNG')
    fake_image.seek(0)

    # Envoyer la requête POST avec le fichier image
    response = client.post('/predict', data={'image': (fake_image, 'test.png')})

    # Le comportement correct est maintenant de renvoyer 503 si le modèle est None.
    assert response.status_code == 503
    json_data = response.get_json()
    assert "Le service n'est pas prêt" in json_data['error']
