import pytest
import numpy as np
from emmanuel_segmentation_package.pipeline import (
    is_url,
    preprocess_image,
    get_class_statistics
)


@pytest.mark.parametrize("url_input, expected", [
    ("https://example.com", True),
    ("http://example.com/path?query=1", True),
    ("ftp://files.example.com", True),
    ("not a url", False),
    ("/local/path/to/file", False),
    ("example.com", False),
])
def test_is_url(url_input, expected):
    """
    Teste la fonction is_url avec des entrées valides et invalides.
    """
    assert is_url(url_input) == expected


def test_preprocess_image_with_array():
    """
    Teste la fonction de prétraitement avec un tableau numpy en entrée.
    """
    # Crée une fausse image (hauteur, largeur, canaux)
    dummy_image_array = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    target_size = (224, 224)

    processed_batch, original_size, original_array = preprocess_image(
        image_array=dummy_image_array,
        img_size=target_size
    )

    # Vérifie que les sorties sont correctes
    assert processed_batch is not None
    assert processed_batch.shape == (1, *target_size, 3)
    assert processed_batch.dtype == 'float32'
    assert np.max(processed_batch) <= 1.0
    assert np.min(processed_batch) >= 0.0
    assert original_size == (480, 640)
    np.testing.assert_array_equal(original_array, dummy_image_array)


def test_get_class_statistics():
    """
    Teste le calcul des statistiques de classe sur un masque de prédiction factice.
    """
    # Crée un masque de prédiction factice et une configuration
    prediction_mask = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [3, 3, 2, 2],
    ])  # Total 12 pixels: 4x(0), 3x(1), 3x(2), 2x(3)

    dummy_config = {
        "num_classes": 4,
        "group_names": ["road", "sky", "car", "person"]
    }

    stats = get_class_statistics(prediction_mask, dummy_config)

    # Vérifie que les statistiques sont correctes
    assert len(stats) == 4  # Doit retourner des stats pour toutes les classes
    stats_dict = {s['class_name']: s for s in stats}

    assert stats_dict["road"]["pixel_count"] == 4
    assert stats_dict["road"]["percentage"] == round((4/12) * 100, 2)

    assert stats_dict["sky"]["pixel_count"] == 3
    assert stats_dict["sky"]["percentage"] == round((3/12) * 100, 2)

    assert stats_dict["car"]["pixel_count"] == 3
    assert stats_dict["car"]["percentage"] == round((3/12) * 100, 2)

    assert stats_dict["person"]["pixel_count"] == 2
    assert stats_dict["person"]["percentage"] == round((2/12) * 100, 2)