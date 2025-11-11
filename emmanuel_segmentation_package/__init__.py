# segmentation_package/__init__.py
"""Package de segmentation sémantique."""

from .pipeline import (
    load_segmentation_model,
    preprocess_image,
    predict_segmentation,
    get_class_statistics,
    create_segmentation_image,
    visualize_prediction,
    download_file_from_url
)

__version__ = "0.1.0"
