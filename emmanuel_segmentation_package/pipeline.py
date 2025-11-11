import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image # Correct import for PIL Image
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from IPython.display import display # Only import display from IPython
import requests # Import requests for downloading images from URL
from urllib.parse import urlparse # Import urlparse for checking if input is a URL
import tempfile # Import tempfile for saving downloaded images
import traceback # Import traceback for printing full error information
from matplotlib.colors import ListedColormap # Import ListedColormap

# URLs for model, config, and class weights from GitHub releases
# Lit les URLs depuis les variables d'environnement, avec des valeurs par défaut si elles ne sont pas définies.
# C'est une bonne pratique pour la configuration des conteneurs.
MODEL_URL = os.getenv(
    "MODEL_URL", "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v0.0.2/final_optimized_model.keras"
)
CONFIG_URL = os.getenv(
    "CONFIG_URL", "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v0.0.2/class_mapping.json"
)
CLASS_WEIGHTS_URL = os.getenv(
    "CLASS_WEIGHTS_URL", "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v0.0.2/class_weights.json"
)

# Directory to cache downloaded model files
MODEL_CACHE_DIR = "model_cache"


# Function to create a weighted loss function (needed for loading the model)
def create_weighted_loss(class_weights):
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    def weighted_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        pixel_weights = tf.gather(class_weights_tensor, y_true)
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        unweighted_loss = scce(y_true, y_pred)
        weighted_loss = unweighted_loss * pixel_weights
        return tf.reduce_mean(weighted_loss)
    return weighted_loss


def load_segmentation_model(model_path, config_path, class_weights_path):
    """Loads the trained segmentation model and its configuration."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Ensure required keys are in config
        required_keys = ['id_to_group', 'group_names', 'group_colors', 'num_classes']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing key in config file: '{key}'")

        config['id_to_group'] = np.array(config['id_to_group'], dtype=np.uint8)

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading or validating config file {config_path}: {e}")
        raise

    # Load class weights within the script
    try:
        if not os.path.exists(class_weights_path):
            raise FileNotFoundError(f"Class weights file not found at: {class_weights_path}")
        with open(class_weights_path, 'r') as f:
            class_weights_values_in_script = json.load(f)
        print("Class weights loaded successfully.")
    except Exception as e:
        print(f"Error loading class weights from {class_weights_path}: {e}")
        raise


    custom_objects = {'weighted_loss': create_weighted_loss(class_weights_values_in_script)}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    return model, config

def is_url(input_string):
    """Checks if a string is a valid URL."""
    try:
        result = urlparse(input_string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def download_image(url):
    """Downloads an image from a URL to a temporary file."""
    print(f"Downloading image from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Create a temporary file to save the image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded image to temporary file: {temp_file.name}")
        return temp_file.name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

def download_file_from_url(url, output_path):
    """Downloads a file from a URL if it doesn't exist at the output path."""
    if os.path.exists(output_path):
        print(f"File already exists, skipping download: {output_path}")
        return True

    print(f"Downloading file from {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def preprocess_image(image_path=None, image_array=None, img_size=(224, 224)):
    """Preprocesses a single image for segmentation."""
    if image_path:
        try:
            original_image = Image.open(image_path).convert('RGB')
            image_array = np.array(original_image)
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None, None, None # Return None for all if file not found
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            return None, None, None # Handle other loading errors
    elif image_array is None:
        raise ValueError("Either image_path or image_array must be provided.")

    original_size = image_array.shape[:2] # (height, width)

    image_resized = tf.image.resize(image_array, img_size, method='bilinear')
    image_normalized = tf.cast(image_resized, tf.float32) / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)

    return image_batch, original_size, image_array

def predict_segmentation(model, processed_image_batch, original_size, config):
    """Performs segmentation prediction and resizes the mask to original size."""
    predictions = model.predict(processed_image_batch, verbose=0)
    pred_mask_resized = tf.argmax(predictions, axis=-1)[0].numpy()

    # Resize the predicted mask back to the original image size
    pred_mask_original_size = tf.image.resize(
        np.expand_dims(pred_mask_resized, axis=-1), # Add channel dimension for resize
        original_size,
        method='nearest' # Use nearest neighbor for masks to preserve class IDs
    )
    pred_mask_original_size = tf.squeeze(pred_mask_original_size, axis=-1).numpy() # Remove channel dimension and convert to numpy

    return pred_mask_original_size

def get_class_statistics(prediction_mask, config):
    """Calculates class statistics for a prediction mask."""
    unique_classes, counts = np.unique(prediction_mask, return_counts=True)
    class_stats = []
    total_pixels = prediction_mask.size
    # Ensure class names and colors are available in config
    group_names = config.get('group_names', [])
    num_classes = config.get('num_classes', len(group_names))

    # Iterate through unique_classes and counts together
    for class_id, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        class_stats.append({
            'class_id': int(class_id),
            'class_name': group_names[class_id] if class_id < len(group_names) else f"Unknown_{class_id}",
            'pixel_count': int(count),
            'percentage': float(f"{percentage:.2f}") # Fixed formatting
        })

    # Ensure all classes are in the stats, even if they have 0 pixels
    present_class_ids = set(unique_classes)
    for class_id in range(num_classes):
        if class_id not in present_class_ids:
            class_stats.append({
                'class_id': int(class_id),
                'class_name': group_names[class_id] if class_id < len(group_names) else f"Unknown_{class_id}",
                'pixel_count': 0,
                'percentage': 0.0
            })

    return class_stats

def create_segmentation_image(prediction_mask, config):
    """
    Crée une image couleur (objet PIL.Image) à partir d'un masque de prédiction.
    """
    if 'group_colors' not in config:
        raise ValueError("La configuration doit contenir 'group_colors'.")

    height, width = prediction_mask.shape
    # S'assurer que les couleurs sont un tableau numpy pour un indexage facile
    colors = np.array(config['group_colors'], dtype=np.uint8)

    # Créer une image RGB vide
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Appliquer les couleurs basées sur les identifiants de classe dans le masque
    rgb_mask = colors[prediction_mask]

    return Image.fromarray(rgb_mask)

def visualize_prediction(original_image_array, prediction_mask, config, save_path=None):
    """Visualizes the original image and the segmented mask with a legend."""
    required_keys = ['group_colors', 'group_names']
    # Get num_classes from config with a fallback
    num_classes_viz = config.get('num_classes', len(config.get('group_names', [])))
    if num_classes_viz == 0:
        print("Warning: num_classes is 0 or not found in config. Cannot visualize.")
        return

    for key in required_keys:
        if key not in config:
            print(f"Warning: Missing key '{{key}}' in visualization config. Cannot visualize.")
            return # Exit visualization if config is incomplete

    # Convert colors from [0, 255] to [0, 1] and ensure it's a list of lists/tuples
    colors = [tuple(c / 255.0 for c in color) for color in config['group_colors']]
    cmap = ListedColormap(colors)

    # Create subplots: 1 row, 2 columns for images, plus space for legend
    fig, axes = plt.subplots(1, 2, figsize=(16, 8)) # Adjusted figsize

    # Display Original Image
    axes[0].imshow(original_image_array)
    axes[0].set_title("Image Originale")
    axes[0].axis("off")

    # Display Predicted Mask (Color-coded)
    # Use the custom colormap and specify vmin/vmax to map class IDs correctly
    img_seg = axes[1].imshow(prediction_mask, cmap=cmap, vmin=0, vmax=num_classes_viz - 1)
    axes[1].set_title("Prédiction (Colorée)")
    axes[1].axis("off")

    # Create Legend - Place it outside the plot area
    legend_handles = []
    for class_id in range(num_classes_viz):
        if class_id < len(config['group_names']) and class_id < len(colors):
            patch = mpatches.Patch(color=colors[class_id], label=config['group_names'][class_id])
            legend_handles.append(patch)

    # Add legend to the figure
    # Adjust bbox_to_anchor to position the legend to the right of the plots
    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for the legend

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to: {{save_path}}")

    # Remove the plt.show() call here

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import Image as IPythonImage, display # Use a different name for IPython Image
    import os # Redundant but safe if not imported globally above
    import tempfile # Redundant but safe if not imported globally above
    import requests # Redundant but safe if not imported globally above
    from urllib.parse import urlparse # Redundant but safe if not imported globally above
    import traceback # Redundant but safe if not imported globally above

    if len(sys.argv) < 2:
        print("Usage: python segmentation_pipeline.py <image_path_or_url>")
        sys.exit(1)

    input_source = sys.argv[1]
    image_to_process_path = None
    temp_file_path = None # Variable to keep track of temporary file

    if is_url(input_source):
        temp_file_path = download_image(input_source)
        if temp_file_path is None:
            sys.exit(1) # Exit if download failed
        image_to_process_path = temp_file_path
        print(f"Processing downloaded image: {{image_to_process_path}}")
    else:
        image_to_process_path = input_source
        print(f"Processing local image: {{image_to_process_path}}")
        if not os.path.exists(image_to_process_path):
            print(f"Error: Local image not found at {{image_to_process_path}}")
            sys.exit(1)

    # --- Model and Config Loading with Download ---
    # Create a directory to cache the model files
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Define local paths for the downloaded files
    final_model_path = os.path.join(MODEL_CACHE_DIR, "final_optimized_model.keras")
    final_config_path = os.path.join(MODEL_CACHE_DIR, "class_mapping.json")
    class_weights_path = os.path.join(MODEL_CACHE_DIR, "class_weights.json")

    # Download files if they don't exist
    if not all([
        download_file_from_url(MODEL_URL, final_model_path),
        download_file_from_url(CONFIG_URL, final_config_path),
        download_file_from_url(CLASS_WEIGHTS_URL, class_weights_path)
    ]):
        print("Failed to download required model files. Exiting.")
        sys.exit(1)

    try:
        # Load the model and config
        model, config = load_segmentation_model(final_model_path, final_config_path, class_weights_path)
        print("Model and configuration loaded.")

        # Preprocess the image
        processed_image_batch, original_size, original_image_array = preprocess_image(image_path=image_to_process_path, img_size=(224, 224))
        if processed_image_batch is None:
             sys.exit(1) # Exit if preprocessing failed (e.g., file not found)
        print(f"Image preprocessed. Original size: {{original_size}}")

        # Perform prediction
        prediction_mask = predict_segmentation(model, processed_image_batch, original_size, config)
        print("Segmentation prediction complete.")

        # Get class statistics
        class_stats = get_class_statistics(prediction_mask, config)
        print("Class statistics calculated.")

        # Visualize the result
        pipeline_output_dir = "outputs" # Save outputs in a local 'outputs' directory
        os.makedirs(pipeline_output_dir, exist_ok=True)
        # Use a simplified, consistent filename for the output visualization
        output_filename = "predicted_segmentation_output.png"
        output_viz_path = os.path.join(pipeline_output_dir, output_filename)
        visualize_prediction(original_image_array, prediction_mask, config, save_path=output_viz_path)

        # Display class statistics
        print("\nClass Statistics:\n")
        stats_df = pd.DataFrame(class_stats)
        # Sort by percentage in descending order for display
        print(stats_df.sort_values(by='percentage', ascending=False).to_string())


        # Display the saved image in the notebook output
        if os.path.exists(output_viz_path):
            print(f"Displaying saved prediction visualization from: {{output_viz_path}}")
            # Note: This display function might not work outside of a Colab/Jupyter notebook environment
            # For a standalone script, consider saving the image and letting the user open it.
            try:
                # Explicitly call display on the IPythonImage object
                display(IPythonImage(filename=output_viz_path))
            except NameError:
                 print("IPython.display not available, skipping image display.")
        else:
            print(f"Error: Saved prediction visualization not found at: {{output_viz_path}}")


    except FileNotFoundError as e:
        print(f"Error: {{e}}")
        sys.exit(1)
    except Exception as e:
        print("\nAn unexpected error occurred:")
        traceback.print_exc() # Print the full traceback
        sys.exit(1)
    finally:
        # Clean up temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {{temp_file_path}}")
