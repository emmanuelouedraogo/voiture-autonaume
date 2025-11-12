# scripts/run_cli.py
import sys
import os
import traceback
import pandas as pd

# Importer les fonctions depuis notre nouveau package
from emmanuel_segmentation_package import (
    load_segmentation_model,
    preprocess_image,
    predict_segmentation,
    get_class_statistics,
    visualize_prediction,
    download_image,
)
# ... et les constantes
from emmanuel_segmentation_package.pipeline import MODEL_PATH, CONFIG_PATH, CLASS_WEIGHTS_PATH

def main():
    """
    Fonction principale pour exécuter le pipeline de segmentation en ligne de commande.
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_cli.py <image_path_or_url>")
        sys.exit(1)

    input_source = sys.argv[1]
    image_to_process_path = None
    temp_file_path = None # Pour garder une trace du fichier temporaire

    if is_url(input_source):
        temp_file_path = download_image(input_source)
        if temp_file_path is None:
            sys.exit(1) # Quitter si le téléchargement a échoué
        image_to_process_path = temp_file_path
    else:
        image_to_process_path = input_source
        if not os.path.exists(image_to_process_path):
            print(f"Erreur: L'image locale n'a pas été trouvée à l'adresse {image_to_process_path}")
            sys.exit(1)

    try:
        # Charger le modèle et la configuration à partir des chemins locaux
        # (lus depuis les variables d'environnement ou les valeurs par défaut)
        model, config = load_segmentation_model(
            MODEL_PATH, CONFIG_PATH, CLASS_WEIGHTS_PATH
        )
        print("Modèle et configuration chargés.")

        # Prétraiter l'image
        processed_image_batch, original_size, original_image_array = preprocess_image(image_path=image_to_process_path, img_size=(224, 224))
        if processed_image_batch is None:
             sys.exit(1)
        print(f"Image prétraitée. Taille originale: {original_size}")

        # Effectuer la prédiction
        prediction_mask = predict_segmentation(model, processed_image_batch, original_size, config)
        print("Prédiction de segmentation terminée.")

        # Obtenir les statistiques des classes
        class_stats = get_class_statistics(prediction_mask, config)
        print("Statistiques des classes calculées.")

        # Visualiser le résultat
        pipeline_output_dir = "outputs"
        os.makedirs(pipeline_output_dir, exist_ok=True)
        output_filename = "predicted_segmentation_output.png"
        output_viz_path = os.path.join(pipeline_output_dir, output_filename)
        visualize_prediction(original_image_array, prediction_mask, config, save_path=output_viz_path)

        # Afficher les statistiques
        print("\nStatistiques des classes :\n")
        stats_df = pd.DataFrame(class_stats)
        print(stats_df.sort_values(by='percentage', ascending=False).to_string())

    except Exception as e:
        print("\nUne erreur inattendue est survenue :")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Nettoyer le fichier temporaire s'il a été créé
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Fichier temporaire nettoyé : {temp_file_path}")

if __name__ == "__main__":
    main()
