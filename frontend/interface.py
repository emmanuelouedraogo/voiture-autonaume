import streamlit as st
import requests
from PIL import Image
import io
import os
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Configuration de la page ---
st.set_page_config(
    page_title="Analyse de Scène Routière",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Configuration de l'API ---
# Utilise une variable d'environnement pour l'URL de l'API, avec une valeur par défaut pour le dev local.
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000/predict")


def call_segmentation_api(image_bytes, filename):
    """Appelle l'API de segmentation avec une image et retourne la réponse."""
    with st.spinner("🧠 Analyse de l'image en cours... Le modèle réfléchit !"):
        try:
            files = {'image': (filename, image_bytes, 'image/jpeg')}
            response = requests.post(API_URL, files=files, timeout=180)
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)

            # Vérifier si la réponse est bien du JSON
            if "application/json" in response.headers.get("Content-Type", ""):
                return response.json()
            else:
                st.error("L'API n'a pas retourné une réponse JSON valide.")
                st.code(response.text) # Affiche le texte brut de la réponse pour le débogage
                return None

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API : {e}")
            st.warning("Veuillez vérifier que le service API est bien démarré et accessible.")
            return None



def display_results(original_image, api_response):
    """Affiche l'image originale, le masque de segmentation et les statistiques."""
    st.subheader("Résultats de la Segmentation")

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Image Originale", use_container_width=True)

    with col2:
        try:
            # Vérifier que les clés nécessaires sont présentes
            if 'segmented_image' not in api_response or 'statistics' not in api_response or 'colors' not in api_response:
                st.error("La réponse de l'API est mal formatée. Clés 'segmented_image', 'statistics' ou 'colors' manquantes.")
                return

            # Décode l'image de segmentation depuis la chaîne Base64
            img_data_str = api_response['segmented_image'].split(',')[1]
            img_data = base64.b64decode(img_data_str)
            segmented_image = Image.open(io.BytesIO(img_data))

            # --- Création de l'image avec légende en utilisant Matplotlib ---
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(segmented_image)
            ax.axis('off') # Masquer les axes

            # Créer les éléments de la légende
            legend_patches = []
            class_colors = api_response['colors']
            for stat in sorted(api_response['statistics'], key=lambda x: x['class_id']):
                class_id = stat['class_id']
                class_name = stat['class_name'].capitalize()
                if class_id < len(class_colors):
                    color = np.array(class_colors[class_id]) / 255.0
                    patch = mpatches.Patch(color=color, label=f"{class_name}")
                    legend_patches.append(patch)
            
            # Ajouter la légende à droite de l'image
            ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='large')
            
            st.pyplot(fig, use_container_width=True)

        except (IndexError, base64.binascii.Error) as e:
            st.error(f"Erreur lors du décodage de l'image de segmentation : {e}")
            return

    st.subheader("📊 Statistiques des Classes Détectées")
    
    # Trier les statistiques par pourcentage pour un meilleur affichage
    sorted_stats = sorted(api_response['statistics'], key=lambda x: x['percentage'], reverse=True)
    
    # Afficher les statistiques dans des colonnes pour un look plus propre
    stats_cols = st.columns(4)
    col_index = 0
    for stat in sorted_stats:
        if stat['percentage'] > 0:
            with stats_cols[col_index % 4]:
                st.metric(label=stat['class_name'].capitalize(), value=f"{stat['percentage']:.2f}%")
            col_index += 1


# --- Interface Principale ---
st.title("🚗 Analyseur de Scène Routière")
st.markdown("Envoyez une image de la route et notre IA identifiera les différents éléments de la scène.")

with st.sidebar:
    st.header("🖼️ Source de l'Image")
    source_option = st.radio(
        "Choisissez une méthode d'envoi :",
        ("Uploader un fichier", "Utiliser une URL")
    )

image_bytes = None
filename = "image.jpg"

if source_option == "Uploader un fichier":
    uploaded_file = st.sidebar.file_uploader(
        "Sélectionnez une image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

elif source_option == "Utiliser une URL":
    image_url = st.sidebar.text_input(
        "Collez l'URL de l'image ici", 
        placeholder="https://exemple.com/image.jpg"
    )
    if image_url and image_url.strip() != "":
        try:
            # Ajouter un User-Agent pour simuler un navigateur et éviter les erreurs 403
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            filename = os.path.basename(image_url)
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Erreur lors du téléchargement de l'image : {e}")

if image_bytes:
    try:
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # Bouton pour lancer l'analyse
        if st.button("Lancer l'Analyse", use_container_width=True, type="primary"):
            api_response = call_segmentation_api(image_bytes, filename)
            if api_response:
                display_results(original_image, api_response)

    except Exception as e:
        st.error(f"L'image fournie n'a pas pu être ouverte. Erreur : {e}")

else:
    st.info("Veuillez uploader une image ou fournir une URL dans la barre latérale pour commencer.")

st.markdown("---")
st.markdown("Développé par Emmanuel OUEDRAOGO - Projet de Voiture Autonome")