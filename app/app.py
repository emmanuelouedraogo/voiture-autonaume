import streamlit as st
import requests
from PIL import Image
import io
import os

# --- Configuration ---
# Utilise une variable d'environnement pour l'URL de l'API, avec une valeur par défaut pour le dev local.
# Dans docker-compose, cette variable sera "http://api:8000/segment/".
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/segment/")

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Segmentation d'Image de Voiture", page_icon="🚗", layout="wide"
)

st.title("🚗 Interface de Segmentation d'Image")
st.write("Envoyez une image de la route et l'algorithme la segmentera pour vous.")

# --- Interface utilisateur ---
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

st.info(
    "ℹ️ Le traitement de l'image peut prendre plus d'une minute, "
    "en particulier lors de la première exécution. Merci de votre patience."
)

if uploaded_file is not None:
    # Afficher l'image originale et l'image segmentée côte à côte
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image Originale")
        image = Image.open(uploaded_file)
        st.image(image, caption="Image envoyée", width="stretch")

    with col2:
        st.subheader("Image Segmentée")
        # Afficher un spinner pendant que l'API traite l'image
        with st.spinner("Segmentation en cours..."):
            try:
                # Préparer le fichier pour l'envoi à l'API
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                # Envoyer la requête POST à l'API FastAPI
                response = requests.post(
                    API_URL, files=files, timeout=180
                )  # Timeout augmenté à 3 minutes

                if response.status_code == 200:
                    # Lire l'image de segmentation retournée par l'API
                    segmented_image = Image.open(io.BytesIO(response.content))
                    st.image(
                        segmented_image,
                        caption="Masque de segmentation",
                        width="stretch",
                    )
                else:
                    st.error(f"Erreur de l'API (code {response.status_code}):")
                    st.error(response.json().get("detail", "Aucun détail fourni."))

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API : {e}")
                st.info("Assurez-vous que l'API FastAPI est bien en cours d'exécution.")
