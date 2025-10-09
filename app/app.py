import streamlit as st
import requests
from PIL import Image
import io
import os
from urllib.parse import urljoin

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
        st.image(image, caption="Image envoyée", use_container_width=True)

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
                # allow_redirects=False est crucial pour les déploiements sur des plateformes
                # comme Hugging Face qui peuvent utiliser des redirections internes.
                # Cela empêche la bibliothèque `requests` de changer la méthode POST en GET.
                response = requests.post(
                    API_URL, files=files, timeout=180, allow_redirects=False
                )  # Timeout augmenté à 3 minutes

                # Gérer manuellement la redirection si nécessaire, en préservant la méthode POST
                if response.status_code in (301, 302, 307, 308):
                    # L'en-tête 'Location' peut être une URL relative. On la reconstruit en URL absolue.
                    redirect_location = response.headers["Location"]
                    absolute_redirect_url = urljoin(API_URL, redirect_location)
                    
                    st.info(f"Redirection détectée, nouvelle requête vers : {absolute_redirect_url}")
                    
                    # On refait la requête POST vers l'URL absolue, en gardant les mêmes fichiers et timeout.
                    response = requests.post(
                        absolute_redirect_url, files=files, timeout=180, allow_redirects=False
                    )

                # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
                response.raise_for_status()

                # Si la requête a réussi, response.raise_for_status() ne lève pas d'exception.
                # On peut donc directement traiter la réponse.
                segmented_image = Image.open(io.BytesIO(response.content))
                st.image(
                    segmented_image,
                    caption="Masque de segmentation",
                    use_container_width=True,
                )

            except requests.exceptions.ConnectionError:
                st.error("Impossible de se connecter à l'API de segmentation.")
                st.warning(
                    "Veuillez vérifier que le service API est bien démarré et que l'URL configurée est correcte."
                )
            except requests.exceptions.Timeout:
                st.error("La requête vers l'API a expiré (timeout).")
                st.warning(
                    "L'API est peut-être surchargée ou le traitement de l'image est trop long."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"Erreur de l'API (code {e.response.status_code}):")
                try:
                    # Essayer de décoder le message d'erreur JSON de l'API
                    detail = e.response.json().get("detail", e.response.text)
                    st.error(f"Détail : {detail}")
                except ValueError:
                    # Si la réponse n'est pas du JSON, afficher le texte brut
                    st.error(f"Réponse brute du serveur : {e.response.text}")
