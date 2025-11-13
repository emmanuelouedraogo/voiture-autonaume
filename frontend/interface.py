import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import os
import base64, math
# ... (autres imports)
# --- Configuration de la page ---
st.set_page_config(
    page_title="Analyse de Scène Routière",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Configuration de l'API ---
# Lit l'URL de l'API depuis la configuration Streamlit, qui est alimentée par la variable d'environnement
# définie dans docker-compose.yml. Cela garantit l'utilisation de 'http://api:8000/predict'.
# La valeur par défaut est pour le développement local sans Docker.

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Construit l'URL complète de l'endpoint de prédiction.
PREDICT_URL = f"{API_BASE_URL.rstrip('/')}/predict"
# ... (le reste de votre code)

def call_segmentation_api(image_bytes, filename):
    """Appelle l'API de segmentation avec une image et retourne la réponse."""
    with st.spinner("🧠 Analyse de l'image en cours... Le modèle réfléchit !"):
        try:
            files = {'image': (filename, image_bytes, 'image/jpeg')}
            response = requests.post(PREDICT_URL, files=files, timeout=180)
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)

            # Vérifier si la réponse est bien du JSON
            if "application/json" in response.headers.get("Content-Type", ""):
                return response.json()
            else:
                st.error("L'API n'a pas retourné une réponse JSON valide.")
                st.code(response.text) # Affiche le texte brut de la réponse pour le débogage
                return None

        except requests.exceptions.HTTPError as e:
            # Gérer spécifiquement les erreurs HTTP (4xx, 5xx)
            st.error(f"L'API a retourné une erreur {e.response.status_code}.")
            try:
                # Essayer d'afficher le message d'erreur détaillé de l'API
                error_details = e.response.json()
                st.error(f"Détails de l'API : {error_details.get('error', '')} - {error_details.get('details', 'Aucun détail fourni.')}")
            except ValueError:
                st.error("La réponse d'erreur de l'API n'était pas au format JSON.")
            return None # Retourner explicitement None en cas d'erreur HTTP
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API : {e}")
            st.warning("Veuillez vérifier que le service API est bien démarré et accessible.")
            return None


def create_legend_image(stats, group_colors, items_per_row=4):
    """Crée une image de légende à partir des statistiques et des couleurs."""
    # Filtrer les classes avec un pourcentage > 0 pour les inclure dans la légende
    present_stats = [s for s in stats if s['percentage'] > 0]
    if not present_stats: # Si aucune classe n'est présente, ne pas créer d'image de légende.
        return None

    # Paramètres de la légende
    padding = 10
    box_size = 20
    font_size = 14
    line_height = font_size + padding

    # Calculer le nombre de lignes et la hauteur de l'image de la légende
    num_rows = math.ceil(len(present_stats) / items_per_row)
    legend_height = num_rows * line_height + padding
    legend_width = 600 # Largeur fixe pour la légende

    # Créer l'image de la légende
    legend_image = Image.new('RGB', (legend_width, legend_height), (255, 255, 255))
    draw = ImageDraw.Draw(legend_image)
    font = ImageFont.load_default() # Utiliser la police par défaut pour la simplicité

    x_pos, y_pos = padding, padding // 2
    # S'assurer que la largeur de l'item est positive, même si items_per_row est 0 ou négatif.
    if items_per_row > 0:
        item_width = legend_width // items_per_row
    else: # Cas de sécurité, ne devrait pas arriver avec la valeur par défaut.
        item_width = legend_width

    for i, stat in enumerate(present_stats):
        if i > 0 and i % items_per_row == 0:
            x_pos = padding
            y_pos += line_height # Passer à la ligne suivante
        
        color = tuple(group_colors[stat['class_id']])
        draw.rectangle([x_pos, y_pos, x_pos + box_size, y_pos + box_size], fill=color, outline=(0,0,0))
        draw.text((x_pos + box_size + 5, y_pos), f"{stat['class_name']}", font=font, fill=(0,0,0))
        x_pos += item_width

    return legend_image

def display_results(original_image, api_response):
    """Affiche l'image originale, le masque de segmentation et les statistiques."""
    st.subheader("Résultats de la Segmentation")

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Image Originale", use_column_width=True)

    with col2:
        try:
            if 'segmented_image' not in api_response or 'statistics' not in api_response:
                st.error("La réponse de l'API est mal formatée. Clés 'segmented_image' ou 'statistics' manquantes.")
                return

            # Décode l'image de segmentation depuis la chaîne Base64
            img_data_str = api_response['segmented_image'].split(',')[1]
            img_data = base64.b64decode(img_data_str)
            segmented_image = Image.open(io.BytesIO(img_data))
            st.image(segmented_image, caption="Prédiction de Segmentation", use_column_width=True)

        except (IndexError, base64.binascii.Error) as e:
            st.error(f"Erreur lors du décodage de l'image de segmentation : {e}")
            return
    
    # Créer et afficher la légende séparément
    legend_img = create_legend_image(api_response['statistics'], api_response['config']['group_colors'])
    if legend_img:
        st.image(legend_img, caption="Légende des classes détectées")

    st.subheader("📊 Statistiques des Classes Détectées")
    
    # Trier les statistiques par pourcentage pour un meilleur affichage
    sorted_stats = sorted(api_response['statistics'], key=lambda x: x['percentage'], reverse=True)
    
    # Filtrer les classes avec un pourcentage supérieur à zéro
    present_stats = [s for s in sorted_stats if s['percentage'] > 0]
    
    # Afficher les statistiques dans des colonnes pour un look plus propre
    if present_stats:
        num_cols = min(len(present_stats), 4) # Utiliser jusqu'à 4 colonnes
        stats_cols = st.columns(num_cols)
        for i, stat in enumerate(present_stats):
            with stats_cols[i % num_cols]:
                st.metric(label=stat['class_name'].capitalize(), value=f"{stat['percentage']:.2f}%")


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
        if st.button("Lancer l'Analyse", use_container_width=True, type="primary"): # use_container_width est correct pour st.button
            api_response = call_segmentation_api(image_bytes, filename)
            if api_response:
                display_results(original_image, api_response)

    except Exception as e:
        st.error(f"L'image fournie n'a pas pu être ouverte. Erreur : {e}")

else:
    st.info("Veuillez uploader une image ou fournir une URL dans la barre latérale pour commencer.")

st.markdown("---")
st.markdown("Développé par Emmanuel OUEDRAOGO - Projet de Voiture Autonome")