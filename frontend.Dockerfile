# --- Étape 1: Image de base ---
# Utilise une image Python slim pour une taille de base réduite.
FROM python:3.9-slim

# --- Étape 2: Définir le répertoire de travail ---
WORKDIR /app

# --- Étape 3: Copier les fichiers de dépendances et les installer ---
# Copier uniquement le fichier de dépendances pour profiter du cache Docker.
# Les dépendances ne seront réinstallées que si ce fichier change.
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Étape 4: Copier le code de l'application ---
COPY frontend/ /app/

# --- Étape 5: Créer un utilisateur non-root pour la sécurité ---
RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app
USER appuser

# --- Étape 6: Exposer le port et définir la commande de démarrage ---
EXPOSE 8501

# Commande pour lancer l'application Streamlit.
# --server.enableCORS=false est une bonne pratique de sécurité.
# --server.headless=true est nécessaire pour fonctionner correctement dans un conteneur.
CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.headless=true"]