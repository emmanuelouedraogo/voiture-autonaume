# --- Étape 1: Le "Builder" ---
FROM python:3.12-slim as builder

# Définir le répertoire de travail pour les dépendances
WORKDIR /install

# Copier et installer les dépendances Python du frontend
COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt


# --- Étape 2: L'image finale ---
FROM python:3.12-slim as final

# Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash appuser

# Copier les dépendances installées depuis l'étape "builder"
COPY --from=builder /install /usr/local

# Copier le code de l'application
WORKDIR /home/appuser/app
COPY --chown=appuser:appuser ./app/app.py .

# Définir l'utilisateur non-root
USER appuser

# Définir la variable d'environnement pour l'URL de l'API.
# Cette valeur sera utilisée si elle n'est pas surchargée par docker-compose ou un autre orchestrateur.
ENV API_URL="http://api:8000/segment/"

# Exposer le port et définir la commande de démarrage
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.headless=true"]