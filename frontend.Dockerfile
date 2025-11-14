# Utilise une image Python slim comme base, légère et efficace.
FROM python:3.12-slim

# Définit le répertoire de travail à l'intérieur du conteneur.
WORKDIR /app

# Copie le fichier des dépendances en premier pour profiter du cache Docker.
# L'installation ne sera relancée que si ce fichier change.
COPY frontend/requirements.txt .

# Installe les dépendances du frontend.
RUN pip install --no-cache-dir -r requirements.txt

# Copie le code source de l'application frontend.
COPY frontend/ ./

# Expose le port sur lequel Streamlit s'exécute.
EXPOSE 8501

# Commande pour lancer l'application Streamlit.
CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0"]