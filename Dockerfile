# Utiliser une image Python slim pour une taille de base réduite
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système minimales (si nécessaire pour certaines bibliothèques comme OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN python -m pip install --upgrade pip

# Copier uniquement le fichier de configuration pour profiter de la mise en cache Docker
COPY pyproject.toml .

# Installer les dépendances, y compris Gunicorn pour la production
# Cette étape sera mise en cache si pyproject.toml ne change pas
RUN pip install --no-cache-dir . "gunicorn" "python-dotenv"

# Créer un utilisateur non-root pour des raisons de sécurité
RUN useradd --create-home --shell /bin/bash appuser

# Copier le code de l'application
COPY . .

# Installer le package en mode éditable pour s'assurer que les chemins sont corrects
RUN pip install -e .

# Changer le propriétaire du répertoire de l'application
RUN chown -R appuser:appuser /app

# Définir l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE 8000

# Commande pour lancer l'API avec Gunicorn quand le conteneur démarre
# 'api.run_api:app' pointe vers l'objet 'app' de Flask dans votre script.
# Le nombre de 'workers' peut être ajusté en fonction des cœurs de CPU disponibles.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "api.run_api:app"]