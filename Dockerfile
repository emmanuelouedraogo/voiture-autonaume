# --- Étape 1: Le "Builder" ---
# Cette étape installe les dépendances dans un environnement isolé.
FROM python:3.12-slim as builder

# Installer les dépendances de construction nécessaires pour certaines bibliothèques Python
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail pour les dépendances
WORKDIR /install

# Copier et installer les dépendances Python
# Utiliser --prefix au lieu d'un venv pour une meilleure mise en cache et une copie plus simple
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt


# --- Étape 2: L'image finale ---
# Cette étape est optimisée pour la production.
FROM python:3.12-slim as final

# Installer les dépendances système minimales (curl pour le healthcheck)
# et nettoyer en une seule commande RUN pour réduire les couches
RUN apt-get update && apt-get install -y --no-install-recommends curl libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root pour des raisons de sécurité
RUN useradd --create-home --shell /bin/bash appuser

# Copier les dépendances installées depuis l'étape "builder"
COPY --from=builder /install /usr/local

# Copier le code de l'application
WORKDIR /home/appuser/app
COPY --chown=appuser:appuser ./api ./api

# Créer le répertoire des modèles, télécharger les modèles pendant la construction de l'image,
# puis s'assurer que l'utilisateur 'appuser' est propriétaire de l'ensemble du répertoire.
# Cela optimise considérablement le temps de démarrage de l'application.
RUN mkdir -p /home/appuser/app/models && \
    curl -L "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v.0.0.1/best_model_final.keras" -o "/home/appuser/app/models/best_model_final.keras" && \
    curl -L "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v.0.0.1/class_mapping.json" -o "/home/appuser/app/models/class_mapping.json" && \
    chown -R appuser:appuser /home/appuser/app

# Définir l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE 8000

# Commande pour lancer l'API quand le conteneur démarre
# Le téléchargement du modèle se fera ici, dans le dossier /home/appuser/app/models
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]