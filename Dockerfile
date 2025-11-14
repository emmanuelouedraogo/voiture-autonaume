# --- Étape 1: Builder ---
# Utilise une image Python complète pour construire les dépendances de manière robuste.
FROM python:3.12-slim AS builder

WORKDIR /app

# Mettre à jour pip pour s'assurer d'avoir la dernière version.
RUN python -m pip install --upgrade pip

# Copier uniquement le fichier de dépendances pour profiter de la mise en cache de Docker.
# L'installation ne sera relancée que si pyproject.toml change.
COPY pyproject.toml .

# Copier tout le code source nécessaire pour construire les wheels.
COPY emmanuel_segmentation_package/ ./emmanuel_segmentation_package/
COPY api/ ./api/
 
# Installer les dépendances de production dans un répertoire local (wheelhouse).
# Cela pré-compile les paquets, ce qui accélère l'étape finale.
# On installe les dépendances du projet.
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -e .

# --- Étape 2: Final ---
# Utilise une image "slim" pour une taille finale réduite.
FROM python:3.12-slim

WORKDIR /app

# Créer un utilisateur non-root pour des raisons de sécurité.
RUN useradd --create-home --shell /bin/bash appuser

# Copier les dépendances pré-compilées de l'étape de build.
COPY --from=builder /app/wheels /wheels

# Installer les dépendances à partir des wheels. C'est plus rapide et ne nécessite pas de compilation.
RUN pip install --no-cache /wheels/*

# Copier le fichier de configuration de Gunicorn
COPY gunicorn_config.py .

# Copier les modèles pré-téléchargés depuis le contexte de build.
# Cette approche est plus robuste que de les télécharger dans le Dockerfile.
COPY models/ /app/models/

# Définir les variables d'environnement pour que l'API trouve les modèles.
ENV MODEL_PATH=/app/models/final_optimized_model.keras CONFIG_PATH=/app/models/class_mapping.json CLASS_WEIGHTS_PATH=/app/models/class_weights.json

# Changer le propriétaire des fichiers et définir l'utilisateur non-root.
RUN chown -R appuser:appuser /app
USER appuser

# Commande pour lancer l'application API Flask avec Gunicorn.
# On utilise le fichier de configuration pour charger le modèle dans chaque worker.
# 'api.run_api:app' fait référence à l'objet 'app' dans le fichier 'api/run_api.py'.
CMD ["gunicorn", "--config", "gunicorn_config.py", "api.run_api:app"]