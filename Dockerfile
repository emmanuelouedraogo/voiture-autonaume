# --- Étape 1: Builder ---
# Utilise une image Python complète pour construire les dépendances de manière robuste.
FROM python:3.12-slim as builder

WORKDIR /app

# Mettre à jour pip pour s'assurer d'avoir la dernière version.
RUN python -m pip install --upgrade pip

# Copier uniquement le fichier de dépendances pour profiter de la mise en cache de Docker.
# L'installation ne sera relancée que si pyproject.toml change.
COPY pyproject.toml ./

# Installer les dépendances de production dans un répertoire local (wheelhouse).
# Cela pré-compile les paquets, ce qui accélère l'étape finale.
# On installe les dépendances du groupe [main] défini dans pyproject.toml.
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

# Copier le code de l'application (le package et le fichier main.py).
COPY emmanuel_segmentation_package/ ./emmanuel_segmentation_package/
COPY api/run_api.py .

# Changer le propriétaire des fichiers et définir l'utilisateur non-root.
RUN chown -R appuser:appuser /app
USER appuser

# Commande pour lancer l'application API Flask avec Gunicorn.
# 'run_api:app' fait référence à l'objet 'app' dans le fichier 'run_api.py'.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "run_api:app"]