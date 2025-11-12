# --- Étape 1: Builder ---
# Utilise une image Python complète pour construire les dépendances
FROM python:3.12-slim as builder

# Définir le répertoire de travail
WORKDIR /app

# Mettre à jour pip
RUN python -m pip install --upgrade pip

# Copier uniquement les fichiers de dépendances pour profiter de la mise en cache Docker
COPY pyproject.toml ./

# Installer les dépendances dans un répertoire local (wheelhouse)
# On installe gunicorn en plus des dépendances du projet
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels . "gunicorn"

# --- Étape 2: Final ---
# Utilise une image slim pour une taille finale réduite
FROM python:3.12-slim

WORKDIR /app

# Créer un utilisateur non-root pour des raisons de sécurité
RUN useradd --create-home --shell /bin/bash appuser

# Installer les dépendances système minimales nécessaires à l'exécution
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copier les dépendances pré-compilées de l'étape de build
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copier le code de l'application
COPY . .

# Installer le package en mode éditable pour s'assurer que les chemins sont corrects
RUN pip install -e .

# Changer le propriétaire des fichiers pour l'utilisateur non-root
RUN chown -R appuser:appuser /app

# Définir l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE 5000

# Commande pour lancer l'API avec Gunicorn quand le conteneur démarre
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "api.run_api:app"]