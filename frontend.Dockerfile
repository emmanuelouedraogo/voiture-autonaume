# --- Étape 1: Builder ---
# Utilise une image Python complète pour construire les dépendances
FROM python:3.10-slim as builder

WORKDIR /app

# Mettre à jour pip
RUN python -m pip install --upgrade pip

# Copier uniquement les fichiers de dépendances pour profiter de la mise en cache Docker
COPY pyproject.toml .

# Installer les dépendances dans un répertoire local (wheelhouse)
# On installe les dépendances de base + streamlit, requests, pillow
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels . streamlit requests Pillow

# --- Étape 2: Final ---
# Utilise une image slim pour une taille finale réduite
FROM python:3.10-slim

WORKDIR /app

# Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash appuser

# Copier les dépendances pré-compilées de l'étape de build
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copier le code de l'application
COPY . .

# Changer le propriétaire et définir l'utilisateur
RUN chown -R appuser:appuser /app
USER appuser

# Exposer le port de Streamlit
EXPOSE 8501

# Commande pour lancer l'application Streamlit
CMD ["streamlit", "run", "frontend/interface.py", "--server.port=8501", "--server.address=0.0.0.0"]