import sys
import os
import logging

# Ajoute le répertoire de l'application au chemin Python
# pour que les imports comme "from api.run_api..." fonctionnent.
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Configuration de Gunicorn
bind = "0.0.0.0:8000"
workers = 1  # Vous pouvez ajuster ce nombre
timeout = 120
loglevel = "info"

# Configuration des logs pour être plus verbeux
logconfig_dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        'gunicorn.error': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': False,
        },
        'gunicorn.access': {
            'handlers': ['console'],
            'propagate': False,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}

def post_fork(server, worker):
    """
    Hook Gunicorn appelé après la création d'un worker.
    C'est ici que nous initialisons le modèle pour chaque processus worker.
    """
    worker.log.info(f"Worker {worker.pid}: Initialisation du modèle...")
    try:
        # L'import devrait maintenant fonctionner grâce à l'ajout au sys.path
        from api.run_api import initialize_model
        initialize_model()
        worker.log.info(f"Worker {worker.pid}: Modèle initialisé avec succès.")
    except Exception as e:
        worker.log.critical(f"Worker {worker.pid}: Échec de l'initialisation du modèle.", exc_info=e)
        # Arrête le worker si l'initialisation échoue
        sys.exit(1)

def on_starting(server):
    """Hook Gunicorn appelé au démarrage du master."""
    server.log.info("Démarrage du serveur Gunicorn master.")

def on_exit(server):
    """Hook Gunicorn appelé à l'arrêt du master."""
    server.log.info("Arrêt du serveur Gunicorn master.")