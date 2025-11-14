"""
Fichier de configuration pour Gunicorn.

Ce fichier permet de s'accrocher au cycle de vie de Gunicorn pour charger
le modèle de manière fiable dans chaque processus worker.
"""

def post_fork(server, worker):
    """
    Hook exécuté après la création d'un processus worker.

    C'est l'endroit idéal pour initialiser des ressources qui ne sont pas
    compatibles avec le "forking", comme les modèles TensorFlow/Keras.
    Chaque worker chargera sa propre instance du modèle.
    """
    server.log.info(f"Worker {worker.pid} : Initialisation du modèle...")

    # Importer l'application Flask et sa fonction d'initialisation
    from api.run_api import initialize_model

    # Charger le modèle dans ce worker spécifique
    initialize_model()
    server.log.info(f"Worker {worker.pid} : Modèle initialisé.")