---title: Analyseur de Scène Routière
---
title: API de Segmentation de Voiture Autonome
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
--- 
# 🚗 Analyseur de Scène Routière

Ce projet fournit une solution complète pour la segmentation sémantique d'images de scènes de conduite. Il inclut :
- Un **package Python** (`emmanuel_segmentation_package`) contenant la logique de segmentation.
- Une **API Flask** pour servir les prédictions via HTTP.
- Une **interface web Streamlit** pour une utilisation interactive.
- Une configuration **Docker** et **Docker Compose** pour un développement et un déploiement faciles et reproductibles.
- Un **pipeline CI/CD** avec GitHub Actions pour automatiser les tests, la publication des images Docker, et le déploiement.

## ✨ Fonctionnalités

- **Package Python Installable** : La logique de segmentation est encapsulée dans un package (`emmanuel_segmentation_package`) pour une réutilisation facile.
- **API Web avec Flask** : Une API simple pour obtenir des prédictions via des requêtes HTTP. Elle retourne les statistiques de segmentation et l'image segmentée encodée en Base64.
- **Interface Web Interactive** : Une application Streamlit conviviale pour uploader des images ou fournir des URLs et visualiser les résultats.
- **Téléchargement Automatique des Modèles** : Le modèle (`.keras`), la configuration et les poids des classes sont téléchargés depuis les *releases* GitHub au premier lancement, évitant d'avoir à stocker de gros fichiers dans le dépôt.
- **Conteneurisation avec Docker** : L'API et le frontend sont conteneurisés, et `docker-compose` orchestre le lancement de l'application complète.

## 📂 Structure du Projet

```
voiture-autonome/
├── .gitignore          # Fichiers à ignorer par Git
├── api/
│   ├── main.py         # Code de l'API FastAPI
│   └── segmentation.py # Logique de l'algorithme de segmentation
├── app/
│   └── app.py          # Code de l'application Streamlit
├── app.Dockerfile      # Instructions pour construire l'image Docker de Streamlit
├── models/             # Dossier pour stocker les modèles entraînés (ex: .h5, .pt)
├── requirements.txt    # Dépendances Python
├── Dockerfile          # Instructions pour construire l'image Docker de l'API
└── README.md           # Ce fichier
`

## 🚀 Démarrage Rapide (avec Docker)

C'est la méthode recommandée pour lancer l'ensemble du projet.

### Prérequis

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Lancement pour le développement (avec monitoring)

Cette commande lance tous les services, y compris l'API, le frontend et la stack de monitoring (Grafana, Loki). C'est idéal pour le développement et le débogage.

```bash
# Construit les images et lance tous les conteneurs en arrière-plan
docker-compose up --build -d
```

Une fois les services démarrés :

- **L'interface web** est accessible à l'adresse `http://localhost:8501`.
- **L'API** est accessible à l'adresse `http://localhost:8000`.
- **Grafana** (pour les logs) est accessible à `http://localhost:3000`.

## 💻 Développement Local (sans Docker)

Suivez ces étapes pour lancer l'API et le frontend directement sur votre machine.

### 1. Prérequis

- Python 3.12+

### 2. Installation

```bash
# Créez un environnement virtuel et activez-le (exemple pour Windows)
python -m venv voitautoenv
.\voitautoenv\Scripts\activate

# Installez les dépendances
pip install -r requirements.txt
```

### 3. Lancement

Vous devez lancer l'API et l'interface Streamlit dans **deux terminaux distincts**.

**Terminal 1 : Lancez l'API FastAPI**

```bash
uvicorn api.main:app --reload
```

> L'API sera accessible à l'adresse `http://127.0.0.1:8000`.

**Terminal 2 : Lancez l'interface Streamlit**

```bash
streamlit run app/app.py
```

> L'interface web sera accessible à l'adresse `http://localhost:8501`.

## 🔧 Personnalisation

Pour utiliser votre propre modèle de segmentation, la méthode recommandée est de configurer les variables d'environnement via Docker Compose.
Pour utiliser votre propre modèle de segmentation, il suffit de modifier les variables d'environnement dans le fichier `docker-compose.yml`.

1. Placez votre fichier de modèle (ex: `best_model_final.keras`) dans le dossier `models/`.
2. Assurez-vous d'avoir un fichier de mapping de classes (ex: `class_mapping.json`) dans le dossier `models/`. Ce fichier doit mapper les index de classe (en tant que chaînes de caractères) à des couleurs RGB.
   Voici un exemple de format valide pour `class_mapping.json`:
1. Ouvrez le fichier `docker-compose.yml`.
2. Localisez la section `environment` du service `api`.
3. Modifiez les URLs pour pointer vers vos propres fichiers de modèle et de mapping.

   ```json
   {
     "0": [128, 64, 128],
     "1": [244, 35, 232],
     "2": [70, 70, 70]
   }
   ```

3. Ouvrez le fichier `docker-compose.yml` et modifiez les variables d'environnement du service `api` pour qu'elles correspondent à vos noms de fichiers.

   ```yaml
   environment:
      - MODEL_FILENAME=best_model_final.keras
      - CLASS_MAPPING_FILENAME=class_mapping.json
      - MODEL_URL=<URL_VERS_VOTRE_MODELE.keras>
      - CLASS_MAPPING_URL=<URL_VERS_VOTRE_MAPPING.json>
   ```

4. Relancez l'application avec `docker-compose up --build`.
4. Relancez l'application avec `docker-compose up --build`. Les fichiers seront automatiquement téléchargés au premier démarrage.

## 🌐 Déploiement

Ce projet est configuré pour un déploiement continu sur **Hugging Face Spaces**, une plateforme gratuite et bien adaptée aux applications de Machine Learning.

### Prérequis

- Un compte Docker Hub où les images sont poussées par le workflow CI.
- Un compte Hugging Face.

### Étapes de déploiement sur Hugging Face

1. **Créez un nouveau "Space"** sur Hugging Face.
2. Choisissez **"Docker"** comme SDK et "Public" comme visibilité.
3. Une fois le Space créé, allez dans l'onglet **"Settings"**.
4. Dans la section "Docker template", cochez **"Use a Docker image from the Hub"**.
5. Entrez le nom de l'image de l'API : `emmanuelouedraogo/voiture-autonome-api:latest`.
6. Assurez-vous que le "Application port" est bien `8000`.
7. Ajoutez les secrets nécessaires (ex: `INTERNAL_API_KEY`) dans la section "Repository secrets".
8. Sauvegardez les changements.

Hugging Face déploiera automatiquement votre conteneur. L'URL publique de votre API sera disponible sur la page principale de votre Space.

> **Note** : Le frontend Streamlit peut également être déployé de la même manière en créant un second Space et en utilisant l'image `emmanuelouedraogo/voiture-autonome-frontend:latest`. N'oubliez pas de configurer la variable d'environnement `API_URL` dans les secrets du Space frontend pour qu'elle pointe vers l'URL de votre API déployée.
