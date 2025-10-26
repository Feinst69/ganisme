# GAN Image Generator

Application Streamlit pour générer des images à partir d'un générateur GAN pré-entraîné.

## Prérequis
- Python 3.11
- Modèle générateur disponible dans `src/data/model/v20_dynamic_gpu_safe/`

## Installation locale
1. Crée et active un environnement virtuel Python 3.11.
2. Installe les dépendances : `pip install -r requirements.txt`.
3. Lance l'application : `streamlit run src/streamlit_app.py`.

## Déploiement Render.com
1. Pousse le dépôt sur GitHub/GitLab/Bitbucket.
2. Render détectera `render.yaml` et proposera de créer le service web.
3. Commande de build : `pip install --upgrade pip && pip install -r requirements.txt`.
4. Commande de démarrage : `streamlit run src/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`.
5. Renseigne `PYTHON_VERSION=3.11.9` et les autres variables d'environnement si besoin (ex. URL de base de données).

## Vérifications
- Exécute `streamlit run src/streamlit_app.py` localement pour valider.
- Sur Render, surveille les logs pour confirmer le chargement du modèle et la génération d'images.

