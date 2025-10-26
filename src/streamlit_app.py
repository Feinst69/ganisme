import math
import sys
from pathlib import Path
from typing import List, Optional

import streamlit as st
import torch
from torchvision.transforms.functional import to_pil_image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from ml.generate import DEFAULT_MODEL_SUBDIR, generate_images, load_generator, resolve_device


st.set_page_config(page_title="GAN Image Generator", layout="wide")


@st.cache_resource(show_spinner="Chargement du générateur...")
def get_generator(latent_dim: int, model_epoch: int, model_subdir: str):
    generator, device = load_generator(
        latent_dim=latent_dim,
        model_epoch=model_epoch,
        model_subdir=model_subdir,
    )
    return generator, device


def to_pil(images: torch.Tensor) -> List["Image.Image"]:
    """Convertir un tenseur [N, C, H, W] en liste d'images PIL."""
    images = images.detach().cpu()
    return [to_pil_image(img) for img in images]


def display_grid(pil_images: List["Image.Image"], columns: int) -> None:
    """Afficher les images dans une grille responsive."""
    if not pil_images:
        return

    columns = max(1, min(columns, len(pil_images)))
    rows = math.ceil(len(pil_images) / columns)
    idx = 0

    for _ in range(rows):
        cols = st.columns(columns)
        for col in cols:
            if idx >= len(pil_images):
                break
            col.image(pil_images[idx], use_container_width=True)
            idx += 1


st.title("🎨 Générateur d'images GAN")
st.write(
    "Sélectionnez le nombre d'images à générer et visualisez-les instantanément. "
    "Le modèle est chargé depuis les poids entraînés existants."
)

with st.sidebar:
    st.header("Paramètres")
    model_epoch = st.number_input("Époque du modèle", min_value=1, value=100, step=1)
    latent_dim = st.number_input("Dimension du latent", min_value=1, value=100, step=1)
    model_subdir = st.text_input("Sous-dossier du modèle", value=DEFAULT_MODEL_SUBDIR)
    num_images = st.slider("Nombre d'images", min_value=1, max_value=64, value=16)
    columns = st.slider("Colonnes d'affichage", min_value=1, max_value=8, value=4)

    use_seed = st.checkbox("Utiliser une graine aléatoire fixe", value=False)
    seed: Optional[int] = None
    if use_seed:
        seed = st.number_input("Graine", min_value=0, value=0, step=1)

    generate_button = st.button("Générer les images", type="primary")

if generate_button:
    try:
        generator, device = get_generator(
            latent_dim=latent_dim,
            model_epoch=model_epoch,
            model_subdir=model_subdir,
        )
    except FileNotFoundError as error:
        st.error(str(error))
    else:
        device = resolve_device(device)
        with st.spinner("Génération des images..."):
            images = generate_images(
                generator,
                num_samples=num_images,
                latent_dim=latent_dim,
                device=device,
                seed=seed,
            )
        st.session_state["gan_images"] = to_pil(images)
        st.session_state["gan_columns"] = columns
        st.success(f"{num_images} image(s) générée(s) avec succès.")


if "gan_images" in st.session_state:
    st.subheader("Résultats")
    display_grid(st.session_state["gan_images"], st.session_state.get("gan_columns", 4))
