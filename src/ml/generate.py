import os
from typing import Optional, Tuple

import torch
from torchvision.utils import save_image

try:
    from .gan import Generator  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback when run as a script
    from gan import Generator  # type: ignore[attr-defined]

DEFAULT_MODEL_SUBDIR = "v20_dynamic_gpu_safe"


def resolve_device(device: Optional[str] = None) -> str:
    """Return the device to use for inference."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_generator(
    latent_dim: int = 100,
    model_epoch: int = 100,
    device: Optional[str] = None,
    model_dir: Optional[str] = None,
    model_subdir: str = DEFAULT_MODEL_SUBDIR,
) -> Tuple[Generator, str]:
    """Instantiate the generator and load pre-trained weights."""
    device = resolve_device(device)

    generator = Generator(latent_dim=latent_dim).to(device)
    generator.eval()

    base_dir = model_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/model")
    )
    model_path = os.path.join(base_dir, model_subdir, f"G_epoch_{model_epoch}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Impossible de trouver le modèle : {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(state_dict)

    return generator, device


def generate_images(
    generator: Generator,
    num_samples: int,
    latent_dim: int,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate images in [0, 1] range."""
    device = device or next(generator.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

    z = torch.randn(num_samples, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        gen_imgs = generator(z)

    return (gen_imgs + 1) / 2


def save_generated_images(
    images: torch.Tensor,
    output_dir: Optional[str] = None,
    model_epoch: int = 100,
    run_name: str = "v20_dynamic",
    nrow: int = 4,
) -> None:
    """Persist generated images on disk as a grid and individually."""
    base_dir = output_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/generated")
    )
    os.makedirs(base_dir, exist_ok=True)

    grid_path = os.path.join(base_dir, f"generated_epoch_{model_epoch}_{run_name}.png")
    save_image(images, grid_path, nrow=nrow)
    print(f" Images générées en grille sauvegardées : {grid_path}")

    for i, img in enumerate(images):
        img_path = os.path.join(
            base_dir, f"generated_epoch_{model_epoch}_{i + 1}.png"
        )
        save_image(img.unsqueeze(0), img_path)
    print(f" {images.size(0)} images individuelles sauvegardées dans : {base_dir}")


def main() -> None:
    """CLI helper replicating the previous script behaviour."""
    latent_dim = 100
    num_samples = 16
    model_epoch = 100
    nrow = 4

    device = resolve_device()
    print(f"🚀 Utilisation de l'appareil : {device}\n")

    generator, _ = load_generator(latent_dim=latent_dim, model_epoch=model_epoch, device=device)
    print("✅ Modèle chargé")

    images = generate_images(generator, num_samples=num_samples, latent_dim=latent_dim, device=device)
    save_generated_images(
        images,
        model_epoch=model_epoch,
        run_name="v20_dynamic",
        nrow=nrow,
    )


if __name__ == "__main__":
    main()
