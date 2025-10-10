import os
import torch
from torchvision.utils import save_image
from gan import Generator

# -----------------------
# CONFIG
# -----------------------
latent_dim = 100
num_samples = 16
model_epoch = 100
nrow = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Utilisation de l'appareil : {device}\n")

# -----------------------
# LOAD GENERATOR
# -----------------------
G = Generator(latent_dim=latent_dim).to(device)
G.eval()

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/model"))
model_path = os.path.join(model_dir, f"v20_dynamic_gpu_safe/G_epoch_{model_epoch}.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Impossible de trouver le modèle : {model_path}")

G.load_state_dict(torch.load(model_path, map_location=device))
print(f"✅ Modèle chargé : {model_path}")

# -----------------------
# GENERATE IMAGES
# -----------------------
z = torch.randn(num_samples, latent_dim, 1, 1, device=device)

with torch.no_grad():
    gen_imgs = G(z)
    gen_imgs = (gen_imgs + 1) / 2  # rescale [-1,1] -> [0,1]

# -----------------------
# SAVE IMAGES
# -----------------------
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/generated"))
os.makedirs(output_dir, exist_ok=True)

# Save grid
grid_path = os.path.join(output_dir, f"generated_epoch_{model_epoch}_v20_dynamic.png")
save_image(gen_imgs, grid_path, nrow=nrow)
print(f" Images générées en grille sauvegardées : {grid_path}")

# Save individual images
for i, img in enumerate(gen_imgs):
    img_path = os.path.join(output_dir, f"generated_epoch_{model_epoch}_{i+1}.png")
    save_image(img.unsqueeze(0), img_path)  # ajoute la dimension batch
print(f" {num_samples} images individuelles sauvegardées dans : {output_dir}")
