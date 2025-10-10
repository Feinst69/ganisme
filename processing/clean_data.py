import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

# === Configuration ===
SOURCE_DIR = "./data/processed"
FINAL_DIR = "./data/final"
IMAGE_SIZE = 128

os.makedirs(FINAL_DIR, exist_ok=True)

# === Transformation des images ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === Traitement des images ===
corrupted_files = []

print("🔄 Démarrage du traitement des images...\n")

for filename in tqdm(os.listdir(SOURCE_DIR), desc="Traitement des fichiers"):
    # Vérifie l’extension du fichier
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    src_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(FINAL_DIR, filename)
    dest_path_pt = os.path.splitext(dest_path)[0] + ".pt"

    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            tensor_img = transform(img)
            torch.save(tensor_img, dest_path_pt)

    except Exception as e:
        corrupted_files.append(filename)
        continue  # Ignore les fichiers corrompus

# === Sauvegarde de la liste des fichiers corrompus ===
if corrupted_files:
    corrupted_txt = os.path.join(FINAL_DIR, "corrupted_files.txt")
    with open(corrupted_txt, "w") as f:
        f.write("\n".join(corrupted_files))
    print(f"\n⚠️ {len(corrupted_files)} fichiers corrompus ignorés.")
    print(f"📄 Liste sauvegardée dans : {corrupted_txt}")
else:
    print("\n✅ Aucun fichier corrompu détecté.")
