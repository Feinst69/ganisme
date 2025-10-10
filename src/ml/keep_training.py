import os
import time
import csv
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from gan import Generator, Discriminator
from utils import TensorImageDataset  

# ===============================
# 1️⃣ CONFIGURATION
# ===============================
REPORT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/reporting/fine_tuning_report.csv"))
assert os.path.exists(REPORT_CSV), f"❌ CSV introuvable : {REPORT_CSV}"

df = pd.read_csv(REPORT_CSV)
best_row = df.loc[df['version'] == 'v20'].iloc[0]  # sélection v20

CONFIG = {
    "version": best_row["version"],
    "lr_G": float(best_row["lr_G"]),
    "lr_D": float(best_row["lr_D"]),
    "batch_size": int(best_row["batch_size"]),
    "latent_dim": int(best_row["latent_dim"]),
    "label_smooth": float(best_row["label_smooth"]),
    "n_critic": int(best_row["n_critic"]) + 1,
    "num_epochs": 100,  # nombre d'epochs supplémentaires
    "dropout_D": 0.0,
    "activation_D": "ReLU"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/model"))
MODEL_DIR = os.path.join(BASE_MODEL_DIR, CONFIG["version"] + "_final")
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 2️⃣ CHEMINS DES CHECKPOINTS
# ===============================
GEN_PATH = os.path.join(MODEL_DIR, "generator_epoch_100.pth")
DIS_PATH = os.path.join(MODEL_DIR, "discriminator_epoch_100.pth")
METRICS_CSV = os.path.join(MODEL_DIR, "continued_training_metrics.csv")

# ===============================
# 3️⃣ DATASET
# ===============================
FINAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/final"))
dataset = TensorImageDataset(FINAL_DIR)

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], shuffle=False)

# ===============================
# 4️⃣ INITIALISATION DES MODÈLES
# ===============================
G = Generator(latent_dim=CONFIG["latent_dim"]).to(device)
D = Discriminator(dropout_rate=CONFIG["dropout_D"], activation=CONFIG["activation_D"]).to(device)

# Charger les checkpoints existants
if os.path.exists(GEN_PATH):
    print("🔄 Chargement du Generator existant...")
    G.load_state_dict(torch.load(GEN_PATH, map_location=device))

if os.path.exists(DIS_PATH):
    print("🔄 Chargement du Discriminator existant...")
    state_dict_D = torch.load(DIS_PATH, map_location=device)
    model_dict = D.state_dict()
    compatible_dict = {k: v for k, v in state_dict_D.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(compatible_dict)
    D.load_state_dict(model_dict)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=CONFIG["lr_G"], betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=CONFIG["lr_D"], betas=(0.5, 0.999))

# ===============================
# 5️⃣ CSV MÉTRIQUES
# ===============================
with open(METRICS_CSV, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "epoch", "phase", "loss_D", "loss_G",
        "D_real_mean", "D_fake_mean",
        "D_real_std", "D_fake_std",
        "entropy_G", "real_correct", "fake_correct", "diversity"
    ])
    writer.writeheader()

# ===============================
# 6️⃣ ENTRAÎNEMENT
# ===============================
START_EPOCH = 100  # continuation à partir de l'epoch 100
END_EPOCH = START_EPOCH + CONFIG["num_epochs"]

start_time = time.time()

for epoch in range(START_EPOCH, END_EPOCH):
    for phase in ["train", "val"]:
        G.train() if phase=="train" else G.eval()
        D.train() if phase=="train" else D.eval()
        loader = train_loader if phase=="train" else val_loader

        loss_D_total, loss_G_total = 0.0, 0.0

        for imgs, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{END_EPOCH} ({phase})", leave=False):
            batch_size_actual = imgs.size(0)
            real_imgs = imgs.to(device)
            valid = torch.ones(batch_size_actual, device=device) * CONFIG["label_smooth"]
            fake = torch.zeros(batch_size_actual, device=device)

            # --- Discriminateur ---
            for _ in range(CONFIG["n_critic"]):
                optimizer_D.zero_grad()
                z = torch.randn(batch_size_actual, CONFIG["latent_dim"], 1, 1, device=device)
                gen_imgs = G(z)
                loss_real = criterion(D(real_imgs), valid)
                loss_fake = criterion(D(gen_imgs.detach()), fake)
                loss_D = (loss_real + loss_fake) / 2
                if phase=="train":
                    loss_D.backward()
                    optimizer_D.step()

            # --- Générateur ---
            optimizer_G.zero_grad()
            z = torch.randn(batch_size_actual, CONFIG["latent_dim"], 1, 1, device=device)
            gen_imgs = G(z)
            loss_G = criterion(D(gen_imgs), valid)
            if phase=="train":
                loss_G.backward()
                optimizer_G.step()

            # --- Métriques ---
            with torch.no_grad():
                D_real_out = D(real_imgs)
                D_fake_out = D(gen_imgs)
                D_real_mean = D_real_out.mean().item()
                D_fake_mean = D_fake_out.mean().item()
                D_real_std = D_real_out.std().item()
                D_fake_std = D_fake_out.std().item()
                gen_probs = torch.sigmoid(D_fake_out)
                entropy_G = -(gen_probs * torch.log(gen_probs + 1e-8) +
                              (1 - gen_probs) * torch.log(1 - gen_probs + 1e-8)).mean().item()
                real_correct = (D_real_out > 0.5).float().mean().item()
                fake_correct = (D_fake_out < 0.5).float().mean().item()
                gen_imgs_flat = gen_imgs.view(batch_size_actual, -1)
                diversity = torch.mean(torch.pdist(gen_imgs_flat, p=2)).item()

            loss_D_total += loss_D.item()
            loss_G_total += loss_G.item()

        # Sauvegarde métriques
        with open(METRICS_CSV, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "phase", "loss_D", "loss_G",
                "D_real_mean", "D_fake_mean",
                "D_real_std", "D_fake_std",
                "entropy_G", "real_correct", "fake_correct", "diversity"
            ])
            writer.writerow({
                "epoch": epoch+1, "phase": phase,
                "loss_D": loss_D_total/len(loader),
                "loss_G": loss_G_total/len(loader),
                "D_real_mean": D_real_mean,
                "D_fake_mean": D_fake_mean,
                "D_real_std": D_real_std,
                "D_fake_std": D_fake_std,
                "entropy_G": entropy_G,
                "real_correct": real_correct,
                "fake_correct": fake_correct,
                "diversity": diversity
            })

    # Sauvegarde et exemples générés tous les 10 epochs
    if (epoch+1) % 10 == 0 or (epoch+1) == END_EPOCH:
        torch.save(G.state_dict(), os.path.join(MODEL_DIR, f"generator_epoch_{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(MODEL_DIR, f"discriminator_epoch_{epoch+1}.pth"))
        print(f"💾 Modèles et échantillons sauvegardés à l'époque {epoch+1}")

print(f"✅ Entraînement terminé en {time.time()-start_time:.2f} sec.")
