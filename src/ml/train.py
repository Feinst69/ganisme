import os
import csv
import time
import numpy as np
from itertools import product
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from gan import Generator, Discriminator
from utils import TensorImageDataset

# ==========================================================
# PARAMÈTRES À FINE-TUNER
# ==========================================================
lr_G_list = [0.0002, 0.0004]
lr_D_list = [0.00005, 0.0001]
batch_size_list = [128]
latent_dim_list = [100]
label_smooth_list = [0.9]
label_noise_list = [0.0, 0.05]
n_critic_list = [1]
n_gen_steps_list = [2]
dropout_D_list = [0.0, 0.3]
activation_D_list = ["LeakyReLU", "ReLU"]
num_epochs = 50
max_images = 5000

# ==========================================================
# DEVICE
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Utilisation de l'appareil : {device}")

# ==========================================================
# DOSSIERS
# ==========================================================
base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/model"))
os.makedirs(base_model_dir, exist_ok=True)
report_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/reporting"))
os.makedirs(report_dir, exist_ok=True)
report_csv = os.path.join(report_dir, "fine_tuning_report_split.csv")

# ==========================================================
# DATASET
# ==========================================================
final_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/final"))
dataset = TensorImageDataset(final_dir)

# Limiter le dataset à max_images
if len(dataset) > max_images:
    dataset, _ = random_split(dataset, [max_images, len(dataset) - max_images])

# Split train / val (90 / 10)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ==========================================================
# CSV GLOBAL REPORT
# ==========================================================
if not os.path.exists(report_csv):
    with open(report_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "version", "lr_G", "lr_D", "batch_size", "latent_dim",
            "label_smooth", "label_noise", "n_critic", "n_gen_steps",
            "dropout_D", "activation_D",
            "epoch", "loss_D", "loss_G",
            "precision_train", "recall_train", "f1_train", "acc_train",
            "precision_val", "recall_val", "f1_val", "acc_val",
            "training_time_sec"
        ])
        writer.writeheader()

# ==========================================================
# FONCTION D'ÉVALUATION
# ==========================================================
def evaluate(D, G, loader, latent_dim, device):
    D.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            batch_size_actual = imgs.size(0)
            z = torch.randn(batch_size_actual, latent_dim, 1, 1, device=device)
            gen_imgs = G(z)

            D_real_out = torch.sigmoid(D(imgs))
            D_fake_out = torch.sigmoid(D(gen_imgs))

            y_true = torch.cat([torch.ones_like(D_real_out), torch.zeros_like(D_fake_out)]).cpu().numpy()
            y_pred = torch.cat([D_real_out, D_fake_out]).cpu().numpy()
            y_pred_labels = (y_pred > 0.5).astype(int)

            all_true.append(y_true)
            all_pred.append(y_pred_labels)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred)
    }

# ==========================================================
# BOUCLE GRID SEARCH AVEC SKIP
# ==========================================================
version_counter = 1
for (lr_G, lr_D, batch_size, latent_dim, label_smooth, label_noise,
     n_critic, n_gen_steps, dropout_D, activation_D) in product(
        lr_G_list, lr_D_list, batch_size_list, latent_dim_list,
        label_smooth_list, label_noise_list,
        n_critic_list, n_gen_steps_list, dropout_D_list, activation_D_list):

    version_name = f"v20_{version_counter}"
    model_dir = os.path.join(base_model_dir, version_name)
    os.makedirs(model_dir, exist_ok=True)

    # --- Vérifier si le modèle existe déjà ---
    existing_epochs = [
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir(model_dir)
        if f.startswith("G_epoch_")
    ]
    if existing_epochs and max(existing_epochs) >= num_epochs:
        print(f"⚡ Skipping {version_name} car déjà entraîné.")
        version_counter += 1
        continue  # passe à la combinaison suivante

    print(f"\n===== Entraînement {version_name} =====")
    version_counter += 1

    # === DataLoaders ===
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Modèles ===
    G = Generator(latent_dim=latent_dim).to(device)
    D = Discriminator(dropout_rate=dropout_D, activation=activation_D).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    start_epoch = max(existing_epochs) + 1 if existing_epochs else 0
    if start_epoch > 0:
        print(f"⚡ Reprise depuis l'époque {start_epoch}")

    start_time = time.time()

    # ======================================================
    # BOUCLE D’ENTRAÎNEMENT
    # ======================================================
    for epoch in range(start_epoch, num_epochs):
        G.train(); D.train()
        loop = tqdm(train_loader, desc=f"{version_name} Epoch [{epoch+1}/{num_epochs}]", leave=False)

        epoch_loss_D = 0
        epoch_loss_G = 0
        num_batches = 0

        for imgs, _ in loop:
            batch_size_actual = imgs.size(0)
            real_imgs = imgs.to(device)

            # Labels avec lissage et bruit
            valid = torch.ones(batch_size_actual, device=device) * label_smooth
            valid += torch.randn_like(valid) * label_noise
            valid = valid.clamp(0, 1)
            fake = torch.zeros(batch_size_actual, device=device)
            fake += torch.randn_like(fake) * label_noise
            fake = fake.clamp(0, 1)

            # === Entraînement Discriminateur ===
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                z = torch.randn(batch_size_actual, latent_dim, 1, 1, device=device)
                gen_imgs = G(z)
                loss_real = criterion(D(real_imgs), valid)
                loss_fake = criterion(D(gen_imgs.detach()), fake)
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                optimizer_D.step()
                epoch_loss_D += loss_D.item()

            # === Entraînement Générateur ===
            for _ in range(n_gen_steps):
                optimizer_G.zero_grad()
                z = torch.randn(batch_size_actual, latent_dim, 1, 1, device=device)
                gen_imgs = G(z)
                loss_G = criterion(D(gen_imgs), valid)
                loss_G.backward()
                optimizer_G.step()
                epoch_loss_G += loss_G.item()

            num_batches += 1

        # Moyenne des losses par epoch
        epoch_loss_D /= num_batches
        epoch_loss_G /= num_batches

        # Évaluation train / val
        metrics_train = evaluate(D, G, train_loader, latent_dim, device)
        metrics_val = evaluate(D, G, val_loader, latent_dim, device)
        elapsed_time = time.time() - start_time

        # Affichage
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss D: {epoch_loss_D:.4f}, Loss G: {epoch_loss_G:.4f} | "
            f"Train Acc: {metrics_train['accuracy']:.4f}, "
            f"Val Acc: {metrics_val['accuracy']:.4f}"
        )

        # Sauvegarde des modèles
        torch.save(G.state_dict(), os.path.join(model_dir, f"G_epoch_{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(model_dir, f"D_epoch_{epoch+1}.pth"))

        # Sauvegarde CSV
        with open(report_csv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "version", "lr_G", "lr_D", "batch_size", "latent_dim",
                "label_smooth", "label_noise", "n_critic", "n_gen_steps",
                "dropout_D", "activation_D",
                "epoch", "loss_D", "loss_G",
                "precision_train", "recall_train", "f1_train", "acc_train",
                "precision_val", "recall_val", "f1_val", "acc_val",
                "training_time_sec"
            ])
            writer.writerow({
                "version": version_name,
                "lr_G": lr_G,
                "lr_D": lr_D,
                "batch_size": batch_size,
                "latent_dim": latent_dim,
                "label_smooth": label_smooth,
                "label_noise": label_noise,
                "n_critic": n_critic,
                "n_gen_steps": n_gen_steps,
                "dropout_D": dropout_D,
                "activation_D": activation_D,
                "epoch": epoch + 1,
                "loss_D": epoch_loss_D,
                "loss_G": epoch_loss_G,
                "precision_train": metrics_train["precision"],
                "recall_train": metrics_train["recall"],
                "f1_train": metrics_train["f1"],
                "acc_train": metrics_train["accuracy"],
                "precision_val": metrics_val["precision"],
                "recall_val": metrics_val["recall"],
                "f1_val": metrics_val["f1"],
                "acc_val": metrics_val["accuracy"],
                "training_time_sec": elapsed_time
            })

    print(f"✅ Fine-tuning {version_name} terminé en {elapsed_time:.2f} sec.")
