# fine_tune_continue_dynamic_fid_adjust_v3.py
import os, time, csv, shutil, pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.models import inception_v3, Inception_V3_Weights
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, random_split

from gan import Generator, Discriminator
from utils import TensorImageDataset

# -----------------------
# CSV REPORT: récupérer latent_dim et batch_size de v20
# -----------------------
REPORT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/reporting/fine_tuning_report.csv"))
assert os.path.exists(REPORT_CSV), f"❌ CSV introuvable : {REPORT_CSV}"
df = pd.read_csv(REPORT_CSV)
best_row = df.loc[df["version"] == "v20"].iloc[0]

# -----------------------
# CONFIG initiale
# -----------------------
CONFIG = {
    "version": "v20_dynamic_gpu_safe",
    "latent_dim": int(best_row["latent_dim"]),
    "batch_size": int(best_row["batch_size"]),
    "num_extra_epochs": 50,
    "fid_sample_size": 512
}

# -----------------------
# Device
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")

# -----------------------
# Modèle et CSV metrics
# -----------------------
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/model/{CONFIG['version']}"))

# Copier metrics v2 vers v3
OLD_CSV = os.path.join(MODEL_DIR, "metrics_dynamic_gpu_safe_v2.csv")
NEW_CSV = os.path.join(MODEL_DIR, "metrics_dynamic_gpu_safe_v3.csv")
if os.path.exists(OLD_CSV):
    shutil.copy2(OLD_CSV, NEW_CSV)
    print(f"✅ Copié {OLD_CSV} vers {NEW_CSV}")
else:
    raise FileNotFoundError(f"❌ Fichier introuvable : {OLD_CSV}")

METRICS_CSV = NEW_CSV

# Charger paramètres dynamiques depuis epoch 100
df_metrics = pd.read_csv(METRICS_CSV)
if 100 in df_metrics["epoch"].values:
    last_epoch_data = df_metrics[df_metrics["epoch"] == 100].iloc[0]
    CONFIG["lr_G"] = float(last_epoch_data["lr_G"])
    CONFIG["lr_D"] = float(last_epoch_data["lr_D"])
    CONFIG["label_smooth"] = float(last_epoch_data.get("label_smooth", 0.9))
    CONFIG["n_critic"] = int(last_epoch_data["n_critic_current"])
    CONFIG["g_steps"] = int(last_epoch_data["g_steps_current"])
else:
    raise ValueError("❌ Epoch 100 non trouvé dans le CSV metrics")

print("✅ Configuration finale pour continuation :")
for k, v in CONFIG.items():
    print(f"{k}: {v}")

# -----------------------
# DATASET
# -----------------------
FINAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/final"))
dataset = TensorImageDataset(FINAL_DIR)
total_size = len(dataset)
train_size = int(0.8*total_size)
val_size = int(0.1*total_size)
test_size = total_size - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader   = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False)

# -----------------------
# MODELS
# -----------------------
G = Generator(CONFIG["latent_dim"]).to(device)
D = Discriminator().to(device)

# Charger modèles existants à l'epoch 100
G_path = os.path.join(MODEL_DIR, "G_epoch_100.pth")
D_path = os.path.join(MODEL_DIR, "D_epoch_100.pth")
if os.path.exists(G_path): G.load_state_dict(torch.load(G_path, map_location=device))
if os.path.exists(D_path): D.load_state_dict(torch.load(D_path, map_location=device))
print("✅ Modèles existants chargés, continuation de l'entraînement")

# Spectral Normalization
for name, module in D.named_children():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        D._modules[name] = nn.utils.spectral_norm(module)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=CONFIG["lr_G"], betas=(0.5,0.999))
optimizer_D = optim.Adam(D.parameters(), lr=CONFIG["lr_D"], betas=(0.5,0.999))

# -----------------------
# EMA Generator
# -----------------------
ema_G = Generator(CONFIG["latent_dim"]).to(device)
ema_G.load_state_dict(G.state_dict())
ema_decay = 0.999
def update_ema(G, ema_G, decay):
    with torch.no_grad():
        for p_ema, p in zip(ema_G.parameters(), G.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1-decay)

# -----------------------
# Inception / FID
# -----------------------
inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
inception_model.eval()
if hasattr(inception_model,'AuxLogits'): inception_model.AuxLogits=None

fid_metric = FrechetInceptionDistance(feature=2048).to(device)
imagenet_mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
imagenet_std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

@torch.no_grad
def compute_inception_score(gen_imgs):
    imgs = F.interpolate(gen_imgs, size=(299,299), mode='bilinear', align_corners=False)
    imgs = (imgs.clamp(-1,1)+1)/2.0
    imgs = (imgs - imagenet_mean)/imagenet_std
    logits = inception_model(imgs)
    probs = F.softmax(logits, dim=1)
    p_y = probs.mean(dim=0, keepdim=True)
    kl = probs * (torch.log(probs+1e-8) - torch.log(p_y+1e-8))
    return torch.exp(kl.sum(dim=1).mean()).item()

# -----------------------
# Dynamic params
# -----------------------
n_critic_current = CONFIG["n_critic"]
g_steps_current = CONFIG["g_steps"]
max_lr, min_lr = 5e-4, 1e-6
prev_fid = None
fake_threshold = 0.85  # seuil minimal pour fake_correct
IS_prev = None

# -----------------------
# Déterminer l’epoch de départ
# -----------------------
start_epoch = 100

# -----------------------
# TRAIN LOOP
# -----------------------
start_time = time.time()
for epoch in range(start_epoch, start_epoch + CONFIG["num_extra_epochs"]):
    print(f"\n=== Epoch {epoch+1} ===")

    for phase in ["train", "val"]:
        is_train = phase=="train"
        G.train(is_train); D.train(is_train)
        loader = train_loader if is_train else val_loader

        loss_D_total=0; loss_G_total=0
        fake_corrects = []

        for imgs,_ in tqdm(loader, desc=f"{phase} epoch {epoch+1}", leave=False):
            B = imgs.size(0)
            real_imgs = imgs.to(device)
            valid = torch.ones(B,device=device)*CONFIG["label_smooth"]
            fake_label = torch.zeros(B,device=device)

            # --- D ---
            for _ in range(n_critic_current):
                optimizer_D.zero_grad()
                z = torch.randn(B, CONFIG["latent_dim"],1,1, device=device)
                gen_imgs = G(z)
                out_real = D(real_imgs)
                out_fake = D(gen_imgs.detach())
                loss_D = (criterion(out_real, valid) + criterion(out_fake, fake_label))/2
                if is_train: loss_D.backward(); optimizer_D.step()

            # --- G ---
            for _ in range(g_steps_current):
                optimizer_G.zero_grad()
                z = torch.randn(B, CONFIG["latent_dim"],1,1, device=device)
                gen_imgs = G(z)
                loss_G = criterion(D(gen_imgs), valid)
                if is_train: loss_G.backward(); optimizer_G.step()

            if is_train: update_ema(G, ema_G, ema_decay)
            with torch.no_grad():
                out_fake = D(gen_imgs)
                fake_corrects.append((out_fake<0.5).float().mean().item())

            loss_D_total += loss_D.item()
            loss_G_total += loss_G.item()
            torch.cuda.empty_cache()

        # ---- Epoch stats
        loss_D_epoch = loss_D_total/len(loader)
        loss_G_epoch = loss_G_total/len(loader)
        fake_correct = np.mean(fake_corrects)

        # ---- FID / IS
        with torch.no_grad():
            fid_metric.reset()
            for _ in range(CONFIG["fid_sample_size"]//CONFIG["batch_size"]):
                z = torch.randn(CONFIG["batch_size"], CONFIG["latent_dim"],1,1, device=device)
                fake_imgs = ema_G(z)
                real_batch = next(iter(val_loader))[0].to(device)
                fid_metric.update(((real_batch.clamp(-1,1)+1)*127.5).to(torch.uint8), real=True)
                fid_metric.update(((fake_imgs.clamp(-1,1)+1)*127.5).to(torch.uint8), real=False)
            fid_value = fid_metric.compute().item()
            IS_value = compute_inception_score(fake_imgs)

        # ---- Ajustement dynamique si FID, IS ou fake_correct déclenche
        adjust = False
        if prev_fid is not None and (fid_value > prev_fid or fake_correct < fake_threshold or (IS_prev is not None and IS_value < IS_prev)):
            adjust = True
            for g in optimizer_G.param_groups:
                g['lr'] = max(min_lr, g['lr'] / 1.1)
            for g in optimizer_D.param_groups:
                g['lr'] = max(min_lr, g['lr'] / 1.1)
            n_critic_current = min(n_critic_current + 1, 10)
            g_steps_current = min(g_steps_current + 1, 5)
            print(f"⚠️ Ajustement dynamique appliqué : FID, IS ou fake_correct a déclenché")

        prev_fid = fid_value
        IS_prev = IS_value

        # ---- Affichage console métriques et params
        print(f"[{phase.upper()}] lossD={loss_D_epoch:.4f}, lossG={loss_G_epoch:.4f}, "
              f"FID={fid_value:.2f}, IS={IS_value:.2f}, fake_correct={fake_correct:.3f}, "
              f"lr_D={optimizer_D.param_groups[0]['lr']:.6f}, lr_G={optimizer_G.param_groups[0]['lr']:.6f}, "
              f"n_critic={n_critic_current}, g_steps={g_steps_current}")

        # ---- Save metrics
        with open(METRICS_CSV,"a",newline="") as f:
            csv.DictWriter(f, fieldnames=df_metrics.columns).writerow({
                "epoch":epoch+1,"phase":phase,"loss_D":loss_D_epoch,"loss_G":loss_G_epoch,
                "D_real_mean":0,"D_fake_mean":0,"entropy_G":0,
                "real_correct":0,"fake_correct":fake_correct,
                "diversity":0,"FID":fid_value,"Inception_Score":IS_value,
                "n_critic_current":n_critic_current,"g_steps_current":g_steps_current,
                "lr_D":optimizer_D.param_groups[0]['lr'],"lr_G":optimizer_G.param_groups[0]['lr']
            })

    # ---- Save models & samples tous les 5 epochs
    if (epoch+1) % 5 == 0:
        torch.save(G.state_dict(), os.path.join(MODEL_DIR, f"G_epoch_{epoch+1}_v3.pth"))
        torch.save(D.state_dict(), os.path.join(MODEL_DIR, f"D_epoch_{epoch+1}_v3.pth"))
        z = torch.randn(16, CONFIG["latent_dim"],1,1, device=device)
        with torch.no_grad(): fake_grid = (ema_G(z).clamp(-1,1)+1)/2
        save_image(fake_grid, os.path.join(MODEL_DIR, f"samples_epoch_{epoch+1}_v3.png"), nrow=4, normalize=True)
        print(f"💾 Saved models and sample grid at epoch {epoch+1}")

print(f"✅ Fine-tuning dynamique terminé en {time.time()-start_time:.1f}s")
