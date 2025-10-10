# train_gan_rl.py
import os, time, csv, pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

from gan import Generator, Discriminator
from utils import TensorImageDataset
from controller_rl import RLAgent
from metrics_utils import setup_inception_and_fid, compute_inception_score, compute_reward
from dynamic_utils import apply_rl_action
from torch.utils.data import DataLoader, random_split


# -----------------------
# CONFIG / LOAD CSV
# -----------------------
REPORT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/reporting/fine_tuning_report.csv"))
assert os.path.exists(REPORT_CSV)
df = pd.read_csv(REPORT_CSV)
best_row = df.loc[df["version"]=="v20"].iloc[0]

CONFIG = {
    "version": best_row["version"],
    "lr_G": float(best_row["lr_G"]),
    "lr_D": float(best_row["lr_D"]),
    "batch_size": int(best_row["batch_size"]),
    "latent_dim": int(best_row["latent_dim"]),
    "label_smooth": float(best_row["label_smooth"]),
    "n_critic": int(best_row["n_critic"]),
    "num_epochs": 100,
    "subset_ratio": 0.01,
    "fid_sample_size": 512
}

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = f"../data/model/{CONFIG['version']}"
os.makedirs(MODEL_DIR, exist_ok=True)
METRICS_CSV = os.path.join(MODEL_DIR, "metrics_v20_rl.csv")

# -----------------------
# DATASET
# -----------------------
FINAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/final"))
dataset = TensorImageDataset(FINAL_DIR)
print(f"Using full dataset: {len(dataset)}")

total_size = len(dataset)
train_size = int(0.8*total_size)
val_size = int(0.1*total_size)
test_size = total_size - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader   = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False)

# -------- MODELS --------
G = Generator(CONFIG["latent_dim"]).to(device)
D = Discriminator().to(device)
criterion = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=CONFIG["lr_G"], betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=CONFIG["lr_D"], betas=(0.5, 0.999))

# EMA
ema_G = Generator(CONFIG["latent_dim"]).to(device)
ema_G.load_state_dict(G.state_dict())
ema_decay = 0.999

def update_ema(G, ema_G, decay):
    with torch.no_grad():
        for p_ema, p in zip(ema_G.parameters(), G.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1 - decay)

# -------- RL Controller --------
agent = RLAgent(device=device)
max_lr, min_lr = 5e-4, 1e-6
n_critic, g_steps = CONFIG["n_critic"], 1

# -------- Metrics --------
inception_model, fid_metric, mean, std = setup_inception_and_fid(device)
fields = ["epoch","phase","loss_D","loss_G","FID","IS","fake_correct","lr_G","lr_D","n_critic","g_steps","reward"]
with open(METRICS_CSV,"w",newline="") as f: csv.DictWriter(f,fieldnames=fields).writeheader()

# -------- TRAIN LOOP --------
for epoch in range(CONFIG["num_epochs"]):
    for phase in ["train","val"]:
        is_train = (phase=="train")
        G.train(is_train); D.train(is_train)
        loader = train_loader if is_train else val_loader
        loss_D_total, loss_G_total, fake_corrects = 0, 0, []
        fid_metric.reset()

        for imgs,_ in tqdm(loader,desc=f"{phase} {epoch+1}",leave=False):
            B = imgs.size(0)
            imgs = imgs.to(device)
            valid = torch.ones(B,device=device)*CONFIG["label_smooth"]
            fake = torch.zeros(B,device=device)

            # D
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                z = torch.randn(B,CONFIG["latent_dim"],1,1,device=device)
                gen = G(z)
                loss_D = (criterion(D(imgs),valid)+criterion(D(gen.detach()),fake))/2
                if is_train: loss_D.backward(); optimizer_D.step()

            # G
            for _ in range(g_steps):
                optimizer_G.zero_grad()
                z = torch.randn(B,CONFIG["latent_dim"],1,1,device=device)
                gen = G(z)
                loss_G = criterion(D(gen),valid)
                if is_train: loss_G.backward(); optimizer_G.step()

            if is_train: update_ema(G,ema_G,ema_decay)
            with torch.no_grad():
                fake_corrects.append((D(gen)<0.5).float().mean().item())

            loss_D_total += loss_D.item(); loss_G_total += loss_G.item()

        lossD = loss_D_total/len(loader)
        lossG = loss_G_total/len(loader)
        fake_corr = np.mean(fake_corrects)

        # FID + IS
        with torch.no_grad():
            fid_metric.reset()
            for _ in range(8):
                z = torch.randn(CONFIG["batch_size"],CONFIG["latent_dim"],1,1,device=device)
                fake_imgs = ema_G(z)
                real_batch = next(iter(val_loader))[0].to(device)
                fid_metric.update(((real_batch.clamp(-1,1)+1)*127.5).to(torch.uint8), real=True)
                fid_metric.update(((fake_imgs.clamp(-1,1)+1)*127.5).to(torch.uint8), real=False)
            FID = fid_metric.compute().item()
            IS = compute_inception_score(inception_model,fake_imgs,mean,std)

        reward = compute_reward(FID,IS,fake_corr)

        if phase=="val":
            state = torch.tensor([FID/300,IS/5,fake_corr,
                                  optimizer_G.param_groups[0]['lr']/max_lr,
                                  optimizer_D.param_groups[0]['lr']/max_lr,
                                  n_critic/10,g_steps/5],device=device)
            action, logp = agent.select_action(state.unsqueeze(0))
            optimizer_G, optimizer_D, n_critic, g_steps = apply_rl_action(action.squeeze().tolist(),
                                                                          optimizer_G,optimizer_D,n_critic,g_steps,min_lr,max_lr)
            agent.update(logp,reward)

        with open(METRICS_CSV,"a",newline="") as f:
            csv.DictWriter(f,fieldnames=fields).writerow({
                "epoch":epoch+1,"phase":phase,"loss_D":lossD,"loss_G":lossG,
                "FID":FID,"IS":IS,"fake_correct":fake_corr,
                "lr_G":optimizer_G.param_groups[0]['lr'],"lr_D":optimizer_D.param_groups[0]['lr'],
                "n_critic":n_critic,"g_steps":g_steps,"reward":reward
            })

    if (epoch+1)%5==0:
        torch.save(G.state_dict(),os.path.join(MODEL_DIR,f"G_epoch_{epoch+1}_rl.pth"))
        torch.save(D.state_dict(),os.path.join(MODEL_DIR,f"D_epoch_{epoch+1}_rl.pth"))
        z = torch.randn(16,CONFIG["latent_dim"],1,1,device=device)
        with torch.no_grad(): grid=(ema_G(z).clamp(-1,1)+1)/2
        save_image(grid, os.path.join(MODEL_DIR,f"samples_{epoch+1}.png"), nrow=4)

print("✅ Finished training with RL controller")
