import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from gan import Generator, Discriminator
from utils import TensorImageDataset


lr = 0.0002
num_epochs = 50
latent_dim = 100
batch_size = 64
max_images = 3000  # nombre maximal d'images à utiliser


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Utilisation de l'appareil : {device}\n")


G = Generator().to(device)
D = Discriminator().to(device)


criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


final_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/final"))
dataset = TensorImageDataset(final_dir)


if len(dataset) > max_images:
    indices = random.sample(range(len(dataset)), max_images)
    dataset = Subset(dataset, indices)
    print(f"⚠️ Utilisation d’un sous-ensemble de {max_images} images pour les tests.")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"✅ Dataset chargé : {len(dataset)} images trouvées dans {final_dir}\n")


for epoch in range(num_epochs):
    loop = tqdm(dataloader, desc=f"🧠 Epoch [{epoch+1}/{num_epochs}]", leave=False)

    for imgs, _ in loop:
        batch_size = imgs.size(0)
        real_imgs = imgs.to(device)


        valid = torch.ones(batch_size, device=device)
        fake = torch.zeros(batch_size, device=device)

        # --- Entraînement du Discriminateur ---
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)  
        gen_imgs = G(z)
        loss_real = criterion(D(real_imgs), valid)
        loss_fake = criterion(D(gen_imgs.detach()), fake)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # --- Entraînement du Générateur ---
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        gen_imgs = G(z)
        loss_G = criterion(D(gen_imgs), valid)
        loss_G.backward()
        optimizer_G.step()

        loop.set_postfix({
            "Loss D": f"{loss_D.item():.4f}",
            "Loss G": f"{loss_G.item():.4f}"
        })

    tqdm.write(f"Epoch {epoch+1}/{num_epochs} terminée | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")


    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/model"))
        os.makedirs(save_dir, exist_ok=True)

        torch.save(G.state_dict(), os.path.join(save_dir, f"generator_epoch_{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(save_dir, f"discriminator_epoch_{epoch+1}.pth"))

        tqdm.write(f"💾 Modèles sauvegardés à l’epoch {epoch+1} dans : {save_dir}")