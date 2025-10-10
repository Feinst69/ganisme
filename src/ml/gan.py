import torch
import torch.nn as nn

# --------------------
# Générateur DCGAN
# # --------------------
# class Generator(nn.Module):
#     def __init__(self, latent_dim=100, ngf=64, nc=3):
#         super(Generator, self).__init__()
#         self.latent_dim = latent_dim  # <-- stocker la dimension du bruit
#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(self.latent_dim, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()  
#         )

#     def forward(self, z):
#         # utiliser self.latent_dim au lieu de latent_dim global
#         return self.main(z.view(z.size(0), self.latent_dim, 1, 1))


# # --------------------
# # Discriminateur DCGAN
# # --------------------
# class Discriminator(nn.Module):
#     def __init__(self, ndf=64, nc=3):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf*4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf*8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid() 
#         )

#     def forward(self, x):
#         return self.main(x).view(-1, 1).squeeze(1)




# --------------------
# 🔹 Générateur DCGAN (128x128)
# --------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            # Entrée: Z latent vector → 4x4
            nn.ConvTranspose2d(self.latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 4x4 → 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 8x8 → 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 16x16 → 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 32x32 → 64x64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            # 🆕 64x64 → 128x128
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z.view(z.size(0), self.latent_dim, 1, 1))


# --------------------
# 🔹 Discriminateur DCGAN (128x128) — avec Dropout & activation paramétrable
# --------------------
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, dropout_rate=0.3, activation='leakyrelu'):
        """
        Args:
            ndf (int): nombre de filtres de base
            nc (int): nombre de canaux (3 pour RGB)
            dropout_rate (float): taux de dropout entre 0 et 1
            activation (str): type d'activation ('leakyrelu', 'relu', 'elu', 'gelu')
        """
        super(Discriminator, self).__init__()

        # Sélection de l’activation
        if activation.lower() == 'leakyrelu':
            act_layer = lambda: nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'relu':
            act_layer = lambda: nn.ReLU(inplace=True)
        elif activation.lower() == 'elu':
            act_layer = lambda: nn.ELU(inplace=True)
        elif activation.lower() == 'gelu':
            act_layer = lambda: nn.GELU()
        else:
            raise ValueError(f"Activation non reconnue : {activation}")

        self.main = nn.Sequential(
            # 128x128 → 64x64
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            act_layer(),
            nn.Dropout(dropout_rate),

            # 64x64 → 32x32
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            act_layer(),
            nn.Dropout(dropout_rate),

            # 32x32 → 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            act_layer(),
            nn.Dropout(dropout_rate),

            # 16x16 → 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            act_layer(),
            nn.Dropout(dropout_rate),

            # 8x8 → 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            act_layer(),
            nn.Dropout(dropout_rate),

            # 4x4 → 1x1 (sortie)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)