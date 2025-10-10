# import pandas as pd
# import os

# data_file = "./data/processed/"



# from pathlib import Path

# # Chemin du dossier
# dossier = Path(data_file)

# # Compter les fichiers images
# nb_images = sum(1 for f in dossier.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"])

# print(f"Nombre d'images : {nb_images}")


import torch

# print(torch.cuda.is_available())  # True si PyTorch détecte le GPU
# print(torch.cuda.device_count())  # Nombre de GPUs disponibles
# print(torch.cuda.get_device_name(0))  # Nom du premier GPU


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)