import pandas as pd
import os

data_file = "./data/processed/"



from pathlib import Path

# Chemin du dossier
dossier = Path(data_file)

# Compter les fichiers images
nb_images = sum(1 for f in dossier.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"])

print(f"Nombre d'images : {nb_images}")
