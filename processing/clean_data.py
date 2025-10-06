import os
from PIL import Image
from torchvision import transforms
import torch


source_dir = "./data/processed"
final_dir = "./data/final"
os.makedirs(final_dir, exist_ok=True)


image_size = 64
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

for filename in os.listdir(source_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(final_dir, filename)

        with Image.open(src_path) as img:
            img = img.convert("RGB") 
            tensor_img = transform(img) 

           
            torch.save(tensor_img, dest_path.replace(".jpg", ".pt").replace(".png", ".pt"))
        print(f"Image traitée et sauvegardée : {dest_path}")
