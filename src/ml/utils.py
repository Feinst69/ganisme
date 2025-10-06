import torch
from torch.utils.data import Dataset, DataLoader
import os

class TensorImageDataset(Dataset):
    def __init__(self, folder_path):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pt")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = torch.load(self.files[idx])
        return img, 0  
