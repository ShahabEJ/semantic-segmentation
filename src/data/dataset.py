from torch.utils.data import Dataset
import torch
import os
import numpy as np
from PIL import Image

class BuildingDataset(Dataset):
    def __init__(self, image_dir, img_folder_name, mask_folder_name, transform=None):
        self.transform = transform
        self.all_image_paths = sorted([os.path.join(image_dir, img_folder_name, img_name)
            for img_name in os.listdir(os.path.join(image_dir, img_folder_name)) if img_name.endswith('.png')])

        self.all_mask_paths = sorted([os.path.join(image_dir, mask_folder_name, mask_name)
            for mask_name in os.listdir(os.path.join(image_dir, mask_folder_name)) if mask_name.endswith('.png')])

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        
        image = np.array(Image.open(self.all_image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.all_mask_paths[idx]).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = np.where(mask == 255, 1, 0)

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
        image = image.float() / 255 

        return image, mask
