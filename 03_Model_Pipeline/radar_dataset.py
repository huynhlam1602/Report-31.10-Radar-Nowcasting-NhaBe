# -*- coding: utf-8 -*-
"""
Radar Dataset & Transform Pipeline
Chuẩn hóa theo Le Hong Trang et al. (2025)
"""

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np

# 1. Custom Transform: Median Blur 5x5 lọc nhiễu hạt địa hình
class MedianBlur5x5(object):
    def __call__(self, img):
        return img.filter(ImageFilter.MedianFilter(size=5))

# 2. Bộ Transforms cho Train và Validation/Testing
def get_transforms():
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandAugment(num_ops=2, magnitude=9),
        MedianBlur5x5(),
        T.RandomAutocontrast(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.25], std=[0.25])
    ])
    
    val_transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.25], std=[0.25])
    ])
    return train_transform, val_transform

# 3. PyTorch Dataset
class RadarNowcastingDataset(Dataset):
    def __init__(self, csv_file, image_dir, target_horizon='Label_2H', transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.target_horizon = target_horizon
        self.transform = transform
        
        self.label_to_idx = {
            "Clear": 0,
            "Light rain": 1,
            "Moderate rain": 2,
            "Heavy rain": 3,
            "Very heavy rain": 4
        }
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['File_Name']
        
        # Đọc ảnh PNG 224x224 (góc quét sweep 0)
        img_name = file_name.replace('.RAW', '') + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            # Fallback nếu thiếu file
            image = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
            
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
            
        label_idx = self.label_to_idx[row[self.target_horizon]]
        return image, torch.tensor(label_idx, dtype=torch.long)
