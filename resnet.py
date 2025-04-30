from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
from typing import Tuple
import os


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['level']
        
        image = Image.open(img_path).convert("RGB") 
        image = image.resize((224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    

def load_data_from_csv(csv_path: str, base_image_dir: str, transform=None) -> Dataset:
    """Read CSV return PyTorch Dataset"""
    df = pd.read_csv(csv_path, usecols=['image', 'level'], nrows=10)
    df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir, 'resized_train_cropped', 'resized_train_cropped', f"{x}.jpeg"))
    df['level'] = (df['level'] > 1).astype(int)  # Disease or no disease
    return DiabeticRetinopathyDataset(df, transform=transform)


def load_model_data(
    csv_path: str,           
    base_image_dir: str,      
    label_list: list[int],  
    model: nn.Module,
    device: str,
    batch_size: int = 16,
    transform=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Use `DiabeticRetinopathyDataset` to load data
    """
    
    dataset = load_data_from_csv(csv_path, base_image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.to(device)
    class_images, class_labels = [], []
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, 1)

            predicted_labels = predicted_labels.cpu()
            mask = torch.isin(predicted_labels, torch.tensor(label_list))
            
            class_images.append(images[mask])
            class_labels.append(labels[mask])
    
    # convert batch list to single Tensor
    class_images = torch.cat(class_images, dim=0)
    class_labels = torch.cat(class_labels, dim=0)
    
    return class_images, class_labels
