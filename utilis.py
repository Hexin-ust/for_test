import torch
import torchvision
from data import InfraredDataset
from torch.utils.data import DataLoader
from data import InfraredDataset
import matplotlib.pyplot as plt


def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    
    train_ds = InfraredDataset(image_dir = train_img_dir, mask_dir = train_mask_dir)

    train_loader = DataLoader(train_ds , batch_size=batch_size , num_workers=num_workers , pin_memory=pin_memory , shuffle=True)

    val_ds = InfraredDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader