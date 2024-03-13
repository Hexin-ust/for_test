import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class InfraredDataset(Dataset):
    def __init__(self, image_dir , mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
    
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png",".jpg"))
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        

        return image, mask

dataset = InfraredDataset("datasets/train_image_dir", "datasets/train_mask_dir")

image, mask = dataset[1]  # Access the 'image' and 'mask' variables from the 'dataset' object
print("image shape:", image.shape)  # Check the shape of the image
print("mask shape:", mask.shape)  # Check the shape of the mask

