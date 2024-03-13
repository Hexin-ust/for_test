import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm   
from model import Unet
from utilis import get_loaders
import matplotlib.pyplot as plt
import numpy as np
import random

#超参数

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping  # 假设你已经定义了一个EarlyStopping类

LEARNING_RATE = 1e-5

PATIENCE = 5
BATCH_SIZE = 6
NUM_EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

TRAIN_IMG_DIR = "./datasets/train_image_dir"
TRAIN_MASK_DIR = "./datasets/train_mask_dir"
VAL_IMG_DIR = "./datasets/val_image_dir"
VAL_MASK_DIR = "./datasets/val_mask_dir"


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

train_losses = []
val_acc = []
val_dice = []




def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE) .float()
        targets = targets.unsqueeze(1).float().to(DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    
        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)


def check_accuracy(loader, model, epoch, DEVICE="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(DEVICE).float()  
            y = y.to(DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            if i == 0:  # 只在每个epoch的第一个batch保存图像
                preds_3channel = preds.repeat(1, 3, 1, 1)
                plt.imsave(f"validation_images/val_segmentation_epoch_{epoch}.png", preds_3channel[0].cpu().numpy().transpose(1,2,0), cmap='gray')

    accuracy = round(float(num_correct) / float(num_pixels), 4)
    dice = round(float(dice_score) / len(loader), 4)
    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"      )
    print(f"Dice score: {dice_score/len(loader):.4f}")
    
    
    model.train()
    return accuracy, dice






# 数据增强

def main():
    train_transform = A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ToTensorV2(),  # convert image to PyTorch tensor
    ])

    val_transform = A.Compose([ToTensorV2(),])

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, 
                                           BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY)
    
    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path='checkpoint.pt')

    for index in range(NUM_EPOCHS):
        print("current Epoch: ", index)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_losses.append(train_loss)

        accuracy, dice = check_accuracy(val_loader, model, index, DEVICE=DEVICE)  # 将epoch传递给check_accuracy函数
        val_acc.append(accuracy)  # Append accuracy to val_acc list
        val_dice.append(dice)

        early_stopping(dice, model)

        if early_stopping.early_stop:
            print("early stopping")
            break

    avg_train_loss = sum(train_losses)/len(train_losses)
    avg_val_acc = sum(val_acc) / len(val_acc)
    avg_val_dice = sum(val_dice) / len(val_dice)
    print(f"Average training loss: {avg_train_loss:.4f}")
    print(f"Average validation accuracy: {avg_val_acc:.4f}")
    print(f"Average validation DICE score: {avg_val_dice:.4f}")


# 假设 model 是你的模型
    
    torch.save(model.state_dict(), 'model_weights.pth')
if __name__ == "__main__":
    main()
    print("Training finished")
