import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from model import get_model

# --- PATHS ---
BASE_PATH  = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\archive"
TRAIN_CSV  = os.path.join(BASE_PATH, "train_1.csv")
VALID_CSV  = os.path.join(BASE_PATH, "valid.csv")
TRAIN_IMGS = os.path.join(BASE_PATH, "train_images", "train_images")
VALID_IMGS = os.path.join(BASE_PATH, "val_images", "val_images")
MODEL_SAVE = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\outputs\best_model.pth"

# --- DATASET ---
class DRDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
        image = Image.open(img_path).convert('RGB')
        label = int(row['diagnosis'])
        if self.transform:
            image = self.transform(image)
        return image, label

# --- TRANSFORMS ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- DATALOADERS ---
train_dataset = DRDataset(TRAIN_CSV, TRAIN_IMGS, train_transform)
valid_dataset = DRDataset(VALID_CSV, VALID_IMGS, valid_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# --- MODEL ---
device = torch.device('cpu')
print(f"Using device: {device}")

model = get_model(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# --- TRAINING ---
NUM_EPOCHS = 10
best_val_acc = 0.0

print("Starting training...")
print("-" * 50)

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_acc = 100. * train_correct / len(train_dataset)

    # Validation phase
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / len(valid_dataset)

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Loss: {train_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"  --> Best model saved (Val Acc: {val_acc:.2f}%)")

    print("-" * 50)

print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}%")