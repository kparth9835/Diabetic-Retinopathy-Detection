import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from model import get_model

# --- PATHS ---
BASE_PATH  = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\archive"
TEST_CSV   = os.path.join(BASE_PATH, "test.csv")
TEST_IMGS  = os.path.join(BASE_PATH, "test_images", "test_images")
MODEL_PATH = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\outputs\best_model.pth"
OUTPUT_DIR = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\outputs"

GRADE_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

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

# --- TRANSFORM ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD DATA ---
test_dataset = DRDataset(TEST_CSV, TEST_IMGS, test_transform)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# --- LOAD MODEL ---
device = torch.device('cpu')
model  = get_model(num_classes=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully.")
print(f"Testing on {len(test_dataset)} images...")

# --- EVALUATE ---
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# --- ACCURACY ---
accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
print(f"\nTest Accuracy: {accuracy:.2f}%")

# --- CLASSIFICATION REPORT ---
print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=GRADE_NAMES))

# --- CONFUSION MATRIX ---
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=GRADE_NAMES,
            yticklabels=GRADE_NAMES,
            ax=ax)
ax.set_xlabel('Predicted Grade')
ax.set_ylabel('Actual Grade')
ax.set_title('Confusion Matrix — DR Detection')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.show()
print(f"\nConfusion matrix saved to outputs folder.")