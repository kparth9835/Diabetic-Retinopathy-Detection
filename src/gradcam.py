import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from model import get_model

# --- PATHS ---
BASE_PATH  = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\archive"
TRAIN_IMGS = os.path.join(BASE_PATH, "train_images", "train_images")
MODEL_PATH = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\outputs\best_model.pth"
OUTPUT_DIR = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\outputs"

GRADE_NAMES = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD MODEL ---
device = torch.device('cpu')
model  = get_model(num_classes=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully.")

# --- GRAD-CAM CLASS ---
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # Hook into the last conv layer of ResNet18
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Last conv layer in ResNet18 is layer4[1].conv2
        target_layer = self.model.layer4[1].conv2
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, image_tensor, target_class):
        # Forward pass
        output = self.model(image_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)

        # Calculate weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to image size
        cam = F.interpolate(cam, size=(224, 224),
                           mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()

        # Normalize to 0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def generate_gradcam(img_path, true_label=None):
    # Load original image
    original = Image.open(img_path).convert('RGB')
    original_np = np.array(original.resize((224, 224)))

    # Prepare tensor
    img_tensor = transform(original).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(1).item()
        confidence = probs[0][pred_class].item() * 100

    # Generate Grad-CAM
    gradcam = GradCAM(model)
    img_tensor_grad = transform(original).unsqueeze(0).to(device)
    img_tensor_grad.requires_grad_(True)
    cam = gradcam.generate(img_tensor_grad, pred_class)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    overlay = (0.5 * original_np + 0.5 * heatmap).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    title = f'Predicted: {GRADE_NAMES[pred_class]} ({confidence:.1f}%)'
    if true_label is not None:
        title += f'\nActual: {GRADE_NAMES[true_label]}'
    axes[2].set_title(title, fontsize=11)
    axes[2].axis('off')

    plt.suptitle('Grad-CAM Explainability — DR Detection', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f'gradcam_{GRADE_NAMES[pred_class]}.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Saved: {save_path}")
    return pred_class, confidence


# --- RUN ON ONE SAMPLE PER GRADE ---
import pandas as pd

train_csv = os.path.join(BASE_PATH, "train_1.csv")
df = pd.read_csv(train_csv)
samples = df.groupby('diagnosis').first().reset_index()

print("\nGenerating Grad-CAM for one sample per grade...\n")
for _, row in samples.iterrows():
    img_path = os.path.join(TRAIN_IMGS, row['id_code'] + '.png')
    true_label = int(row['diagnosis'])
    print(f"Processing Grade {true_label} ({GRADE_NAMES[true_label]})...")
    pred, conf = generate_gradcam(img_path, true_label)
    print(f"  Predicted: {GRADE_NAMES[pred]} ({conf:.1f}%)")

print("\nAll Grad-CAM images saved to outputs folder.")