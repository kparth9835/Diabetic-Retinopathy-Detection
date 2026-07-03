import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=5):
    model = models.resnet18(weights='IMAGENET1K_V1')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_model()
    print("=== ResNet18 Model Loaded ===")
    print(f"Final layer: {model.fc}")
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"Output shape: {output.shape}")
    print("Model ready.")