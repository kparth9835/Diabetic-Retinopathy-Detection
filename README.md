# Diabetic Retinopathy Detection
### ResNet18 + Transfer Learning + Grad-CAM Explainability

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

An automated Diabetic Retinopathy (DR) severity classification system using deep learning. The model classifies retinal fundus images into 5 severity grades and generates **Grad-CAM heatmaps** to visually explain which regions of the eye influenced the prediction.

This project directly addresses the explainability gap identified in:
> Uppamma & Bhattacharya, *"A Multidomain Bio-Inspired Feature Extraction and Selection Model for Diabetic Retinopathy Severity Classification"*, Scientific Reports, Nature (2023). DOI: [10.1038/s41598-023-45886-7](https://doi.org/10.1038/s41598-023-45886-7)

The referenced paper achieved 96.5% accuracy using traditional ML with handcrafted features but explicitly stated explainability as future work. This project implements that future work using modern deep learning with Grad-CAM.

---

## Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 84.15% |
| Test Accuracy | 83.33% |
| Training Epochs | 10 |
| Dataset | APTOS 2019 (3,662 images) |
| Model | ResNet18 (Transfer Learning) |

### Classification Report

| Grade | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 — No DR | 0.98 | 0.99 | 0.98 |
| 1 — Mild | 0.60 | 0.60 | 0.60 |
| 2 — Moderate | 0.67 | 0.85 | 0.75 |
| 3 — Severe | 0.30 | 0.18 | 0.22 |
| 4 — Proliferative | 0.93 | 0.39 | 0.55 |

---

## Grad-CAM Explainability

The model generates visual heatmaps showing which regions of the fundus image it focused on to make its prediction. Red/warm areas = high attention. Blue/cool areas = low attention.

| Grade | Confidence | Heatmap |
|-------|-----------|---------|
| No DR | 99.7% | Focused on optic disc region |
| Mild | 57.9% | Diffuse attention - subtle early disease |
| Moderate | 79.9% | Vessel boundary regions |
| Severe | 96.0% | Tight focus on damaged region |
| Proliferative | 84.4% | Two focal lesion areas |

---

## Dataset

**APTOS 2019 Blindness Detection** - Kaggle Competition Dataset

| Split | Images |
|-------|--------|
| Train | 2,930 |
| Validation | 365 |
| Test | 868 |
| **Total** | **4,163** |

DR Severity Grades: 0 (No DR), 1 (Mild), 2 (Moderate), 3 (Severe), 4 (Proliferative DR)

---

## Methodology

### Why ResNet18 + Transfer Learning
Medical imaging datasets are small. Training from scratch on 3,662 images causes overfitting. ResNet18 pre-trained on ImageNet already understands image features (edges, textures, shapes). We fine-tune only the final layer (512 → 5 outputs) for DR grade classification.

### Why Grad-CAM
DR detection models are used in clinical settings. Doctors need to understand and verify model decisions. Grad-CAM generates visual explanations by computing gradients of the predicted class with respect to the last convolutional layer activations - highlighting the exact retinal regions that drove the prediction.

### Key Differences from Reference Paper
| Aspect | Reference Paper (2023) | This Project |
|--------|----------------------|--------------|
| Approach | Traditional ML | Deep Learning |
| Features | Handcrafted (Gabor, Wavelet, DFT) | Learned automatically |
| Explainability | None (black box) | Grad-CAM heatmaps |
| Dataset | 593 images | 3,662 images |
| Framework | NumPy + TensorFlow | PyTorch |

---

## Setup

```bash
git clone https://github.com/kparth9838/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
pip install torch torchvision matplotlib scikit-learn pandas numpy opencv-python seaborn Pillow
```

Download APTOS 2019 dataset from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data) and place in `archive/` folder.

---

## Run

```bash
python src/dataset.py    # Visualize dataset
python src/model.py      # Verify model architecture
python src/train.py      # Train the model
python src/evaluate.py   # Evaluate on test set
python src/gradcam.py    # Generate Grad-CAM heatmaps
```

---

## Related Work

- **Reference Paper:** Uppamma & Bhattacharya (2023), Scientific Reports — DOI: [10.1038/s41598-023-45886-7](https://doi.org/10.1038/s41598-023-45886-7)

---

## Author

**Parth Khunt** — B.Tech ICT, PDEU Gandhinagar (2026)

[![GitHub](https://img.shields.io/badge/GitHub-kparth9838-black)](https://github.com/kparth9835)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-parthkhunt160904-blue)](https://linkedin.com/in/parthkhunt160904/)
