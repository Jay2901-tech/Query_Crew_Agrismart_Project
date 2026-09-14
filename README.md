# AgriSmart AI

**Crop Disease Detection System**  
**SIH 2026 — Problem Statement 1**

## Project Overview
AgriSmart AI is a production-grade machine learning pipeline designed to automatically detect crop diseases from leaf images and provide farmer-friendly precautionary guidance. 

## Architecture
- **Model:** EfficientNetV2-S (pretrained on ImageNet, fine-tuned for crop diseases)
- **Frameworks:** PyTorch, torchvision, timm, Albumentations
- **API:** FastAPI (Planned)
- **Experiment Tracking:** Weights & Biases

## Dataset and Preprocessing
We utilize a subset of the PlantVillage dataset representing 12 common disease classes and healthy variants for crops including Tomato, Potato, Corn, Apple, Grape, and Pepper.

### Data Split Strategy
- 70% Training / 15% Validation / 15% Held-out Test Set
- **CRITICAL:** The test set is isolated and completely unused during training or hyperparameter tuning.

### Augmentation Strategy (Training)
- Albumentations for robust field-condition simulation:
  - RandomBrightnessContrast (p=0.5)
  - RandomShadow (p=0.3)
  - GaussianBlur (p=0.3)
  - CoarseDropout (p=0.3)
  - HorizontalFlip & RandomRotate90 (p=0.5)
  - Normalization (ImageNet means/stds)

## Training Methodology
- **Optimizer:** AdamW (weight decay 1e-4)
- **Scheduler:** CosineAnnealingLR
- **Loss Function:** CrossEntropyLoss
- **Model Selection:** Based strictly on Validation **Macro-F1** score (not raw accuracy).

## Evaluation
- Primary metric: Macro-F1
- Per-class precision, recall, F1, and accuracy.
- Final evaluation is performed *only* on the held-out test set to generate the confusion matrix and final model report.

## Setup and Usage

*(Instructions pending implementation of API and final training).*
