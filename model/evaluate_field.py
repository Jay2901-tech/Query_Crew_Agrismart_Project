import os
import json
import argparse
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import timm
from tqdm import tqdm
from PIL import Image

class PlantDocDataset(Dataset):
    def __init__(self, root_dir, class_mapping_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        with open(class_mapping_path, 'r') as f:
            self.mapping = json.load(f)
            
        # Create PlantDoc name -> Official name mapping
        self.pd_to_official = {
            v["plantdoc"]: k for k, v in self.mapping.items() if v["plantdoc"] is not None
        }
        
        self.image_paths = []
        self.labels = []
        self.official_classes = sorted(list(self.mapping.keys()))
        
        # We need to map the model's 38 classes to know what it predicted.
        # But for ground truth, we only use the official 18 classes.
        for pd_class, official_class in self.pd_to_official.items():
            class_dir = os.path.join(root_dir, pd_class)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(official_class)
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        if self.transform:
            image_np = self.transform(image=image_np)['image']
            
        return image_np, self.labels[idx]

def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

@torch.no_grad()
def evaluate_model(model, dataloader, model_classes, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # We want to evaluate strictly on the official classes.
    official_classes = dataloader.dataset.official_classes
    
    pbar = tqdm(dataloader, desc="Evaluating Field Performance")
    for images, batch_labels in pbar:
        images = images.to(device)
        outputs = model(images)
        _, pred_indices = torch.max(outputs, 1)
        
        for i in range(len(pred_indices)):
            pred_class_name = model_classes[pred_indices[i].item()]
            
            # If the model predicts a class in PlantVillage that's not official,
            # map it using the reverse mapping if it's one of the 18 official classes.
            # But the model_classes are PlantVillage folder names.
            # We need PV -> Official mapping.
            pv_to_official = {v["plantvillage"]: k for k, v in dataloader.dataset.mapping.items()}
            
            predicted_official = pv_to_official.get(pred_class_name, "UNKNOWN")
            
            all_preds.append(predicted_official)
            all_labels.append(batch_labels[i])
            
    # Calculate metrics ONLY for the 18 official classes
    labels_idx = {name: idx for idx, name in enumerate(official_classes)}
    
    y_true = [labels_idx[l] for l in all_labels]
    y_pred = [labels_idx[p] if p in labels_idx else -1 for p in all_preds]
    
    # Remove UNKNOWN predictions by assigning them to a dummy class, but they will drag down F1
    # which is correct, since it's a wrong prediction. We can just compute F1 with labels=range(18)
    macro_f1 = f1_score(y_true, y_pred, labels=list(range(len(official_classes))), average='macro')
    
    print(f"\n--- Field Evaluation Results ---")
    print(f"Macro-F1 (Official 18 Classes): {macro_f1:.4f}\n")
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(official_classes))))
    print("Confusion Matrix:")
    print(cm)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(official_classes))), zero_division=0
    )
    
    print("\nPer-class Metrics (PlantDoc Field Images):")
    for i, cls_name in enumerate(official_classes):
        print(f"{cls_name}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model_classes = checkpoint['classes']
    
    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(model_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    dataset = PlantDocDataset(
        root_dir=args.data_dir,
        class_mapping_path=args.mapping,
        transform=get_inference_transform()
    )
    
    if len(dataset) == 0:
        print(f"No valid field images found in {args.data_dir}.")
        return
        
    print(f"Found {len(dataset)} field images mapped to official classes.")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    evaluate_model(model, dataloader, model_classes, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/PlantDoc/test")
    parser.add_argument("--weights", type=str, default="model/weights/best_model.pth")
    parser.add_argument("--mapping", type=str, default="config/class_mapping.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args)
