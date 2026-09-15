import os
import json
import argparse
import time
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from PIL import Image
import timm
from tqdm import tqdm

class CSVDataset(Dataset):
    def __init__(self, csv_file, classes, transform=None):
        self.data = []
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # We skip missing files just in case
                if os.path.exists(row['filepath']):
                    self.data.append(row)
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        image_path = row['filepath']
        label = self.class_to_idx[row['class']]
        
        # Open image and convert to RGB
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Fallback to random noise if corrupted
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        image_np = np.array(image)
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = ToTensorV2()(image=image_np)['image']
            
        return image_tensor, label, row['domain']

def get_transforms():
    train_transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.5),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform

def evaluate_domains(model, dataloader, criterion, device):
    model.eval()
    
    # We will track predictions per domain
    domain_preds = {'plantvillage': [], 'plantdoc': []}
    domain_labels = {'plantvillage': [], 'plantdoc': []}
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for images, labels, domains in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        _, preds = torch.max(outputs, 1)
        preds_cpu = preds.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        
        for i, d in enumerate(domains):
            if d.lower() in domain_preds:
                domain_preds[d.lower()].append(preds_cpu[i])
                domain_labels[d.lower()].append(labels_cpu[i])
                
    pv_f1 = f1_score(domain_labels['plantvillage'], domain_preds['plantvillage'], average='macro') if domain_labels['plantvillage'] else 0.0
    pd_f1 = f1_score(domain_labels['plantdoc'], domain_preds['plantdoc'], average='macro') if domain_labels['plantdoc'] else 0.0
    
    # 0.35 * PV Macro F1 + 0.65 * PlantDoc Macro F1
    field_score = 0.35 * pv_f1 + 0.65 * pd_f1
    
    return pv_f1, pd_f1, field_score

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.weights_dir, exist_ok=True)
    
    with open("config/class_mapping.json", "r") as f:
        mapping = json.load(f)
    classes = [c for c in mapping.keys() if "UNKNOWN" not in c]
    classes.sort()
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = CSVDataset("data/train.csv", classes, transform=train_transform)
    val_dataset = CSVDataset("data/val.csv", classes, transform=val_transform)
    
    # Weighted Random Sampler for Domain Balancing
    # PV vs PlantDoc balance
    weights = []
    for row in train_dataset.data:
        if row['domain'].lower() == 'plantdoc':
            weights.append(5.0) # Upweight PlantDoc to balance PV
        else:
            weights.append(1.0)
            
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Building EfficientNetV2-S for {len(classes)} classes...")
    model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=len(classes)).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device.type, enabled=device.type == 'cuda')
    
    best_score = 0.0
    best_weights_path = os.path.join(args.weights_dir, 'best_model.pth')
    metadata_path = os.path.join(args.weights_dir, 'best_model_metadata.json')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=loss.item())
            
        with torch.no_grad():
            pv_f1, pd_f1, field_score = evaluate_domains(model, val_loader, criterion, device)
            
        scheduler.step()
        
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - {time_elapsed:.0f}s - PV F1: {pv_f1:.4f} | PD F1: {pd_f1:.4f} | Field Score: {field_score:.4f}")
        
        if field_score > best_score:
            best_score = field_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_score,
                'classes': classes
            }, best_weights_path)
            
            with open(metadata_path, 'w') as f:
                json.dump({
                    "architecture": "tf_efficientnetv2_s",
                    "classes": classes,
                    "best_field_score": field_score,
                    "pv_f1": pv_f1,
                    "pd_f1": pd_f1,
                    "epoch": epoch,
                    "image_size": [224, 224]
                }, f, indent=4)
                
            print(f"--> Saved new best model with Field Score: {best_score:.4f}")
            
    print("Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=str, default="models", help="Directory to save model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers")
    
    args = parser.parse_args()
    main(args)
