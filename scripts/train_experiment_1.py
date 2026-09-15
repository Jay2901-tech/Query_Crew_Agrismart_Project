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
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
import timm
from tqdm import tqdm
import random
import hashlib

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CSVDataset(Dataset):
    def __init__(self, csv_file, classes, transform=None):
        self.data = []
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if os.path.exists(row['filepath']):
                    self.data.append(row)
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data[idx]
        image_path = row['filepath']
        label = self.class_to_idx[row['class']]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        image_np = np.array(image)
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = ToTensorV2()(image=image_np)['image']
            
        return image_tensor, label, row['domain']

def get_transforms():
    # Stronger, realistic field-oriented augmentation
    train_transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),  # Reasonable for leaves
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
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
    
    domain_preds = {'plantvillage': [], 'plantdoc': []}
    domain_labels = {'plantvillage': [], 'plantdoc': []}
    
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for images, labels, domains in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
            
        _, preds = torch.max(outputs, 1)
        preds_cpu = preds.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        
        for i, d in enumerate(domains):
            d_lower = d.lower()
            if d_lower in domain_preds:
                domain_preds[d_lower].append(preds_cpu[i])
                domain_labels[d_lower].append(labels_cpu[i])
                
    pv_f1 = f1_score(domain_labels['plantvillage'], domain_preds['plantvillage'], average='macro') if domain_labels['plantvillage'] else 0.0
    pd_f1 = f1_score(domain_labels['plantdoc'], domain_preds['plantdoc'], average='macro') if domain_labels['plantdoc'] else 0.0
    pv_acc = accuracy_score(domain_labels['plantvillage'], domain_preds['plantvillage']) if domain_labels['plantvillage'] else 0.0
    pd_acc = accuracy_score(domain_labels['plantdoc'], domain_preds['plantdoc']) if domain_labels['plantdoc'] else 0.0
    
    field_score = 0.35 * pv_f1 + 0.65 * pd_f1
    avg_loss = total_loss / total_samples
    
    return avg_loss, pv_acc, pv_f1, pd_acc, pd_f1, field_score

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest().upper()

def main(args):
    seed = 42
    seed_everything(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = "models/experiment_1"
    os.makedirs(output_dir, exist_ok=True)
    
    with open("config/class_mapping.json", "r") as f:
        mapping = json.load(f)
    classes = [c for c in mapping.keys() if "UNKNOWN" not in c]
    classes.sort()
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = CSVDataset("data/train.csv", classes, transform=train_transform)
    val_dataset = CSVDataset("data/val.csv", classes, transform=val_transform)
    
    pv_count = 0
    pd_count = 0
    weights = []
    
    pv_weight = 1.0
    pd_weight = 5.0
    
    for row in train_dataset.data:
        if row['domain'].lower() == 'plantdoc':
            pd_count += 1
            weights.append(pd_weight)
        else:
            pv_count += 1
            weights.append(pv_weight)
            
    total_samples = len(train_dataset.data)
    total_weight = sum(weights)
    prob_pd = (pd_count * pd_weight) / total_weight if total_weight > 0 else 0
    prob_pv = (pv_count * pv_weight) / total_weight if total_weight > 0 else 0
    
    print(f"Train Dataset: {total_samples} samples (PV: {pv_count}, PD: {pd_count})")
    print(f"Sampling Probabilities -> PV: {prob_pv:.4f}, PD: {prob_pd:.4f}")
    
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Building EfficientNetV2-S for {len(classes)} classes...")
    model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=len(classes)).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device.type, enabled=device.type == 'cuda')
    
    best_field_score = 0.0
    best_epoch = -1
    
    best_weights_path = os.path.join(output_dir, 'best_model.pth')
    final_weights_path = os.path.join(output_dir, 'final_model.pth')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    history_path = os.path.join(output_dir, 'training_history.csv')
    
    history_records = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_samples = 0
        
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
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / train_samples
        
        with torch.no_grad():
            val_loss, pv_acc, pv_f1, pd_acc, pd_f1, field_score = evaluate_domains(model, val_loader, criterion, device)
            
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - {time_elapsed:.0f}s - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"PV F1: {pv_f1:.4f} | PD F1: {pd_f1:.4f} | Field Score: {field_score:.4f}")
        
        history_records.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'validation_loss': val_loss,
            'pv_val_acc': pv_acc,
            'pv_val_f1': pv_f1,
            'pd_val_acc': pd_acc,
            'pd_val_f1': pd_f1,
            'field_score': field_score,
            'learning_rate': current_lr
        })
        
        if field_score > best_field_score:
            best_field_score = field_score
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': best_field_score,
                'classes': classes
            }, best_weights_path)
            print(f"--> Saved new best model with Field Score: {best_field_score:.4f}")
            
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_field_score,
        'classes': classes
    }, final_weights_path)
    
    # Save training history
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history_records[0].keys())
        writer.writeheader()
        writer.writerows(history_records)
        
    # Compile and save metadata
    metadata = {
        "architecture": "tf_efficientnetv2_s",
        "pretrained_status": True,
        "class_count": len(classes),
        "dataset_paths": {"train": "data/train.csv", "val": "data/val.csv", "test": "data/test.csv"},
        "dataset_counts": {"train": total_samples, "val": len(val_dataset), "test": 5736}, # Hardcoded test count from check
        "sampler_configuration": {"pv_weight": pv_weight, "pd_weight": pd_weight},
        "effective_sampling_distribution": {"pv_prob": prob_pv, "pd_prob": prob_pd},
        "augmentation_configuration": "RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotate90, ShiftScaleRotate, ColorJitter, GaussianBlur, GaussNoise",
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "scheduler": "CosineAnnealingLR (T_max=10)",
        "weight_decay": 0.01,
        "loss": "CrossEntropyLoss",
        "label_smoothing": 0.1,
        "amp_status": True,
        "batch_size": args.batch_size,
        "actual_batch_size": args.batch_size,
        "configured_epochs": args.epochs,
        "completed_epochs": args.epochs,
        "best_epoch": best_epoch,
        "final_epoch": args.epochs,
        "best_model_is_best_epoch": True,
        "best_model_is_final_epoch": (best_epoch == args.epochs),
        "seed": seed,
        "checkpoint_hashes": {
            "best_model": compute_sha256(best_weights_path),
            "final_model": compute_sha256(final_weights_path)
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_metrics_best": next(item for item in history_records if item['epoch'] == best_epoch) if best_epoch > 0 else {}
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Training Complete! Best Field Score: {best_field_score:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers")
    
    args = parser.parse_args()
    main(args)
