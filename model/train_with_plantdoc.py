import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import timm
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler, ConcatDataset
import timm
from tqdm import tqdm

from download_data import check_and_download_dataset

class AlbumentationsDataset(Dataset):
    """Wrapper to apply Albumentations transforms to an ImageFolder dataset."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image_np = np.array(image)
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = ToTensorV2()(image=image_np)['image']
        return image_tensor, label

def get_transforms():
    # Stronger augmentations to simulate field conditions
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.RandomResizedCrop(224, 224, scale=(0.5, 1.0), p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.5),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform

def load_classes(config_path="config/classes.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_class_mapping(mapping_path="config/class_mapping.json"):
    with open(mapping_path, 'r') as f:
        return json.load(f)

def get_dataloaders(data_dir, plantdoc_dir=None, batch_size=32, num_workers=0, samples_per_class=None):
    expected_classes = set(load_classes())
    base_dataset = ImageFolder(root=data_dir)
    dataset_classes = set(base_dataset.classes)
    
    missing_classes = expected_classes - dataset_classes
    if missing_classes:
        print(f"Warning: Expected classes missing: {missing_classes}")
        
    if samples_per_class is not None:
        import random
        random.seed(42)
        class_indices = {c: [] for c in range(len(base_dataset.classes))}
        for idx, (_, class_idx) in enumerate(base_dataset.samples):
            class_indices[class_idx].append(idx)
        
        subset_indices = []
        for class_idx, indices in class_indices.items():
            subset_indices.extend(random.sample(indices, min(samples_per_class, len(indices))))
            
        dataset_to_split = torch.utils.data.Subset(base_dataset, subset_indices)
        total_size = len(dataset_to_split)
        print(f"Subset created with {total_size} images (up to {samples_per_class} per class).")
    else:
        dataset_to_split = base_dataset
        total_size = len(base_dataset)
        
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        dataset_to_split, 
        [train_size, val_size, test_size], 
        generator=generator
    )
    
    # If plantdoc_dir is provided, we combine its training set into train_ds
    sampler = None
    if plantdoc_dir and os.path.exists(plantdoc_dir):
        print(f"Loading PlantDoc dataset from {plantdoc_dir}...")
        plantdoc_dataset = ImageFolder(root=plantdoc_dir)
        
        # Load mapping
        mapping = load_class_mapping()
        # PlantDoc name -> PV name
        pd_to_pv = {}
        for official_name, info in mapping.items():
            if info["plantdoc"] is not None:
                pd_to_pv[info["plantdoc"]] = info["plantvillage"]
                
        # We need to map PlantDoc's class indices to base_dataset's class indices
        base_class_to_idx = base_dataset.class_to_idx
        pd_class_to_idx = plantdoc_dataset.class_to_idx
        
        # Create a target transform for PlantDoc
        def target_transform(pd_idx):
            pd_class_name = plantdoc_dataset.classes[pd_idx]
            pv_class_name = pd_to_pv.get(pd_class_name)
            if pv_class_name and pv_class_name in base_class_to_idx:
                return base_class_to_idx[pv_class_name]
            return -1 # Should be filtered out
            
        plantdoc_dataset.target_transform = target_transform
        
        # Filter out images that don't map to a PV class
        valid_indices = []
        for i, (_, label) in enumerate(plantdoc_dataset.samples):
            if target_transform(label) != -1:
                valid_indices.append(i)
                
        plantdoc_subset = torch.utils.data.Subset(plantdoc_dataset, valid_indices)
        print(f"Added {len(plantdoc_subset)} PlantDoc images to training set.")
        
        # Combine datasets
        combined_train = ConcatDataset([train_ds, plantdoc_subset])
        
        # Create WeightedRandomSampler
        weights = []
        for _ in range(len(train_ds)):
            weights.append(1.0) # Weight 1 for PlantVillage
        for _ in range(len(plantdoc_subset)):
            weights.append(4.0) # Weight 4 for PlantDoc (oversampling)
            
        sampler = WeightedRandomSampler(weights, len(weights))
        train_ds = combined_train
        
    train_transform, val_transform = get_transforms()
    train_dataset = AlbumentationsDataset(train_ds, transform=train_transform)
    val_dataset = AlbumentationsDataset(val_ds, transform=val_transform)
    test_dataset = AlbumentationsDataset(test_ds, transform=val_transform)
    
    if sampler:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, base_dataset.classes

def build_model(num_classes):
    print(f"Building EfficientNetV2-S for {num_classes} classes...")
    model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=num_classes)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, macro_f1, all_labels, all_preds

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # Automatically download the dataset if it's not present
    check_and_download_dataset(args.data_dir)
    
    if args.dry_run:
        print("Dry run enabled. Validating data loading only...")
        train_loader, _, _, _ = get_dataloaders(
            args.data_dir, 
            plantdoc_dir=args.plantdoc_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        for images, labels in train_loader:
            print(f"Successfully loaded a batch of shape {images.shape}")
            break
        print("Dry run complete!")
        return

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir, 
        plantdoc_dir=args.plantdoc_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_class=args.samples_per_class
    )
    num_classes = len(classes)
    
    model = build_model(num_classes).to(device)
    # We use label smoothing to prevent the model from being over-confident on clean images
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize scaler, disable it explicitly if CUDA is not available
    scaler = GradScaler(device.type, enabled=device.type == 'cuda')
    
    best_f1 = 0.0
    best_weights_path = os.path.join(args.weights_dir, 'best_model.pth')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - {time_elapsed:.0f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Macro-F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'classes': classes
            }, best_weights_path)
            print(f"--> Saved new best model with Macro-F1: {best_f1:.4f}")
            
    print("\nTraining Complete! Evaluating on Held-out Test Set...")
    
    if os.path.exists(best_weights_path):
        checkpoint = torch.load(best_weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best weights for testing.")
    
    test_loss, test_f1, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Macro-F1: {test_f1:.4f}")
    
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average=None)
    print("\nPer-class Metrics:")
    for i, class_name in enumerate(classes):
        print(f"{class_name}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriSmart AI Model Training")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to PlantVillage dataset")
    parser.add_argument("--plantdoc-dir", type=str, default=None, help="Path to PlantDoc training dataset for combined training")
    parser.add_argument("--weights-dir", type=str, default="model/weights", help="Directory to save model weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (0 is safer on Windows)")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Limit number of images per class for faster training")
    parser.add_argument("--dry-run", action="store_true", help="Validate data loading without training")
    
    args = parser.parse_args()
    main(args)
