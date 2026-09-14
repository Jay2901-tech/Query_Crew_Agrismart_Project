import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import timm
from evaluate_field import PlantDocDataset, get_inference_transform
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_finetune_transforms():
    # Stronger augmentations for field conditions
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = get_inference_transform()
    return train_transform, val_transform

def finetune_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Fine-tuning")
    for images, batch_labels in pbar:
        images = images.to(device)
        
        # We need to map official class names to the model's indices
        # We assume batch_labels are the official class strings, 
        # so we need to map them to the PV index that the model expects.
        
        official_to_pv_idx = dataloader.dataset.official_to_pv_idx
        
        labels_idx = torch.tensor([official_to_pv_idx[l] for l in batch_labels]).to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels_idx)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(dataloader.dataset)

class FinetuneDataset(PlantDocDataset):
    def __init__(self, root_dir, class_mapping_path, model_classes, transform=None):
        super().__init__(root_dir, class_mapping_path, transform)
        
        # Map Official Name -> PV Name -> Index in model_classes
        # model_classes are the PlantVillage names the model was trained on
        pv_name_to_idx = {name: idx for idx, name in enumerate(model_classes)}
        
        self.official_to_pv_idx = {}
        for official_name in self.official_classes:
            pv_name = self.mapping[official_name]["plantvillage"]
            if pv_name in pv_name_to_idx:
                self.official_to_pv_idx[official_name] = pv_name_to_idx[pv_name]
            else:
                raise ValueError(f"Mapping failed: PV name '{pv_name}' not found in model classes.")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model_classes = checkpoint['classes']
    
    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(model_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Datasets
    train_transform, val_transform = get_finetune_transforms()
    
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    
    train_dataset = FinetuneDataset(train_dir, args.mapping, model_classes, transform=train_transform)
    test_dataset = FinetuneDataset(test_dir, args.mapping, model_classes, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # We use label smoothing for finetuning to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Very low learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scaler = GradScaler(enabled=device.type == 'cuda')
    
    from evaluate_field import evaluate_model
    
    print("\n--- Initial Evaluation (Before Fine-tuning) ---")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    evaluate_model(model, test_loader, model_classes, device)
    
    best_loss = float('inf')
    best_weights_path = "model/weights/best_model_field.pth"
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = finetune_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # We don't evaluate fully every epoch to save time, but we save best training loss 
        # or we could evaluate on the test set. Let's just save the final model for simplicity,
        # since it's a short fine-tune.
        
    print("\n--- Final Evaluation (After Fine-tuning) ---")
    evaluate_model(model, test_loader, model_classes, device)
    
    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {best_weights_path}")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': model_classes
    }, best_weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/PlantDoc")
    parser.add_argument("--weights", type=str, default="model/weights/best_model.pth")
    parser.add_argument("--mapping", type=str, default="config/class_mapping.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)
