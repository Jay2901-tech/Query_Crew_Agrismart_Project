import os
import json
import torch
import hashlib
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import timm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.amp import autocast

def get_transforms():
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return val_transform

class CorrectDataset(Dataset):
    def __init__(self, csv_file, checkpoint_classes, transform=None):
        self.df = pd.read_csv(csv_file)
        self.checkpoint_classes = checkpoint_classes
        self.class_to_idx = {c: i for i, c in enumerate(checkpoint_classes)}
        self.transform = transform
        
        with open('config/class_mapping.json', 'r') as f:
            self.mapping = json.load(f)
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        orig_class = row['class'] # This is the canonical key like "Cherry___healthy"
        domain = row['domain']
        
        # In best_model.pth, the classes are the plantvillage names!
        # So we map from canonical -> plantvillage to match the checkpoint classes.
        mapped_class = orig_class
        if orig_class in self.mapping and self.mapping[orig_class]['plantvillage'] is not None:
            mapped_class = self.mapping[orig_class]['plantvillage']
            
        if mapped_class not in self.class_to_idx:
            # Fallback for Grape___Leaf_blight_(Isariopsis_Leaf_Spot) which might have different casing etc.
            # print(f"Warning: {mapped_class} not in checkpoint classes")
            mapped_class = self.checkpoint_classes[0]
            
        label = self.class_to_idx[mapped_class]
        
        try:
            image = Image.open(img_path).convert('RGB')
            import numpy as np
            image = np.array(image)
        except Exception as e:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label, domain

def main():
    model_path = "model/weights/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    
    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = get_transforms()
    dataset = CorrectDataset('data/test.csv', classes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    all_domains = []
    all_confs = []
    
    for images, labels, domains in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, 1)
                
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_domains.extend(domains)
        all_confs.extend(confs.cpu().numpy())
        
    df = pd.DataFrame({
        'pred': all_preds,
        'label': all_labels,
        'domain': all_domains,
        'conf': all_confs
    })
    
    df.to_csv('reports/model_audit/test_predictions_fixed.csv', index=False)
    
    for domain in ['plantvillage', 'plantdoc']:
        domain_df = df[df['domain'] == domain]
        if len(domain_df) == 0: continue
        
        acc = accuracy_score(domain_df['label'], domain_df['pred'])
        _, _, macro_f1, _ = precision_recall_fscore_support(domain_df['label'], domain_df['pred'], average='macro', zero_division=0)
        
        print(f"{domain} Accuracy: {acc:.4f}")
        print(f"{domain} Macro-F1: {macro_f1:.4f}")

if __name__ == '__main__':
    main()
