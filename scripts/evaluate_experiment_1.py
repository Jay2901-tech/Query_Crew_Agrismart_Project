"""
Evaluate the Experiment 1 candidate model on the held-out test set.
Outputs per-domain accuracy and macro-F1. READ-ONLY — does not modify any weights.
"""
import json, os, torch, timm, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, classification_report

CHECKPOINT = "models/experiment_1/best_model.pth"
TEST_CSV   = "data/test.csv"
IMG_SIZE   = 224
BATCH_SIZE = 64

class CSVDataset(Dataset):
    def __init__(self, csv_path, classes, transform=None):
        self.df = pd.read_csv(csv_path)
        self.cls2idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transform: img = self.transform(img)
        label = self.cls2idx.get(row['class'], 0)
        domain = row.get('domain', 'plantvillage')
        return img, label, domain

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    classes = ckpt['classes']
    print(f"Classes: {len(classes)}")

    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds = CSVDataset(TEST_CSV, classes, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(ds)}")

    preds_all, labels_all, domains_all = [], [], []
    with torch.no_grad():
        for imgs, lbls, doms in loader:
            out = model(imgs.to(device))
            preds_all.extend(out.argmax(1).cpu().tolist())
            labels_all.extend(lbls.tolist())
            domains_all.extend(doms)

    # Split by domain
    for domain in ['plantvillage', 'plantdoc']:
        idx = [i for i, d in enumerate(domains_all) if d.lower() == domain]
        if not idx:
            print(f"{domain}: no samples found")
            continue
        p = [preds_all[i] for i in idx]
        l = [labels_all[i] for i in idx]
        acc = accuracy_score(l, p)
        f1  = f1_score(l, p, average='macro', zero_division=0)
        print(f"\n{'='*40}")
        print(f"{domain.upper()}")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  Macro-F1 : {f1*100:.2f}%")

    # Overall
    acc_all = accuracy_score(labels_all, preds_all)
    f1_all  = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    print(f"\n{'='*40}")
    print(f"OVERALL  Accuracy: {acc_all*100:.2f}%  Macro-F1: {f1_all*100:.2f}%")

if __name__ == '__main__':
    main()
