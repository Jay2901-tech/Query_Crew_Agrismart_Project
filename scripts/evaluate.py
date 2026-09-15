import os
import json
import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
import pandas as pd

from train import CSVDataset, get_transforms
import timm
from tqdm import tqdm
from torch.amp import autocast

def evaluate_on_test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_domains = []
    
    pbar = tqdm(dataloader, desc="Evaluating on Test Set")
    for images, labels, domains in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
            
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_domains.extend(domains)
        
    return all_preds, all_labels, all_domains

def generate_report(preds, labels, domains, classes, output_path):
    report_lines = []
    report_lines.append("# AgriSmart-AI Evaluation Report\n")
    
    df = pd.DataFrame({'pred': preds, 'label': labels, 'domain': domains})
    
    # Combined Metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds)
    
    report_lines.append("## Overall Metrics (Combined Field)")
    report_lines.append(f"- **Macro Precision:** {macro_p:.4f}")
    report_lines.append(f"- **Macro Recall:** {macro_r:.4f}")
    report_lines.append(f"- **Macro F1-Score:** {macro_f1:.4f}")
    report_lines.append(f"- **Balanced Accuracy:** {bal_acc:.4f}\n")
    
    # Per-domain metrics
    report_lines.append("## Domain-Specific Metrics\n")
    
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        dp, dr, df1, _ = precision_recall_fscore_support(domain_df['label'], domain_df['pred'], average='macro', zero_division=0)
        d_acc = balanced_accuracy_score(domain_df['label'], domain_df['pred'])
        
        report_lines.append(f"### {domain}")
        report_lines.append(f"- **Samples:** {len(domain_df)}")
        report_lines.append(f"- **Macro F1-Score:** {df1:.4f}")
        report_lines.append(f"- **Balanced Accuracy:** {d_acc:.4f}\n")
        
    # Per-class metrics
    report_lines.append("## Per-Class Metrics\n")
    report_lines.append("| Class | Precision | Recall | F1-Score | Support |")
    report_lines.append("|---|---|---|---|---|")
    
    p, r, f1, s = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    for i, cls in enumerate(classes):
        report_lines.append(f"| {cls} | {p[i]:.4f} | {r[i]:.4f} | {f1[i]:.4f} | {s[i]} |")
        
    with open(output_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Report generated at {output_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found at {args.weights}")
        return
        
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    _, val_transform = get_transforms()
    test_dataset = CSVDataset(args.test_csv, classes, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    preds, labels, domains = evaluate_on_test(model, test_loader, device)
    
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    generate_report(preds, labels, domains, classes, args.output_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate crop disease model")
    parser.add_argument("--test-csv", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--weights", type=str, default="models/best_model.pth", help="Path to the model weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers")
    parser.add_argument("--output-report", type=str, default="reports/evaluation_report.md", help="Output MD report path")
    
    args = parser.parse_args()
    main(args)
