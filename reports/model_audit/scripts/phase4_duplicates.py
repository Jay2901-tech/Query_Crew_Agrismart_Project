import os
import hashlib
from collections import defaultdict
import csv
import pandas as pd

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_files_with_hash(directory):
    file_hashes = defaultdict(list)
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, f)
                file_hashes[hash_file(path)].append(path)
    return file_hashes

def main():
    print("Hashing all images...")
    pv_hashes = get_files_with_hash("data/PlantVillage")
    pd_hashes = get_files_with_hash("data/PlantDoc")
    
    # Load splits
    def load_splits(csv_path):
        if not os.path.exists(csv_path): return {}
        df = pd.read_csv(csv_path)
        return {os.path.normpath(row['filepath']).replace('\\', '/'): split_name for _, row in df.iterrows()}
    
    splits = {}
    for s in ['train', 'val', 'test']:
        df = pd.read_csv(f'data/{s}.csv')
        for _, row in df.iterrows():
            splits[os.path.normpath(row['filepath']).replace('\\', '/')] = s
            
    all_hashes = defaultdict(list)
    for h, paths in pv_hashes.items():
        all_hashes[h].extend(paths)
    for h, paths in pd_hashes.items():
        all_hashes[h].extend(paths)
        
    duplicates = {h: paths for h, paths in all_hashes.items() if len(paths) > 1}
    
    out_csv = "reports/model_audit/duplicate_report.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    counts = {
        'PV_train_test': 0,
        'PV_train_val': 0,
        'PD_train_test': 0,
        'PD_train_val': 0,
        'Cross_Dataset': 0,
        'Other': 0
    }
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "path1", "split1", "dataset1", "path2", "split2", "dataset2", "category"])
        
        for h, paths in duplicates.items():
            for i in range(len(paths)):
                for j in range(i+1, len(paths)):
                    p1 = paths[i].replace('\\', '/')
                    p2 = paths[j].replace('\\', '/')
                    
                    s1 = splits.get(p1, 'unknown')
                    s2 = splits.get(p2, 'unknown')
                    
                    d1 = 'PlantVillage' if 'PlantVillage' in p1 else 'PlantDoc'
                    d2 = 'PlantVillage' if 'PlantVillage' in p2 else 'PlantDoc'
                    
                    cat = 'Other'
                    if d1 == 'PlantVillage' and d2 == 'PlantVillage':
                        if set([s1, s2]) == {'train', 'test'}: cat = 'PV_train_test'
                        elif set([s1, s2]) == {'train', 'val'}: cat = 'PV_train_val'
                    elif d1 == 'PlantDoc' and d2 == 'PlantDoc':
                        if set([s1, s2]) == {'train', 'test'}: cat = 'PD_train_test'
                        elif set([s1, s2]) == {'train', 'val'}: cat = 'PD_train_val'
                    elif d1 != d2:
                        cat = 'Cross_Dataset'
                        
                    counts[cat] += 1
                    
                    writer.writerow([h, p1, s1, d1, p2, s2, d2, cat])
                    
    print(f"Total duplicate groups: {len(duplicates)}")
    for k, v in counts.items():
        print(f"{k}: {v} pairs")

if __name__ == '__main__':
    main()
