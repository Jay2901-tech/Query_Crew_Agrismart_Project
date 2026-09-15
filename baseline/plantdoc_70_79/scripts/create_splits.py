import os
import json
import csv
import hashlib
from pathlib import Path
import random
from collections import defaultdict
from tqdm import tqdm

random.seed(42)

def compute_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def create_splits():
    with open("config/class_mapping.json", "r") as f:
        mapping = json.load(f)
        
    datasets_base = Path("data")
    
    # We will track hashes to prevent leakage
    global_hashes = {} # hash -> list of file paths
    
    # For each dataset, collect images by canonical class
    # structure: collected[canonical_class][dataset_name] = [list of filepaths]
    collected = defaultdict(lambda: defaultdict(list))
    
    for canon_class, sources in mapping.items():
        if "UNKNOWN" in canon_class:
            continue
            
        for ds_name, original_name in sources.items():
            if not original_name:
                continue
                
            # If PlantDoc, it's organized differently (train/test folders)
            # Actually, the user script said PlantDoc is in data/PlantDoc/train/class and data/PlantDoc/test/class
            ds_path = datasets_base / ds_name.replace("plantvillage", "PlantVillage").replace("plantdoc", "PlantDoc")
            
            if not ds_path.exists():
                continue
                
            if ds_name == "plantvillage":
                class_dir = ds_path / original_name
                if class_dir.exists():
                    for f in class_dir.iterdir():
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            collected[canon_class][ds_name].append(str(f))
            
            elif ds_name == "plantdoc":
                # Check both train and test dirs inside PlantDoc
                for split in ["train", "test"]:
                    class_dir = ds_path / split / original_name
                    if class_dir.exists():
                        for f in class_dir.iterdir():
                            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                collected[canon_class][ds_name].append(str(f))

    print("Hashing all images to detect duplicates...")
    
    # Check for duplicates and build valid lists
    valid_files = defaultdict(lambda: defaultdict(list))
    duplicate_count = 0
    
    all_files_to_hash = []
    for c, ds_dict in collected.items():
        for d, files in ds_dict.items():
            all_files_to_hash.extend([(c, d, f) for f in files])
            
    for c, d, f in tqdm(all_files_to_hash, desc="Hashing"):
        h = compute_hash(f)
        if h in global_hashes:
            global_hashes[h].append(f)
            duplicate_count += 1
            # We skip adding to valid_files if it's a duplicate across datasets or within,
            # wait, if it's identical, just skip it to prevent leakage.
        else:
            global_hashes[h] = [f]
            valid_files[c][d].append(f)
            
    print(f"Found {duplicate_count} exact duplicates. Ignored them to prevent leakage.")
    
    # Now split into train/val/test per class per dataset
    # PV: 80/10/10
    # PlantDoc: 80/10/10
    
    train_set, val_set, test_set = [], [], []
    
    for c, ds_dict in valid_files.items():
        for d, files in ds_dict.items():
            random.shuffle(files)
            n = len(files)
            if n == 0:
                continue
                
            # split sizes
            if n < 10:
                # If very few, put most in train, maybe 1 in val, 1 in test
                tr_end = max(1, int(0.8 * n))
                val_end = tr_end + max(1, int(0.1 * n))
            else:
                tr_end = int(0.8 * n)
                val_end = tr_end + int(0.1 * n)
                
            train_files = files[:tr_end]
            val_files = files[tr_end:val_end]
            test_files = files[val_end:]
            
            for f in train_files:
                train_set.append([f, c, d])
            for f in val_files:
                val_set.append([f, c, d])
            for f in test_files:
                test_set.append([f, c, d])
                
    # Save to CSV
    def save_csv(filename, data):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "class", "domain"])
            for row in data:
                writer.writerow(row)
                
    save_csv("data/train.csv", train_set)
    save_csv("data/val.csv", val_set)
    save_csv("data/test.csv", test_set)
    
    print(f"Splits created: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

if __name__ == "__main__":
    create_splits()
