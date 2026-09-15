import os
import requests
import zipfile
import shutil
import json
import csv
from pathlib import Path
from tqdm import tqdm

def download_url_with_resume(url, output_path):
    output_path = Path(output_path)
    resume_header = {}
    mode = 'ab'
    
    if output_path.exists():
        file_size = output_path.stat().st_size
        resume_header = {'Range': f'bytes={file_size}-'}
    else:
        file_size = 0
        mode = 'wb'
        
    response = requests.get(url, headers=resume_header, stream=True)
    
    if response.status_code == 416:
        print("File is already fully downloaded.")
        return
        
    if response.status_code not in (200, 206):
        raise Exception(f"Failed to download: {response.status_code}")
        
    if response.status_code == 200 and file_size > 0:
        print("\n[WARNING] Server does not support resuming. Restarting download from the beginning...")
        mode = 'wb'
        file_size = 0
        
    total_size = int(response.headers.get('content-length', 0)) + file_size
    
    with open(output_path, mode) as f, tqdm(
        desc=output_path.name,
        total=total_size,
        initial=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def check_and_download_dataset(data_dir: str, name: str, url: str, extract_filter=None, nested_dir=None):
    target_dir = Path(data_dir)
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[OK] {name} already found at {data_dir}.")
        return True

    print(f"[WARNING] {name} not found at {data_dir}. Preparing to download...")
    
    zip_path = Path(f"{name.lower()}_master.zip")
    temp_dir = Path(f"temp_{name.lower()}_extract")
    
    try:
        print(f"[DOWNLOAD] Downloading {name} dataset from GitHub...")
        download_url_with_resume(url, zip_path)
        
        print("\n[EXTRACT] Extracting dataset (this may take a minute)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if extract_filter:
                members = [m for m in zip_ref.namelist() if extract_filter(m)]
            else:
                members = zip_ref.namelist()
            for member in tqdm(members, desc="Extracting"):
                zip_ref.extract(member, path=temp_dir)
                
        print(f"\n[MOVE] Moving files to target directory: {data_dir}")
        os.makedirs(target_dir, exist_ok=True)
        
        if nested_dir:
            extracted_dir = temp_dir / nested_dir
        else:
            extracted_dir = temp_dir
            
        if not extracted_dir.exists():
            # Find the actual extracted root folder
            extracted_dir = next(temp_dir.iterdir())
            if nested_dir:
                extracted_dir = extracted_dir / nested_dir

        if extracted_dir.exists():
            for item in extracted_dir.iterdir():
                dest_item = target_dir / item.name
                if dest_item.exists() and dest_item.is_dir():
                    shutil.rmtree(dest_item)
                elif dest_item.exists():
                    os.remove(dest_item)
                shutil.move(str(item), str(target_dir))
        
        print(f"[SUCCESS] {name} dataset successfully downloaded and prepared!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download or extract {name}: {e}")
        return False
        
    finally:
        print("[CLEANUP] Cleaning up temporary files...")
        if zip_path.exists():
            os.remove(zip_path)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def generate_inventory(base_data_dir: str):
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    datasets = ["PlantVillage", "PlantDoc", "PlantWild"]
    inventory = {}
    
    for ds in datasets:
        ds_path = Path(base_data_dir) / ds
        if not ds_path.exists():
            continue
            
        inventory[ds] = {"total_images": 0, "classes": {}}
        
        for root, _, files in os.walk(ds_path):
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG'))]
            if img_files:
                class_name = Path(root).name
                count = len(img_files)
                if class_name not in inventory[ds]["classes"]:
                    inventory[ds]["classes"][class_name] = 0
                inventory[ds]["classes"][class_name] += count
                inventory[ds]["total_images"] += count
                
    with open(reports_dir / "dataset_inventory.json", "w") as f:
        json.dump(inventory, f, indent=4)
        
    with open(reports_dir / "dataset_inventory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Class", "Image Count"])
        for ds, data in inventory.items():
            for cls, count in data["classes"].items():
                writer.writerow([ds, cls, count])
                
    print("[INFO] Dataset inventory generated in reports/")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    check_and_download_dataset(
        data_dir="data/PlantVillage",
        name="PlantVillage",
        url="https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip",
        extract_filter=lambda m: "raw/color/" in m,
        nested_dir="raw/color"
    )
    
    check_and_download_dataset(
        data_dir="data/PlantDoc",
        name="PlantDoc",
        url="https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip",
        extract_filter=lambda m: "train/" in m or "test/" in m,
        nested_dir=""
    )
    
    print("[INFO] Attempting PlantWild...")
    success = check_and_download_dataset(
        data_dir="data/PlantWild",
        name="PlantWild",
        url="https://github.com/Project-AgML/PlantWild/archive/refs/heads/main.zip",
        extract_filter=None,
        nested_dir=""
    )
    if not success:
        print("[WARNING] PlantWild unavailable or failed to integrate reliably. Continuing with PV + PlantDoc.")
        
    generate_inventory("data")
