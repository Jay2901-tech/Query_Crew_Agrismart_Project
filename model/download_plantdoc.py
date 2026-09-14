import os
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
from download_data import download_url_with_resume
import json

def check_and_download_plantdoc(data_dir: str):
    """
    Checks if the PlantDoc dataset exists at data_dir.
    If not, downloads it from GitHub and extracts it.
    """
    target_dir = Path(data_dir)
    
    if target_dir.exists() and (target_dir / "train").exists() and (target_dir / "test").exists():
        print(f"[OK] PlantDoc dataset already found at {data_dir}.")
        return True

    print(f"[WARNING] PlantDoc dataset not found at {data_dir}. Preparing to download...")
    
    url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = Path("plantdoc_master.zip")
    temp_dir = Path("temp_plantdoc_extract")
    
    try:
        # 1. Download
        print(f"[DOWNLOAD] Downloading PlantDoc dataset from GitHub...")
        download_url_with_resume(url, zip_path)
        
        # 2. Extract
        print("\n[EXTRACT] Extracting dataset (this may take a minute)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in train and test directories
            members = [m for m in zip_ref.namelist() if "train/" in m or "test/" in m]
            for member in tqdm(members, desc="Extracting"):
                zip_ref.extract(member, path=temp_dir)
                
        # 3. Move to final data_dir
        print(f"\n[MOVE] Moving files to target directory: {data_dir}")
        os.makedirs(target_dir, exist_ok=True)
        
        extracted_base = temp_dir / "PlantDoc-Dataset-master"
        
        for split in ["train", "test"]:
            src_split = extracted_base / split
            dst_split = target_dir / split
            if src_split.exists():
                if dst_split.exists():
                    shutil.rmtree(dst_split)
                shutil.move(str(src_split), str(target_dir))
                
        print("[SUCCESS] PlantDoc dataset successfully downloaded and prepared!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download or extract dataset: {e}")
        return False
        
    finally:
        # 4. Cleanup
        print("[CLEANUP] Cleaning up temporary files...")
        if zip_path.exists():
            os.remove(zip_path)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download PlantDoc Dataset")
    parser.add_argument("--data-dir", type=str, default="data/PlantDoc", help="Destination directory for the dataset")
    args = parser.parse_args()
    
    check_and_download_plantdoc(args.data_dir)
