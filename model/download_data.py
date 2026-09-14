import os
import requests
import zipfile
import shutil
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
    
    if response.status_code == 416: # Range Not Satisfiable
        print("File is already fully downloaded.")
        return
        
    if response.status_code not in (200, 206):
        raise Exception(f"Failed to download: {response.status_code}")
        
    # If the server ignores the Range header (like GitHub dynamic archives) and returns 200 instead of 206,
    # we MUST NOT append, but rather overwrite from the beginning.
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

def check_and_download_dataset(data_dir: str):
    """
    Checks if the dataset exists at data_dir.
    If not, downloads the PlantVillage dataset from GitHub and extracts it to data_dir.
    """
    target_dir = Path(data_dir)
    
    # Basic check: if directory exists and has some subdirectories (classes), we assume it's good.
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[OK] Dataset already found at {data_dir}.")
        return True

    print(f"[WARNING] Dataset not found at {data_dir}. Preparing to download...")
    
    url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    zip_path = Path("plantvillage_master.zip")
    temp_dir = Path("temp_plantvillage_extract")
    
    try:
        # 1. Download
        print(f"[DOWNLOAD] Downloading PlantVillage dataset from GitHub...")
        download_url_with_resume(url, zip_path)
        
        # 2. Extract
        print("\n[EXTRACT] Extracting dataset (this may take a minute)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in the raw/color directory
            members = [m for m in zip_ref.namelist() if "raw/color/" in m]
            for member in tqdm(members, desc="Extracting"):
                zip_ref.extract(member, path=temp_dir)
                
        # 3. Move to final data_dir
        print(f"\n[MOVE] Moving files to target directory: {data_dir}")
        os.makedirs(target_dir, exist_ok=True)
        
        # The structure inside temp_dir will be: temp_plantvillage_extract/PlantVillage-Dataset-master/raw/color/...
        extracted_color_dir = temp_dir / "PlantVillage-Dataset-master" / "raw" / "color"
        
        for class_dir in extracted_color_dir.iterdir():
            if class_dir.is_dir():
                dest_dir = target_dir / class_dir.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.move(str(class_dir), str(target_dir))
                
        print("[SUCCESS] Dataset successfully downloaded and prepared!")
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
    parser = argparse.ArgumentParser(description="Download PlantVillage Dataset")
    parser.add_argument("--data-dir", type=str, default="data/PlantVillage", help="Destination directory for the dataset")
    args = parser.parse_args()
    
    check_and_download_dataset(args.data_dir)
