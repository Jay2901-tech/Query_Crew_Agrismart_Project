"""
download_models.py — Run this ONCE before starting the AgriSmart app.

Downloads the pretrained YOLO leaf-detection model from Hugging Face and saves
it next to this script so the app can load it locally without internet access.

Usage:
    python download_models.py

Requirements:
    pip install huggingface_hub

The model file (yolo11x_leaf.pt) is ~200 MB. Download takes a few minutes on
a typical connection. Once downloaded, subsequent app restarts load it from
disk instantly.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Config — change these if you want a different model
# ---------------------------------------------------------------------------
REPO_ID   = "pedromiguelsanchez/yolo-plant-leaf-detection"
FILENAME  = "yolo11x_leaf.pt"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script


def main():
    # Ensure huggingface_hub is available
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import hf_hub_download

    target_path = os.path.join(LOCAL_DIR, FILENAME)

    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"Model already exists at: {target_path}  ({size_mb:.1f} MB)")
        print("Nothing to download. Run the app with:  streamlit run app.py")
        return

    print(f"Downloading '{FILENAME}' from '{REPO_ID}' ...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=LOCAL_DIR,
    )

    # hf_hub_download may save inside a cache subdirectory; copy to LOCAL_DIR root
    if os.path.abspath(downloaded_path) != os.path.abspath(target_path):
        import shutil
        shutil.copy2(downloaded_path, target_path)
        print(f"Copied to: {target_path}")
    else:
        print(f"Saved to:  {target_path}")

    size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"Done! ({size_mb:.1f} MB)")
    print("You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
