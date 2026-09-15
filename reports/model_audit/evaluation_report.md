# Model Audit Report (Phases 1-5)

## Phase 1: Baseline Verification
**1. Checkpoint SHA-256:**
- Target File: `model/weights/best_model.pth`
- Expected SHA-256: `94cd7dff369884f3c77ea7d15490fdd1556d77439a78a76a27251808d48b31cf`
- Actual SHA-256: `94cd7dff369884f3c77ea7d15490fdd1556d77439a78a76a27251808d48b31cf` (Match)

**2. Model Architecture:**
- Architecture: `tf_efficientnetv2_s`
- Classes: 38 output classes (uses `plantvillage` specific string naming rather than canonical keys).

**3. Current PlantVillage and PlantDoc Performance:**
An evaluation was run directly against the test set for `model/weights/best_model.pth`:
- **PlantVillage Accuracy**: 99.96%
- **PlantVillage Macro-F1**: 99.94%
- **PlantDoc Accuracy**: 16.85%
- **PlantDoc Macro-F1**: 12.28%

> [!WARNING]
> **Severe Generalization Failure**: The production checkpoint claims 99.94% PlantVillage F1 but plummets to 12.28% F1 on PlantDoc. This indicates severe overfitting to the clean PlantVillage lab setting and a near-complete failure to generalize to real-world noisy images.

**4. Inference Pipeline Behavior (`model/predict.py`)**
- **Uncertainty Rejection**: Uses `CONFIDENCE_THRESHOLD = 0.60`. If the top predicted class confidence is below 60%, the prediction is rejected as "UNCERTAIN".
- **Non-Plant Rejection**: Uses a lightweight `mobilenetv3_small_100` ImageNet model. If none of the top 5 predicted ImageNet classes contain plant-related keywords (e.g., "plant", "leaf", "flower"), the image is rejected to prevent out-of-distribution (OOD) errors (e.g., classifying a car as a diseased plant).

## Phase 2: Training History Audit
**Training Logs and Metadata:**
- No epoch-by-epoch checkpoints exist. The only saved artifact is `models/best_model_metadata.json` (Note: this is in `models/`, NOT `model/weights/`).
- The metadata in `models/best_model_metadata.json` shows:
  - `epoch`: 2
  - `pv_f1`: 99.24%
  - `pd_f1`: 73.94%
- The production checkpoint (`model/weights/best_model.pth`) exhibits a PlantDoc F1 of 12.28%, which severely contradicts the 73.94% logged in `best_model_metadata.json`. This implies the production checkpoint was likely trained purely on PlantVillage or is an overfitted early snapshot that did not utilize the balanced sampling strategy implemented in `train.py`.

## Phase 3: PlantDoc Error Analysis
Due to the drastic performance drop on PlantDoc (12.28% F1), the current production model is virtually unusable for real-world images. The model is overconfident on incorrect predictions when presented with PlantDoc images because it has not learned to generalize past the uniform background characteristics of PlantVillage. 

## Phase 4: Duplicate / Leakage Analysis
A strict SHA-256 cryptographic hash check was performed across all images in `data/PlantVillage` and `data/PlantDoc`.
- **Total Duplicate Groups Found**: 33 pairs.
- **Leakage Assessment**: All 33 duplicates occurred between images not assigned to splits, or identical copies within the source directories. There was **0 leakage** between train/test, train/val, or between datasets. The dataset splits themselves are structurally sound.

## Phase 5: Class Mapping Validation
The production checkpoint (`model/weights/best_model.pth`) expects `plantvillage` string identifiers (e.g., `Cherry_(including_sour)___healthy`), whereas the official evaluation script (`scripts/evaluate.py`) and training script (`scripts/train.py`) have been updated to use unified canonical keys (e.g., `Cherry___healthy`). This mismatch caused `evaluate.py` to crash out of the box with a `KeyError`. The evaluation had to be performed using a patched script that translated canonical keys back to PlantVillage strings to match the production checkpoint's expected outputs.
