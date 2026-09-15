"""
cv_utils.py — Bridge between the Core CV Model and the Bonus Modules

This is the "translator" that lets every bonus module (B, C, D, G) speak
a common language ("none" / "mild" / "moderate" / "severe") instead of
each one having to understand raw class names and confidence scores.

USE_REAL_MODEL = True tries to load your trained Keras model via
predict.py's predict_image(pil_image) -> (idx, raw_label, confidence).
If that import fails (model file missing, wrong path, still training,
TensorFlow not installed, etc.) it automatically falls back to mock
data, so the rest of the app/bonus modules keep working during
development/demos.

New in v2 (YOLO pipeline):
  - extract_crop_from_label(raw_label) -> canonical crop type string
  - get_leaf_model()                   -> cached YOLO leaf detector
  - detect_and_crop(pil_or_path, ...) -> (img, crops, boxes, is_fallback)
"""

import os
import numpy as np
from PIL import Image

USE_REAL_MODEL = True

if USE_REAL_MODEL:
    try:
        from predict import predict_image, clean_label
        print("[cv_utils] Using real trained model.")
    except Exception as e:
        print(f"[cv_utils] Could not load real model ({e}) — falling back to mock.")
        USE_REAL_MODEL = False

if not USE_REAL_MODEL:
    def clean_label(raw_label):
        return raw_label.replace("___", " - ").replace(",", "").replace("_", " ").strip()

    def predict_image(pil_image):
        """Placeholder used when the real model isn't available yet."""
        return 14, "Tomato___Early_blight", 0.91


# ---------------------------------------------------------------------------
# YOLO leaf detector config
# ---------------------------------------------------------------------------
LEAF_MODEL_FILE = "yolo11x_leaf.pt"          # must be in same dir as app.py
CONF_THRESHOLD   = 0.15                       # YOLO detection confidence
MIN_BOX_AREA_FRAC = 0.02                      # drop boxes < 2 % of image area
MAX_BOXES        = 15                         # max crops to keep, by confidence

_yolo_cache = {}


# ---------------------------------------------------------------------------
# Label → canonical crop type mapping
# Covers all 24 TARGET_CLASSES the classifier can predict.
# Used by app.py to auto-update the sidebar "Crop" selector.
# ---------------------------------------------------------------------------
# Mapping: prefix of raw class label → sidebar crop key
_LABEL_PREFIX_TO_CROP = {
    "Apple":       "apple",
    "Blueberry":   "blueberry",
    "Cherry":      "cherry",
    "Corn":        "corn",
    "Grape":       "grape",
    "Peach":       "peach",
    "Pepper":      "pepper",
    "Potato":      "potato",
    "Raspberry":   "raspberry",
    "Soybean":     "soybean",
    "Squash":      "squash",
    "Strawberry":  "strawberry",
    "Tomato":      "tomato",
}


def extract_crop_from_label(raw_label: str) -> str:
    """
    Maps a raw class label (e.g. 'Tomato___Early_blight') to a canonical
    crop type string used by downstream modules (e.g. 'tomato').

    Design notes:
    - Matches on the FIRST word/prefix before '___' or '_' or ','
      so it's robust to new class names following the same PlantVillage
      naming convention.
    - Returns 'other' for any label not covered (safe default).

    Examples:
        'Tomato___Early_blight'   → 'tomato'
        'Potato___Late_blight'    → 'potato'
        'Apple___Apple_scab'      → 'apple'
        'Corn_(maize)___...'      → 'corn'
        'Pepper,_bell___healthy'  → 'pepper'
    """
    if not raw_label:
        return "other"
    # The prefix is everything before the first '___', '(', or ','
    prefix = raw_label.split("___")[0]   # e.g. 'Tomato' or 'Corn_(maize)'
    prefix = prefix.split("(")[0]        # strip parenthetical
    prefix = prefix.split(",")[0]        # strip comma (Pepper,_bell)
    prefix = prefix.strip().rstrip("_")  # tidy up
    return _LABEL_PREFIX_TO_CROP.get(prefix, "other")


# ---------------------------------------------------------------------------
# YOLO leaf detector — loaded once and cached
# ---------------------------------------------------------------------------

def get_leaf_model():
    """
    Loads yolo11x_leaf.pt using Ultralytics YOLO and caches it in-process.
    Requires the model file to exist locally (run download_models.py first).

    Returns the YOLO model object, or None if the file is missing or
    Ultralytics is not installed (allows the rest of the app to keep working).
    """
    if "leaf" in _yolo_cache:
        return _yolo_cache["leaf"]

    if not os.path.exists(LEAF_MODEL_FILE):
        print(
            f"[cv_utils] YOLO model not found at '{LEAF_MODEL_FILE}'. "
            "Run 'python download_models.py' first. "
            "Falling back to full-image prediction."
        )
        _yolo_cache["leaf"] = None
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(LEAF_MODEL_FILE)
        _yolo_cache["leaf"] = model
        print(f"[cv_utils] Leaf detection model loaded from '{LEAF_MODEL_FILE}'.")
        return model
    except Exception as e:
        print(f"[cv_utils] Could not load YOLO model ({e}). Falling back to full-image.")
        _yolo_cache["leaf"] = None
        return None


def detect_and_crop(
    image_input,
    conf: float = CONF_THRESHOLD,
    min_area_frac: float = MIN_BOX_AREA_FRAC,
    max_boxes: int = MAX_BOXES,
):
    """
    Step 1 / 4 / 7 of the YOLO leaf-detection pipeline.

    Args:
        image_input: PIL Image object OR a file path string.
        conf:          YOLO detection confidence threshold.
        min_area_frac: Drop boxes smaller than this fraction of total image area.
                       Filters false-positive specks (< 2 % is a safe default).
        max_boxes:     Cap on returned crops; highest-confidence first.

    Returns:
        (original_img, crops, boxes, is_fallback)
        - original_img  : PIL Image of the full input
        - crops         : list of PIL Images (individual leaf crops)
        - boxes         : np.ndarray of shape (N, 4) in xyxy format
        - is_fallback   : True if no valid leaf survived filtering → crops == [original_img]

    Pipeline steps implemented here:
        Step 1  — run YOLO leaf detector
        Step 4  — drop tiny boxes (area < min_area_frac of image)
        (cap)   — keep at most max_boxes, highest confidence first
        Step 7  — if nothing survived, return full image as single crop

    Sources for thresholds:
        CONF_THRESHOLD=0.15   : permissive; allows the later area filter to cull
                                junk detections rather than missing real leaves.
        MIN_BOX_AREA_FRAC=0.02: a box covering < 2% of the frame is almost
                                always a false positive or background artifact.
        MAX_BOXES=15          : balances UX (gallery size) with runtime cost.
    """
    # Accept both PIL Image and file path
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
        predict_input = image_input   # YOLO can accept file paths directly
    else:
        img = image_input.convert("RGB")
        predict_input = img           # YOLO accepts PIL images too

    img_area = img.size[0] * img.size[1]

    leaf_model = get_leaf_model()

    # If model unavailable, fall back immediately to full image
    if leaf_model is None:
        full_box = np.array([[0, 0, img.size[0], img.size[1]]])
        return img, [img], full_box, True

    results = leaf_model.predict(predict_input, conf=conf, verbose=False)

    # Step 7: nothing detected at all
    if len(results[0].boxes) == 0:
        full_box = np.array([[0, 0, img.size[0], img.size[1]]])
        return img, [img], full_box, True

    boxes_raw = results[0].boxes.xyxy.cpu().numpy()
    confs_raw = results[0].boxes.conf.cpu().numpy()

    # Step 4: drop tiny boxes
    kept = []
    for box, c in zip(boxes_raw, confs_raw):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area / img_area >= min_area_frac:
            kept.append((box, float(c)))

    if not kept:
        # Every box was too tiny → fall back to full image (Step 7)
        full_box = np.array([[0, 0, img.size[0], img.size[1]]])
        return img, [img], full_box, True

    # Cap to max_boxes, keeping the highest-confidence ones first
    kept.sort(key=lambda bc: bc[1], reverse=True)
    kept = kept[:max_boxes]

    boxes_final = np.array([b for b, c in kept])
    crops = [img.crop(tuple(map(int, b))) for b, c in kept]
    return img, crops, boxes_final, False


# ---------------------------------------------------------------------------
# Existing helpers — unchanged, keep for backward compatibility
# ---------------------------------------------------------------------------

def predict(pil_image):
    """
    Wrapper so every bonus module (agent.py, irrigation.py, disease_risk.py,
    sustainability.py) can call one consistent shape regardless of whether
    the real model or the mock is active:

        predict(pil_image) -> {"class": str, "confidence": float}

    Note: this takes a PIL Image object (not a file path), matching what
    Streamlit's camera_input/file_uploader give you after Image.open().
    """
    _, raw_label, confidence = predict_image(pil_image)
    return {"class": clean_label(raw_label), "confidence": round(float(confidence), 4)}


def map_prediction_to_severity(disease_class, confidence):
    """
    Convert a raw CV prediction into a simple severity label used across
    irrigation (B), disease risk (C), and sustainability (D) logic.

    Thresholds:
        < 0.60  → "mild"      : low-confidence positive — treat cautiously
        0.60–0.85 → "moderate"
        >= 0.85 → "severe"

    The "healthy" shortcut avoids the confidence thresholds entirely, since
    any confidence level for a healthy prediction means no disease.
    """
    if "healthy" in disease_class.strip().lower():
        return "none"
    if confidence < 0.60:
        return "mild"       # low-confidence positive — treat cautiously
    elif confidence < 0.85:
        return "moderate"
    else:
        return "severe"


if __name__ == "__main__":
    # Quick manual test — replace with a real image path when testing locally
    img = Image.open("sample_leaf.jpg")
    pred = predict(img)
    sev = map_prediction_to_severity(pred["class"], pred["confidence"])
    print(pred, "->", sev)

    # Test label mapping
    test_labels = [
        "Tomato___Early_blight",
        "Potato___Late_blight",
        "Apple___Apple_scab",
        "Corn_(maize)___Common_rust_",
        "Pepper,_bell___healthy",
        "Squash___Powdery_mildew",
    ]
    print("\nextract_crop_from_label tests:")
    for lbl in test_labels:
        print(f"  {lbl!r:45s} -> {extract_crop_from_label(lbl)!r}")