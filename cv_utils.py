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

Leaf Detection (v3 — OpenCV, no PyTorch):
  Instead of a separate YOLO model file, leaf regions are found using
  classical computer vision:
    1. Convert to HSV colour space (more robust than RGB for green detection)
    2. Threshold for green + yellow-green (covers healthy AND diseased leaves)
    3. Morphological close+open to merge fragments and drop noise
    4. Find external contours → bounding boxes → crop
  This removes the PyTorch / Ultralytics dependency entirely.
  Fallback: if no green region is found, the full image is used (same
  behaviour as the previous YOLO fallback path).
"""

import os
import cv2
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
# Leaf detection config
# ---------------------------------------------------------------------------
MIN_BOX_AREA_FRAC = 0.02    # drop boxes < 2 % of image area (same as before)
MAX_BOXES         = 15       # max crops to return, largest-first (same as before)

# HSV green range — covers healthy green leaves
_HSV_GREEN_LO = np.array([25,  40,  40])
_HSV_GREEN_HI = np.array([95, 255, 255])

# HSV yellow-green range — covers chlorotic / early-disease leaves
_HSV_YELLOW_LO = np.array([15, 40,  40])
_HSV_YELLOW_HI = np.array([30, 255, 255])

# Morphology kernel for closing small gaps and opening noise
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))


# ---------------------------------------------------------------------------
# Label → canonical crop type mapping
# Covers all 24 TARGET_CLASSES the classifier can predict.
# Used by app.py to auto-update the sidebar "Crop" selector.
# ---------------------------------------------------------------------------
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
    prefix = raw_label.split("___")[0]
    prefix = prefix.split("(")[0]
    prefix = prefix.split(",")[0]
    prefix = prefix.strip().rstrip("_")
    return _LABEL_PREFIX_TO_CROP.get(prefix, "other")


# ---------------------------------------------------------------------------
# OpenCV-based leaf detector
# ---------------------------------------------------------------------------

def _build_leaf_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Build a binary mask that covers leaf-coloured pixels (green + yellow-green).

    Steps:
      1. Convert RGB → BGR → HSV  (HSV separates hue from brightness so
         shadows and highlights don't break the colour threshold)
      2. Threshold for green range AND yellow-green range separately
      3. Combine with bitwise OR
      4. Morphological CLOSE to fill small holes within a leaf region
      5. Morphological OPEN  to remove small isolated noise blobs

    Returns a uint8 mask (255 = leaf pixel, 0 = background).
    """
    bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv  = cv2.cvtColor(bgr,     cv2.COLOR_BGR2HSV)

    mask_green  = cv2.inRange(hsv, _HSV_GREEN_LO,  _HSV_GREEN_HI)
    mask_yellow = cv2.inRange(hsv, _HSV_YELLOW_LO, _HSV_YELLOW_HI)
    mask = cv2.bitwise_or(mask_green, mask_yellow)

    # Close: bridge small gaps inside a leaf (e.g. veins, spots)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    # Open:  remove tiny specks that aren't real leaf regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)

    return mask


def detect_and_crop(
    image_input,
    conf: float = None,              # kept for API compatibility — unused by OpenCV
    min_area_frac: float = MIN_BOX_AREA_FRAC,
    max_boxes: int = MAX_BOXES,
):
    """
    Detect leaf regions in an image using OpenCV colour segmentation
    and return individual crops for the classifier.

    Args:
        image_input: PIL Image object OR a file path string.
        conf:          Ignored (was the YOLO confidence threshold).
                       Kept so call sites in app.py don't need changes.
        min_area_frac: Drop bounding boxes smaller than this fraction of
                       total image area. Filters background noise.
        max_boxes:     Cap on returned crops; largest-area first.

    Returns:
        (original_img, crops, boxes, is_fallback)
        - original_img : PIL Image of the full input
        - crops        : list of PIL Images (individual leaf crops)
        - boxes        : np.ndarray of shape (N, 4) in xyxy format
        - is_fallback  : True when no valid leaf region was found →
                         crops == [original_img] (full image used)
    """
    # ── Load image ──────────────────────────────────────────────────────────
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")
    except Exception as e:
        print(f"[cv_utils] Could not open image ({e}) — using blank fallback.")
        img = Image.new("RGB", (224, 224), (200, 200, 200))

    img_w, img_h = img.size
    img_area = img_w * img_h
    img_np   = np.array(img)

    # ── Build colour mask & find contours ───────────────────────────────────
    try:
        mask     = _build_leaf_mask(img_np)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception as e:
        print(f"[cv_utils] OpenCV segmentation failed ({e}) — using full image.")
        full_box = np.array([[0, 0, img_w, img_h]])
        return img, [img], full_box, True

    # ── Filter by minimum area ───────────────────────────────────────────────
    valid = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = w * h
        if box_area / img_area >= min_area_frac:
            valid.append((x, y, x + w, y + h, box_area))

    # ── Fallback: no leaf region found → use full image ──────────────────────
    if not valid:
        print("[cv_utils] No leaf region detected by colour segmentation — using full image.")
        full_box = np.array([[0, 0, img_w, img_h]])
        return img, [img], full_box, True

    # ── Sort largest-first, cap at max_boxes ────────────────────────────────
    valid.sort(key=lambda b: b[4], reverse=True)
    valid = valid[:max_boxes]

    boxes  = np.array([[b[0], b[1], b[2], b[3]] for b in valid])
    crops  = [img.crop((b[0], b[1], b[2], b[3])) for b in valid]

    return img, crops, boxes, False


# ---------------------------------------------------------------------------
# Existing helpers — unchanged, keep for backward compatibility
# ---------------------------------------------------------------------------

def predict_mock(image_path=None):
    """Mock prediction — used by agent.py when no real image is supplied."""
    return {"class": "Tomato Early blight", "confidence": 0.91}


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
        return "mild"
    elif confidence < 0.85:
        return "moderate"
    else:
        return "severe"


if __name__ == "__main__":
    # Quick manual test — replace with a real image path when testing locally
    img = Image.open("test_leaf.jpg")
    original, crops, boxes, is_fallback = detect_and_crop(img)
    print(f"is_fallback={is_fallback}, crops found={len(crops)}")
    for i, crop in enumerate(crops):
        print(f"  Crop {i+1}: {crop.size}")

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