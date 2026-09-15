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
import sys
import numpy as np
from PIL import Image

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

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
# ---------------------------------------------------------------------------
# YOLO leaf detector config
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POSSIBLE_YOLO_PATHS = [
    os.path.join(_BASE_DIR, "model", "yolo11x_leaf.pt"),
    os.path.join(_BASE_DIR, "yolo11x_leaf.pt"),
    "model/yolo11x_leaf.pt",
    "yolo11x_leaf.pt",
]
LEAF_MODEL_FILE = next((p for p in _POSSIBLE_YOLO_PATHS if os.path.exists(p)), _POSSIBLE_YOLO_PATHS[0])
CONF_THRESHOLD   = 0.15                       # YOLO detection confidence
MIN_BOX_AREA_FRAC = 0.02                      # drop boxes < 2 % of image area
MAX_BOXES        = 15                         # max crops to keep, by confidence

_yolo_cache = {}


def opencv_crop_leaf(pil_img):
    """
    OpenCV HSV green-channel segmentation fallback.
    Detects the leaf bounding box and crops it when YOLO is not active.
    """
    try:
        import cv2
        np_img = np.array(pil_img.convert("RGB"))
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)

        # Green color range in HSV
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = np_img.shape[0] * np_img.shape[1]

        valid_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area / img_area >= 0.02:  # at least 2% of area
                valid_boxes.append((x, y, x + w, y + h))

        if valid_boxes:
            # Sort by area largest first
            valid_boxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
            crops = [pil_img.crop(b) for b in valid_boxes[:5]]
            boxes_final = np.array(valid_boxes[:5])
            return pil_img, crops, boxes_final, False
    except Exception:
        pass

    full_box = np.array([[0, 0, pil_img.size[0], pil_img.size[1]]])
    return pil_img, [pil_img], full_box, True


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
    """
    if "leaf" in _yolo_cache:
        return _yolo_cache["leaf"]

    model_path = LEAF_MODEL_FILE
    if not os.path.exists(model_path):
        for p in _POSSIBLE_YOLO_PATHS:
            if os.path.exists(p):
                model_path = p
                break

    if not os.path.exists(model_path):
        print(
            f"[cv_utils] YOLO model not found at '{model_path}'. "
            "Falling back to OpenCV green-channel leaf segmentation."
        )
        _yolo_cache["leaf"] = None
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        _yolo_cache["leaf"] = model
        print(f"[cv_utils] Leaf detection model loaded from '{model_path}'.")
        return model
    except Exception as e:
        print(f"[cv_utils] Could not load YOLO model ({e}). Falling back to OpenCV segmentation.")
        _yolo_cache["leaf"] = None
        return None


def detect_and_crop(
    image_input,
    conf: float = CONF_THRESHOLD,
    min_area_frac: float = MIN_BOX_AREA_FRAC,
    max_boxes: int = MAX_BOXES,
):
    # Accept both PIL Image and file path
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
        predict_input = image_input   # YOLO can accept file paths directly
    else:
        img = image_input.convert("RGB")
        predict_input = img           # YOLO accepts PIL images too

    img_area = img.size[0] * img.size[1]

    leaf_model = get_leaf_model()

    # If YOLO model unavailable, fall back to OpenCV leaf segmentation
    if leaf_model is None:
        return opencv_crop_leaf(img)

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