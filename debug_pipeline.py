"""
debug_pipeline.py — Run this to test the full detection pipeline
Usage: python debug_pipeline.py <path_to_image>
       python debug_pipeline.py  (uses a sample URL download if no image given)
"""
import sys
import os

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

print("=" * 60)
print("STEP 1: Checking model files...")
for f in ["model/yolo11x_leaf.pt", "model/phase2_best_model.keras", "model/agri_mobilenetv2_plantwild.keras"]:
    exists = os.path.exists(f)
    size = f"{os.path.getsize(f) / 1e6:.1f} MB" if exists else "MISSING"
    print(f"  {'OK' if exists else 'MISSING'} {f}: {size}")

print()
print("STEP 2: Loading YOLO leaf detector...")
try:
    from ultralytics import YOLO
    yolo = YOLO("model/yolo11x_leaf.pt" if os.path.exists("model/yolo11x_leaf.pt") else "yolo11x_leaf.pt")
    print("  YOLO loaded OK")
except Exception as e:
    print(f"  YOLO load notice: {e}")

print()
print("STEP 3: Loading Keras classifier...")
try:
    import tensorflow as tf
    m_path = "model/phase2_best_model.keras" if os.path.exists("model/phase2_best_model.keras") else "phase2_best_model.keras"
    model = tf.keras.models.load_model(m_path)
    print(f"  Keras model loaded OK - output shape: {model.output_shape}")
except Exception as e:
    print(f"  Keras load failed: {e}")
    sys.exit(1)

# Get image
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    import urllib.request
    img_path = "test_leaf.jpg"
    print()
    print("STEP 3.5: Downloading test leaf image...")
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_leaf_shiny.jpg/640px-Simple_leaf_shiny.jpg"
    try:
        urllib.request.urlretrieve(url, img_path)
        print(f"  Downloaded to {img_path}")
    except Exception as e:
        print(f"  Download failed: {e}. Please provide an image path as argument.")
        sys.exit(1)

print()
print(f"STEP 4: Running YOLO detection on '{img_path}'...")
try:
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    print(f"  Image size: {img.size}")

    results = yolo.predict(img_path, conf=0.15, verbose=False)
    boxes = results[0].boxes
    print(f"  Raw detections: {len(boxes)} boxes")

    if len(boxes) == 0:
        print("  WARNING: YOLO found 0 boxes - will use FULL IMAGE as fallback")
    else:
        for i, (box, conf) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())):
            x1, y1, x2, y2 = box
            area_frac = ((x2-x1) * (y2-y1)) / (img.size[0] * img.size[1])
            kept = area_frac >= 0.02
            print(f"    Box {i+1}: xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  conf={conf:.3f}  area={area_frac*100:.1f}%  {'KEPT' if kept else 'DROPPED (< 2%)'}")
except Exception as e:
    print(f"  YOLO predict failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print("STEP 5: Running full cv_utils.detect_and_crop...")
try:
    from src.cv_utils import detect_and_crop, extract_crop_from_label
    orig, crops, boxes_arr, is_fallback = detect_and_crop(img_path)
    print(f"  is_fallback: {is_fallback}")
    print(f"  Number of crops: {len(crops)}")
    for i, c in enumerate(crops):
        print(f"    Crop {i+1}: size={c.size}")
except Exception as e:
    print(f"  detect_and_crop failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print("STEP 6: Classifying first crop...")
try:
    from src.predict import predict_image_full, clean_label
    from src.cv_utils import extract_crop_from_label, map_prediction_to_severity
    idx, raw_label, confidence, _ = predict_image_full(crops[0])
    label = clean_label(raw_label)
    crop_type = extract_crop_from_label(raw_label)
    severity = map_prediction_to_severity(label, confidence)
    print(f"  Raw label  : {raw_label}")
    print(f"  Clean label: {label}")
    print(f"  Confidence : {confidence*100:.1f}%")
    print(f"  Crop type  : {crop_type}")
    print(f"  Severity   : {severity}")
except Exception as e:
    print(f"  Classification failed: {e}")
    import traceback; traceback.print_exc()

print()
print("=" * 60)
print("Pipeline debug complete.")
