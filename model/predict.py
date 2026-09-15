import os
import json
import argparse
import numpy as np
import torch
import math
from PIL import Image, ImageStat
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import urllib.request

def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def load_config(config_path="config/guidance.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_class_mapping(mapping_path="config/class_mapping.json"):
    with open(mapping_path, 'r') as f:
        return json.load(f)

def is_valid_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        stat = ImageStat.Stat(img)
        # Check if the image is extremely uniform (blank/solid color/pure noise)
        std_sum = sum(stat.stddev)
        if std_sum < 5.0:
            return False, "Image appears to be blank or a solid color."
        return True, ""
    except Exception as e:
        return False, f"Invalid or corrupted image: {e}"

# Cache for ImageNet classes
IMAGENET_CLASSES = None

def load_imagenet_classes():
    global IMAGENET_CLASSES
    if IMAGENET_CLASSES is None:
        try:
            url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
            response = urllib.request.urlopen(url)
            IMAGENET_CLASSES = [line.decode('utf-8').strip().lower() for line in response.readlines()]
        except Exception:
            # Fallback if download fails (some common ones)
            IMAGENET_CLASSES = [""] * 1000
    return IMAGENET_CLASSES

def check_plant_relevance(image_path, device):
    # Lightweight ImageNet model to check for non-plant objects
    imagenet_model = timm.create_model('mobilenetv3_small_100', pretrained=True).to(device)
    imagenet_model.eval()
    
    classes = load_imagenet_classes()
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    transform = get_inference_transform()
    tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = imagenet_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.topk(probs, 5)
        
    top_catid = top_catid[0].cpu().numpy()
    
    plant_keywords = [
        "plant", "leaf", "flower", "fruit", "tree", "pot", "greenhouse", "vegetable", 
        "crop", "grass", "mushroom", "daisy", "apple", "orange", "lemon", "fig", 
        "strawberry", "pineapple", "banana", "pomegranate", "corn", "acorn", "hip", 
        "buckeye", "coral fungus", "agaric", "bolete", "stinkhorn", "earthstar",
        "jackfruit", "custard apple", "macaque", "hay", "ear", "vase", "bell pepper",
        "cucumber", "broccoli", "cauliflower", "zucchini", "squash", "pumpkin"
    ]
    
    is_plant = False
    top_preds = []
    
    for i in range(5):
        idx = top_catid[i]
        class_name = classes[idx] if idx < len(classes) else ""
        top_preds.append(class_name)
        
        for kw in plant_keywords:
            if kw in class_name:
                is_plant = True
                break
        if is_plant:
            break
            
    # Also if the top confidence is very low, it might be a weird abstract image
    # But we mainly want to reject "car", "laptop", "building" with high confidence.
    return is_plant, top_preds

def predict_image(image_path, model, classes, device):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    transform = get_inference_transform()
    tensor = transform(image=image_np)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_prob, top_catid = torch.topk(probabilities, 3)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9)).item()
        
    results = []
    for i in range(3):
        predicted_class = classes[top_catid[0][i].item()]
        confidence = top_prob[0][i].item()
        results.append((predicted_class, confidence))
        
    return results, entropy

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found at {args.weights}")
        return
        
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
        
    print(f"\n--- Processing {os.path.basename(args.image)} ---")
    
    # Gate 1: Image Quality
    image_quality_pass, iq_msg = is_valid_image(args.image)
    if not image_quality_pass:
        print("\nPrediction Status: REJECTED")
        print(f"Reason: {iq_msg}")
        return

    # Gate 2: Plant Relevance (OOD non-plant rejection)
    plant_relevance_pass, top_imagenet_preds = check_plant_relevance(args.image, device)
    
    if not plant_relevance_pass:
        print("\nPrediction Status: REJECTED")
        print("Reason: No recognizable plant detected.")
        print(f"(ImageNet sees: {', '.join(top_imagenet_preds[:3])})")
        return
        
    try:
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        # Use our best_model.pth classes if they exist, otherwise try to load from mapping
        if 'classes' in checkpoint:
            classes = checkpoint['classes']
        else:
            print("Error: Model weights do not contain 'classes'.")
            return
            
        num_classes = len(classes)
        
        model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        mapping = load_class_mapping()
        # Create mapping from training classes (which might be PV or Canonical) to Canonical
        # Our train.py uses Canonical classes directly! 
        # But wait, earlier model/predict.py had pv_to_official. Let's handle both cases.
        # Let's assume `classes` are canonical if they don't exactly match PV.
        # Actually our train.py used canonical classes natively.
        
        results, entropy = predict_image(args.image, model, classes, device)
        
        best_pred = results[0][0]
        best_conf = results[0][1]
        top2_conf = results[1][1]
        margin = best_conf - top2_conf
        
        # Translate to official name if necessary (it might already be official)
        official_name = best_pred
        
        # Gate 3: Uncertainty
        uncertainty_gate_pass = (best_conf >= args.conf_threshold)
        
        print("\n--- Model Predictions ---")
        print(f"Top 1: {results[0][0]} ({results[0][1]:.4f})")
        print(f"Top 2: {results[1][0]} ({results[1][1]:.4f})")
        print(f"Top 3: {results[2][0]} ({results[2][1]:.4f})")
        
        print("\n--- Prediction Statistics ---")
        print(f"Top-1 Confidence:  {best_conf:.6f}")
        print(f"Top-2 Confidence:  {top2_conf:.6f}")
        print(f"Top-1/Top-2 Margin: {margin:.6f}")
        print(f"Entropy:           {entropy:.4f}")
        
        if not uncertainty_gate_pass:
            print("\nPrediction Status: UNCERTAIN")
            print(f"Reason: Threshold Comparison: {best_conf:.6f} >= {args.conf_threshold:.6f} -> FALSE")
            return
            
        print("\nPrediction Status: ACCEPTED")
        
        print("\n--- Actionable Guidance ---")
        guidance_path = "config/guidance.json"
        if os.path.exists(guidance_path):
            guidance = load_config(guidance_path)
            if official_name in guidance:
                info = guidance[official_name]
                print(f"Detected: {info.get('label', official_name)}")
                print(f"Precaution: {info.get('guidance', 'No guidance available.')}")
            else:
                print("No guidance available for this specific disease/crop.")
                
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict crop disease from an image with safety gates.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--weights", type=str, default="model/weights/best_model.pth", help="Path to the model weights")
    parser.add_argument("--conf-threshold", type=float, default=0.60, help="Confidence threshold for prediction gate")
    
    args = parser.parse_args()
    main(args)
