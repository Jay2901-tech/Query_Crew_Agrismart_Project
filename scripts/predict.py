import os
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageStat
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def load_config(config_path="config/guidance.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def is_valid_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        stat = ImageStat.Stat(img)
        
        # Check if the image is extremely uniform (blank/solid color)
        # stddev will be very low
        std_sum = sum(stat.stddev)
        if std_sum < 5.0:
            return False, "Image appears to be blank or a solid color."
            
        return True, ""
    except Exception as e:
        return False, f"Invalid or corrupted image: {e}"

def predict_image(image_path, model, classes, device):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    transform = get_inference_transform()
    tensor = transform(image=image_np)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_prob, top_catid = torch.topk(probabilities, 3)
        
    results = []
    for i in range(3):
        predicted_class = classes[top_catid[0][i].item()]
        confidence = top_prob[0][i].item()
        results.append((predicted_class, confidence))
        
    return results

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found at {args.weights}")
        return
        
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
        
    is_valid, msg = is_valid_image(args.image)
    if not is_valid:
        print(f"\n--- SAFETY GATE REJECTION ---")
        print(f"Reason: {msg}")
        return
        
    try:
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        classes = checkpoint['classes']
        num_classes = len(classes)
        
        model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        results = predict_image(args.image, model, classes, device)
        
        best_official = results[0][0]
        best_conf = results[0][1]
        
        if best_conf < 0.6:
            print(f"\n--- SAFETY GATE REJECTION ---")
            print(f"Reason: Out-of-Domain or low confidence ({best_conf:.4f} < 0.6). Model is uncertain.")
            print(f"Top guess was {best_official} but confidence is too low to provide treatment guidance.")
            return
            
        print("\n--- Prediction Results ---")
        
        for i, (official_name, conf) in enumerate(results):
            if i == 0:
                print(f"Top 1 (class_label): {official_name} ({conf:.4f})")
            else:
                print(f"Top {i+1}: {official_name} ({conf:.4f})")
                
        print("\n--- Actionable Guidance ---")
        guidance_path = "config/guidance.json"
        if os.path.exists(guidance_path) and best_official:
            guidance = load_config(guidance_path)
            if best_official in guidance:
                info = guidance[best_official]
                print(f"Detected: {info.get('label', best_official)}")
                print(f"Precaution: {info.get('guidance', 'No precaution specified.')}")
            else:
                print("No guidance available for this specific disease/crop.")
                
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict crop disease from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--weights", type=str, default="models/best_model.pth", help="Path to the model weights")
    
    args = parser.parse_args()
    main(args)
