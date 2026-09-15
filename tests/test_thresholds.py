import sys
import os
import json

# Ensure we can import from model/predict.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.predict import load_config

def evaluate_threshold(best_conf, threshold):
    return best_conf >= threshold

def test_deterministic_thresholds():
    print("--- Running Deterministic Threshold Tests ---")
    
    # Test cases: (confidence, threshold, expected_status)
    test_cases = [
        (0.5999, 0.60, "UNCERTAIN"),
        (0.59999, 0.60, "UNCERTAIN"),
        (0.6000, 0.60, "ACCEPTED"),
        (0.6001, 0.60, "ACCEPTED"),
        (0.70, 0.70, "ACCEPTED")
    ]
    
    all_passed = True
    for conf, thresh, expected in test_cases:
        passed_gate = evaluate_threshold(conf, thresh)
        status = "ACCEPTED" if passed_gate else "UNCERTAIN"
        
        if status == expected:
            print(f"[PASS] Conf: {conf:.6f}, Threshold: {thresh:.6f} -> Status: {status}")
        else:
            print(f"[FAIL] Conf: {conf:.6f}, Threshold: {thresh:.6f} -> Expected {expected}, got {status}")
            all_passed = False
            
    assert all_passed, "One or more threshold tests failed."
    print("All threshold tests passed.\n")

def test_guidance_mapping():
    print("--- Running Guidance Mapping Tests ---")
    guidance = load_config("config/guidance.json")
    
    # Test Exact lookup for Tomato Septoria Leaf Spot
    predicted_class = "Tomato___Septoria_leaf_spot"
    
    assert predicted_class in guidance, f"Key {predicted_class} not found in guidance.json"
    
    info = guidance[predicted_class]
    label = info.get("label", "")
    
    assert "Septoria Leaf Spot" in label, f"Expected Septoria guidance, got label: {label}"
    assert "Early Blight" not in label, f"Incorrectly retrieved Early Blight guidance for Septoria"
    
    print(f"[PASS] Canonical lookup for {predicted_class} successfully retrieved: {label}")
    
    # Test missing guidance
    missing_class = "Some___Unknown_Disease"
    assert missing_class not in guidance, f"Key {missing_class} should not exist."
    print(f"[PASS] Missing guidance gracefully handled (not mapped to a fallback).")
    
    print("All guidance tests passed.\n")

if __name__ == "__main__":
    test_deterministic_thresholds()
    test_guidance_mapping()
