import json
import csv
from pathlib import Path

inventory_path = Path("reports/dataset_inventory.json")
with open(inventory_path, "r") as f:
    inventory = json.load(f)

pv_classes = inventory.get("PlantVillage", {}).get("classes", {}).keys()
pd_classes = inventory.get("PlantDoc", {}).get("classes", {}).keys()

canonical_mapping = {}

def normalize_name(name):
    # simple standardisation
    name = name.replace("_(including_sour)", "")
    name = name.replace("_(maize)", "")
    name = name.replace(",_bell", "___bell")
    return name

# Map PlantVillage classes to canonical
for pvc in pv_classes:
    canonical = normalize_name(pvc)
    canonical_mapping[canonical] = {
        "plantvillage": pvc,
        "plantdoc": None
    }

# Map PlantDoc classes to existing canonical
mapping_rules = {
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Bell_pepper leaf": "Pepper___bell___healthy",
    "Bell_pepper leaf spot": "Pepper___bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry___healthy",
    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn___Common_rust_",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite"
}

for pdc in pd_classes:
    if pdc in mapping_rules:
        canon = mapping_rules[pdc]
        if canon in canonical_mapping:
            canonical_mapping[canon]["plantdoc"] = pdc
        else:
            print(f"Warning: Canonical target {canon} not found for {pdc}")
    else:
        print(f"Warning: No mapping rule for PlantDoc class: {pdc}")
        canonical_mapping[f"UNKNOWN___{pdc}"] = {
            "plantvillage": None,
            "plantdoc": pdc
        }

# Save class_mapping.json
with open("config/class_mapping.json", "w") as f:
    json.dump(canonical_mapping, f, indent=4)

# Save class_mapping_report.csv
with open("reports/class_mapping_report.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Canonical_Class", "PlantVillage_Class", "PlantDoc_Class", "Status"])
    for canon, sources in canonical_mapping.items():
        status = "OK" if "UNKNOWN" not in canon else "UNSUPPORTED_CLASS"
        writer.writerow([canon, sources["plantvillage"], sources["plantdoc"], status])

print("Canonical taxonomy built and saved.")
