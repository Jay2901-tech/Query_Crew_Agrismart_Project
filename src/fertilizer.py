"""
fertilizer.py — Recommended Fertilizer Lookup (feeds Bonus Module D)

Mirrors CROP_MOISTURE_THRESHOLDS in irrigation.py: recommended values are
looked up automatically by crop + growth stage, since this is agronomic
fact the farmer shouldn't have to supply.

IMPORTANT: The numbers below are PLACEHOLDERS for demo purposes.
Before your final submission, replace them with real published figures
(e.g. ICAR crop-wise fertilizer recommendations, or FAO fertilizer-use
guidelines for your target region/crop) and cite the exact source in
your README and model report — judges check this.
"""

# Recommended nitrogen-equivalent fertilizer (kg per week, per acre)
# by crop and growth stage. PLACEHOLDER VALUES — replace + cite source.
RECOMMENDED_FERTILIZER_KG = {
    "tomato":  {"seedling": 2, "vegetative": 5, "flowering": 8,  "maturity": 4},
    "potato":  {"seedling": 3, "vegetative": 6, "flowering": 7,  "maturity": 3},
    "corn":    {"seedling": 4, "vegetative": 8, "flowering": 10, "maturity": 5},
    "grape":   {"seedling": 1, "vegetative": 3, "flowering": 4,  "maturity": 2},
    "apple":   {"seedling": 2, "vegetative": 4, "flowering": 5,  "maturity": 3},
    "pepper":  {"seedling": 2, "vegetative": 5, "flowering": 7,  "maturity": 4},
    "default": {"seedling": 3, "vegetative": 5, "flowering": 7,  "maturity": 4},
}


def get_recommended_fertilizer(crop_type, growth_stage):
    crop_table = RECOMMENDED_FERTILIZER_KG.get(crop_type, RECOMMENDED_FERTILIZER_KG["default"])
    return crop_table.get(growth_stage, crop_table["vegetative"])


if __name__ == "__main__":
    print(get_recommended_fertilizer("tomato", "flowering"))  # -> 8
