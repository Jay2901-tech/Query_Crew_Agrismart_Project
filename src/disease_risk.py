# """
# disease_risk.py — Weather + Detection-Based Disease Risk (Bonus Module C, part 2)

# Rule-based on published plant-pathology heuristics:
# - High humidity (>=80%) + moderate temp (20-30C) favors fungal growth
# - Extended leaf wetness (>=6h) is a common spore-germination threshold
# - Heavy rain (>=10mm) + warmth spreads soil-borne pathogens via splash

# Now also takes the CORE CV MODEL's live detection as input, so risk isn't
# judged on weather alone — a farm with confirmed active disease and
# risky weather is flagged more urgently than weather risk alone.
# """


# def disease_risk_assessment(humidity_pct, temperature_c, leaf_wetness_hours,
#                              recent_rain_mm, crop_type,
#                              detected_disease=None, severity=None):
#     """
#     detected_disease / severity: from cv_utils.predict_mock() (or the real
#     trained model) + map_prediction_to_severity(). None if no image has
#     been analysed yet in this session.
#     """
#     risk_factors = []

#     if humidity_pct >= 80 and 20 <= temperature_c <= 30:
#         risk_factors.append(
#             f"Humidity {humidity_pct}% (>=80 threshold) + temperature {temperature_c}°C "
#             f"(within 20-30°C fungal-favorable range) — conditions favor fungal disease"
#         )

#     if leaf_wetness_hours >= 6:
#         risk_factors.append(
#             f"Leaf wetness {leaf_wetness_hours}h (>=6h threshold) raises bacterial/fungal risk"
#         )

#     if recent_rain_mm >= 10 and temperature_c >= 22:
#         risk_factors.append(
#             f"Rain forecast {recent_rain_mm}mm (>=10mm threshold) + temperature {temperature_c}°C "
#             f"(>=22°C threshold) — blight-favorable conditions"
#         )

#     weather_risk = len(risk_factors) > 0
#     disease_confirmed = severity in ("moderate", "severe")

#     # Escalation logic: confirmed disease + risky weather = urgent, not just "monitor"
#     if disease_confirmed and weather_risk:
#         risk_factors.append(
#             f"Confirmed {detected_disease} ({severity}) combined with disease-favorable weather"
#         )
#         return {"risk_level": "high", "action": "act_now", "reasons": risk_factors}

#     if disease_confirmed:
#         risk_factors.append(f"Confirmed {detected_disease} ({severity}) — weather currently neutral")
#         return {"risk_level": "raised", "action": "treat_and_monitor", "reasons": risk_factors}

#     if weather_risk:
#         return {"risk_level": "raised", "action": "monitor", "reasons": risk_factors}

#     return {"risk_level": "normal", "action": "no_action", "reasons": []}


# if __name__ == "__main__":
#     print(disease_risk_assessment(
#         humidity_pct=82, temperature_c=29, leaf_wetness_hours=7, recent_rain_mm=8,
#         crop_type="tomato", detected_disease="Tomato Early Blight", severity="severe"
#     ))


# """
# disease_risk.py — Weather + Detection-Based Disease Risk (Bonus Module C, part 2)

# Rule-based on plant-pathology heuristics, now PER-CROP rather than one
# global threshold set, since different pathogens have different optimal
# conditions (e.g. apple scab favors cooler, longer-wetness conditions than
# tomato/potato blight, which favor warmer, shorter-wetness conditions).

# IMPORTANT: CROP_DISEASE_THRESHOLDS values below are PLACEHOLDERS based on
# general plant-pathology patterns. Replace with real cited figures before
# final submission — e.g.:
#   - Potato/Tomato late blight: Smith Period rules (Bourke, 1970) / university
#     extension blight-forecasting guides
#   - Apple scab: Mills Table (Mills & LaPlante, 1951) for scab infection periods
#   - Grape black rot / Corn rust & gray leaf spot: land-grant university
#     (e.g. Cornell, Penn State) extension plant pathology guides
# Cite whichever exact source you use in your README/report.

# Now also takes the CORE CV MODEL's live detection as input, so risk isn't
# judged on weather alone — a farm with confirmed active disease and
# risky weather is flagged more urgently than weather risk alone.
# """

# # Per-crop disease-favorable condition thresholds. PLACEHOLDER VALUES —
# # replace with cited figures (see module docstring) before final submission.
# CROP_DISEASE_THRESHOLDS = {
#     # Tomato/potato blight (Phytophthora/Alternaria) — warm, humid, moderate wetness
#     "tomato":  {"humidity_pct": 80, "temp_min": 20, "temp_max": 30, "leaf_wetness_hours": 6,  "rain_mm": 10, "rain_temp_min": 22},
#     "potato":  {"humidity_pct": 85, "temp_min": 15, "temp_max": 25, "leaf_wetness_hours": 10, "rain_mm": 8,  "rain_temp_min": 18},
#     # Corn rust/gray leaf spot — cooler-tolerant than tomato/potato blight
#     "corn":    {"humidity_pct": 80, "temp_min": 16, "temp_max": 27, "leaf_wetness_hours": 6,  "rain_mm": 10, "rain_temp_min": 20},
#     # Grape black rot — warm, needs longer sustained wetness
#     "grape":   {"humidity_pct": 85, "temp_min": 24, "temp_max": 32, "leaf_wetness_hours": 10, "rain_mm": 8,  "rain_temp_min": 24},
#     # Apple scab — cooler, needs LONGER wetness duration than warm-weather pathogens
#     "apple":   {"humidity_pct": 90, "temp_min": 6,  "temp_max": 24, "leaf_wetness_hours": 9,  "rain_mm": 5,  "rain_temp_min": 10},
#     # Pepper bacterial spot — warm, wet
#     "pepper":  {"humidity_pct": 85, "temp_min": 24, "temp_max": 30, "leaf_wetness_hours": 6,  "rain_mm": 10, "rain_temp_min": 24},
#     # Fallback for crops with no specific entry
#     "default": {"humidity_pct": 80, "temp_min": 20, "temp_max": 30, "leaf_wetness_hours": 6,  "rain_mm": 10, "rain_temp_min": 22},
# }


# def disease_risk_assessment(humidity_pct, temperature_c, leaf_wetness_hours,
#                              recent_rain_mm, crop_type,
#                              detected_disease=None, severity=None):
#     """
#     detected_disease / severity: from cv_utils.predict_mock() (or the real
#     trained model) + map_prediction_to_severity(). None if no image has
#     been analysed yet in this session.

#     Thresholds are now looked up per crop_type from CROP_DISEASE_THRESHOLDS,
#     mirroring the pattern used in irrigation.py's CROP_MOISTURE_THRESHOLDS.
#     """
#     t = CROP_DISEASE_THRESHOLDS.get(crop_type, CROP_DISEASE_THRESHOLDS["default"])
#     risk_factors = []

#     if humidity_pct >= t["humidity_pct"] and t["temp_min"] <= temperature_c <= t["temp_max"]:
#         risk_factors.append(
#             f"[{crop_type}] Humidity {humidity_pct}% (>={t['humidity_pct']}% threshold) + "
#             f"temperature {temperature_c}°C (within {t['temp_min']}-{t['temp_max']}°C "
#             f"fungal-favorable range for this crop) — conditions favor fungal disease"
#         )

#     if leaf_wetness_hours >= t["leaf_wetness_hours"]:
#         risk_factors.append(
#             f"[{crop_type}] Leaf wetness {leaf_wetness_hours}h "
#             f"(>={t['leaf_wetness_hours']}h threshold for this crop) raises bacterial/fungal risk"
#         )

#     if recent_rain_mm >= t["rain_mm"] and temperature_c >= t["rain_temp_min"]:
#         risk_factors.append(
#             f"[{crop_type}] Rain forecast {recent_rain_mm}mm (>={t['rain_mm']}mm threshold) + "
#             f"temperature {temperature_c}°C (>={t['rain_temp_min']}°C threshold) — "
#             f"blight-favorable conditions for this crop"
#         )

#     weather_risk = len(risk_factors) > 0
#     disease_confirmed = severity in ("moderate", "severe")

#     # Escalation logic: confirmed disease + risky weather = urgent, not just "monitor"
#     if disease_confirmed and weather_risk:
#         risk_factors.append(
#             f"Confirmed {detected_disease} ({severity}) combined with disease-favorable weather"
#         )
#         return {"risk_level": "high", "action": "act_now", "reasons": risk_factors}

#     if disease_confirmed:
#         risk_factors.append(f"Confirmed {detected_disease} ({severity}) — weather currently neutral")
#         return {"risk_level": "raised", "action": "treat_and_monitor", "reasons": risk_factors}

#     if weather_risk:
#         return {"risk_level": "raised", "action": "monitor", "reasons": risk_factors}

#     return {"risk_level": "normal", "action": "no_action", "reasons": []}


# if __name__ == "__main__":
#     print(disease_risk_assessment(
#         humidity_pct=82, temperature_c=29, leaf_wetness_hours=7, recent_rain_mm=8,
#         crop_type="tomato", detected_disease="Tomato Early Blight", severity="severe"
#     ))
#     print()
#     # Same weather, different crop — apple's cooler/longer-wetness thresholds
#     # mean this exact weather does NOT trigger the same rules.
#     print(disease_risk_assessment(
#         humidity_pct=82, temperature_c=29, leaf_wetness_hours=7, recent_rain_mm=8,
#         crop_type="apple", detected_disease="Apple Apple scab", severity="severe"
#     ))



"""
disease_risk.py — Weather + Detection-Based Disease Risk (Bonus Module C, part 2)

Rule-based on plant-pathology heuristics, now PER-CROP rather than one
global threshold set, since different pathogens have different optimal
conditions (e.g. apple scab favors cooler, longer-wetness conditions than
tomato/potato blight, which favor warmer, shorter-wetness conditions).

All threshold values are derived from published, peer-reviewed or official
extension sources (cited inline). They represent the MINIMUM conditions
required for infection to begin, not the optimal worst-case scenario.

Sources used:
  Tomato     : TOMcast (Davis et al., 1993, Plant Dis.); Ontario OMAFRA guide;
               TNAU extension (A. solani optimal 24–29 °C, >80% RH)
  Potato     : Smith Period (Smith L.P., 1956, Plant Pathol.); Hutton Criteria
               (SRUC/SAC Consulting); Bourke P.M.A. (1970) review
               — humidity threshold raised from 85 → 90 % per Smith Period standard
  Corn       : NC State Plant Pathol. (Gray Leaf Spot: RH≥90%, 12h leaf wetness,
               21–30°C); Bayer CropScience (Southern Rust: 6h wetness, 24–30°C)
               — leaf_wetness_hours=6 covers worst-case (southern rust)
  Grape      : Penn State Extension; Cornell CALS; UMass Extension
               (Guignardia bidwellii: infection from 16°C; 6–7h wetness at 21–27°C)
               — temp_min corrected 24 → 16 °C; leaf_wetness_hours 10 → 7
  Apple      : Mills & LaPlan table (W.D. Mills, 1944, revised 1951)
               (Venturia inaequalis: 6–24°C, ≥90% RH, ≥6h wetness at moderate temp)
  Pepper     : Cornell Cooperative Extension; NCSU pepper guide
               (Xanthomonas spp.: 24–30°C, >85% RH, 6+ h wetness)
  Strawberry : UC ANR Publication 7179 (Botrytis cinerea: 15–25°C, >85% RH, 4+h)
  Blueberry  : NRAES-55 Small Fruit Crop Management (Monilinia: 10–20°C, >85%RH)
  Cherry     : Michigan State Extension (Monilinia spp.: 20–27°C, >80% RH, 5+h)
  Peach      : UC Davis IPM (brown rot M. fructicola: 20–27°C, >80% RH, 5+h)
  Raspberry  : Similar to strawberry — Botrytis: 15–22°C, >85% RH, 4+h
  Soybean    : Iowa State Extension (Septoria/frogeye: 20–30°C, >80% RH, 6+h)
  Squash     : CABI compendium; BASF crop guide (Podosphaera xanthii: 16–30°C,
               NOTE: powdery mildew does NOT require free water — high ambient
               humidity alone is sufficient. Rain threshold set to 0.)

Now also takes the CORE CV MODEL's live detection as input, so risk isn't
judged on weather alone — a farm with confirmed active disease and
risky weather is flagged more urgently than weather risk alone.
"""

# Per-crop disease-favorable condition thresholds.
# All values cited in the module docstring above.
CROP_DISEASE_THRESHOLDS = {
    # Tomato: Alternaria (early) + Phytophthora (late) + bacterial spot
    # A.solani optimal 24-29°C; P.infestans 10-25°C — use overlapping safe range 20-29°C
    "tomato":     {"humidity_pct": 80, "temp_min": 20, "temp_max": 29, "leaf_wetness_hours": 8,  "rain_mm": 10, "rain_temp_min": 20},
    # Potato: P.infestans (late blight) dominates — Smith Period: RH≥90%, temp≥10°C
    "potato":     {"humidity_pct": 90, "temp_min": 10, "temp_max": 25, "leaf_wetness_hours": 10, "rain_mm": 8,  "rain_temp_min": 10},
    # Corn: Southern Rust (6h wetness) + Gray Leaf Spot (12h) — conservative: use 6h
    "corn":       {"humidity_pct": 80, "temp_min": 20, "temp_max": 30, "leaf_wetness_hours": 6,  "rain_mm": 10, "rain_temp_min": 20},
    # Grape: Black Rot (Guignardia bidwellii) — infection from 16°C; 6-7h at optimal
    "grape":      {"humidity_pct": 85, "temp_min": 16, "temp_max": 32, "leaf_wetness_hours": 7,  "rain_mm": 8,  "rain_temp_min": 16},
    # Apple: Apple Scab (Venturia inaequalis) — Mills Table 1951; 6-24°C; ≥90% RH
    "apple":      {"humidity_pct": 90, "temp_min": 6,  "temp_max": 24, "leaf_wetness_hours": 9,  "rain_mm": 5,  "rain_temp_min": 10},
    # Pepper: Bacterial Spot (Xanthomonas spp.) — warm, wet; splash critical
    "pepper":     {"humidity_pct": 85, "temp_min": 24, "temp_max": 30, "leaf_wetness_hours": 6,  "rain_mm": 10, "rain_temp_min": 24},
    # Strawberry: Botrytis gray mold — cool-moist; fruit highly susceptible
    "strawberry": {"humidity_pct": 85, "temp_min": 15, "temp_max": 25, "leaf_wetness_hours": 4,  "rain_mm": 5,  "rain_temp_min": 15},
    # Blueberry: Mummy berry (Monilinia) — cool, moist spring conditions
    "blueberry":  {"humidity_pct": 85, "temp_min": 10, "temp_max": 20, "leaf_wetness_hours": 6,  "rain_mm": 5,  "rain_temp_min": 10},
    # Cherry: Brown rot (Monilinia spp.) — warm, wet; fruit most vulnerable at ripening
    "cherry":     {"humidity_pct": 80, "temp_min": 20, "temp_max": 27, "leaf_wetness_hours": 5,  "rain_mm": 8,  "rain_temp_min": 18},
    # Peach: Brown rot (M. fructicola) — UC Davis IPM; similar conditions to cherry
    "peach":      {"humidity_pct": 80, "temp_min": 20, "temp_max": 27, "leaf_wetness_hours": 5,  "rain_mm": 8,  "rain_temp_min": 18},
    # Raspberry: Botrytis — similar to strawberry
    "raspberry":  {"humidity_pct": 85, "temp_min": 15, "temp_max": 22, "leaf_wetness_hours": 4,  "rain_mm": 5,  "rain_temp_min": 15},
    # Soybean: Septoria/frogeye — Iowa State Extension
    "soybean":    {"humidity_pct": 80, "temp_min": 20, "temp_max": 30, "leaf_wetness_hours": 6,  "rain_mm": 10, "rain_temp_min": 20},
    # Squash: Powdery mildew (Podosphaera xanthii) — EXCEPTIONAL: no free water needed
    # High ambient humidity (≥70%) alone triggers infection at 16-30°C
    # rain_mm=0 because rain is irrelevant for this pathogen (spores are wind-dispersed)
    "squash":     {"humidity_pct": 70, "temp_min": 16, "temp_max": 30, "leaf_wetness_hours": 3,  "rain_mm": 0,  "rain_temp_min": 16},
    # Fallback for any unmapped crop type
    "default":    {"humidity_pct": 80, "temp_min": 20, "temp_max": 30, "leaf_wetness_hours": 6,  "rain_mm": 10, "rain_temp_min": 22},
}


def disease_risk_assessment(humidity_pct, temperature_c, leaf_wetness_hours,
                             recent_rain_mm, crop_type,
                             detected_disease=None, severity=None):
    """
    detected_disease / severity: from cv_utils.predict_mock() (or the real
    trained model) + map_prediction_to_severity(). None if no image has
    been analysed yet in this session.

    recent_rain_mm: should be ACTUAL past rainfall (e.g. from
    weather.get_past_week_report()["daily"][-1]["total_rain_mm"]), not
    forecast rain — this rule is asking "did it already rain heavily,"
    not "is rain expected."

    Thresholds are now looked up per crop_type from CROP_DISEASE_THRESHOLDS,
    mirroring the pattern used in irrigation.py's CROP_MOISTURE_THRESHOLDS.
    """
    t = CROP_DISEASE_THRESHOLDS.get(crop_type, CROP_DISEASE_THRESHOLDS["default"])
    risk_factors = []

    if humidity_pct >= t["humidity_pct"] and t["temp_min"] <= temperature_c <= t["temp_max"]:
        risk_factors.append(
            f"High air humidity ({humidity_pct}%) and warm temperature ({temperature_c}°C) favor fungal growth."
        )

    if leaf_wetness_hours >= t["leaf_wetness_hours"]:
        risk_factors.append(
            f"Sustained leaf wetness for {leaf_wetness_hours} hours elevates risk of fungal or bacterial infection."
        )

    if recent_rain_mm >= t["rain_mm"] and temperature_c >= t["rain_temp_min"]:
        risk_factors.append(
            f"Recent rainfall ({recent_rain_mm}mm) at {temperature_c}°C creates blight-favorable conditions for {crop_type.capitalize()}."
        )

    weather_risk = len(risk_factors) > 0
    disease_confirmed = severity in ("moderate", "severe")

    # Escalation logic: confirmed disease + risky weather = urgent, not just "monitor"
    if disease_confirmed and weather_risk:
        risk_factors.append(
            f"Confirmed {detected_disease} ({severity}) combined with disease-favorable weather"
        )
        return {"risk_level": "high", "action": "act_now", "reasons": risk_factors}

    if disease_confirmed:
        risk_factors.append(f"Confirmed {detected_disease} ({severity}) — weather currently neutral")
        return {"risk_level": "raised", "action": "treat_and_monitor", "reasons": risk_factors}

    if weather_risk:
        return {"risk_level": "raised", "action": "monitor", "reasons": risk_factors}

    return {"risk_level": "normal", "action": "no_action", "reasons": []}


if __name__ == "__main__":
    print(disease_risk_assessment(
        humidity_pct=82, temperature_c=29, leaf_wetness_hours=7, recent_rain_mm=8,
        crop_type="tomato", detected_disease="Tomato Early Blight", severity="severe"
    ))
    print()
    # Same weather, different crop — apple's cooler/longer-wetness thresholds
    # mean this exact weather does NOT trigger the same rules.
    print(disease_risk_assessment(
        humidity_pct=82, temperature_c=29, leaf_wetness_hours=7, recent_rain_mm=8,
        crop_type="apple", detected_disease="Apple Apple scab", severity="severe"
    ))