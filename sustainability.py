# """
# sustainability.py — Composite Sustainability Score (Bonus Module D)

# PUBLISHED FORMULA (state this exact formula in your README/report —
# required for reproducibility credit):

#     Sustainability Score = 0.40 x WaterEfficiencyScore
#                           + 0.30 x ResourceUseScore
#                           + 0.30 x CropHealthScore

# Weight rationale: water efficiency weighted highest because irrigation
# is the largest controllable resource lever in this challenge's data;
# resource use and crop health weighted equally since both directly
# affect yield/sustainability outcomes.
# """


# def water_efficiency_score(irrigation_log, liters_per_event=500, baseline_liters_per_day=500):
#     """
#     Compares actual irrigation events (from the agent's logged history)
#     against a naive "water every day regardless of conditions" baseline.
#     Fewer irrigations (because the system smartly skipped unnecessary
#     ones) scores higher.
#     """
#     days = len(irrigation_log)
#     if days == 0:
#         return 100.0

#     actual_events = sum(1 for e in irrigation_log if e["action"].startswith("irrigate"))
#     actual_liters = actual_events * liters_per_event
#     baseline_liters = days * baseline_liters_per_day

#     usage_ratio = actual_liters / baseline_liters if baseline_liters > 0 else 1.0
#     score = 100 * (1 - usage_ratio)
#     return round(max(0, min(100, score)), 1)


# def resource_use_score(actual_fertilizer_kg, recommended_fertilizer_kg):
#     """
#     actual_fertilizer_kg: farmer-supplied input (form/CLI) — no sensor
#     or API can know real-world fertilizer application.
#     recommended_fertilizer_kg: looked up from fertilizer.get_recommended_fertilizer().
#     """
#     if actual_fertilizer_kg is None or recommended_fertilizer_kg in (None, 0):
#         return 100.0  # neutral default if not tracked
#     excess_ratio = max(0, (actual_fertilizer_kg - recommended_fertilizer_kg) / recommended_fertilizer_kg)
#     score = 100 * (1 - excess_ratio)
#     return round(max(0, min(100, score)), 1)


# SEVERITY_HEALTH_SCORE = {
#     "none": 100,
#     "mild": 70,
#     "moderate": 45,
#     "severe": 15,
# }


# def crop_health_score(severity):
#     """severity comes from cv_utils.map_prediction_to_severity() — the core CV model's output."""
#     return SEVERITY_HEALTH_SCORE.get(severity, 50)  # neutral default if unknown


# def sustainability_score(irrigation_log, severity,
#                           actual_fertilizer_kg=None, recommended_fertilizer_kg=None):
#     water_score = water_efficiency_score(irrigation_log)
#     resource_score = resource_use_score(actual_fertilizer_kg, recommended_fertilizer_kg)
#     health_score = crop_health_score(severity)

#     composite = round(0.40 * water_score + 0.30 * resource_score + 0.30 * health_score, 1)

#     scores = {"water": water_score, "resource": resource_score, "health": health_score}
#     weakest = min(scores, key=scores.get)

#     suggestions = {
#         "water": "Water usage is above optimal — consider tightening irrigation triggers to reduce waste.",
#         "resource": "Fertilizer/resource use exceeds recommended levels — review application rates.",
#         "health": "Crop health is declining — inspect for disease spread and apply precautionary measures now.",
#     }

#     return {
#         "sustainability_score": composite,
#         "sub_scores": scores,
#         "weakest_area": weakest,
#         "suggestion": suggestions[weakest],
#         "method": "0.40*water_efficiency + 0.30*resource_use + 0.30*crop_health (see README)",
#     }


# if __name__ == "__main__":
#     fake_log = [
#         {"action": "delay"}, {"action": "irrigate_today"}, {"action": "no_action"},
#         {"action": "no_action"}, {"action": "irrigate_now"}, {"action": "delay"}, {"action": "no_action"},
#     ]
#     print(sustainability_score(fake_log, severity="severe",
#                                 actual_fertilizer_kg=12, recommended_fertilizer_kg=10))



"""
sustainability.py — Composite Sustainability Score (Bonus Module D)

PUBLISHED FORMULA (state this exact formula in your README/report —
required for reproducibility credit):

    Sustainability Score = 0.40 x WaterEfficiencyScore
                          + 0.30 x ResourceUseScore
                          + 0.30 x CropHealthScore

Weight rationale: water efficiency weighted highest because irrigation
is the largest controllable resource lever in this challenge's data;
resource use and crop health weighted equally since both directly
affect yield/sustainability outcomes.
"""


def water_efficiency_score(irrigation_log, rain_mm_7d=None,
                            liters_per_event=500, baseline_liters_per_day=500):
    """
    Compares actual irrigation events (from the agent's logged history)
    against a naive "water every day regardless of conditions" baseline.
    Fewer irrigations (because the system smartly skipped unnecessary
    ones) scores higher.

    rain_mm_7d: ACTUAL past rainfall total over the same period, from
    weather.get_past_week_report()["total_rain_mm_7d"] — NOT forecast rain.
    Adds a rain-awareness adjustment on top of the base ratio:
      - Irrigating heavily during a genuinely rainy week is wasteful ->
        extra penalty (rain fell AND water was still used).
      - Irrigating during a genuinely dry week is justified, not wasteful ->
        small credit, since the naive baseline alone can't tell efficient
        irrigation apart from careless irrigation when both happen on a
        dry week.
    Rule thresholds below are project-defined heuristics (not from an
    external source) — document as such in your report.
    """
    days = len(irrigation_log)
    if days == 0:
        return 100.0

    actual_events = sum(1 for e in irrigation_log if e["action"].startswith("irrigate"))
    actual_liters = actual_events * liters_per_event
    baseline_liters = days * baseline_liters_per_day

    usage_ratio = actual_liters / baseline_liters if baseline_liters > 0 else 1.0
    score = 100 * (1 - usage_ratio)

    # Rain-awareness adjustment — only applied if real 7-day rain data is supplied.
    if rain_mm_7d is not None:
        rain_per_day = rain_mm_7d / days

        if rain_per_day >= 5 and actual_events > 0:
            # Rained a lot (>=5mm/day avg) and ANY irrigation still happened -> wasteful.
            # Uses actual_events (not usage_ratio) so even light irrigation during a
            # genuinely rainy week is flagged, not just heavy overwatering.
            penalty = min(30, rain_per_day)  # capped so one extreme storm day doesn't zero the score alone
            score -= penalty
        elif rain_per_day < 1 and usage_ratio > 0:
            # Genuinely dry week (<1mm/day avg) but irrigation still happened -> that's
            # justified water use, not waste; small credit vs. the plain baseline comparison
            score += 5

    return round(max(0, min(100, score)), 1)


def resource_use_score(actual_fertilizer_kg, recommended_fertilizer_kg):
    """
    actual_fertilizer_kg: farmer-supplied input (form/CLI) — no sensor
    or API can know real-world fertilizer application.
    recommended_fertilizer_kg: looked up from fertilizer.get_recommended_fertilizer().
    """
    if actual_fertilizer_kg is None or recommended_fertilizer_kg in (None, 0):
        return 100.0  # neutral default if not tracked
    excess_ratio = max(0, (actual_fertilizer_kg - recommended_fertilizer_kg) / recommended_fertilizer_kg)
    score = 100 * (1 - excess_ratio)
    return round(max(0, min(100, score)), 1)


SEVERITY_HEALTH_SCORE = {
    "none": 100,
    "mild": 70,
    "moderate": 45,
    "severe": 15,
}


def crop_health_score(severity):
    """severity comes from cv_utils.map_prediction_to_severity() — the core CV model's output."""
    return SEVERITY_HEALTH_SCORE.get(severity, 50)  # neutral default if unknown


def sustainability_score(irrigation_log, severity, rain_mm_7d=None,
                          actual_fertilizer_kg=None, recommended_fertilizer_kg=None):
    water_score = water_efficiency_score(irrigation_log, rain_mm_7d=rain_mm_7d)
    resource_score = resource_use_score(actual_fertilizer_kg, recommended_fertilizer_kg)
    health_score = crop_health_score(severity)

    composite = round(0.40 * water_score + 0.30 * resource_score + 0.30 * health_score, 1)

    scores = {"water": water_score, "resource": resource_score, "health": health_score}
    weakest = min(scores, key=scores.get)

    suggestions = {
        "water": "Water usage is above optimal — consider tightening irrigation triggers to reduce waste.",
        "resource": "Fertilizer/resource use exceeds recommended levels — review application rates.",
        "health": "Crop health is declining — inspect for disease spread and apply precautionary measures now.",
    }

    return {
        "sustainability_score": composite,
        "sub_scores": scores,
        "weakest_area": weakest,
        "suggestion": suggestions[weakest],
        "method": (
            "0.40*water_efficiency + 0.30*resource_use + 0.30*crop_health (see README). "
            "Water efficiency includes a rain-awareness adjustment using actual past-7-day "
            "rainfall (Open-Meteo historical) when available."
        ),
    }


if __name__ == "__main__":
    fake_log = [
        {"action": "delay"}, {"action": "irrigate_today"}, {"action": "no_action"},
        {"action": "no_action"}, {"action": "irrigate_now"}, {"action": "delay"}, {"action": "no_action"},
    ]
    print("--- No rain data (old behavior) ---")
    print(sustainability_score(fake_log, severity="severe",
                                actual_fertilizer_kg=12, recommended_fertilizer_kg=10))

    print("\n--- Rainy week (avg 8mm/day) — irrigating anyway should score LOWER ---")
    print(sustainability_score(fake_log, severity="severe", rain_mm_7d=56,
                                actual_fertilizer_kg=12, recommended_fertilizer_kg=10))

    print("\n--- Dry week (avg 0.3mm/day) — irrigating is justified, should score HIGHER ---")
    print(sustainability_score(fake_log, severity="severe", rain_mm_7d=2,
                                actual_fertilizer_kg=12, recommended_fertilizer_kg=10))