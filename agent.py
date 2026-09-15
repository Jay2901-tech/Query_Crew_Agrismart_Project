"""
agent.py — Agentic Advisor (Bonus Module G)

Ties sensors (F) + weather (C) + irrigation (B) + disease risk (C) +
the core CV model's detection into ONE decision loop that:
  reads -> reasons -> decides -> notifies

This is what makes the system feel autonomous rather than a set of
functions you call manually. Each cycle also logs irrigation history,
which sustainability.py later uses for Module D.
"""

import sys
import os

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.sensors import SimulatedFarmSensors
from src.weather import get_forecast
from src.irrigation import irrigation_decision
from src.disease_risk import disease_risk_assessment
from src.cv_utils import predict, map_prediction_to_severity


class AgentState:
    """Tracks what's already been reported (avoids repeat alerts) + keeps history for Module D."""

    def __init__(self):
        self.last_irrigation_action = None
        self.last_disease_risk_level = None
        self.irrigation_log = []  # consumed by sustainability.sustainability_score()


def should_notify(irrigation, disease_risk, state):
    """Decide if this cycle's findings are worth telling the farmer about."""
    notifications = []

    state.irrigation_log.append({"action": irrigation["action"], "reason": irrigation["reason"]})

    if irrigation["action"] != "no_action" and irrigation["action"] != state.last_irrigation_action:
        notifications.append({"type": "irrigation", "message": f"\U0001F4A7 {irrigation['reason']}"})

    if disease_risk["risk_level"] != "normal" and disease_risk["risk_level"] != state.last_disease_risk_level:
        reasons = "; ".join(disease_risk["reasons"]) if disease_risk["reasons"] else "see detection details"
        notifications.append({
            "type": "disease_risk",
            "message": f"\u26A0\uFE0F {disease_risk['risk_level'].upper()} risk — {disease_risk['action']}. ({reasons})",
        })

    state.last_irrigation_action = irrigation["action"]
    state.last_disease_risk_level = disease_risk["risk_level"]

    return notifications


def run_agent_cycle(sensors, lat, lon, growth_stage, state, cycle_num, image_path=None):
    """
    One full check: read sensor + weather + (optionally) a new leaf photo,
    reason over all of it together, decide, notify.
    """
    reading = sensors.read()
    weather = get_forecast(lat, lon)

    # Core CV step — swap predict_mock for the real trained model when ready
    if image_path:
        from PIL import Image
        pil_img = Image.open(image_path) if isinstance(image_path, str) and os.path.exists(image_path) else None
        if pil_img:
            prediction = predict(pil_img)
        else:
            prediction = {"class": "Tomato - Early Blight", "confidence": 0.88}
        severity = map_prediction_to_severity(prediction["class"], prediction["confidence"])
        detected_disease = prediction["class"]
    else:
        severity, detected_disease = None, None

    irrigation = irrigation_decision(
        soil_moisture_pct=reading["soil_moisture_pct"],
        temperature_c=reading["temperature_c"],
        crop_type=reading["crop_type"],
        growth_stage=growth_stage,
        rain_probability_pct=weather["rain_probability_pct"],
        rain_expected_mm=weather["rain_expected_mm"],
        disease_severity=severity,
    )

    disease_risk = disease_risk_assessment(
        humidity_pct=weather["humidity_pct"],
        temperature_c=weather["temperature_c"],
        leaf_wetness_hours=weather["leaf_wetness_hours"],
        recent_rain_mm=weather["rain_expected_mm"],
        crop_type=reading["crop_type"],
        detected_disease=detected_disease,
        severity=severity,
    )

    notifications = should_notify(irrigation, disease_risk, state)

    print(f"\n[Cycle {cycle_num}] Checking conditions...")
    print(f"  Soil moisture: {reading['soil_moisture_pct']}%  |  Humidity: {weather['humidity_pct']}%  |  Rain: {weather['rain_probability_pct']}%")
    if severity:
        print(f"  CV detection: {detected_disease} ({severity})")
    print(f"  Irrigation: {irrigation['action']}  |  Disease risk: {disease_risk['risk_level']}")

    if notifications:
        for n in notifications:
            print(f"  -> NOTIFYING FARMER: {n['message']}")
    else:
        print("  -> Nothing new to report this cycle.")

    return {
        "reading": reading, "weather": weather, "irrigation": irrigation,
        "disease_risk": disease_risk, "notifications": notifications,
        "severity": severity, "detected_disease": detected_disease,
    }


if __name__ == "__main__":
    sensors = SimulatedFarmSensors(crop_type="tomato", seed=42)
    state = AgentState()

    for day in range(1, 8):
        # simulate a photo being taken on day 3 only, as a demo
        img = "leaf_photo.jpg" if day == 3 else None
        run_agent_cycle(sensors, lat=23.02, lon=72.57, growth_stage="flowering",
                         state=state, cycle_num=day, image_path=img)
