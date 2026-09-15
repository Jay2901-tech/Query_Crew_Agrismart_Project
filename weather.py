# """
# weather.py — Weather Data Layer (Bonus Module C, part 1)

# Data source: Open-Meteo (https://open-meteo.com) — free, no API key required.
# Cite this exact source in your README as required by the challenge brief.
# """

# import requests


# def get_forecast(lat, lon):
#     """
#     Fetch the next-24-hour hourly forecast for a location and reduce it
#     to the farm-relevant summary values used by irrigation and disease-risk logic.
#     """
#     url = (
#         "https://api.open-meteo.com/v1/forecast"
#         f"?latitude={lat}&longitude={lon}"
#         "&hourly=precipitation_probability,precipitation,temperature_2m,relative_humidity_2m"
#         "&forecast_days=1"
#     )
#     r = requests.get(url, timeout=10)
#     r.raise_for_status()
#     data = r.json()["hourly"]

#     # Leaf wetness proxy: hours where humidity stayed high enough to keep leaves damp.
#     # >=85% RH is a common agronomic proxy threshold for sustained leaf surface moisture.
#     leaf_wetness_hours = sum(1 for h in data["relative_humidity_2m"][:24] if h >= 85)

#     return {
#         "rain_probability_pct": max(data["precipitation_probability"][:24]),
#         "rain_expected_mm": round(sum(data["precipitation"][:24]), 1),
#         "temperature_c": data["temperature_2m"][12],       # midday reading
#         "humidity_pct": data["relative_humidity_2m"][12],  # midday reading
#         "leaf_wetness_hours": leaf_wetness_hours,
#     }


# if __name__ == "__main__":
#     print(get_forecast(lat=23.02, lon=72.57))  # Ahmedabad — swap for your demo location



"""
weather.py — Weather Data Layer (Bonus Module C, part 1)

Data source: Open-Meteo (https://open-meteo.com) — free, no API key required.
Cite this exact source in your README as required by the challenge brief.
"""

import requests


def get_forecast(lat, lon):
    """
    Fetch the next-24-hour hourly forecast for a location and reduce it
    to the farm-relevant summary values used by irrigation and disease-risk logic.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation_probability,precipitation,temperature_2m,relative_humidity_2m"
        "&forecast_days=1"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()["hourly"]

    # Leaf wetness proxy: hours where humidity stayed high enough to keep leaves damp.
    # >=85% RH is a common agronomic proxy threshold for sustained leaf surface moisture.
    leaf_wetness_hours = sum(1 for h in data["relative_humidity_2m"][:24] if h >= 85)

    return {
        "rain_probability_pct": max(data["precipitation_probability"][:24]),
        "rain_expected_mm": round(sum(data["precipitation"][:24]), 1),
        "temperature_c": data["temperature_2m"][12],       # midday reading
        "humidity_pct": data["relative_humidity_2m"][12],  # midday reading
        "leaf_wetness_hours": leaf_wetness_hours,
    }


def get_past_week_report(lat, lon, days=7):
    """
    Fetch ACTUAL past weather (not forecast) using Open-Meteo's past_days
    parameter on the same forecast endpoint. This gives a genuine
    "what really happened this week" report, useful for:
      - a real recent_rain_mm value (the earlier version of disease_risk.py
        used FORECAST rain under a "recent_rain" name, which was a naming
        mismatch — this is the real thing)
      - a farmer-facing weekly weather summary
      - a more accurate disease-risk read for the days that already passed
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=precipitation,temperature_2m,relative_humidity_2m"
        f"&past_days={days}&forecast_days=1"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()["hourly"]
    times = data["time"]

    n_past_hours = days * 24  # first N hours are the past_days window; rest is today's forecast
    past_rain = data["precipitation"][:n_past_hours]
    past_temp = data["temperature_2m"][:n_past_hours]
    past_humidity = data["relative_humidity_2m"][:n_past_hours]

    # Group into daily buckets (24 hourly readings per day) for a clean day-by-day report
    daily_report = []
    for day_idx in range(days):
        start, end = day_idx * 24, (day_idx + 1) * 24
        day_rain = past_rain[start:end]
        day_temp = past_temp[start:end]
        day_humidity = past_humidity[start:end]
        daily_report.append({
            "date": times[start][:10],  # YYYY-MM-DD
            "total_rain_mm": round(sum(day_rain), 1),
            "avg_temp_c": round(sum(day_temp) / len(day_temp), 1),
            "avg_humidity_pct": round(sum(day_humidity) / len(day_humidity), 1),
            "max_humidity_pct": round(max(day_humidity), 1),
        })

    total_rain_7d = round(sum(past_rain), 1)
    avg_temp_7d = round(sum(past_temp) / len(past_temp), 1)
    avg_humidity_7d = round(sum(past_humidity) / len(past_humidity), 1)
    rainy_days = sum(1 for d in daily_report if d["total_rain_mm"] >= 1.0)  # >=1mm counts as a rainy day

    return {
        "daily": daily_report,
        "total_rain_mm_7d": total_rain_7d,
        "avg_temp_c_7d": avg_temp_7d,
        "avg_humidity_pct_7d": avg_humidity_7d,
        "rainy_days_7d": rainy_days,
    }


if __name__ == "__main__":
    print("--- 24h forecast ---")
    print(get_forecast(lat=23.02, lon=72.57))
    print("\n--- Past 7 days report ---")
    report = get_past_week_report(lat=23.02, lon=72.57)
    for day in report["daily"]:
        print(day)
    print(f"\n7-day totals: {report['total_rain_mm_7d']}mm rain, "
          f"avg temp {report['avg_temp_c_7d']}°C, "
          f"avg humidity {report['avg_humidity_pct_7d']}%, "
          f"{report['rainy_days_7d']} rainy days")