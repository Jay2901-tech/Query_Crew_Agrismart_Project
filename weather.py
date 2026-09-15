"""
weather.py — Weather Data Layer (Bonus Module C, part 1)

Data source: Open-Meteo (https://open-meteo.com) — free, no API key required.
Cite this exact source in your README as required by the challenge brief.

Both public functions return None on any network/API failure so that callers
(app.py) can show a graceful fallback instead of crashing the whole app.
"""

import requests


# ---------------------------------------------------------------------------
# Shared fallback values — returned when the API is unreachable
# ---------------------------------------------------------------------------
_FORECAST_FALLBACK = {
    "rain_probability_pct": 40,
    "rain_expected_mm":     2.0,
    "temperature_c":        28.0,
    "humidity_pct":         65.0,
    "leaf_wetness_hours":   3,
    "_is_fallback":         True,   # flag so callers can show a notice
}

_PAST_WEEK_FALLBACK = {
    "daily":               [],
    "total_rain_mm_7d":    14.0,
    "avg_temp_c_7d":       27.0,
    "avg_humidity_pct_7d": 65.0,
    "rainy_days_7d":       2,
    "_is_fallback":        True,
}


def get_forecast(lat, lon):
    """
    Fetch the next-24-hour hourly forecast for a location and reduce it
    to the farm-relevant summary values used by irrigation and disease-risk logic.

    Returns a dict on success, or the fallback dict if the API is unreachable
    (network error, invalid coordinates, API down, etc.).
    The returned dict includes '_is_fallback': True only when fallback data is used.
    """
    try:
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
            "rain_expected_mm":     round(sum(data["precipitation"][:24]), 1),
            "temperature_c":        data["temperature_2m"][12],       # midday reading
            "humidity_pct":         data["relative_humidity_2m"][12], # midday reading
            "leaf_wetness_hours":   leaf_wetness_hours,
        }

    except requests.exceptions.Timeout:
        print("[weather] get_forecast timed out — using fallback values.")
        return _FORECAST_FALLBACK.copy()
    except requests.exceptions.ConnectionError:
        print("[weather] get_forecast connection error — using fallback values.")
        return _FORECAST_FALLBACK.copy()
    except Exception as e:
        print(f"[weather] get_forecast unexpected error ({e}) — using fallback values.")
        return _FORECAST_FALLBACK.copy()


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

    Returns a dict on success, or the fallback dict on any API failure.
    """
    try:
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
        past_rain     = data["precipitation"][:n_past_hours]
        past_temp     = data["temperature_2m"][:n_past_hours]
        past_humidity = data["relative_humidity_2m"][:n_past_hours]

        # Guard against empty slices (e.g. API returns fewer hours than expected)
        if not past_temp or not past_humidity:
            return _PAST_WEEK_FALLBACK.copy()

        # Group into daily buckets (24 hourly readings per day) for a clean day-by-day report
        daily_report = []
        for day_idx in range(days):
            start, end = day_idx * 24, (day_idx + 1) * 24
            day_rain     = past_rain[start:end]
            day_temp     = past_temp[start:end]
            day_humidity = past_humidity[start:end]
            if not day_temp:
                continue
            daily_report.append({
                "date":             times[start][:10],  # YYYY-MM-DD
                "total_rain_mm":    round(sum(day_rain), 1),
                "avg_temp_c":       round(sum(day_temp) / len(day_temp), 1),
                "avg_humidity_pct": round(sum(day_humidity) / len(day_humidity), 1),
                "max_humidity_pct": round(max(day_humidity), 1),
            })

        total_rain_7d    = round(sum(past_rain), 1)
        avg_temp_7d      = round(sum(past_temp) / len(past_temp), 1)
        avg_humidity_7d  = round(sum(past_humidity) / len(past_humidity), 1)
        rainy_days       = sum(1 for d in daily_report if d["total_rain_mm"] >= 1.0)

        return {
            "daily":               daily_report,
            "total_rain_mm_7d":    total_rain_7d,
            "avg_temp_c_7d":       avg_temp_7d,
            "avg_humidity_pct_7d": avg_humidity_7d,
            "rainy_days_7d":       rainy_days,
        }

    except requests.exceptions.Timeout:
        print("[weather] get_past_week_report timed out — using fallback values.")
        return _PAST_WEEK_FALLBACK.copy()
    except requests.exceptions.ConnectionError:
        print("[weather] get_past_week_report connection error — using fallback values.")
        return _PAST_WEEK_FALLBACK.copy()
    except Exception as e:
        print(f"[weather] get_past_week_report unexpected error ({e}) — using fallback values.")
        return _PAST_WEEK_FALLBACK.copy()


if __name__ == "__main__":
    print("--- 24h forecast ---")
    print(get_forecast(lat=23.02, lon=72.57))
    print("\n--- Past 7 days report ---")
    report = get_past_week_report(lat=23.02, lon=72.57)
    for day in report.get("daily", []):
        print(day)
    print(f"\n7-day totals: {report['total_rain_mm_7d']}mm rain, "
          f"avg temp {report['avg_temp_c_7d']}°C, "
          f"avg humidity {report['avg_humidity_pct_7d']}%, "
          f"{report['rainy_days_7d']} rainy days")