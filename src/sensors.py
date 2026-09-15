"""
sensors.py — Simulated IoT Sensor Feed (Bonus Module F)

No physical hardware is used. Readings start from realistic baseline
values and drift gradually between calls (rather than generating a fresh
random number each time) to mimic how a real soil/climate sensor behaves.

This is documented and reproducible (seedable), which the challenge
explicitly says is scored equally to real hardware.
"""

import random
from datetime import datetime


class SimulatedFarmSensors:
    # Realistic physical ranges
    RANGES = {
        "soil_moisture_pct": (10, 95),
        "temperature_c": (10, 42),
        "humidity_pct": (20, 100),
        "ph": (4.5, 8.5),
    }

    # Max change allowed per reading (keeps drift realistic, not jumpy)
    MAX_STEP = {
        "soil_moisture_pct": 3.0,
        "temperature_c": 1.5,
        "humidity_pct": 4.0,
        "ph": 0.1,
    }

    def __init__(self, crop_type="tomato", seed=None):
        if seed is not None:
            random.seed(seed)  # reproducible for judges/demo

        self.crop_type = crop_type
        # Sensible starting point, not the extreme ends of the range
        self.values = {
            "soil_moisture_pct": random.uniform(45, 65),
            "temperature_c": random.uniform(22, 30),
            "humidity_pct": random.uniform(55, 75),
            "ph": random.uniform(6.0, 7.0),
        }
        self.history = []

    def _drift(self, key):
        low, high = self.RANGES[key]
        step = random.uniform(-self.MAX_STEP[key], self.MAX_STEP[key])
        self.values[key] = max(low, min(high, self.values[key] + step))

    def read(self):
        """One sensor reading — call whenever B/C/D/G need current conditions."""
        for key in self.values:
            self._drift(key)

        reading = {
            **{k: round(v, 2) for k, v in self.values.items()},
            "crop_type": self.crop_type,
            "timestamp": datetime.now().isoformat(),
            "source": "simulated_sensor_feed",
        }
        self.history.append(reading)
        return reading

    def stream(self, n_readings=10):
        """Generate a sequence of readings — useful for demo graphs/logs."""
        return [self.read() for _ in range(n_readings)]


if __name__ == "__main__":
    # Quick manual test
    s = SimulatedFarmSensors(crop_type="tomato", seed=42)
    for _ in range(3):
        print(s.read())
