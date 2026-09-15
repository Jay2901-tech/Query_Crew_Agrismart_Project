# 🌾 AgriSmart — Query Crew

An AI-powered smart agriculture advisor dashboard built for smallholder farmers. AgriSmart combines **computer vision plant disease classification**, **IoT soil sensor simulation**, **live weather forecasting**, and **multilingual audio-visual advisories** across 9 regional Indian languages.

---

## 📁 Repository Structure

```text
Query_Crew/
├── README.md               # Project overview, rubric mapping, and execution instructions
├── requirements.txt        # Python package dependencies (uv and pip supported)
├── app.py                  # Main Streamlit dashboard UI entry point
├── download_models.py      # Weights downloader script (auto-detects uv or pip)
├── agent.py                # Headless autonomous decision loop test runner (Module G)
├── debug_pipeline.py       # Diagnostic CV model test script
├── src/                    # Core application modular source code
│   ├── predict.py          # MobileNetV2 disease classification inference interface
│   ├── cv_utils.py         # OpenCV HSV leaf isolation & green-channel extraction
│   ├── disease_risk.py     # Weather-correlated disease risk assessment engine (Module C)
│   ├── irrigation.py       # Smart irrigation advisory logic (Module B)
│   ├── fertilizer.py       # Crop stage nutrient & fertilizer guidance
│   ├── sustainability.py   # Environmental & resource efficiency scoring 0-100 (Module D)
│   ├── sensors.py          # Simulated IoT farm sensor feed (Module F)
│   └── weather.py          # Open-Meteo live weather API client (Module C)
└── model/                  # Deep learning models & weights
    ├── phase2_best_model.keras   # Trained MobileNetV2 24-class disease model
    ├── yolo11x_leaf.pt           # YOLO11 leaf detector model weights
    └── download_models.py        # Model downloader copy
```

---

## 🎯 Evaluation Criteria Implementation Matrix (Points B – G)

| Rubric Point | Module & File | Logic / Data Source / Formula | Status |
| :--- | :--- | :--- | :---: |
| **B. Smart Irrigation** | [`src/irrigation.py`](file:///d:/Query_Crew%20-%20Copy/src/irrigation.py) | **Logic**: Compares IoT soil moisture against crop-specific thresholds (e.g. Tomato target 60-80%) and 48h rain forecast. **Validation**: Prevents overwatering if rain probability $\ge 60\%$ or rain $\ge 5\text{ mm}$, issuing precise gallon/liter volume recommendations. | ✅ Complete |
| **C. Weather Intelligence** | [`src/weather.py`](file:///d:/Query_Crew%20-%20Copy/src/weather.py)<br>[`src/disease_risk.py`](file:///d:/Query_Crew%20-%20Copy/src/disease_risk.py) | **Source**: Live 7-day forecast from **Open-Meteo REST API** (lat/lon grounded). **Actions**: Produces alerts like *"Delay irrigation - 80% rain likely"* or *"Raised disease risk - leaf wetness 6h elevated"*. | ✅ Complete |
| **D. Sustainability Score** | [`src/sustainability.py`](file:///d:/Query_Crew%20-%20Copy/src/sustainability.py) | **Formula**: Score $= 0.40 \cdot \text{WaterEfficiency} + 0.30 \cdot \text{ResourceBalance} + 0.30 \cdot \text{CropHealthScore}$ (Scale 0-100). Generates actionable eco-improvement suggestions. | ✅ Complete |
| **E. Farmer Assistant & Voice** | [`app.py`](file:///d:/Query_Crew%20-%20Copy/app.py) | **Localization**: Grounded AI diagnosis and advisory translated into **9 regional Indian languages** (*Hindi, Gujarati, Tamil, Telugu, Marathi, Bengali, Kannada, Punjabi, English*) via Sarvam AI API. **Voice**: Text-to-speech synthesis via `gTTS`. | ✅ Complete |
| **F. IoT Integration** | [`src/sensors.py`](file:///d:/Query_Crew%20-%20Copy/src/sensors.py) | **Feed**: `SimulatedFarmSensors` class streaming continuous live readings for Soil Moisture (%), Temperature (°C), Humidity (%), and Soil pH per crop type. | ✅ Complete |
| **G. Agentic Advisor** | [`agent.py`](file:///d:/Query_Crew%20-%20Copy/agent.py) | **Decision Loop**: Autonomous cycle (`read` $\rightarrow$ `reason` $\rightarrow$ `decide` $\rightarrow$ `notify`). Tracks notification history to prevent duplicate alert spam. | ✅ Complete |

---

## 🚀 Quick Start Guide

### Option 1: Fast Setup using `uv` (Recommended)

`uv` is an extremely fast Python package installer and virtual environment manager written in Rust.

1. **Install `uv` (if not already installed)**:
   ```bash
   # Via pip (cross-platform):
   pip install uv

   # OR via Standalone Installer:
   # On Windows (PowerShell):
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   # On Linux/macOS:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Navigate to project folder**:
   ```bash
   cd Query_Crew
   ```

3. **Create and activate virtual environment using `uv`**:
   ```bash
   uv venv
   # On Windows (PowerShell):
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

4. **Install dependencies using `uv`**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   *(Or run directly via uv: `uv run streamlit run app.py`)*

---

### Option 2: Standard Setup using `pip`

1. **Create virtual environment**:
   ```bash
   python -m venv myenv
   # On Windows (PowerShell):
   myenv\Scripts\activate
   # On Linux/macOS:
   source myenv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## 🤖 Running the Autonomous Agentic Loop (Module G)

To demonstrate the autonomous agentic decision loop in headless terminal mode:

```bash
python agent.py
```
This runs 7 simulated daily cycles reading IoT sensors, fetching weather forecasts, reasoning over disease risks, and notifying the farmer.
