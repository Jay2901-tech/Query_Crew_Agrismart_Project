import streamlit as st
from PIL import Image
from gtts import gTTS
from io import BytesIO
import tempfile
import os

from predict import predict_image, predict_image_full, clean_label
from cv_utils import (
    map_prediction_to_severity,
    extract_crop_from_label,
    detect_and_crop,
)
from sensors import SimulatedFarmSensors
from weather import get_forecast, get_past_week_report
from irrigation import irrigation_decision
from disease_risk import disease_risk_assessment
from fertilizer import get_recommended_fertilizer
from sustainability import sustainability_score

# Page Config
st.set_page_config(page_title="AgriSmart - Crop Disease Detector", page_icon="🌾", layout="centered")

# Language Localization Mapping
LANGUAGES = {
    "English":          {"code": "en", "title": "🌾 AgriSmart Disease Detector",       "take_pic": "Take a photo of the plant leaf", "upload_pic": "Or upload an image file",          "pred_btn": "🔍 Detect & Classify Leaves", "speak_btn": "🔊 Hear Diagnosis"},
    "Hindi (हिंदी)":   {"code": "hi", "title": "🌾 एग्रीस्मार्ट पौधा रोग पहचान",    "take_pic": "पत्ते की तस्वीर लें",           "upload_pic": "या फोटो अपलोड करें",            "pred_btn": "🔍 रोग पहचानें",              "speak_btn": "🔊 आवाज सुनें"},
    "Gujarati (ગુજરાતી)": {"code": "gu", "title": "🌾 એગ્રીસ્માર્ટ પાક રોગ ડિટેક્ટર", "take_pic": "છોડના પાંદડાનો ફોટો લો",       "upload_pic": "અથવા ઈમેજ અપલોડ કરો",         "pred_btn": "🔍 રોગ ઓળખો",                "speak_btn": "🔊 અવાજ સાંભળો"},
}

# All crop types across 24 model classes (sidebar + threshold dicts)
ALL_CROP_TYPES = [
    "tomato", "potato", "corn", "grape", "apple", "pepper",
    "strawberry", "blueberry", "cherry", "peach", "raspberry", "soybean", "squash",
]

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "sensors"         not in st.session_state: st.session_state.sensors         = None
if "irrigation_log"  not in st.session_state: st.session_state.irrigation_log  = []
# YOLO pipeline state
if "leaf_crops"      not in st.session_state: st.session_state.leaf_crops       = None  # list[PIL]
if "leaf_boxes"      not in st.session_state: st.session_state.leaf_boxes       = None
if "is_fallback"     not in st.session_state: st.session_state.is_fallback      = False
if "detection_done"  not in st.session_state: st.session_state.detection_done   = False
if "selected_crop_idx" not in st.session_state: st.session_state.selected_crop_idx = 0
if "final_result"    not in st.session_state: st.session_state.final_result     = None  # dict
if "detected_crop_type" not in st.session_state: st.session_state.detected_crop_type = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
selected_lang = st.sidebar.selectbox("🌐 Choose Language / भाषा चुनें", list(LANGUAGES.keys()))
lang = LANGUAGES[selected_lang]

st.sidebar.markdown("---")
st.sidebar.subheader("🌱 Farm Details")

# Auto-update the sidebar crop selector when YOLO+classifier detects a different crop.
# We store the auto-detected type in session_state and set the sidebar index to match.
detected_type = st.session_state.detected_crop_type
default_crop_idx = ALL_CROP_TYPES.index(detected_type) if (detected_type and detected_type in ALL_CROP_TYPES) else 0

crop_type = st.sidebar.selectbox(
    "Crop",
    ALL_CROP_TYPES,
    index=default_crop_idx,
    help="Auto-updates to match the detected plant after classification.",
)

# Show auto-update notice if the sidebar was just changed by the classifier
if detected_type and detected_type == crop_type and st.session_state.final_result is not None:
    st.sidebar.caption(f"🤖 Auto-set to **{crop_type}** based on leaf detection.")

growth_stage = st.sidebar.selectbox("Growth Stage", ["seedling", "vegetative", "flowering", "maturity"])
lat = st.sidebar.number_input("Latitude",  value=23.02, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=72.57, format="%.4f")

# Compute fertilizer recommendation BEFORE the input box so its default follows the selection
recommended_fert = get_recommended_fertilizer(crop_type, growth_stage)
st.sidebar.caption(f"Recommended for {crop_type} ({growth_stage}): {recommended_fert} kg")
actual_fertilizer_kg = st.sidebar.number_input(
    "Fertilizer applied this week (kg)",
    value=float(recommended_fert),
    min_value=0.0,
    key=f"fert_{crop_type}_{growth_stage}",
)

# ---------------------------------------------------------------------------
# Main title + image input
# ---------------------------------------------------------------------------
st.title(lang["title"])

input_option = st.radio("Choose Input Method:", ("Camera", "Upload Local File"), horizontal=True)

image_data = None
if input_option == "Camera":
    camera_file = st.camera_input(lang["take_pic"])
    if camera_file:
        image_data = camera_file
else:
    uploaded_file = st.file_uploader(lang["upload_pic"], type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_data = uploaded_file

# Reset pipeline state when a new image is loaded
if image_data is not None:
    img = Image.open(image_data)
    current_img_id = getattr(image_data, "file_id", id(image_data))
    if st.session_state.get("_last_img_id") != current_img_id:
        # New image → wipe previous detection/result
        st.session_state["_last_img_id"]    = current_img_id
        st.session_state.leaf_crops         = None
        st.session_state.leaf_boxes         = None
        st.session_state.is_fallback        = False
        st.session_state.detection_done     = False
        st.session_state.selected_crop_idx  = 0
        st.session_state.final_result       = None

    st.image(img, caption="Selected Leaf Image", use_container_width=True)

    # =========================================================================
    # PHASE 1 — Leaf Detection (YOLO)
    # Runs when the user clicks the "Detect & Classify" button.
    # If YOLO finds only 1 crop (or falls back), goes straight to classification.
    # If 2+ crops found, shows the gallery for the user to pick one.
    # =========================================================================
    if not st.session_state.detection_done:
        if st.button(lang["pred_btn"], type="primary"):
            try:
                with st.spinner("🔍 Detecting leaves in image..."):
                    # Save PIL image to a temp file so YOLO can accept a file path
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        img.convert("RGB").save(tmp.name)
                        tmp_path = tmp.name

                    original_img, crops, boxes, is_fallback = detect_and_crop(tmp_path)

                    try:
                        os.unlink(tmp_path)   # clean up temp file
                    except OSError:
                        pass

                st.session_state.leaf_crops     = crops
                st.session_state.leaf_boxes     = boxes
                st.session_state.is_fallback    = is_fallback
                st.session_state.detection_done = True

                if is_fallback:
                    st.info("ℹ️ No clear leaf region detected — using the full image.")
                else:
                    st.success(f"✅ Found **{len(crops)}** leaf region(s).")

                # If only 1 crop (or fallback), skip gallery → classify immediately
                if len(crops) == 1:
                    st.session_state.selected_crop_idx = 0
                    with st.spinner("🧠 Classifying leaf..."):
                        idx, raw_label, confidence, _ = predict_image_full(crops[0])
                    detected_type_now = extract_crop_from_label(raw_label)
                    st.session_state.detected_crop_type = detected_type_now
                    st.session_state.final_result = {
                        "raw_label":  raw_label,
                        "label":      clean_label(raw_label),
                        "confidence": confidence,
                        "severity":   map_prediction_to_severity(clean_label(raw_label), confidence),
                        "crop_type":  detected_type_now,
                    }
                    st.rerun()   # rerun to update sidebar index + show results
            except Exception as _detect_err:
                st.error(f"❌ Detection failed: {_detect_err}")
                st.exception(_detect_err)

    # =========================================================================
    # PHASE 1b — Gallery (only when 2+ crops detected)
    # Displayed after detection if multiple crops were found and user hasn't
    # chosen one yet.
    # =========================================================================
    if (
        st.session_state.detection_done
        and st.session_state.final_result is None
        and st.session_state.leaf_crops is not None
        and len(st.session_state.leaf_crops) >= 2
    ):
        crops = st.session_state.leaf_crops
        selected_idx = st.session_state.selected_crop_idx  # default 0

        st.markdown("---")
        st.subheader(f"🌿 Select a Leaf to Diagnose ({len(crops)} detected)")
        st.caption(
            "Click **'Select'** on the leaf you want diagnosed, then press **Classify**. "
            f"Currently selected: **Leaf {selected_idx + 1}**"
        )

        # Display crops in a 3-column grid with a Select button under each image
        COLS_PER_ROW = 3
        rows = [crops[i:i+COLS_PER_ROW] for i in range(0, len(crops), COLS_PER_ROW)]

        for r_idx, row in enumerate(rows):
            cols = st.columns(COLS_PER_ROW)
            for col_idx, crop_img in enumerate(row):
                global_idx = r_idx * COLS_PER_ROW + col_idx
                if global_idx >= len(crops):
                    break
                is_selected = (global_idx == selected_idx)
                with cols[col_idx]:
                    caption = f"✅ Leaf {global_idx + 1} — SELECTED" if is_selected else f"Leaf {global_idx + 1}"
                    st.image(crop_img, caption=caption, use_container_width=True)
                    btn_label = f"☑ Selected" if is_selected else f"Select Leaf {global_idx + 1}"
                    if st.button(btn_label, key=f"sel_{global_idx}", disabled=is_selected):
                        st.session_state.selected_crop_idx = global_idx
                        st.rerun()

        st.markdown("")
        chosen_idx = st.session_state.selected_crop_idx
        if st.button(f"🔬 Classify Leaf {chosen_idx + 1} →", type="primary"):
            chosen_crop = crops[chosen_idx]
            try:
                with st.spinner(f"🧠 Classifying Leaf {chosen_idx + 1}..."):
                    idx, raw_label, confidence, _ = predict_image_full(chosen_crop)
                detected_type_now = extract_crop_from_label(raw_label)
                st.session_state.detected_crop_type = detected_type_now
                st.session_state.final_result = {
                    "raw_label":  raw_label,
                    "label":      clean_label(raw_label),
                    "confidence": confidence,
                    "severity":   map_prediction_to_severity(clean_label(raw_label), confidence),
                    "crop_type":  detected_type_now,
                }
                st.rerun()
            except Exception as _cls_err:
                st.error(f"❌ Classification failed: {_cls_err}")
                st.exception(_cls_err)

    # =========================================================================
    # PHASE 2 — Results + Downstream Modules
    # Shown after the user has selected a leaf and classification is complete.
    # =========================================================================
    if st.session_state.final_result is not None:
        result     = st.session_state.final_result
        label      = result["label"]
        confidence = result["confidence"]
        severity   = result["severity"]
        raw_label  = result["raw_label"]

        # Auto-detected crop type drives downstream modules (sidebar reflects this)
        detected_crop = result["crop_type"]
        # Use detected crop if it's in our known list; else fall back to sidebar selection
        effective_crop = detected_crop if (detected_crop in ALL_CROP_TYPES) else crop_type

        st.markdown("---")
        st.success("✅ Classification complete!")

        # ---- Confidence gating: ≥70 % → confirmed diagnosis; < 70 % → uncertain ----
        CONFIDENCE_THRESHOLD_DISPLAY = 0.70
        if confidence >= CONFIDENCE_THRESHOLD_DISPLAY:
            st.metric(label="🌿 Diagnosed Condition", value=label)
            st.metric(label="Confidence",             value=f"{confidence * 100:.2f}%")
        else:
            st.warning(
                f"⚠️ **Low confidence ({confidence * 100:.1f}%)** — result is uncertain.\n\n"
                f"Possible condition: **{label}**\n\n"
                "For a reliable diagnosis, try a clearer photo with the leaf well-lit and filling the frame."
            )
            st.metric(label="Possible Condition", value=label)
            st.metric(label="Confidence (low)",   value=f"{confidence * 100:.2f}%")

        # Auto-update notice
        if detected_crop in ALL_CROP_TYPES and detected_crop != crop_type:
            st.info(
                f"🤖 **Crop auto-detected as '{detected_crop}'** based on the leaf classification. "
                f"The sidebar has been updated — all advice below uses **{detected_crop}**. "
                f"You can override this manually in the sidebar at any time."
            )

        # ---- Stage 2: Simulated IoT sensor reading (Bonus F) ----
        if st.session_state.sensors is None or st.session_state.sensors.crop_type != effective_crop:
            st.session_state.sensors = SimulatedFarmSensors(crop_type=effective_crop)
        reading = st.session_state.sensors.read()

        st.subheader("📡 Simulated Farm Sensors (IoT)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Soil Moisture", f"{reading['soil_moisture_pct']}%")
        c2.metric("Temperature",   f"{reading['temperature_c']}°C")
        c3.metric("Humidity",      f"{reading['humidity_pct']}%")
        c4.metric("Soil pH",       f"{reading['ph']}")
        st.caption("Source: documented simulated sensor feed (no physical hardware) — see Bonus Module F.")

        # ---- Stage 3: Weather forecast (Bonus C data layer) ----
        st.subheader("⛅ Weather Forecast (Open-Meteo)")
        try:
            weather = get_forecast(lat, lon)
        except Exception as e:
            st.warning(f"Could not reach weather API ({e}) — using fallback demo values.")
            weather = {
                "rain_probability_pct": 40,
                "rain_expected_mm":     2.0,
                "temperature_c":        reading["temperature_c"],
                "humidity_pct":         reading["humidity_pct"],
                "leaf_wetness_hours":   3,
            }

        w1, w2, w3 = st.columns(3)
        w1.metric("Rain Chance",    f"{weather['rain_probability_pct']}%")
        w2.metric("Rain Expected",  f"{weather['rain_expected_mm']} mm")
        w3.metric("Leaf Wetness",   f"{weather['leaf_wetness_hours']} h")

        # ---- Stage 4: Irrigation decision (Bonus B) ----
        irrigation = irrigation_decision(
            soil_moisture_pct=reading["soil_moisture_pct"],
            rain_probability_pct=weather["rain_probability_pct"],
            rain_expected_mm=weather["rain_expected_mm"],
            crop_type=effective_crop,
            growth_stage=growth_stage,
            temperature_c=reading["temperature_c"],
            disease_severity=severity,
        )
        st.session_state.irrigation_log.append({"action": irrigation["action"], "reason": irrigation["reason"]})

        st.subheader("💧 Irrigation Advice")
        st.info(f"**{irrigation['action'].replace('_', ' ').title()}** — {irrigation['reason']}")

        # ---- Stage 5: Disease risk assessment (Bonus C reasoning layer) ----
        disease_risk = disease_risk_assessment(
            humidity_pct=weather["humidity_pct"],
            temperature_c=weather["temperature_c"],
            leaf_wetness_hours=weather["leaf_wetness_hours"],
            recent_rain_mm=weather["rain_expected_mm"],
            crop_type=effective_crop,
            detected_disease=label,
            severity=severity,
        )
        st.subheader("⚠️ Disease Risk")
        risk_display = {"normal": st.success, "raised": st.warning, "high": st.error}
        risk_display.get(disease_risk["risk_level"], st.info)(
            f"**{disease_risk['risk_level'].upper()}** — action: {disease_risk['action']}"
        )
        if disease_risk["reasons"]:
            for r in disease_risk["reasons"]:
                st.caption(f"• {r}")

        # ---- Stage 6: Sustainability score (Bonus D) ----
        recommended_fert_for_result = get_recommended_fertilizer(effective_crop, growth_stage)
        sustainability = sustainability_score(
            irrigation_log=st.session_state.irrigation_log,
            severity=severity,
            actual_fertilizer_kg=actual_fertilizer_kg,
            recommended_fertilizer_kg=recommended_fert_for_result,
        )
        st.subheader("📊 Sustainability Score")
        st.metric("Overall Score", f"{sustainability['sustainability_score']} / 100")
        s1, s2, s3 = st.columns(3)
        s1.metric("Water",      sustainability["sub_scores"]["water"])
        s2.metric("Resource",   sustainability["sub_scores"]["resource"])
        s3.metric("Crop Health", sustainability["sub_scores"]["health"])
        st.caption(f"💡 {sustainability['suggestion']}")
        st.caption(f"Formula: {sustainability['method']}")

        # ---- Stage 7: Voice readout (Bonus E) ----
        st.subheader("🔊 Farmer Voice Summary")
        confidence_word = "high" if confidence >= 0.70 else "low"
        speech_text = (
            f"The predicted condition is {label} with {confidence * 100:.1f} percent confidence "
            f"({confidence_word} confidence). "
            f"Irrigation advice: {irrigation['reason']}. "
            f"Disease risk is {disease_risk['risk_level']}. "
            f"Your sustainability score this week is {sustainability['sustainability_score']} out of 100. "
            f"{sustainability['suggestion']}"
        )
        if st.button(lang["speak_btn"]):
            with st.spinner("Generating audio..."):
                tts = gTTS(text=speech_text, lang=lang["code"])
                sound_fp = BytesIO()
                tts.write_to_fp(sound_fp)
                st.audio(sound_fp, format="audio/mp3", autoplay=True)

        # ---- Reset button ----
        st.markdown("---")
        if st.button("🔄 Analyse Another Image"):
            st.session_state.leaf_crops         = None
            st.session_state.leaf_boxes         = None
            st.session_state.is_fallback        = False
            st.session_state.detection_done     = False
            st.session_state.selected_crop_idx  = 0
            st.session_state.final_result       = None
            st.session_state["_last_img_id"]    = None
            st.rerun()