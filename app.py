import sys
import os
import streamlit as st
from PIL import Image
from gtts import gTTS
from io import BytesIO
import tempfile
import requests
import re
import hashlib

# Ensure src directory is in sys.path
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.predict import predict_image, predict_image_full, clean_label
from src.cv_utils import (
    map_prediction_to_severity,
    extract_crop_from_label,
    detect_and_crop,
)
from src.sensors import SimulatedFarmSensors
from src.weather import get_forecast, get_past_week_report
from src.irrigation import irrigation_decision
from src.disease_risk import disease_risk_assessment
from src.fertilizer import get_recommended_fertilizer
from src.sustainability import sustainability_score

# ---------------------------------------------------------------------------
# Sarvam Translation Layer (Configurable via ENV / Secrets or fallback)
# ---------------------------------------------------------------------------
_DEFAULT_SARVAM_KEY = "sk_49k3bz3i_nIiaXKsIzySEDGkS3rbcYSpD"
SARVAM_API_KEY = os.environ.get(
    "SARVAM_API_KEY",
    st.secrets.get("SARVAM_API_KEY", _DEFAULT_SARVAM_KEY) if hasattr(st, "secrets") else _DEFAULT_SARVAM_KEY
)
SARVAM_ENDPOINT = "https://api.sarvam.ai/translate"

# Maps our lang codes → Sarvam BCP-47 codes
SARVAM_LANG_MAP = {
    "en": None,       # no translation needed
    "hi": "hi-IN",
    "gu": "gu-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "kn": "kn-IN",
    "pa": "pa-IN",
}


# Maximum characters per Sarvam API call (stay safely under their limit)
_CHUNK_SIZE = 400


def _split_into_chunks(text: str, max_chars: int = _CHUNK_SIZE) -> list[str]:
    """
    Splits text into sentence-level chunks that each fit within max_chars.
    Strategy:
      1. Split on sentence boundaries (. ! ?)
      2. If a single sentence is still > max_chars, hard-split on word boundaries.
    This ensures Sarvam never receives a truncation-prone payload.
    """
    # Split on sentence-ending punctuation, keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        # If adding this sentence keeps us under the limit, accumulate
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            # Flush current chunk
            if current:
                chunks.append(current)
            # If single sentence is too long, hard-split on words
            if len(sentence) > max_chars:
                words = sentence.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_chars:
                        current = (current + " " + word).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = word
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


# HTTP status codes that mean "stop trying for this session"
_SARVAM_FATAL_CODES = {
    401: "Invalid API key",
    403: "Access forbidden",
    402: "Quota / credits exhausted",
    429: "Rate limit / tokens exhausted",
}


@st.cache_data(ttl=86400, show_spinner=False)
def _raw_sarvam_translate_call(chunk: str, target_sarvam_code: str) -> str:
    """
    Cached API call to Sarvam AI.
    Results are saved in RAM for 24 hours per (chunk, language).
    Subsequent renders with the same string return instantaneously.
    """
    if not chunk or not chunk.strip():
        return chunk

    try:
        resp = requests.post(
            SARVAM_ENDPOINT,
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "input": chunk,
                "source_language_code": "en-IN",
                "target_language_code": target_sarvam_code,
                "speaker_gender": "Male",
                "mode": "formal",
                "enable_preprocessing": True,   # handles agri/technical terms
            },
            timeout=5,
        )

        # ── Fatal errors: return marker string to trigger session flag ──
        if resp.status_code in _SARVAM_FATAL_CODES:
            return f"__FATAL_{resp.status_code}__"

        # ── Transient server errors: fail this chunk only ──
        if resp.status_code >= 500:
            return chunk

        resp.raise_for_status()
        return resp.json().get("translated_text", chunk)

    except Exception:
        return chunk   # safe English fallback on network error/timeout


def _sarvam_translate_chunk(chunk: str, target_sarvam_code: str) -> str:
    """
    Translates a SINGLE chunk via Sarvam API.
    - Uses cached _raw_sarvam_translate_call for zero-latency UI reruns.
    - On fatal errors (bad key, no credits, rate-limit) sets a session flag
      so ALL future calls skip the API and return English immediately.
    """
    if not chunk or not chunk.strip():
        return chunk

    # If a fatal error was already detected this session, skip the API entirely
    if st.session_state.get("_sarvam_disabled"):
        return chunk

    res = _raw_sarvam_translate_call(chunk, target_sarvam_code)
    if res.startswith("__FATAL_"):
        try:
            status_code = int(res.split("_")[2])
            reason = _SARVAM_FATAL_CODES.get(status_code, "API failure")
            st.session_state["_sarvam_disabled"] = True
            st.session_state["_sarvam_error_msg"] = (
                f"⚠️ Translation unavailable ({status_code} — {reason}). "
                f"Showing content in English."
            )
        except Exception:
            pass
        return chunk

    return res


def _sarvam_translate(text: str, target_sarvam_code: str) -> str:
    """
    Public translation entry point.
    Splits text into ≤400-char chunks, translates each independently,
    then rejoins — so even long paragraphs come back fully translated.
    """
    chunks = _split_into_chunks(text)
    translated_chunks = [_sarvam_translate_chunk(c, target_sarvam_code) for c in chunks]
    return " ".join(translated_chunks)


def t(text: str) -> str:
    """
    Translate `text` to the currently selected language.
    No-op when English is selected.
    """
    sarvam_code = SARVAM_LANG_MAP.get(st.session_state.get("_lang_code", "en"))
    if sarvam_code is None:
        return text
    return _sarvam_translate(text, sarvam_code)


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriSmart - Crop Disease Detector",
    page_icon="🌾",
    layout="centered",
)

# Language Localization Mapping (static UI strings — already localised)
LANGUAGES = {
    "English":                {"code": "en", "title": "🌾 AgriSmart Disease Detector",              "take_pic": "Take a photo of the plant leaf",    "upload_pic": "Or upload an image file",          "pred_btn": "🔍 Detect & Classify Leaves",   "speak_btn": "🔊 Hear Diagnosis"},
    "Hindi (हिंदी)":         {"code": "hi", "title": "🌾 एग्रीस्मार्ट पौधा रोग पहचान",         "take_pic": "पत्ते की तस्वीर लें",               "upload_pic": "या फोटो अपलोड करें",              "pred_btn": "🔍 रोग पहचानें",               "speak_btn": "🔊 आवाज सुनें"},
    "Gujarati (ગુજરાતી)":    {"code": "gu", "title": "🌾 એગ્રીસ્માર્ટ પાક રોગ ડિટેક્ટર",      "take_pic": "છોડના પાંદડાનો ફોટો લો",           "upload_pic": "અથવા ઈમેજ અપલોડ કરો",            "pred_btn": "🔍 રોગ ઓળખો",                 "speak_btn": "🔊 અવાજ સાંભળો"},
    "Tamil (தமிழ்)":         {"code": "ta", "title": "🌾 அக்ரிஸ்மார்ட் பயிர் நோய் கண்டறிதல்", "take_pic": "இலையின் புகைப்படம் எடுக்கவும்",   "upload_pic": "அல்லது படத்தை பதிவேற்றவும்",    "pred_btn": "🔍 நோயை கண்டறியவும்",          "speak_btn": "🔊 கண்டறிதலை கேளுங்கள்"},
    "Telugu (తెలుగు)":       {"code": "te", "title": "🌾 అగ్రిస్మార్ట్ పంట వ్యాధి గుర్తింపు",  "take_pic": "ఆకు ఫోటో తీయండి",               "upload_pic": "లేదా చిత్రాన్ని అప్‌లోడ్ చేయండి", "pred_btn": "🔍 వ్యాధిని గుర్తించండి",       "speak_btn": "🔊 నిర్ణయాన్ని వినండి"},
    "Marathi (मराठी)":       {"code": "mr", "title": "🌾 अॅग्रीस्मार्ट पीक रोग ओळखणे",         "take_pic": "पानाचा फोटो घ्या",                "upload_pic": "किंवा प्रतिमा अपलोड करा",        "pred_btn": "🔍 रोग ओळखा",                  "speak_btn": "🔊 निदान ऐका"},
    "Bengali (বাংলা)":       {"code": "bn", "title": "🌾 অ্যাগ্রিস্মার্ট ফসল রোগ শনাক্তকরণ",  "take_pic": "পাতার ছবি তুলুন",               "upload_pic": "অথবা ছবি আপলোড করুন",           "pred_btn": "🔍 রোগ শনাক্ত করুন",           "speak_btn": "🔊 রোগ নির্ণয় শুনুন"},
    "Kannada (ಕನ್ನಡ)":      {"code": "kn", "title": "🌾 ಅಗ್ರಿಸ್ಮಾರ್ಟ್ ಬೆಳೆ ರೋಗ ಪತ್ತೆ",       "take_pic": "ಎಲೆಯ ಫೋಟೋ ತೆಗೆಯಿರಿ",            "upload_pic": "ಅಥವಾ ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",       "pred_btn": "🔍 ರೋಗ ಪತ್ತೆ ಮಾಡಿ",            "speak_btn": "🔊 ರೋಗ ನಿರ್ಣಯ ಕೇಳಿ"},
    "Punjabi (ਪੰਜਾਬੀ)":     {"code": "pa", "title": "🌾 ਐਗਰੀਸਮਾਰਟ ਫ਼ਸਲ ਰੋਗ ਖੋਜ",            "take_pic": "ਪੱਤੇ ਦੀ ਫ਼ੋਟੋ ਲਓ",              "upload_pic": "ਜਾਂ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ",          "pred_btn": "🔍 ਰੋਗ ਲੱਭੋ",                  "speak_btn": "🔊 ਨਿਦਾਨ ਸੁਣੋ"},
}

# All crop types across 24 model classes (sidebar + threshold dicts)
ALL_CROP_TYPES = [
    "tomato", "potato", "corn", "grape", "apple", "pepper",
    "strawberry", "blueberry", "cherry", "peach", "raspberry", "soybean", "squash",
]

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "sensors"           not in st.session_state: st.session_state.sensors           = None
if "irrigation_log"    not in st.session_state: st.session_state.irrigation_log    = []
if "leaf_crops"        not in st.session_state: st.session_state.leaf_crops        = None
if "leaf_boxes"        not in st.session_state: st.session_state.leaf_boxes        = None
if "is_fallback"       not in st.session_state: st.session_state.is_fallback       = False
if "detection_done"    not in st.session_state: st.session_state.detection_done    = False
if "selected_crop_idx" not in st.session_state: st.session_state.selected_crop_idx = 0
if "final_result"      not in st.session_state: st.session_state.final_result      = None
if "detected_crop_type" not in st.session_state: st.session_state.detected_crop_type = None
if "_lang_code"        not in st.session_state: st.session_state["_lang_code"]     = "en"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
selected_lang = st.sidebar.selectbox("🌐 Choose Language / भाषा चुनें", list(LANGUAGES.keys()))
lang = LANGUAGES[selected_lang]

# Keep _lang_code in session_state so t() helper always knows the current language
st.session_state["_lang_code"] = lang["code"]

# ── One-time warning if Sarvam API is unavailable ──────────────────────────
if st.session_state.get("_sarvam_disabled"):
    st.sidebar.warning(st.session_state.get(
        "_sarvam_error_msg",
        "⚠️ Translation unavailable. Showing content in English."
    ))

st.sidebar.markdown("---")
st.sidebar.subheader(t("🌱 Farm Details"))

# Auto-update the sidebar crop selector when YOLO+classifier detects a different crop.
detected_type = st.session_state.detected_crop_type
default_crop_idx = ALL_CROP_TYPES.index(detected_type) if (detected_type and detected_type in ALL_CROP_TYPES) else 0

crop_type = st.sidebar.selectbox(
    t("Crop"),
    ALL_CROP_TYPES,
    index=default_crop_idx,
    help=t("Auto-updates to match the detected plant after classification."),
)

if detected_type and detected_type == crop_type and st.session_state.final_result is not None:
    st.sidebar.caption(t(f"🤖 Auto-set to **{crop_type}** based on leaf detection."))

growth_stage = st.sidebar.selectbox(t("Growth Stage"), ["seedling", "vegetative", "flowering", "maturity"])
lat = st.sidebar.number_input(t("Latitude"),  value=23.02, format="%.4f")
lon = st.sidebar.number_input(t("Longitude"), value=72.57, format="%.4f")

recommended_fert = get_recommended_fertilizer(crop_type, growth_stage)
st.sidebar.caption(t(f"Recommended for {crop_type} ({growth_stage}): {recommended_fert} kg"))
actual_fertilizer_kg = st.sidebar.number_input(
    t("Fertilizer applied this week (kg)"),
    value=float(recommended_fert),
    min_value=0.0,
    key=f"fert_{crop_type}_{growth_stage}",
)

# ---------------------------------------------------------------------------
# Main title + image input
# ---------------------------------------------------------------------------
st.title(lang["title"])

uploaded_file = st.file_uploader(
    lang["upload_pic"],
    type=["jpg", "jpeg", "png"],
    key="leaf_image_uploader",
)

if uploaded_file is not None:
    st.session_state["active_image_bytes"] = uploaded_file.getvalue()

image_bytes = st.session_state.get("active_image_bytes")

# Reset pipeline state ONLY when a genuinely new image is loaded
if image_bytes is not None:
    img = Image.open(BytesIO(image_bytes))
    current_img_id = hashlib.md5(image_bytes).hexdigest()
    if st.session_state.get("_last_img_id") != current_img_id:
        st.session_state["_last_img_id"]    = current_img_id
        st.session_state.leaf_crops         = None
        st.session_state.leaf_boxes         = None
        st.session_state.is_fallback        = False
        st.session_state.detection_done     = False
        st.session_state.selected_crop_idx  = 0
        st.session_state.final_result       = None

    st.image(img, caption=t("Selected Leaf Image"), use_container_width=True)

    # =========================================================================
    # PHASE 1 — Leaf Detection (YOLO)
    # =========================================================================
    if not st.session_state.detection_done:
        if st.button(lang["pred_btn"], type="primary"):
            try:
                with st.spinner(t("🔍 Detecting leaves in image...")):
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        img.convert("RGB").save(tmp.name)
                        tmp_path = tmp.name

                    original_img, crops, boxes, is_fallback = detect_and_crop(tmp_path)

                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                st.session_state.leaf_crops     = crops
                st.session_state.leaf_boxes     = boxes
                st.session_state.is_fallback    = is_fallback
                st.session_state.detection_done = True

                if is_fallback:
                    st.info(t("ℹ️ No clear leaf region detected — using the full image."))
                else:
                    st.success(t(f"✅ Found {len(crops)} leaf region(s)."))

                if len(crops) == 1:
                    st.session_state.selected_crop_idx = 0
                    with st.spinner(t("🧠 Classifying leaf...")):
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
                    st.rerun()
            except Exception as _detect_err:
                st.error(t(f"❌ Detection failed: {_detect_err}"))
                st.exception(_detect_err)

    # =========================================================================
    # PHASE 1b — Gallery (only when 2+ crops detected)
    # =========================================================================
    if (
        st.session_state.detection_done
        and st.session_state.final_result is None
        and st.session_state.leaf_crops is not None
        and len(st.session_state.leaf_crops) >= 2
    ):
        crops = st.session_state.leaf_crops
        selected_idx = st.session_state.selected_crop_idx

        st.markdown("---")
        st.subheader(t(f"🌿 Select a Leaf to Diagnose ({len(crops)} detected)"))
        st.caption(
            t("Click 'Select' on the leaf you want diagnosed, then press Classify. ")
            + t(f"Currently selected: Leaf {selected_idx + 1}")
        )

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
                    caption  = t(f"✅ Leaf {global_idx + 1} — SELECTED") if is_selected else t(f"Leaf {global_idx + 1}")
                    st.image(crop_img, caption=caption, use_container_width=True)
                    btn_label = t("☑ Selected") if is_selected else t(f"Select Leaf {global_idx + 1}")
                    if st.button(btn_label, key=f"sel_{global_idx}", disabled=is_selected):
                        st.session_state.selected_crop_idx = global_idx
                        st.rerun()

        st.markdown("")
        chosen_idx = st.session_state.selected_crop_idx
        if st.button(t(f"🔬 Classify Leaf {chosen_idx + 1} →"), type="primary"):
            chosen_crop = crops[chosen_idx]
            try:
                with st.spinner(t(f"🧠 Classifying Leaf {chosen_idx + 1}...")):
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
                st.error(t(f"❌ Classification failed: {_cls_err}"))
                st.exception(_cls_err)

    # =========================================================================
    # PHASE 2 — Results + Downstream Modules
    # =========================================================================
    if st.session_state.final_result is not None:
        result     = st.session_state.final_result
        label      = result["label"]
        confidence = result["confidence"]
        severity   = result["severity"]
        raw_label  = result["raw_label"]

        detected_crop  = result["crop_type"]
        effective_crop = detected_crop if (detected_crop in ALL_CROP_TYPES) else crop_type

        st.markdown("---")
        st.success(t("✅ Classification complete!"))

        # ---- Confidence gating ----
        CONFIDENCE_THRESHOLD_DISPLAY = 0.70
        label_translated = t(label)   # translate disease name once, reuse everywhere

        if confidence >= CONFIDENCE_THRESHOLD_DISPLAY:
            st.metric(label=t("🌿 Diagnosed Condition"), value=label_translated)
            st.metric(label=t("Confidence"),             value=f"{confidence * 100:.2f}%")
        else:
            st.warning(
                t(f"⚠️ Image lighting or focus is uncertain ({confidence * 100:.1f}% confidence).") + "\n\n"
                + t(f"Possible match: {label}") + "\n\n"
                + t("Tip: For best accuracy, take a clear photo in bright daylight with the leaf filling the center of the frame.")
            )
            st.metric(label=t("Possible Condition"), value=label_translated)
            st.metric(label=t("Confidence (low)"),   value=f"{confidence * 100:.2f}%")

        # Auto-update notice
        if detected_crop in ALL_CROP_TYPES and detected_crop != crop_type:
            st.info(
                t(f"🤖 Crop auto-detected as '{detected_crop}' based on the leaf classification. "
                  f"The sidebar has been updated — all advice below uses {detected_crop}. "
                  f"You can override this manually in the sidebar at any time.")
            )

        # ---- Stage 2: Simulated IoT sensor reading (Bonus F) ----
        if st.session_state.sensors is None or st.session_state.sensors.crop_type != effective_crop:
            st.session_state.sensors = SimulatedFarmSensors(crop_type=effective_crop)
        reading = st.session_state.sensors.read()

        st.subheader(t("📡 Farm Sensors (IoT)"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("Soil Moisture"), f"{reading['soil_moisture_pct']}%")
        c2.metric(t("Temperature"),   f"{reading['temperature_c']}°C")
        c3.metric(t("Humidity"),      f"{reading['humidity_pct']}%")
        c4.metric(t("Soil pH"),       f"{reading['ph']}")

        # ---- Stage 3: Weather forecast (Bonus C) ----
        st.subheader(t("⛅ Weather Forecast (Open-Meteo)"))
        try:
            weather = get_forecast(lat, lon)
        except Exception as e:
            st.warning(t(f"Could not reach weather API ({e}) — using fallback demo values."))
            weather = {
                "rain_probability_pct": 40,
                "rain_expected_mm":     2.0,
                "temperature_c":        reading["temperature_c"],
                "humidity_pct":         reading["humidity_pct"],
                "leaf_wetness_hours":   3,
            }

        w1, w2, w3 = st.columns(3)
        w1.metric(t("Rain Chance"),   f"{weather['rain_probability_pct']}%")
        w2.metric(t("Rain Expected"), f"{weather['rain_expected_mm']} mm")
        w3.metric(t("Leaf Wetness"),  f"{weather['leaf_wetness_hours']} h")

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

        irrigation_action_text = t(irrigation["action"].replace("_", " ").title())
        irrigation_reason_text = t(irrigation["reason"])

        st.subheader(t("💧 Irrigation Advice"))
        st.info(f"**{irrigation_action_text}** — {irrigation_reason_text}")

        # ---- Stage 5: Disease risk assessment (Bonus C) ----
        disease_risk = disease_risk_assessment(
            humidity_pct=weather["humidity_pct"],
            temperature_c=weather["temperature_c"],
            leaf_wetness_hours=weather["leaf_wetness_hours"],
            recent_rain_mm=weather["rain_expected_mm"],
            crop_type=effective_crop,
            detected_disease=label,
            severity=severity,
        )

        risk_level_text  = t(disease_risk["risk_level"].upper())
        risk_action_text = t(disease_risk["action"])

        st.subheader(t("⚠️ Disease Risk"))
        risk_display = {"normal": st.success, "raised": st.warning, "high": st.error}
        risk_display.get(disease_risk["risk_level"], st.info)(
            f"**{risk_level_text}** — {t('action')}: {risk_action_text}"
        )
        if disease_risk["reasons"]:
            for r in disease_risk["reasons"]:
                st.caption(f"• {t(r)}")

        # ---- Stage 6: Sustainability score (Bonus D) ----
        recommended_fert_for_result = get_recommended_fertilizer(effective_crop, growth_stage)
        sustainability = sustainability_score(
            irrigation_log=st.session_state.irrigation_log,
            severity=severity,
            actual_fertilizer_kg=actual_fertilizer_kg,
            recommended_fertilizer_kg=recommended_fert_for_result,
        )

        sustainability_suggestion = t(sustainability["suggestion"])
        sustainability_method     = t(sustainability["method"])

        st.subheader(t("📊 Sustainability Score"))
        st.metric(t("Overall Score"), f"{sustainability['sustainability_score']} / 100")
        s1, s2, s3 = st.columns(3)
        s1.metric(t("Water"),       sustainability["sub_scores"]["water"])
        s2.metric(t("Resource"),    sustainability["sub_scores"]["resource"])
        s3.metric(t("Crop Health"), sustainability["sub_scores"]["health"])
        st.caption(f"💡 {sustainability_suggestion}")

        # ---- Stage 7: Voice readout (Bonus E) ----
        st.subheader(t("🔊 Farmer Voice Summary"))
        confidence_word = t("high") if confidence >= 0.70 else t("low")

        # Build speech text in English first, then translate the whole block once
        speech_text_en = (
            f"The predicted condition is {label} with {confidence * 100:.1f} percent confidence "
            f"({('high' if confidence >= 0.70 else 'low')} confidence). "
            f"Irrigation advice: {irrigation['reason']}. "
            f"Disease risk is {disease_risk['risk_level']}. "
            f"Your sustainability score this week is {sustainability['sustainability_score']} out of 100. "
            f"{sustainability['suggestion']}"
        )
        speech_text = t(speech_text_en)   # translate the full summary in one call

        if st.button(lang["speak_btn"]):
            with st.spinner(t("Generating audio...")):
                tts = gTTS(text=speech_text, lang=lang["code"])
                sound_fp = BytesIO()
                tts.write_to_fp(sound_fp)
                st.audio(sound_fp, format="audio/mp3", autoplay=True)

        # ---- Reset button ----
        st.markdown("---")
        if st.button(t("🔄 Analyse Another Image")):
            # Clear image/detection pipeline state
            st.session_state.leaf_crops          = None
            st.session_state.leaf_boxes          = None
            st.session_state.is_fallback         = False
            st.session_state.detection_done      = False
            st.session_state.selected_crop_idx   = 0
            st.session_state.final_result        = None
            st.session_state["_last_img_id"]     = None
            st.session_state["active_image_bytes"] = None
            # Also clear downstream module state so the next image
            # starts fresh (otherwise irrigation_log accumulates, sensors
            # keep stale crop, and sustainability scores are wrong)
            st.session_state.irrigation_log      = []
            st.session_state.sensors             = None
            st.session_state.detected_crop_type  = None
            st.rerun()