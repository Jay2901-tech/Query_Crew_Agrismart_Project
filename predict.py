"""
predict.py — Keras model loader + inference helpers

The model is loaded LAZILY (on first predict call), not at import time.
This prevents a blank crash page in Streamlit if the .keras file is
missing or TensorFlow fails to initialise — the app still starts and
shows a clear error only when the user actually tries to run detection.
"""

import streamlit as st
import tensorflow as tf
import numpy as np

# EDIT THIS if your model file lives elsewhere relative to the app.
# Keep it relative (not C:\\Users\\...) so it works on any machine/judge's laptop.
MODEL_PATH = "phase2_best_model.keras"

TARGET_CLASSES = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Grape___Black_rot",
    "Grape___healthy",
    "Peach___healthy",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


@st.cache_resource
def get_model():
    """
    Load the Keras model once per Streamlit session and cache it.
    Returns the model on success, or None if the file is missing /
    TensorFlow fails — callers must check for None before using.
    """
    import os
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file '{MODEL_PATH}' not found. "
            "Make sure the .keras file is in the same folder as app.py."
        )
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None          # (model, error_message)
    except Exception as e:
        return None, f"Failed to load model: {e}"


def _require_model():
    """
    Returns the loaded model, or raises a clear RuntimeError so the
    Streamlit UI can show a user-friendly message instead of a traceback.
    """
    model, err = get_model()
    if model is None:
        raise RuntimeError(err)
    return model


def clean_label(raw_label):
    """
    Normalize labels like 'Pepper,_bell___healthy' or
    'Corn_(maize)___Common_rust_' into readable, consistent strings.
    Used both for display AND for the healthy-check in map_prediction_to_severity,
    so it must be applied consistently everywhere.
    """
    label = raw_label.replace("___", " - ")
    label = label.replace(",", "")
    label = label.replace("_", " ")
    label = " ".join(label.split())  # collapse any double spaces
    return label.strip()


def predict_image(pil_image):
    """
    Accepts a PIL Image, pre-processes it, and returns prediction details.
    Returns: (predicted_index: int, predicted_class_raw: str, confidence: float)
    Raises RuntimeError with a human-readable message if the model is unavailable.
    """
    model = _require_model()

    img = pil_image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = TARGET_CLASSES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return predicted_index, predicted_class, confidence


def predict_image_full(pil_image):
    """
    Extended version of predict_image that also returns the full softmax
    probability vector. Used by the YOLO multi-crop pipeline to show per-crop
    confidence across all classes.

    Returns:
        (predicted_index: int,
         predicted_class_raw: str,
         confidence: float,
         softmax_vector: np.ndarray of shape (num_classes,))
    Raises RuntimeError with a human-readable message if the model is unavailable.
    """
    model = _require_model()

    img = pil_image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]   # shape: (num_classes,)

    predicted_index = int(np.argmax(predictions))
    predicted_class = TARGET_CLASSES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_index, predicted_class, confidence, predictions