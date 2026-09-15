import streamlit as st
import tensorflow as tf
import numpy as np

# EDIT THIS if your model file lives elsewhere relative to the app.
# Keep it relative (not C:\Users\...) so it works on any machine/judge's laptop.
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
    st.cache_resource makes Streamlit load the model file ONCE per session
    and reuse it across reruns/button clicks, instead of reloading from disk
    every time (which is what the previous register_keras_serializable
    decorator was accidentally NOT preventing).
    """
    return tf.keras.models.load_model(MODEL_PATH)


model = get_model()


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
    """
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
    """
    img = pil_image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]   # shape: (num_classes,)

    predicted_index = int(np.argmax(predictions))
    predicted_class = TARGET_CLASSES[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_index, predicted_class, confidence, predictions