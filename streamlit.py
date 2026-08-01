import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
import keras

# --- Configuration & Setup ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CLASS_NAMES = ['Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy']

# --- Page Configuration ---
st.set_page_config(
    page_title="Tomato Plant Disease Detector",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🍅 Tomato Plant Disease Detector")
st.write("Upload an image of a tomato leaf to detect if it is healthy or infected with Yellow Leaf Curl Virus.")


# --- Custom Layer to Bypass Deserialization / Quantization Config Errors ---
class CustomDense(tf.keras.layers.Dense):
    """
    Custom Dense layer wrapper to filter out unrecognized arguments like
    'quantization_config' during model deserialization on Streamlit Cloud.
    """
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        # Filter out unrecognized deserialization kwargs
        kwargs.pop('quantization_config', None)
        super().__init__(units, activation=activation, use_bias=use_bias,
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, **kwargs)


# --- Safe Model Loader ---
@st.cache_resource
def load_model_file(model_path):
    try:
        # Using native keras loader with compile=False completely ignores 
        # training/quantization metadata causing the deserialization error.
        return keras.models.load_model(model_path, compile=False, safe_mode=False)
    except Exception as e:
        # Secondary fallback if Keras 3 native fails
        return tf.keras.models.load_model(
            model_path, 
            custom_objects={'Dense': tf.keras.layers.Dense}, 
            compile=False, 
            safe_mode=False
        )


def check_input_shape(name, model):
    """Warn in the sidebar if the model's expected shape drifts from app settings."""
    try:
        shape = model.input_shape  # e.g. (None, 128, 128, 3)
        expected = (shape[1], shape[2])
    except Exception:
        return
    if expected != (IMAGE_HEIGHT, IMAGE_WIDTH):
        st.sidebar.warning(
            f"⚠️ {name} expects input shape {expected}, but app resizes to "
            f"{(IMAGE_HEIGHT, IMAGE_WIDTH)}. Predictions may be unreliable."
        )


# --- Model Registry ---
MODEL_REGISTRY = [
    {"name": "Custom CNN", "path": "models/custom_cnn.keras"}
]

loaded_models = {}

with st.spinner("Loading model..."):
    for entry in MODEL_REGISTRY:
        name, path = entry["name"], entry["path"]
        if not os.path.exists(path):
            st.sidebar.warning(f"{name} model not found at '{path}'")
            continue
        try:
            model = load_model(path)
            check_input_shape(name, model)
            loaded_models[name] = model
            st.sidebar.success(f"Loaded {name}")
        except Exception as e:
            st.sidebar.error(f"Error loading {name} model: {e}")


# --- Sidebar Model Selection ---
st.sidebar.header("Model Selection")

model_to_use = None
selected_model_name = None

if loaded_models:
    selected_model_name = st.sidebar.radio(
        "Choose model for inference:",
        list(loaded_models.keys()),
        index=0
    )
    model_to_use = loaded_models[selected_model_name]
else:
    st.error("No valid models were loaded. Please verify model files in 'models/'.")


# --- Main Image Upload & Inference Flow ---
st.header("Upload Tomato Leaf Image")
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_to_use is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except UnidentifiedImageError:
        st.error("Invalid image format. Please upload a valid JPG or PNG file.")
        st.stop()
    except Exception as e:
        st.error(f"Could not read image file: {e}")
        st.stop()

    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")

    # Resize image to match training specification (128x128).
    # Raw [0, 255] float32 values are passed because the internal Rescaling layer processes them.
    img_array = np.array(image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)), dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

    try:
        with st.spinner("Analyzing image..."):
            predictions = model_to_use.predict(img_array)
    except Exception as e:
        st.error(f"Prediction execution failed: {e}")
        st.stop()

    # Calculate probabilities based on output layer configuration (Sigmoid vs Softmax)
    if model_to_use.output_shape[-1] == 1:
        score = float(predictions[0][0])
        predicted_class_index = 1 if score >= 0.5 else 0
        confidence = score if predicted_class_index == 1 else (1.0 - score)
        probs = [1.0 - score, score]
    else:
        probs = predictions[0]
        predicted_class_index = int(np.argmax(probs))
        confidence = float(probs[predicted_class_index])

    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence_pct = confidence * 100.0

    st.subheader("Analysis Results")
    
    # Display result alert
    if "healthy" in predicted_class_name.lower():
        st.success(f"🌱 Model Prediction: **Healthy Leaf**")
    else:
        st.error(f"⚠️ Model Prediction: **{predicted_class_name.replace('_', ' ')}**")
        
    st.metric("Confidence Score", f"{confidence_pct:.2f}%")

    # Output detailed raw probability dataframe
    st.write("Detailed Class Probabilities:")
    pred_df = pd.DataFrame({"Condition / Class": CLASS_NAMES, "Probability": probs})
    st.dataframe(pred_df.style.format({'Probability': '{:.4f}'}))


# --- Sidebar Information ---
st.sidebar.markdown("""
---
### Deployment Checklist:
1. GitHub repo directory layout:
    * `streamlit.py`
    * `models/custom_cnn.keras`
    * `requirements.txt` containing `streamlit`, `tensorflow`, `numpy`, `Pillow`, and `pandas`.
""")
