import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Configuration --- #
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
class_names = ['Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy']

# --- Load the Model --- #
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    model_path = "models/mobilenetv3_transfer.keras"
    # Native Keras 3 load without compilation constraints
    return tf.keras.models.load_model(model_path, compile=False)

st.title("Tomato Plant Disease Detector")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    import keras

# Load directly with standalone Keras 3
model = keras.models.load_model("path_to_your_model.keras")
# --- Prediction Function (similar to your notebook) ---
def predict_image_streamlit(img, model, class_names, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    img_resized = img.resize(image_size)
    img_array = tf.keras.utils.img_to_array(img_resized)  # [0, 255] float32
    img_array = np.expand_dims(img_array, axis=0)       # (1, H, W, 3)

    # Preprocess for MobileNetV3 if the model expects it
    # Assuming the loaded model already has the preprocessing layer, otherwise add it:
    # from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    # img_array = preprocess_input(img_array)

    prob  = model.predict(img_array, verbose=0)[0][0]
    # Assuming binary classification where prob < 0.5 is class_names[0] and prob >= 0.5 is class_names[1]
    # Adjust this logic if your model outputs probabilities for each class (e.g., softmax)
    # For softmax output for 2 classes, prob would be [prob_class0, prob_class1]
    # If it's a sigmoid output for class_names[1], then:
    # label_index = int(prob >= 0.5)
    # label = class_names[label_index]

    # Based on the notebook's predict_image, it's a sigmoid output for class_names[1]
    if prob >= 0.5:
        label = class_names[1] # Tomato___healthy
        confidence = prob
    else:
        label = class_names[0] # Tomato___Tomato_Yellow_Leaf_Curl_Virus
        confidence = 1 - prob

    return label, confidence

# --- Streamlit App --- #
st.set_page_config(page_title="Tomato Disease Classifier", page_icon=":tomato:")

st.title("Disease Detection for Tomatoes :tomato:")
st.write("Upload an image of a tomato leaf, and I'll predict if it's healthy or has 'Tomato Yellow Leaf Curl Virus'.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label, confidence = predict_image_streamlit(image, model, class_names)

    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{confidence:.2f}**")

st.markdown("--- Source Code ---")
st.code("""
# Your original notebook code for model training and prediction
# This Streamlit app loads the saved 'mobilenetv3_transfer.keras' model.
""")
