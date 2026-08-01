import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Configuration ---
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
class_names = ['Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy']

# --- Load the Model ---
@st.cache_resource
def load_model():
    model_path = 'models/custom_cnn.keras'
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}. Please ensure it's in your repository.")
        st.stop()
    # Using tf.keras.models.load_model is crucial for Keras models saved with TensorFlow 2.x
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# --- Prediction Function ---
def predict_image_streamlit(img, model, class_names, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    img_resized = img.resize(image_size)
    img_array = tf.keras.utils.img_to_array(img_resized)  # [0, 255] float32
    img_array = np.expand_dims(img_array, axis=0)       # (1, H, W, 3)

    # The custom CNN model includes the Rescaling layer internally, so no external preprocessing here.
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    prob = predictions[0][0] # Assuming it outputs a single sigmoid probability for binary classification

    # Based on the notebook's predict_image, if prob >= 0.5, it's class_names[1] (healthy)
    if prob >= 0.5:
        label = class_names[1] # Tomato___healthy
        confidence = prob
    else:
        label = class_names[0] # Tomato___Tomato_Yellow_Leaf_Curl_Virus
        confidence = 1 - prob

    return label, confidence

# --- Streamlit App ---
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
# This Streamlit app loads the saved 'custom_cnn.keras' model.
""")
'''

# Define the path in Google Drive
drive_path = '/content/drive/MyDrive/'
streamlit_file_path = os.path.join(drive_path, 'streamlit.py')

# Write the Streamlit app code to the file
with open(streamlit_file_path, 'w') as f:
    f.write(streamlit_app_code_cnn)

print(f'New streamlit.py (using custom CNN) saved to {streamlit_file_path}')
