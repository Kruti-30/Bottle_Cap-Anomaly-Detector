import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os

# Load the TFLite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Get input & output details of the model
def get_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

# Load class labels from labels.txt
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Preprocess the uploaded image
def preprocess_image(image, input_shape):
    image = image.convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    return image_array

# Predict the class of the image
def predict(image_array, interpreter, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit UI
st.set_page_config(page_title="Bottle Cap Anomaly Detector", layout="centered")
st.title("🧪 Bottle Cap Anomaly Detector")
st.markdown("Upload an image of a bottle cap to detect if it is normal or anomalous.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Model and label paths
model_path = "model.tflite"
label_path = "labels.txt"

# Check for model and label files
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at `{model_path}`. Please ensure it's placed correctly.")
    st.stop()

if not os.path.exists(label_path):
    st.error(f"❌ Labels file not found at `{label_path}`. Please ensure it's placed correctly.")
    st.stop()

# Load model and labels
interpreter = load_model(model_path)
input_details, output_details = get_io_details(interpreter)
labels = load_labels(label_path)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    resized_image = image.resize((300, 300))  # Resize to 300x300 pixels (adjust as needed)
    st.image(resized_image, caption='Uploaded Image')


    with st.spinner('🧠 Analyzing the image...'):
        image_array = preprocess_image(image, input_details[0]['shape'])
        predictions = predict(image_array, interpreter, input_details, output_details)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        label = labels[predicted_index] if predicted_index < len(labels) else "Unknown"
        time.sleep(1)

    st.success(f"✅ Prediction: **{label}** with **{confidence:.2f}%** confidence")
