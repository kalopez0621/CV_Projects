
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Path to your saved model
MODEL_PATH = r"C:\Users\kalop\CV_Projects\HappyorSadPrediction\happy_sad_model_vgg16.keras"

# Load the trained model
model = load_model(MODEL_PATH)

# Constants for image preprocessing
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ["Sad", "Happy"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to match model input
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Streamlit UI
st.title("Happy or Sad Emotion Detector")
st.write("Upload an image, and the model will predict whether the emotion is Happy or Sad.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB format
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)[0][0]
    predicted_class = CLASS_NAMES[int(prediction > 0.5)]
    confidence = prediction if predicted_class == "Happy" else 1 - prediction

    # Display the result
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
