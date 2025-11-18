# Import the computer vision library
import cv2
# Import the numerical operations library (arrays)
import numpy as np
# Import Streamlit for creating the web application UI
import streamlit as st
# Import MobileNetV2 for image classification and necessary utilities
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,      # The classification model
    preprocess_input, # Function to prepare image for the model
    decode_predictions # Function to interpret model output
)
# Import Image from PIL for image file handling
from PIL import Image

# Function to load the deep learning model
def load_model():
    # Load MobileNetV2 with weights trained on the ImageNet dataset
    model = MobileNetV2(weights="imagenet")
    # Return the ready-to-use model
    return model

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        precessed_image = preprocess_image(image)
        predictions = model.predict(preprocess_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    
    except Exception as e:
        st.error(f"Error classifying image: {srt(e)}")

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

    st.title("AI Image Classifier")
    

