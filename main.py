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

# Function to prepare the input image for the MobileNetV2 model
def preprocess_image(image):

    # Convert the PIL Image object into a NumPy array for processing
    img = np.array(image)

    # Resize the image to 224x224 pixels, the required input size for MobileNetV2
    img = cv2.resize(img, (224, 224))

    # Apply MobileNetV2-specific preprocessing (e.g., scaling pixel values)
    img = preprocess_input(img)

    # Add an extra dimension (batch dimension) to make the array shape (1, 224, 224, 3)
    # This is required because the model expects a batch of images, even if it's just one.
    img = np.expand_dims(img, axis=0)

    # Return the fully preprocessed image array
    return img



