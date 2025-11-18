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


def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")
    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")

    @st.cache_resource
    def load_cached_model():
        return load_model()
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        Image = st.image(
            uploaded_file, caption="Uploaded Image", use_column_width=True
        )
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image..."):
                Image = Image.open(uploaded_file)
                predictions = classify_image(image)
                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")



