# ---------------------------
# Import required libraries
# ---------------------------

# OpenCV for image resizing and processing
import cv2

# NumPy for array operations
import numpy as np

# Streamlit for building the web UI
import streamlit as st

# MobileNetV2 model and helper functions from TensorFlow/Keras
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# PIL for opening uploaded images
from PIL import Image as PILImage



# ---------------------------
# Load the MobileNetV2 model
# ---------------------------
def load_model():
    """
    Loads the MobileNetV2 model pre-trained on ImageNet.
    This model can classify images into 1,000 categories.
    """
    model = MobileNetV2(weights="imagenet")
    return model



# ------------------------------------------
# Preprocess image so the model can use it
# ------------------------------------------
def preprocess_image(image):
    """
    Takes a PIL Image and prepares it for MobileNetV2 by:
    - converting to NumPy array
    - resizing to 224x224
    - applying MobileNet preprocessing (scaling)
    - adding batch dimension
    """

    # Convert PIL image to NumPy array
    img = np.array(image)

    # Resize to MobileNetV2 input size
    img = cv2.resize(img, (224, 224))

    # Apply MobileNetV2 image preprocessing
    img = preprocess_input(img)

    # Add batch dimension (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    return img



# ------------------------------------------
# Use the model to classify the image
# ------------------------------------------
def classify_image(model, image):
    """
    Runs the image through the model and returns the top 3 predictions.
    Handles errors gracefully.
    """
    try:
        processed_image = preprocess_image(image)   # Prepare image
        predictions = model.predict(processed_image) # Run model inference
        decoded_predictions = decode_predictions(predictions, top=3)[0] # Convert to labels

        return decoded_predictions

    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None



# ---------------------------
# Main Streamlit Application
# ---------------------------
def main():

    # Configure Streamlit page layout and title
    st.set_page_config(
        page_title="AI Image Classifier",
        page_icon="🖼️",
        layout="centered"
    )

    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what's inside!")



    # Cache the model so it's only loaded once
    @st.cache_resource
    def load_cached_model():
        """
        Loads and caches the deep learning model.
        Prevents reloading on every user interaction.
        """
        return load_model()

    model = load_cached_model()



    # ---------------------------
    # Image Upload Interface
    # ---------------------------
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png"]
    )



    # When user uploads an image
    if uploaded_file is not None:

        # Display the uploaded image
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Button to classify the image
        if st.button("Classify Image"):

            with st.spinner("Analyzing Image..."):

                # Load image using PIL
                image = PILImage.open(uploaded_file)

                # Perform classification
                predictions = classify_image(model, image)

                # If predictions successful, display results
                if predictions:
                    st.subheader("Predictions")

                    # Format each prediction nicely
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")



# Run the app
if __name__ == "__main__":
    main()
