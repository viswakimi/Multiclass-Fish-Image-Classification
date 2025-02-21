
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

# Function to load the model with caching for better performance
@st.cache_resource
def load_best_model():
    model_path = r"D:\\App\\vscode1\\fishimage\Dataset\\images.cv_jzk6llhf18tm3k0kyttxz\\data\\best_fish_model.h5"
    return load_model(model_path)

# Load the model
best_model = load_best_model()

# Class labels (Ensure these match the dataset used for training)
class_names = [
    "Fish", "Fish Bass", "Black Sea Spart", "Gilt Heard Bream", "Hourse Mackerel",
    "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet", "Trout"
]

# Image Preprocessing Function
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((224, 224))  # Resize correctly
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand batch dimension
    return image    

# Streamlit UI
st.title("🐠 Fish Classification App")
st.write("Upload images of fish, and the model will predict their types!")

# Sidebar UI
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)", min_value=0, max_value=100, value=50, step=5
)

# Upload multiple images
uploaded_files = st.file_uploader("📷 Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()  # Add a separator between images
        st.subheader(f"📷 Image: {uploaded_file.name}")

        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        # Preprocess image
        processed_image = preprocess_image(image)

        # Predict
        with st.spinner("🔍 Classifying..."):
            prediction = best_model.predict(processed_image)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        # Display Prediction
        if confidence >= confidence_threshold:
            st.success(f"🎯 **Prediction: {predicted_class}**")
            st.write(f"🔵 **Confidence:** {confidence:.2f}%")
        else:
            st.warning("⚠️ Prediction confidence is too low. Try uploading a clearer image.")

        # Show class probabilities
        st.subheader("📊 Class Probabilities")
        prob_dict = {class_names[i]: f"{pred*100:.2f}%" for i, pred in enumerate(prediction[0])}
        st.json(prob_dict)

        # Show class probabilities bar chart
        prob_df = pd.DataFrame({"Class": class_names, "Confidence (%)": prediction[0] * 100})
        prob_df = prob_df.sort_values(by="Confidence (%)", ascending=False)
        st.bar_chart(prob_df.set_index("Class"))
