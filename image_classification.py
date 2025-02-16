# Streamlit Deployment
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
st.title("Fish Species Classification")
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).resize(IMG_SIZE)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    model = load_model(MODEL_SAVE_PATH)
    predictions = model.predict(image)
    class_names = list(train_gen.class_indices.keys())
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    st.write(f"Predicted Fish Category: {predicted_class}")
    st.write(f"Confidence Score: {confidence:.2f}")
