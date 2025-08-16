import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cd_model.h5")

model = load_model()

st.title("🐱🐶 Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess (resize to model input size, e.g., 224x224)
    img = image.resize((224, 224))   # adjust to your training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(img_array)
        
        # Assuming binary classification [cat=0, dog=1]
        label = np.argmax(prediction, axis=1)[0]   # 0=Cat, 1=Dog
        if label == 0:
            st.success("It's a **Cat 🐱**")
        else:
            st.success("It's a **Dog 🐶**")