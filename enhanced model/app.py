# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Page config
st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")

# Title
st.title("🚀 YOLOv8 Object Detection App")
st.write("Upload an image and detect objects using your trained model")

# Load model (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Make sure best.pt is in same folder
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # Detection button
    if st.button("🔍 Detect Objects"):
        with st.spinner("Running detection..."):
            results = model(temp_path)

            # Plot results
            result_image = results[0].plot()

            # Show output
            st.image(result_image, caption="✅ Detected Output", use_column_width=True)

            # Optional: show detected classes
            boxes = results[0].boxes
            if boxes is not None:
                classes = boxes.cls.tolist()
                st.write("### 📊 Detected Class IDs:")
                st.write(classes)
