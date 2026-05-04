# app.py (UPGRADED VERSION)

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Page config
st.set_page_config(page_title="YOLOv8 Detection App", layout="centered")

# Title
st.title("🚀 YOLOv8 Smart Object Detection")
st.markdown("Upload an image and detect objects with adjustable confidence threshold")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Confidence slider
confidence = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.25)

# File uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    if st.button("🔍 Run Detection"):
        with st.spinner("Detecting objects..."):
            results = model(temp_path, conf=confidence)

            # Show image with detections
            result_img = results[0].plot()
            st.image(result_img, caption="✅ Detection Result", use_column_width=True)

            # Show class names instead of IDs
            if results[0].boxes is not None:
                names = model.names
                detected_classes = [names[int(cls)] for cls in results[0].boxes.cls.tolist()]

                st.subheader("📊 Detected Objects")
                st.write(detected_classes)

                # Count occurrences
                counts = {}
                for item in detected_classes:
                    counts[item] = counts.get(item, 0) + 1

                st.subheader("📈 Object Count")
                st.write(counts)
