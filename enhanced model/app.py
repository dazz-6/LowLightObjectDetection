import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import av
import time

# ==============================
# LOAD MODEL
# ==============================
model = YOLO("best.pt")

st.title("Occlusion-Robust Real-Time Object Detection")

# ==============================
# OPTIONAL CLAHE ENHANCEMENT
# ==============================
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

# ==============================
# VIDEO TRANSFORMER
# ==============================
class VideoTransformer(VideoTransformerBase):

    prev_time = 0

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Resize for speed
        img = cv2.resize(img, (640, 480))

        # OPTIONAL CLAHE
        # img = apply_clahe(img)

        # YOLO DETECTION
        results = model(img, conf=0.4)

        annotated = results[0].plot()

        # FPS Calculation
        curr_time = time.time()

        fps = (
            1 / (curr_time - self.prev_time)
            if self.prev_time != 0
            else 0
        )

        self.prev_time = curr_time

        cv2.putText(
            annotated,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return annotated

# ==============================
# LIVE WEBRTC STREAM
# ==============================
webrtc_streamer(
    key="yolov8-live",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)