import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="Real-Time Face & Lighting Detector", layout="wide")

st.title("🎥 Real-Time Face Detection with Lighting Analysis")
st.markdown("Detects faces and evaluates lighting condition (Low / Normal / Bright) in real time.")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Lighting detection function
def analyze_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 70:
        return "Low Light", brightness
    elif brightness > 170:
        return "Too Bright", brightness
    else:
        return "Normal", brightness

# Video Transformer Class
class FaceLightingDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw face boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Lighting analysis
        lighting_status, brightness = analyze_lighting(img)

        # Overlay text
        cv2.putText(
            img,
            f"Lighting: {lighting_status} ({brightness:.0f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            img,
            f"Faces Detected: {len(faces)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        return img

# WebRTC Configuration
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.sidebar.header("⚙️ Settings")
st.sidebar.write("This app uses Haar Cascade for face detection.")
st.sidebar.write("Lighting is calculated using average grayscale brightness.")

webrtc_streamer(
    key="face-lighting-detector",
    video_transformer_factory=FaceLightingDetector,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
)
