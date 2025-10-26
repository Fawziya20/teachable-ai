import streamlit as st
import av
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import tensorflow as tf

# --- PAGE SETTINGS ---
st.set_page_config(page_title="AI Object Detector", layout="wide")
st.title("🎥 AI Object Detection (Teachable Machine)")
st.write("Show any object or person to the webcam — or upload an image.")

# --- LOAD TFLITE MODEL ---
interpreter = tf.lite.Interpreter(model_path="models/object_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("models/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- VIDEO PROCESSOR ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize and normalize
        img_resized = Image.fromarray(img).resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        label = labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Draw text
        cv2.putText(img, f"{label} ({confidence:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- WEBCAM STREAM ---
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

# --- IMAGE UPLOAD FALLBACK ---
st.subheader("Or upload an image for detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    label = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.image(img, caption=f"Prediction: {label} ({confidence:.1f}%)", use_column_width=True)
