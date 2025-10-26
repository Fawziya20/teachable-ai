import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- PAGE SETTINGS ---
st.set_page_config(page_title="AI Object Detector", layout="wide")
st.title("🎥 AI Object Detection (Teachable Machine)")
st.write("Show any object or person to the webcam — or upload an image!")

# --- LOAD TFLITE MODEL ---
interpreter = tf.lite.Interpreter(model_path="models/object_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("models/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- HELPER FUNCTION ---
def predict_image(img: np.ndarray):
    img_resized = Image.fromarray(img).resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    label = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return f"Hey, it looks like a {label} ({confidence:.1f}%)!"

# --- VIDEO PROCESSOR ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        text = predict_image(img)

        # Draw text on frame
        cv2.putText(img, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- WEBCAM STREAM ---
st.subheader("📹 Live Webcam Detection (optional)")
try:
    webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        async_processing=True
    )
except Exception as e:
    st.warning("Webcam unavailable. You can still upload an image below.")

# --- IMAGE UPLOAD ---
st.subheader("🖼️ Upload an Image for Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)
    result_text = predict_image(img_array)
    st.image(img_array, caption="Uploaded Image", use_column_width=True)
    st.success(result_text)
