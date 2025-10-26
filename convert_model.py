import streamlit as st
import av
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import tensorflow as tf

# --- PAGE SETTINGS ---
st.set_page_config(page_title="🎥 AI Object Detector", layout="wide")
st.title("🎥 AI Object Detection (Teachable Machine)")
st.write("Show any object or person to the webcam — or upload an image!")

# --- LOAD TFLITE MODEL ---
interpreter = tf.lite.Interpreter(model_path="models/object_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("models/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- HELPER FUNCTION FOR PREDICTION ---
def predict(img_array):
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(predictions)
    pred_label = labels[pred_idx]
    confidence = np.max(predictions) * 100
    display_text = f"It's a {pred_label}! ({confidence:.0f}% sure)"
    return display_text

# --- VIDEO PROCESSOR FOR WEBCAM ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = Image.fromarray(img).resize((224, 224))
        img_array = np.array(img_resized)
        display_text = predict(img_array)

        # Draw friendly text on frame
        cv2.putText(img, display_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
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

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Or upload an image to detect objects", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    display_text = predict(img_array)

    st.image(img, caption=display_text, use_column_width=True)

st.caption("🔹 Move objects in front of the camera or upload an image to test your AI model.")
