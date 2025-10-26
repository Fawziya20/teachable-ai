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
st.write("Show any object or person to the webcam — or upload an image — to identify it!")

# --- LOAD TFLITE MODEL ---
interpreter = tf.lite.Interpreter(model_path="models/object_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Detect if model is quantized
input_dtype = input_details[0]['dtype']
quantized = input_dtype != np.float32
if quantized:
    scale, zero_point = input_details[0]['quantization']
    st.write("Quantized model detected.")
else:
    st.write("Float32 model detected.")

# Load labels
with open("models/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- VIDEO PROCESSOR ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = Image.fromarray(img).resize((224, 224))
        img_array = np.array(img_resized)

        if quantized:
            img_array = img_array.astype(np.uint8)
        else:
            img_array = (img_array / 255.0).astype(np.float32)

        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        label = labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        cv2.putText(img, f"{label} ({confidence:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- WEBCAM STREAM ---
st.subheader("📷 Webcam Detection")
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# --- IMAGE UPLOAD ---
st.subheader("🖼️ Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)

    if quantized:
        img_array = img_array.astype(np.uint8)
    else:
        img_array = (img_array / 255.0).astype(np.float32)

    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    label = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    st.success(f"Prediction: {label} ({confidence:.1f}%)")
