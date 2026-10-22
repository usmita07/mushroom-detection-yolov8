import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="Mushroom Detection | YOLOv8s", page_icon="🍄", layout="wide")
MODEL_PATH = os.path.join("models", "best.pt")

st.title("🍄 Mushroom Object Detection")
st.write("Upload an image to run the YOLOv8s detector trained on **CoW** and **chanterelle** classes.")
st.warning("Educational demo only. Predictions may be incorrect. Never use this app to decide whether a mushroom is safe to eat.")

@st.cache_resource
def load_model(path):
    return YOLO(path)

if not os.path.exists(MODEL_PATH):
    st.error("Model weights not found. Add your trained checkpoint at `models/best.pt`, then redeploy.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.21, 0.01)
    image_size = st.selectbox("Image size", [640, 416], index=0)

uploaded_file = st.file_uploader("Choose a mushroom image", type=["jpg", "jpeg", "png", "webp", "jfif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with st.spinner("Detecting objects..."):
        result = model.predict(source=image, imgsz=image_size, conf=confidence, verbose=False)[0]
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Detection result")
        st.image(annotated_rgb, use_container_width=True)

    rows = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            rows.append({
                "Class": result.names.get(class_id, str(class_id)),
                "Confidence": round(float(box.conf.item()), 4),
                "X1": round(float(box.xyxy[0][0].item()), 1),
                "Y1": round(float(box.xyxy[0][1].item()), 1),
                "X2": round(float(box.xyxy[0][2].item()), 1),
                "Y2": round(float(box.xyxy[0][3].item()), 1),
            })
    st.subheader("Detection details")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.success(f"Detected {len(rows)} object(s).")
    else:
        st.info("No objects detected at this confidence threshold.")

st.divider()
st.caption("YOLOv8s | Classes: CoW, chanterelle | Evaluation set: 5 images / 9 instances; metrics are preliminary.")
