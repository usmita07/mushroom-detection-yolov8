
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from collections import Counter
import tempfile
import io

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Mushroom AI Detector",
    page_icon="🍄",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #aab4c3;
        margin-bottom: 25px;
    }
    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
    }
    .warning-box {
        padding: 14px;
        border-radius: 10px;
        background-color: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.45);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOCATION ----------------
MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🍄 Mushroom AI")
    st.write("YOLOv8s Object Detection")
    st.divider()

    st.subheader("Detection Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Higher values filter out more low-confidence detections."
    )

    image_size = st.selectbox(
        "Inference image size",
        options=[416, 640],
        index=1
    )

    st.divider()
    st.subheader("Model Information")
    st.write("**Architecture:** YOLOv8s")
    st.write("**Task:** Object detection")
    st.write("**Classes:** CoW, chanterelle")
    st.caption("Predictions are experimental and may be incorrect.")

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🍄 Mushroom Detection Studio</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered mushroom object detection using YOLOv8s</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="warning-box">
<b>⚠️ Educational demonstration only.</b>
This model can make incorrect predictions. Never use it to determine
whether a mushroom is safe to eat.
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))

if not MODEL_PATH.is_file():
    st.error(
        "Model weights not found. Please ensure that "
        "`models/best.pt` exists in the project folder."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load the YOLO model: {error}")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a mushroom photograph",
    type=["jpg", "jpeg", "png", "webp"],
    help="For the clearest test, upload a photograph rather than a screenshot."
)

if uploaded_file is None:
    st.info("👆 Upload an image above to begin detection.")

    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1. Upload")
        st.write("Choose a mushroom image from your device.")

    with col2:
        st.markdown("#### 2. Detect")
        st.write("YOLOv8s predicts object locations and classes.")

    with col3:
        st.markdown("#### 3. Review")
        st.write("Inspect the predictions and download the result.")

else:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(
            original_image,
            caption="Original image",
            use_container_width=True
        )

        run_detection = st.button(
            "🔍 Run Mushroom Detection",
            type="primary",
            use_container_width=True
        )

        if run_detection:
            with st.spinner("Analyzing image..."):
                # Use a temporary image file for inference.
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp_file:
                    original_image.save(temp_file.name, format="JPEG")
                    temp_path = temp_file.name

                try:
                    results = model.predict(
                        source=temp_path,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False
                    )
                finally:
                    Path(temp_path).unlink(missing_ok=True)

                result = results[0]

                # Render the annotated image.
                annotated_array = result.plot()
                annotated_image = Image.fromarray(
                    annotated_array[:, :, ::-1]
                )

                # Summarize predictions.
                boxes = result.boxes
                total_detections = len(boxes) if boxes is not None else 0

                class_counts = Counter()
                confidence_scores = []

                if boxes is not None and total_detections > 0:
                    for box in boxes:
                        class_id = int(box.cls.item())
                        class_name = model.names[class_id]
                        class_counts[class_name] += 1
                        confidence_scores.append(float(box.conf.item()))

                st.divider()
                st.subheader("Detection Results")

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric("Objects detected", total_detections)

                metric2.metric(
                    "Classes detected",
                    len(class_counts)
                )

                average_confidence = (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores else 0
                )

                metric3.metric(
                    "Average confidence",
                    f"{average_confidence:.1%}"
                    if confidence_scores else "N/A"
                )

                left, right = st.columns(2)

                with left:
                    st.image(
                        original_image,
                        caption="Original image",
                        use_container_width=True
                    )

                with right:
                    st.image(
                        annotated_image,
                        caption="Image with predictions",
                        use_container_width=True
                    )

                st.subheader("Class Summary")

                if class_counts:
                    for class_name, count in class_counts.items():
                        st.write(f"**{class_name}:** {count}")
                else:
                    st.warning(
                        "No detections were found at this confidence threshold. "
                        "Try lowering the threshold."
                    )

                st.subheader("Download Result")

                image_buffer = io.BytesIO()
                annotated_image.save(image_buffer, format="PNG")

                st.download_button(
                    label="⬇️ Download annotated image",
                    data=image_buffer.getvalue(),
                    file_name="mushroom_detection_result.png",
                    mime="image/png",
                    use_container_width=True
                )

    except Exception as error:
        st.error(f"Something went wrong while processing the image: {error}")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Mushroom Detection Studio | YOLOv8s | "
    "Educational project — predictions are not safety advice."
)
# ---------------- FOOTER ----------------
# ---------------- MODEL PERFORMANCE ----------------
st.divider()

st.header("📊 Model Performance")

st.write(
    "The following results were obtained from two YOLOv8s "
    "experiments conducted during model evaluation."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Experiment 1 — Image Size 640")

    st.metric("Precision", "96.13%")
    st.metric("Recall", "95.16%")
    st.metric("F1-score", "95.64%")
    st.metric("mAP@50", "99.50%")
    st.metric("mAP@50–95", "63.07%")

with col2:
    st.subheader("Experiment 2 — Image Size 416")

    st.metric("Precision", "88.45%")
    st.metric("Recall", "98.07%")
    st.metric("F1-score", "93.01%")
    st.metric("mAP@50", "99.50%")
    st.metric("mAP@50–95", "67.52%")

st.info(
    "Important limitation: The test set contained only five images. "
    "These results are preliminary and may not represent performance "
    "on a larger, independent dataset."
)

