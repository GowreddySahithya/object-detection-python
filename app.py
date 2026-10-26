import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(page_title="Object Detection with Python", page_icon="🔎", layout="wide")

st.title("🔎 Object Detection with Python")
st.write(
    "Upload an image to detect objects using a pre-trained YOLO model. "
    "Detected objects are shown with bounding boxes, labels, and confidence scores."
)

@st.cache_resource
def load_model():
    # The model weights are downloaded automatically on first run if needed.
    return YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear image containing objects you want to detect."
)

confidence = st.slider("Confidence threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    with st.spinner("Detecting objects..."):
        try:
            model = load_model()
            results = model.predict(source=image_array, conf=confidence, verbose=False)
            annotated_bgr = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            with right:
                st.subheader("Detection output")
                st.image(annotated_rgb, use_container_width=True)

            boxes = results[0].boxes
            detected_rows = []
            if boxes is not None and len(boxes) > 0:
                names = results[0].names
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    detected_rows.append({
                        "Object": names.get(class_id, str(class_id)),
                        "Confidence": f"{score:.1%}"
                    })
                st.subheader("Detected objects")
                st.dataframe(detected_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No objects were detected above the selected confidence threshold.")

            output_image = Image.fromarray(annotated_rgb)
            import io
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            st.download_button(
                "Download output image",
                data=buffer.getvalue(),
                file_name="object_detection_output.png",
                mime="image/png"
            )
        except Exception as exc:
            st.error("The model could not run. Check the deployment logs or try again.")
            st.caption(f"Technical details: {exc}")
else:
    st.info("Upload an image to begin object detection.")

st.caption("Educational demo. Detection quality depends on image quality and the model's supported classes.")
