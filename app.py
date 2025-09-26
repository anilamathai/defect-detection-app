import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # replace with your local path if needed

st.title("Real-time Defect Detection")

# Sidebar options
option = st.sidebar.radio("Choose Input:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Run YOLO detection
        results = model(image_np)
        annotated_image = results[0].plot()

        st.image(annotated_image, caption="Detected Defects", use_container_width=False)

elif option == "Use Webcam":
    st.info("Click 'Start Webcam' to detect defects in real-time")
    run = st.button("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLO detection
            results = model(frame_rgb)
            annotated_frame = results[0].plot()

            # ✅ Fixed: no warning + control width
            stframe.image(
                annotated_frame,
                channels="RGB",
                use_container_width=False,
                width=640  # adjust as needed
            )

            # Stop if 'q' is pressed (only works locally, not in browser UI)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

