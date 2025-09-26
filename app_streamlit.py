import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # make sure best.pt is in the project folder

st.title("Defect Detection for Ceramic Tiles")
st.write("Upload an image to detect defects in tiles.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run YOLO detection
    results = model(image_np)
    annotated_image = results[0].plot()

    # Display annotated image
    st.image(
        annotated_image,
        caption="Detected Defects",
        use_container_width=True
    )

st.write("Note: Webcam detection is not supported on Streamlit Cloud. Please upload an image.")
