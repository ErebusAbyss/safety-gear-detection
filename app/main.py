import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Page configuration
st.set_page_config(page_title="AI Safety Guard", layout="wide")

st.title("Construction Site Safety Detection 👷‍♂️")
st.sidebar.title("Settings")

# Load model (Adjust path if needed)
# Since we will run this from project root, path is './runs/...'
model_path = './runs/detect/safety_v1_run/weights/best.pt'

@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Sidebar options
source = st.sidebar.radio("Select Source", ("Image", "Video (Upload)"))
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

if source == "Image":
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Run detection
        results = model.predict(source=img_array, conf=conf_threshold)
        
        # Plot results
        res_plotted = results[0].plot()
        
        # Layout columns
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(res_plotted, caption="Detection Results", use_container_width=True)
            
        # Show detected classes
        st.subheader("Detected Objects Statistics")
        counts = results[0].boxes.cls.unique().tolist()
        names = model.names
        for c in counts:
            num = (results[0].boxes.cls == c).sum()
            st.write(f"- **{names[int(c)]}**: {int(num)}")

elif source == "Video (Upload)":
    st.info("Video processing module is under construction. We will add it in the next step!")