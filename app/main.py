import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd

# 1. Page configuration
st.set_page_config(page_title="AI Safety Guard", layout="wide")

st.title("Construction Site Safety Detection 👷‍♂️")
st.sidebar.title("Settings")

# 2. Load model
# Ensure this path matches your training results folder
model_path = './runs/detect/safety_v1_run/weights/best.pt'

@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. Sidebar options
source = st.sidebar.radio("Select Source", ("Image", "Video (Upload)"))
# Defaulting to 0.5 to match standard YOLO behavior
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# 4. Image Processing Logic
if source == "Image":
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Run detection
        results = model.predict(source=img_array, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(res_plotted, caption="Detection Results", use_container_width=True)
            
        st.subheader("Detected Objects Statistics")
        counts = results[0].boxes.cls.unique().tolist()
        names = model.names
        for c in counts:
            num = (results[0].boxes.cls == c).sum()
            st.write(f"- **{names[int(c)]}**: {int(num)}")

# 5. Video Processing Logic
elif source == "Video (Upload)":
    uploaded_video = st.file_uploader("Upload a video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Create directories for logs
        log_dir = "violations"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        st_frame = st.empty() # Placeholder for video frames
        
        st.subheader("⚠️ Real-time Violation Log (with Tracking ID)")
        log_placeholder = st.empty()
        
        violations_data = []
        # Dictionary to keep track of last recorded time for each (track_id, violation_type)
        last_recorded_vlog = {} 

        stop_btn = st.button("Stop Processing")
        
        # Frame-by-frame processing loop
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            
            # Run tracking on the frame
            results = model.track(source=frame, persist=True, conf=conf_threshold, verbose=False)
            
            # Generate the annotated frame (with boxes and IDs)
            res_plotted = results[0].plot()
            
            # Display frame in Streamlit (convert BGR to RGB for display)
            st_frame.image(res_plotted, channels="BGR", use_container_width=True)
            
            # Check if tracking IDs are available
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_indices = results[0].boxes.cls.int().cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

                for track_id, cls_idx, conf in zip(track_ids, class_indices, confidences):
                    class_name = model.names[cls_idx]
                    
                    # LOGIC: Only record safety violations (NO-...)
                    if "NO-" in class_name:
                        # Unique key: specific person ID + specific violation type
                        violation_key = (track_id, class_name)
                        
                        # Cooldown: Don't record the same person for 10 minutes (600s)
                        cooldown_period = 600 
                        current_video_time = timestamp_sec
                        
                        if violation_key not in last_recorded_vlog or (current_video_time - last_recorded_vlog[violation_key] > cooldown_period):
                            
                            file_name = f"id{track_id}_{class_name}_{timestamp_sec}s.jpg"
                            file_path = os.path.join(log_dir, file_name)
                            
                            # SAVE EVIDENCE: Use 'res_plotted' instead of raw 'frame'
                            cv2.imwrite(file_path, res_plotted) # <-- ОСЬ ТУТ БУЛА ЗМІНА
                            
                            violations_data.append({
                                "Video Time": f"{timestamp_sec}s",
                                "Person ID": track_id,
                                "Violation": class_name,
                                "Confidence": f"{conf:.2f}",
                                "Evidence File": file_name
                            })
                            
                            # Update last recorded time
                            last_recorded_vlog[violation_key] = current_video_time
            
            # Update the table in UI
            if violations_data:
                log_placeholder.table(pd.DataFrame(violations_data).tail(5))
            
        cap.release()
        
        # Save full report to CSV
        if violations_data:
            report_path = os.path.join(log_dir, "session_report.csv")
            pd.DataFrame(violations_data).to_csv(report_path, index=False)
            st.success(f"Processing finished. Full report saved to {report_path}")