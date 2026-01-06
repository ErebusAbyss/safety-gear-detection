import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd
from datetime import datetime

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
    uploaded_files = st.file_uploader("Upload images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Total images uploaded: {len(uploaded_files)}")
        
        for uploaded_file in uploaded_files:
            with st.expander(f"Results for {uploaded_file.name}", expanded=True):
                # Convert to image
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Run detection
                results = model.predict(source=img_array, conf=conf_threshold)
                res_plotted = results[0].plot()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_container_width=True)
                with col2:
                    st.image(res_plotted, caption="Detection", use_container_width=True)
                
                # Statistics for each image
                counts = results[0].boxes.cls.unique().tolist()
                names = model.names
                stats = [f"{names[int(c)]}: {int((results[0].boxes.cls == c).sum())}" for c in counts]
                st.write(f"**Found:** {', '.join(stats)}")

# 5. Video Processing Logic
elif source == "Video (Upload)":
    uploaded_video = st.file_uploader("Upload a video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        log_dir = "violations"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 1. Generate unique report filename with timestamp
        video_name = os.path.splitext(uploaded_video.name)[0]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{video_name}_{timestamp_str}.csv"
        report_path = os.path.join(log_dir, report_filename)

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        st_frame = st.empty()
        
        st.subheader("⚠️ Real-time Violation Log (with Tracking ID)")
        log_placeholder = st.empty()
        
        # List for displaying records in the Streamlit UI table
        violations_list_ui = []
        last_recorded_vlog = {} 

        stop_btn = st.button("Stop Processing")
        
        while cap.isOpened():
            if stop_btn:
                st.warning("Processing stopped by user. Report is saved.")
                break

            ret, frame = cap.read()
            if not ret:
                st.success("Video processing finished.")
                break
                
            timestamp_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            
            # Object Tracking using ByteTrack/BoT-SORT
            results = model.track(source=frame, persist=True, conf=conf_threshold, verbose=False)
            res_plotted = results[0].plot()
            st_frame.image(res_plotted, channels="BGR", use_container_width=True)
            
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_indices = results[0].boxes.cls.int().cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

                for track_id, cls_idx, conf in zip(track_ids, class_indices, confidences):
                    class_name = model.names[cls_idx]
                    
                    # Logic: Identify safety violations (prefixed with 'NO-')
                    if "NO-" in class_name:
                        violation_key = (track_id, class_name)
                        cooldown_period = 600 # 10-minute cooldown for the same person/violation
                        
                        if violation_key not in last_recorded_vlog or (timestamp_sec - last_recorded_vlog[violation_key] > cooldown_period):
                            
                            file_name = f"id{track_id}_{class_name}_{timestamp_sec}s_{timestamp_str}.jpg"
                            file_path = os.path.join(log_dir, file_name)
                            
                            # 1. Save annotated screenshot as evidence
                            cv2.imwrite(file_path, res_plotted)
                            
                            # Violation data record
                            violation_record = {
                                "Video Time": f"{timestamp_sec}s",
                                "Person ID": track_id,
                                "Violation": class_name,
                                "Confidence": f"{conf:.2f}",
                                "Evidence File": file_name
                            }
                            
                            # 2. INSTANTLY APPEND TO CSV (Append mode)
                            # Check if file exists to determine if headers are needed
                            file_exists = os.path.isfile(report_path)
                            pd.DataFrame([violation_record]).to_csv(
                                report_path, 
                                mode='a', # 'a' for append mode
                                header=not file_exists, # Write header only if file is new
                                index=False
                            )
                            
                            # Update UI table
                            violations_list_ui.append(violation_record)
                            last_recorded_vlog[violation_key] = timestamp_sec
            
            # Update the UI table with the 5 most recent violations
            if violations_list_ui:
                log_placeholder.table(pd.DataFrame(violations_list_ui).tail(5))
            
        cap.release()