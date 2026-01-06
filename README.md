# AI Safety Guard: Construction Site Monitoring System 👷‍♂️🛡️

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green.svg)](https://ultralytics.com/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A computer vision application designed to automate safety monitoring on construction sites. This tool detects Personal Protective Equipment (PPE) like helmets and vests in real-time, providing an automated "violation log" with visual evidence.

## ✨ Core Functionality
- **Multi-Image Detection:** Batch process multiple images to identify safety gear violations.
- **Video Tracking (MVP):** Real-time monitoring with unique Tracking IDs for every person using BoT-SORT.
- **Evidence Management:** Automatically captures annotated screenshots of violations and generates CSV reports for site managers.
- **Custom Confidence Control:** Adjustable sensitivity to balance between detection recall and precision.

## 🛠️ Technology Stack
- **AI/ML:** Ultralytics YOLOv8 (Inference & Tracking).
- **Backend/Frontend:** Streamlit.
- **Image Processing:** OpenCV, PIL.
- **Data:** Pandas for CSV reporting.

## 📂 Project Structure
- `app/main.py` - Main Streamlit application.
- `notebooks/research.ipynb` - Initial research and model training logic.
- `runs/` - Model weights and training history.
- `violations/` - Generated evidence reports (screenshots + CSV).

## 🚀 Installation
1. Clone the repo: `git clone https://github.com/ErebusAbyss/safety_gear_detection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run app: `streamlit run app/main.py`

## 📈 Future Roadmap
- [ ] Implement User Authentication (Admin/Manager roles).
- [ ] Direct RTSP stream support for IP Cameras.
- [ ] Retrain model on larger datasets for 90%+ accuracy.
- [ ] Email/Telegram notifications for critical violations.