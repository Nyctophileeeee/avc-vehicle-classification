# 🚦 Automated Vehicle Classification & Counting (AVC&C)
> AI-powered real-time traffic monitoring system built for Bradford Council  
> University of Bradford · BSc Applied Artificial Intelligence · Final Year Project 2026

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![YOLO](https://img.shields.io/badge/YOLO-11n-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Detection%20Accuracy-91.4%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-orange?style=flat-square)

---

## 📌 Overview

The AVC&C system is a prototype AI solution developed for **Bradford Council** to automate road vehicle counting and classification — replacing costly and inconsistent manual traffic surveys with a scalable, privacy-compliant, real-time monitoring tool.

The system uses computer vision to detect, track, and count vehicles from live camera feeds or recorded video, classifying them by type (car, bus, truck) and logging all data to a CSV for downstream analysis. A live Streamlit dashboard visualises traffic flows in real time, enabling council officers to make data-driven decisions for urban planning and road safety.

---

## 🎯 Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Detection Accuracy (mAP@0.5) | ≥ 85% | **91.4%** ✅ |
| Classification Accuracy | ≥ 80% | **88.6%** ✅ |
| Processing Speed | Real-time | **25 FPS** ✅ |
| Dashboard Refresh | < 5 min | **5 seconds** ✅ |
| Weather Conditions Tested | Multiple | **5 conditions** ✅ |

---

## 🧠 System Architecture

```
📷 Camera / Video Feed
        ↓
🤖 YOLO11n Detection Model
        ↓
🔄 DeepSORT Vehicle Tracking
        ↓
📊 Vehicle Counter (Line Crossing Logic)
        ↓
📁 vehicle_log.csv (Timestamped Logs)
        ↓
📈 Streamlit Dashboard (Live Visualisation)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv11n (Ultralytics) |
| Vehicle Tracking | DeepSORT |
| Model Training | Custom YOLO11n on weather-specific dataset |
| Dashboard | Streamlit + Plotly |
| Data Logging | CSV via Python |
| Camera Input | OpenCV (webcam / DroidCam / video file) |
| Language | Python 3.13 |

---

## 📂 Project Structure

```
avc-vehicle-classification/
│
├── vehicle_counter.py      # Main detection & counting script
├── app.py                  # Streamlit dashboard
├── train.py                # Model training script
├── prepare_dataset.py      # Dataset organisation script
├── start_demo.bat          # One-click demo launcher
├── vehicle_log.csv         # Detection logs (auto-generated)
├── yolo11n.pt              # Base YOLO11n model
│
├── src/
│   └── data/
│       ├── data.yaml       # Dataset configuration
│       └── train data/     # Training dataset
│           ├── images/     # train / val / test splits
│           └── labels/     # YOLO format annotations
│
└── runs/
    └── detect/
        └── runs/train/
            └── avc_model_v15/
                └── weights/
                    └── best.pt   # Trained model weights
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics streamlit plotly pandas opencv-python
```

### 1. Clone the Repository

```bash
git clone https://github.com/Myctophileeeee/avc-vehicle-classification.git
cd avc-vehicle-classification
```

### 2. Run Vehicle Counter

```bash
python vehicle_counter.py
```

### 3. Launch Dashboard

```bash
python -m streamlit run app.py
```

### 4. One-Click Demo (Windows)

Simply double-click `start_demo.bat` — opens both the counter and dashboard automatically.

---

## 📊 Dataset

The model was trained on a comprehensive dataset of **11,582 images** across 5 weather conditions:

| Split | Images |
|-------|--------|
| Training (70%) | 7,337 |
| Validation (20%) | 2,532 |
| Test (10%) | 1,713 |
| **Total** | **11,582** |

**Weather conditions covered:**

| Condition | Images (approx) |
|-----------|----------------|
| Normal (Day + Night) | ~2,100 |
| Rain | ~2,800 |
| Fog | ~2,200 |
| Snow | ~2,400 |
| Sand | ~2,082 |

- **Original raw dataset:** ~14,000+ images collected
- **After duplicate removal & cleaning:** 11,582 quality images retained
- **Classes:** Car, Bus, Truck
- **Format:** YOLO darknet annotation format
- **Sources:** Roboflow annotated datasets + custom weather-condition datasets

## 🎥 Demo Setup

**For live demonstration:**

1. Run `start_demo.bat`
2. Play a traffic video on a phone
3. Point phone camera at laptop webcam
4. Watch live detection with bounding boxes
5. Dashboard updates every 5 seconds with counts

**Camera options:**
- Laptop webcam (`VIDEO_PATH = 0`)
- Phone via DroidCam WiFi (`VIDEO_PATH = "http://IP:4747/video"`)
- Pre-recorded video (`VIDEO_PATH = r'path/to/video.mp4'`)

**Press W** during demo to cycle through weather conditions live.

---

## 📋 Bradford Council Requirements Met

Based on the client brief presented by **Yunus Mayat, Enterprise Architect, Bradford Council:**

- ✅ Real-time vehicle detection and counting from video feeds
- ✅ Vehicle classification (car, bus, truck)
- ✅ Virtual counting line to avoid double-counting
- ✅ Dashboard showing traffic flows by type and time
- ✅ Tested under varied weather conditions (rain, fog, snow, sand, normal)
- ✅ Privacy compliant — only anonymised, aggregated counts stored
- ✅ Exceeds 85% detection accuracy target (91.4% achieved)
- ✅ Exceeds 80% classification accuracy target (88.6% achieved)
- ✅ Dashboard updates within 5 seconds

---

## 👥 Team — Group G25

| Name | Role |
|------|------|
| **Ayush Acharya** | AI/ML Lead — Model training, detection pipeline, system architecture |
| **Pratham Patel** | Data & Testing — Dataset preparation, validation, testing |
| **Aman Gill** | Dashboard & Docs — Streamlit dashboard, documentation |

**Supervised by:** Dr Kulwinder Panesar  
**Client:** Yunus Mayat, Enterprise Architect, Bradford Council  
**Institution:** University of Bradford, School of Computing  
**Year:** 2026

---

## 📄 License

This project was developed as part of the BSc Applied Artificial Intelligence final year programme at the University of Bradford. All code is submitted for academic assessment purposes.

---

> *"Moving Bradford Council away from manual traffic surveys towards automated, ethical, and scalable AI-powered monitoring."*