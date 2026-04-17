# 🚗 AVC&C — Automated Vehicle Classification & Counting

> **Client:** Bradford Council — Yunus Mayat (Enterprise Architect)
> **Module:** COS6032-E Industrial AI Project | University of Bradford
> **Group:** G25 — BSc Applied Artificial Intelligence | Academic Year 2025/2026
> **Supervisor:** Dr Kulwinder Panesar

---

## 🏆 Key Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Detection Accuracy (mAP@0.5) | ≥ 85% | **91.4%** | ✅ EXCEEDED |
| Classification Accuracy | ≥ 80% | **88.6%** | ✅ EXCEEDED |
| Precision (UA-DETRAC) | ≥ 80% | **89.2%** | ✅ EXCEEDED |
| Recall (UA-DETRAC) | ≥ 80% | **83.6%** | ✅ EXCEEDED |
| Processing Speed | ≥ 15 FPS | **25 FPS** | ✅ EXCEEDED |
| GDPR Compliance | 100% | **100%** | ✅ MET |
| Dashboard Load Time | < 5s | **< 2s** | ✅ EXCEEDED |

---

## 👥 Team

| Name | Role | Key Contributions |
|------|------|-------------------|
| **Ayush Acharya** | AI/ML Lead & Scrum Master | YOLO11n training pipeline, vehicle_counter.py, DeepSORT integration, GDPR audit |
| **Pratham Patel** | Data & Testing Lead | 11,582-image dataset annotation, UA-DETRAC benchmark validation |
| **Aman Gill** | Dashboard & Docs Lead | Streamlit dashboard (app.py), exhibition poster, documentation |

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [CDIO Phases](#-cdio-phases)
- [Technical Stack](#-technical-stack)
- [Quick Start](#-quick-start)
- [File Structure](#-file-structure)
- [Model & Dataset](#-model--dataset)
- [Results](#-results)
- [Documentation](#-documentation)
- [Ethics & Compliance](#-ethics--compliance)
- [Future Work](#-future-work)

---

## 🎯 Project Overview

Bradford Council currently deploys 2–3 officers per road junction to conduct manual vehicle counts — expensive, inconsistent, and limited to snapshot data. The AVC&C system replaces this with an automated computer vision pipeline that:

- **Detects** cars, buses, and trucks in real-time video using YOLO11n
- **Tracks** each vehicle across frames using DeepSORT with persistent track IDs
- **Counts** unique vehicles crossing a virtual trip-line without duplicates
- **Displays** live traffic data on a Streamlit dashboard with weather filtering
- **Logs** anonymised counts to CSV — fully GDPR compliant, no personal data captured

The system runs entirely on a standard laptop CPU (no GPU required) at 25 FPS, making it deployable on Bradford Council's existing hardware with zero additional infrastructure cost.

---

## 🔄 CDIO Phases

### Conceive — Sprints 1 & 2 (Sep–Nov 2025)
- Client requirements gathering with Yunus Mayat (Bradford Council)
- YOLO model comparison (v8 / v9 / v11) → **YOLO11n selected**
- Dataset strategy: 11,582 images across 5 weather conditions via Roboflow
- GitHub repository setup and team role allocation
- 📄 See: `docs/AVC_G25_Research_Document.docx`

### Design — Sprints 3 & 4 (Nov 2025–Jan 2026)
- `vehicle_counter.py` — OpenCV + YOLO11n + COCO class filter (car=2, bus=5, truck=7)
- Colour-coded bounding boxes: green=car, blue=bus, orange=truck
- DeepSORT integration — persistent vehicle IDs across frames
- Virtual trip-line at y=300, centroid crossing detection
- Double-counting bug identified and fixed (centroid history approach)
- CSV logging: timestamp, track_id, class, confidence, x, y, weather
- 📄 See: `docs/AVC_G25_Design_Document.docx`

### Implement — Sprint 5 (Feb 2026)
- Custom YOLO11n training: 50 epochs, batch=8, imgsz=640, ~3 hours on CPU
- Achieved mAP@0.5: **91.4%** (target was 85%) — exceeded
- Training output: `runs/detect/avc_model_v15/weights/best.pt`
- 📄 See: `docs/AVC_G25_Testing_Validation_Report.docx`

### Operate — Sprints 6, 7 & 8 (Mar–Apr 2026)
- Streamlit dashboard (`app.py`) with Plotly charts, weather filter, 5s auto-refresh
- UA-DETRAC benchmark validation: Precision 89.2% | Recall 83.6%
- GDPR compliance audit — 100% compliant
- DroidCam phone camera integration for live exhibition demo
- `start_demo.bat` one-click launcher
- 📄 See: `docs/AVC_G25_Agile_Artefacts.docx`

---

## 🛠 Technical Stack

| Component | Technology | Version | Cost |
|-----------|-----------|---------|------|
| Object Detection | Ultralytics YOLO11n | yolo11n.pt | Free (AGPL-3.0) |
| Multi-Object Tracking | DeepSORT | — | Free (open source) |
| Video Processing | OpenCV | 4.x | Free |
| Dashboard | Streamlit + Plotly + Pandas | Latest | Free |
| Dataset Management | Roboflow | Free tier | Free |
| Language | Python | 3.13 | Free |
| Version Control | GitHub | Free tier | Free |
| **Total Cost** | | | **~£15 (poster printing only)** |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install ultralytics opencv-python streamlit plotly pandas
```

### Option 1 — One-Click (Windows)
```
Double-click: start_demo.bat
```
Opens vehicle counter + dashboard automatically in browser at `localhost:8501`

### Option 2 — Manual (All Platforms)

**Terminal 1 — Start vehicle counter:**
```bash
python vehicle_counter.py --source 0 --weather Normal
# --source 0 = webcam | --source video.mp4 = video file
# --weather options: Normal | Rain | Fog | Snow | Sand
```

**Terminal 2 — Start dashboard:**
```bash
streamlit run app.py
# Opens at localhost:8501
```

### Option 3 — DroidCam (Phone as Camera)
1. Install DroidCam on phone + PC
2. Connect phone and PC to same WiFi network
3. Note DroidCam IP address (shown in app)
```bash
python vehicle_counter.py --source "http://192.168.x.x:4747/video" --weather Normal
```

---

## 📁 File Structure

```
avc-vehicle-classification/
│
├── vehicle_counter.py          # Core: YOLO11n + DeepSORT + trip-line counting
├── app.py                      # Streamlit dashboard with weather filter
├── train.py                    # Custom YOLO11n training script
├── prepare_dataset.py          # Auto dataset split (70/22/15%)
├── start_demo.bat              # One-click Windows launcher
│
├── vehicle_log.csv             # Output: anonymised vehicle counts
├── data.yaml                   # YOLO training config (classes + paths)
├── requirements.txt            # Python dependencies
│
├── runs/
│   └── detect/
│       └── avc_model_v15/
│           └── weights/
│               ├── best.pt     # ← USE THIS: 91.4% mAP@0.5
│               └── last.pt     # Final epoch weights
│
└── docs/
    ├── AVC_G25_Testing_Validation_Report.docx
    ├── AVC_G25_FAIR_Report.docx
    ├── AVC_G25_Ethics_Implications.docx
    ├── AVC_G25_Research_Document.docx
    ├── AVC_G25_Design_Document.docx
    └── AVC_G25_Agile_Artefacts.docx
```

---

## 🧠 Model & Dataset

### YOLO11n Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | yolo11n.pt (COCO pretrained) |
| Total Images | 11,582 |
| Train / Val / Test | 7,337 / 2,532 / 1,713 |
| Epochs | 50 |
| Batch Size | 8 |
| Image Size | 640 × 640 |
| Device | CPU — AMD Ryzen 5 5600H |
| Training Time | ~3 hours |
| Output | runs/detect/avc_model_v15/weights/best.pt |

### Dataset Weather Conditions

| Condition | Images | Notes |
|-----------|--------|-------|
| Normal | ~2,400 | Clear conditions — baseline |
| Rain | ~2,300 | Wet roads, spray, reduced contrast |
| Fog | ~2,200 | Low visibility — most challenging |
| Snow | ~2,300 | White road surfaces reduce contrast |
| Sand/Haze | ~2,382 | Improves fog-like condition robustness |
| **Total** | **11,582** | 70% train / 22% val / 15% test |

Dataset sourced and annotated via [Roboflow](https://roboflow.com) by Pratham Patel (~25 hours annotation effort).

---

## 📊 Results

### Model Performance

| Metric | Car | Bus | Truck | Overall |
|--------|-----|-----|-------|---------|
| Precision | 90.1% | 88.4% | 89.1% | **89.2%** |
| Recall | 84.3% | 83.0% | 83.5% | **83.6%** |
| mAP@0.5 | 92.1% | 90.8% | 91.3% | **91.4%** |

### Per-Weather Performance (mAP@0.5)

| Normal | Rain | Fog | Snow | Sand |
|--------|------|-----|------|------|
| 93.2% | 90.4% | 89.8% | 90.1% | 92.1% |

All conditions exceed the 85% Bradford Council target ✅

### Manual Ground Truth Validation
- **48 vehicles** counted on Bradford test footage
- **48 vehicles** counted by Group G25 manually
- **100% match** — confirmed post double-counting bug fix

---

## 📄 Documentation

All project documentation is in the `/docs` folder and on the Canvas submission:

| Document | Description |
|----------|-------------|
| `AVC_G25_Testing_Validation_Report.docx` | Unit tests, integration tests, UA-DETRAC benchmark, GDPR audit |
| `AVC_G25_FAIR_Report.docx` | AI/GenAI disclosure — all tools declared, responsible use |
| `AVC_G25_Ethics_Implications.docx` | CDEI framework, GDPR, bias assessment, Bradford Net Zero 2038 |
| `AVC_G25_Research_Document.docx` | Model comparison, dataset research, literature review (Conceive phase) |
| `AVC_G25_Design_Document.docx` | System architecture, data flow, component design (Design phase) |
| `AVC_G25_Agile_Artefacts.docx` | Sprint plans, backlog, risk register, burndown, team blogs |

---

## ⚖️ Ethics & Compliance

### GDPR / UK Data Protection Act 2018
- ✅ **No personal data captured** — no faces, no licence plates, no individuals
- ✅ **Data minimisation** — CSV stores counts only (timestamp, class, anonymised track_id)
- ✅ **Local processing only** — no cloud upload, no external data transmission
- ✅ **Privacy-by-design** — COCO class filter architecturally prevents personal data capture
- ✅ **100% GDPR compliant** — verified by audit of 200 CSV rows (0 personal data found)

### AI Transparency (FAIR Report)
All AI tools used in this project are declared in `docs/AVC_G25_FAIR_Report.docx`:
- Claude (Anthropic) — code debugging and documentation guidance
- YOLO11n (Ultralytics) — core detection model
- DeepSORT — multi-object tracking
- Roboflow — dataset annotation assistance

All model training, dataset preparation, system architecture, and validation results are the original work of Group G25.

### CDEI AI Assurance Framework
The AVC&C system is assessed as compliant across all five CDEI principles: Safety & Security, Transparency, Fairness, Accountability, and Contestability. Full assessment in `docs/AVC_G25_Ethics_Implications.docx`.

---

## 🔮 Future Work

| Feature | Priority | Notes |
|---------|----------|-------|
| Motorcycle & bicycle detection | High | MoSCoW: Could Have (US9) — v2.0 |
| NVIDIA Jetson edge deployment | High | GPU acceleration for junction-side processing |
| Multi-camera junction support | Medium | MoSCoW: Won't Have v1 (US10) |
| GIS traffic hotspot mapping | Medium | Bradford-wide visualisation |
| Night-time / low-light testing | High | Untested in v1.0 — needed for full deployment |
| Bradford Council CCTV integration | High | Requires separate DPIA and Council IT approval |
| Docker containerisation | Medium | Reproducible deployment across Council hardware |

---

## 📜 Licence

This project is developed for academic purposes at the University of Bradford as part of COS6032-E Industrial AI Project. All code is original work by Group G25 unless otherwise stated in source files.

- **YOLO11n:** [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **DeepSORT:** Open source
- **Streamlit / Plotly / OpenCV:** Open source (Apache 2.0 / BSD)

---

*Group G25 — BSc Applied Artificial Intelligence | University of Bradford | 2025/2026*
*Supervised by Dr Kulwinder Panesar | Client: Bradford Council*
