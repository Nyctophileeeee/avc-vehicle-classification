<div align="center">

# 🚦 AVC&C
## Automated Vehicle Classification & Counting

*Replacing manual traffic surveys with real-time AI — built for Bradford Council*

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLO11n](https://img.shields.io/badge/YOLO11n-Ultralytics-00DBDE?style=for-the-badge&logo=pytorch&logoColor=white)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![GDPR](https://img.shields.io/badge/GDPR-100%25%20Compliant-2EA44F?style=for-the-badge)](https://ico.org.uk)
[![License](https://img.shields.io/badge/Licence-AGPL--3.0-blue?style=for-the-badge)](LICENSE)

---

**🎓 University of Bradford — COS6032-E Industrial AI Project 2025/2026**

**Group G25** · BSc Applied Artificial Intelligence · Supervised by Dr Kulwinder Panesar

**Client:** Bradford Council · Yunus Mayat (Enterprise Architect)

</div>

---

<div align="center">

## 🏆 Every Target Exceeded

| Metric | Bradford Council Target | Our Result | |
|:-------|:----------------------:|:----------:|:--|
| Detection Accuracy (mAP@0.5) | ≥ 85% | **91.4%** | 🟢 +6.4% |
| Classification Accuracy | ≥ 80% | **88.6%** | 🟢 +8.6% |
| Precision — UA-DETRAC | ≥ 80% | **89.2%** | 🟢 +9.2% |
| Recall — UA-DETRAC | ≥ 80% | **83.6%** | 🟢 +3.6% |
| Processing Speed | ≥ 15 FPS | **25 FPS** | 🟢 +67% |
| Dashboard Load Time | < 5 seconds | **< 2 seconds** | 🟢 60% faster |
| GDPR Compliance | 100% | **100%** | 🟢 Verified |
| Manual Count Match | 100% | **48/48** | 🟢 Perfect |

</div>

---

## 📖 Table of Contents

| | | |
|--|--|--|
| [🎯 What We Built](#-what-we-built) | [🔄 CDIO Journey](#-cdio-journey) | [🧠 How It Works](#-how-it-works) |
| [🚀 Run It Yourself](#-run-it-yourself) | [📊 Results In Depth](#-results-in-depth) | [🗂️ File Structure](#%EF%B8%8F-file-structure) |
| [📄 Documentation](#-documentation) | [⚖️ Ethics & GDPR](#%EF%B8%8F-ethics--gdpr) | [👥 Team](#-team) |

---

## 🎯 What We Built

> Bradford Council spends **£15.30/hour × 2–3 officers × every junction** doing traffic counts by hand.
> We replaced that with a laptop, a camera, and this system.

The AVC&C system is a **real-time computer vision pipeline** that automatically detects, classifies, and counts vehicles passing a road junction — no GPU, no cloud, no personal data, no £50,000 ANPR contract.

```
📷 Camera Feed  →  🧠 YOLO11n  →  🔁 DeepSORT  →  📏 Trip-Line  →  📊 Dashboard
  (any source)     (detect)       (track IDs)     (count once)     (live charts)
```

**What it detects:** 🚗 Cars · 🚌 Buses · 🚛 Trucks

**What it ignores (by design):** faces · licence plates · individuals · personal data of any kind

**Where it runs:** your existing laptop — AMD Ryzen, no GPU needed

---

## 🔄 CDIO Journey

The project followed the **Conceive → Design → Implement → Operate** engineering lifecycle across 8 Agile sprints from September 2025 to April 2026.

<details>
<summary><strong>🔵 CONCEIVE — Sprints 1 & 2 &nbsp;|&nbsp; Sep–Nov 2025</strong></summary>

### What we did
- Met with Yunus Mayat at Bradford Council — understood the manual counting problem first-hand
- Compared YOLO v8 / v9 / v11 on CPU performance and accuracy → **YOLO11n selected**
- Researched 6 public datasets — none met UK weather requirements
- Decided to build a custom 11,582-image dataset via Roboflow across 5 weather conditions
- Set up GitHub, agreed team roles, configured branching strategy

### Key decision
> **No existing dataset covered Bradford's weather.** Rather than compromise on training data quality, we invested 25 hours of manual annotation to build the right dataset from scratch.

📄 **Full research:** `docs/AVC_G25_Research_Document.docx`

</details>

<details>
<summary><strong>🟡 DESIGN — Sprints 3 & 4 &nbsp;|&nbsp; Nov 2025–Jan 2026</strong></summary>

### What we built
- `vehicle_counter.py` — core detection pipeline with COCO class filter [2, 5, 7]
- Virtual trip-line at y=300 with centroid-based directional crossing detection
- DeepSORT integration — persistent track IDs across occlusions
- Colour-coded bounding boxes: 🟢 car · 🔵 bus · 🟠 truck
- CSV logging schema: `timestamp, track_id, class, confidence, x, y, weather`

### The bug we had to fix
> Vehicles slowing near the trip-line were being counted 2–4× due to proximity triggering. We fixed this by storing **centroid history per track_id** — a count only registers on confirmed directional crossing, not proximity. After the fix: **48/48 manual count match**.

📄 **Full design:** `docs/AVC_G25_Design_Document.docx`

</details>

<details>
<summary><strong>🟠 IMPLEMENT — Sprint 5 &nbsp;|&nbsp; Feb 2026</strong></summary>

### Training run
```
Model:    yolo11n.pt (COCO pretrained)
Dataset:  11,582 images | Normal/Rain/Fog/Snow/Sand
Epochs:   50  |  Batch: 8  |  imgsz: 640
Device:   CPU — AMD Ryzen 5 5600H
Duration: ~3 hours
Output:   runs/detect/avc_model_v15/weights/best.pt
```

### Results
```
mAP@0.5:   91.4%  ← target was 85%  ✅
Precision: 89.2%  ← target was 80%  ✅
Recall:    83.6%  ← target was 80%  ✅
FPS:       25     ← target was 15   ✅
```

### The path bug
> `data.yaml` kept failing because our folder was named `train data` (with a space). Fixed with raw string: `r'C:\Users\...\train data\data.yaml'`

📄 **Full results:** `docs/AVC_G25_Testing_Validation_Report.docx`

</details>

<details>
<summary><strong>🟢 OPERATE — Sprints 6, 7 & 8 &nbsp;|&nbsp; Mar–Apr 2026</strong></summary>

### What we shipped
- `app.py` — Streamlit dashboard with Plotly bar/line charts, weather dropdown, 5s auto-refresh
- `start_demo.bat` — double-click to launch everything (for Bradford Council officers)
- UA-DETRAC benchmark validation by Pratham — confirmed results on external dataset
- Full GDPR audit — 200 CSV rows reviewed, 0 personal data found
- DroidCam phone-as-webcam integration for live exhibition demo
- A1 exhibition poster, demo plan, all documentation

📄 **Full sprint log:** `docs/AVC_G25_Agile_Artefacts.docx`

</details>

---

## 🧠 How It Works

### The Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AVC&C PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📷 INPUT          🧠 DETECT           🔁 TRACK                    │
│  ─────────         ────────────        ────────────                 │
│  OpenCV            YOLO11n             DeepSORT                     │
│  VideoCapture  →   91.4% mAP@0.5  →   Kalman filter +              │
│                    Class filter:       appearance features           │
│  webcam /          car  (COCO 2)       Persistent track_id          │
│  DroidCam /        bus  (COCO 5)       per vehicle                  │
│  video file        truck(COCO 7)                                    │
│                                              │                      │
│                                              ▼                      │
│  📊 DISPLAY        📝 LOG              📏 COUNT                    │
│  ─────────         ──────────          ────────────                 │
│  Streamlit    ←    vehicle_log.csv ←   Trip-line y=300              │
│  dashboard         anonymised          Centroid history             │
│  weather filter    counts only         Directional check            │
│  5s refresh        GDPR ✅             Count once ✅                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Technical Decisions

| Decision | Why |
|----------|-----|
| **YOLO11n nano** over larger models | 25 FPS on CPU — no GPU needed at Bradford Council |
| **Centroid history** for counting | Prevents double-counting when vehicles decelerate near trip-line |
| **COCO class filter [2,5,7]** | Architecturally blocks all non-vehicle classes — privacy enforcement |
| **DeepSORT** over ByteTrack/SORT | Best tracking accuracy vs CPU overhead balance |
| **Streamlit** over Flask/Dash | Python-native — Council officers need zero HTML/JS knowledge |
| **Append-mode CSV** | Data never lost if counter restarts mid-session |
| **5 weather conditions** in training | Bradford's climate demands fog/rain/snow robustness |

---

## 🚀 Run It Yourself

### Install dependencies
```bash
pip install ultralytics opencv-python streamlit plotly pandas
```

### ⚡ Option 1 — One click (Windows)
```
Double-click  →  start_demo.bat
```
Launches counter + dashboard. Browser opens at `localhost:8501` automatically.

### 💻 Option 2 — Manual (any OS)

**Terminal 1:**
```bash
python vehicle_counter.py --source 0 --weather Normal
# --source 0          → webcam
# --source video.mp4  → video file
# --weather           → Normal | Rain | Fog | Snow | Sand
```

**Terminal 2:**
```bash
streamlit run app.py
# → opens at localhost:8501
```

### 📱 Option 3 — DroidCam (phone as live camera)
```bash
# 1. Install DroidCam on phone + PC
# 2. Connect both to same WiFi
# 3. Note the IP shown in DroidCam app

python vehicle_counter.py --source "http://192.168.x.x:4747/video" --weather Rain
```

### 🧠 Re-train the model
```bash
python train.py
# Trains YOLO11n on your dataset
# Output: runs/detect/avc_model_v15/weights/best.pt
```

---

## 📊 Results In Depth

### Per-Class Performance

| Class | COCO ID | Precision | Recall | mAP@0.5 |
|:------|:-------:|:---------:|:------:|:-------:|
| 🚗 Car | 2 | 90.1% | 84.3% | 92.1% |
| 🚌 Bus | 5 | 88.4% | 83.0% | 90.8% |
| 🚛 Truck | 7 | 89.1% | 83.5% | 91.3% |
| **Overall** | — | **89.2%** | **83.6%** | **91.4%** |

### Per-Weather Performance (mAP@0.5)

| ☀️ Normal | 🌧️ Rain | 🌫️ Fog | ❄️ Snow | 🏜️ Sand |
|:--------:|:------:|:-----:|:------:|:------:|
| 93.2% | 90.4% | 89.8% | 90.1% | 92.1% |

> All five weather conditions exceed the 85% Bradford Council target ✅

### Manual Validation (Bradford Footage)

```
Manual count:   48 vehicles
System count:   48 vehicles
Match:          100%  ✅
```

### UA-DETRAC External Benchmark

| Metric | Result | Industry Standard |
|--------|--------|------------------|
| Precision | 89.2% | ~85% typical |
| Recall | 83.6% | ~80% typical |
| F1 Score | 0.862 | ~0.82 typical |
| False Positive Rate | 10.8% | <15% target |

---

## 🗂️ File Structure

```
avc-vehicle-classification/
│
├── 🔧 CORE SYSTEM
│   ├── vehicle_counter.py      ← detection + tracking + counting + CSV log
│   ├── app.py                  ← Streamlit dashboard (weather filter, live charts)
│   ├── start_demo.bat          ← one-click launcher for exhibitions / Council use
│   └── requirements.txt        ← all dependencies
│
├── 🧠 MODEL TRAINING
│   ├── train.py                ← custom YOLO11n training script
│   ├── prepare_dataset.py      ← auto 70/22/15% split organiser
│   └── data.yaml               ← YOLO config (class names + dataset paths)
│
├── 📦 TRAINED WEIGHTS
│   └── runs/detect/avc_model_v15/weights/
│       ├── best.pt             ← ⭐ USE THIS — 91.4% mAP@0.5
│       └── last.pt             ← final epoch (backup)
│
├── 📝 OUTPUT
│   └── vehicle_log.csv         ← anonymised counts (timestamp/class/weather)
│
└── 📄 DOCUMENTATION
    ├── AVC_G25_Research_Document.docx        ← Conceive phase
    ├── AVC_G25_Design_Document.docx          ← Design phase
    ├── AVC_G25_Testing_Validation_Report.docx← Implement + Operate
    ├── AVC_G25_Agile_Artefacts.docx          ← sprints, backlog, blogs
    ├── AVC_G25_Ethics_Implications.docx      ← CDEI, GDPR, bias, sustainability
    └── AVC_G25_FAIR_Report.docx              ← AI/GenAI disclosure
```

---

## 📄 Documentation

Full project documentation covers all CDIO phases and is available in `/docs`:

| Document | Phase | Contents |
|----------|-------|----------|
| 📘 [Research Document](docs/AVC_G25_Research_Document.docx) | Conceive | Model comparison, dataset analysis, literature review |
| 📐 [Design Document](docs/AVC_G25_Design_Document.docx) | Design | System architecture, data flow, component specs |
| 🧪 [Testing & Validation](docs/AVC_G25_Testing_Validation_Report.docx) | Implement/Operate | 48 tests, UA-DETRAC benchmark, GDPR audit |
| 📋 [Agile Artefacts](docs/AVC_G25_Agile_Artefacts.docx) | All | Sprint plans, risk register, burndown, team blogs |
| ⚖️ [Ethics & Implications](docs/AVC_G25_Ethics_Implications.docx) | All | CDEI, GDPR, bias, Bradford Net Zero 2038 |
| 🤖 [FAIR Report](docs/AVC_G25_FAIR_Report.docx) | All | AI/GenAI disclosure, responsible use declaration |

---

## ⚖️ Ethics & GDPR

### Privacy by Design

This system was built with privacy as a **hard architectural constraint**, not an afterthought.

```
What the system DOES store:          What the system NEVER stores:
─────────────────────────────        ──────────────────────────────
✅ timestamp                         ❌ video frames
✅ vehicle class (car/bus/truck)     ❌ facial data
✅ anonymised track_id (session)     ❌ licence plates
✅ confidence score                  ❌ individual identities
✅ centroid x, y position            ❌ anything transmitted externally
✅ weather condition
```

**Verification:** 200 CSV rows manually audited — **0 instances of personal data found.**

### CDEI AI Assurance Framework

| Principle | Status | Evidence |
|-----------|--------|----------|
| Safety & Security | ✅ Compliant | Local processing, no network exposure |
| Transparency | ✅ Compliant | FAIR Report, open-source codebase |
| Fairness | ✅ Compliant | Tested across 5 weather conditions equally |
| Accountability | ✅ Compliant | Clear team ownership, supervisor oversight |
| Contestability | ✅ Compliant | Auditable CSV, explainable dashboard |

Full assessment in `docs/AVC_G25_Ethics_Implications.docx`

### AI Tools Declared (FAIR Report)

All AI tool usage is disclosed in `docs/AVC_G25_FAIR_Report.docx`. Summary:

| Tool | Purpose | Our Work |
|------|---------|---------|
| Claude (Anthropic) | Code debugging guidance, doc structure | All implementation, testing, results by G25 |
| YOLO11n (Ultralytics) | Core detection model | Training, fine-tuning, validation by G25 |
| DeepSORT | Tracking algorithm | Integration and counting logic by G25 |
| Roboflow | Dataset annotation platform | 25hrs manual labelling by Pratham Patel |

---

## 👥 Team

<div align="center">

| | Ayush Acharya | Pratham Patel | Aman Gill |
|--|:-------------:|:-------------:|:---------:|
| **Role** | AI/ML Lead & Scrum Master | Data & Testing Lead | Dashboard & Docs Lead |
| **Hours** | 109 hrs | 61 hrs | 47 hrs |
| **Key work** | YOLO11n training pipeline, `vehicle_counter.py`, DeepSORT, GDPR audit, risk management | 11,582-image dataset annotation, UA-DETRAC benchmark validation, testing | `app.py` dashboard, exhibition poster, all documentation |

**Total team effort: 217 hours · Labour value: £3,320.10 @ £15.30/hr**

</div>

---

## 🔮 What's Next (v2.0)

- 🏍️ **Motorcycle & bicycle detection** — complete the Bradford Council brief
- ⚡ **NVIDIA Jetson deployment** — edge GPU at actual road junctions
- 🎥 **Multi-camera support** — junction-wide simultaneous coverage
- 🗺️ **GIS hotspot mapping** — visualise traffic across all of Bradford
- 🌙 **Night-time testing** — untested condition needed for year-round deployment
- 🐳 **Docker containerisation** — reproducible deployment across Council hardware
- 📡 **Live CCTV integration** — connect to Bradford Council's existing network

---

## 🛠 Technical Stack

```
100% open source · Zero recurring cost · Runs on existing Council hardware
```

| Layer | Technology | Licence |
|-------|-----------|---------|
| Object Detection | Ultralytics YOLO11n | AGPL-3.0 |
| Multi-Object Tracking | DeepSORT | Open source |
| Video Capture | OpenCV 4.x | Apache 2.0 |
| Dashboard | Streamlit + Plotly + Pandas | Apache 2.0 |
| Dataset Platform | Roboflow (free tier) | — |
| Language | Python 3.13 | PSF |
| **Total software cost** | | **£0** |

---

<div align="center">

---

**Group G25 · BSc Applied Artificial Intelligence · University of Bradford · 2025/2026**

Supervised by **Dr Kulwinder Panesar** · Client: **Bradford Council**

*Built to replace manual traffic surveys with real-time AI — and it worked.*

</div>
