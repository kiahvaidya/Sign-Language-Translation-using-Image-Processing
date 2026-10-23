# Sign Language Translation Using Digital Image Processing & Vision

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.28%25-brightgreen.svg)](#evaluation-metrics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, real-time American Sign Language (ASL) to English translation system integrating classical **Digital Image Processing (DIP Tasks 1–8)** with deep vision classification and two-tier natural language synthesis. Features a modern, minimal, Google Translate-style web application.

---

## Key Features

- **Multi-Modal Input:** Upload images (with drag-and-drop sequence reordering), capture live webcam frames with a target viewfinder, or process recorded video files.
- **Full DIP Pipeline Integration (Tasks 1–8):**
  - **Task 1:** Multi-format image acquisition & channel validation.
  - **Task 2:** Spatial resolution normalization ($224 \times 224$) and color space transforms (RGB, Grayscale, HSV, YCrCb).
  - **Task 3:** Dynamic range contrast stretching ($f_{\text{norm}}$).
  - **Task 4:** $5 \times 5$ median spatial noise filtering.
  - **Task 5:** High-boost frequency amplification ($A = 1.5$) to sharpen finger boundaries.
  - **Task 6 & 7:** Adaptive background-aware segmentation (Inverted Otsu for studio white backgrounds, HSV+YCrCb fusion for room backgrounds, and seed-based region growing).
  - **Task 8:** Morphological opening/closing and Euclidean distance transform inradius calculation for safe forearm excision.
- **Deep Vision Classification:** MobileNetV3-Small neural network trained with multi-background synthesis on Apple Silicon GPU (`mps`), achieving **98.28% test accuracy** across all 29 ASL alphabet classes.
- **Two-Tier NLP & Sentence Synthesis:** Exact dictionary validation against 237,000+ English words + length-bounded fuzzy autocorrection (e.g. `H-O-L-L-O-W` $\to$ `"Hollow."`, `H-B-L-L-O` $\to$ `"Hello!"`).
- **Technical Inspection Modal:** *"Inspect DIP Pipeline"* inspection window allowing examiners and researchers to view all intermediate filter stages and quantitative metrics.

---

## Evaluation Metrics

Evaluated on a **stratified holdout test set of 174 unseen samples across all 29 ASL classes**:

| Metric | Measured Score | Status |
| :--- | :---: | :---: |
| **Overall Accuracy** | **98.28%** | **Exceeds Target ($\ge 93\%$)** |
| **Macro Precision** | **98.52%** | **Optimal** |
| **Weighted Precision** | **98.52%** | **Optimal** |
| **Macro Recall** | **98.28%** | **Optimal** |
| **Weighted Recall** | **98.28%** | **Optimal** |
| **Macro F1-Score** | **98.26%** | **Optimal** |
| **Weighted F1-Score** | **98.26%** | **Optimal** |
| **Inference Latency** | **$\sim 18\text{ ms}$** | **$> 45\text{ FPS}$ Live** |

Detailed per-class performance and confusion metrics are documented in [`PRESENTATION_OVERVIEW.md`](PRESENTATION_OVERVIEW.md).

---

## Project Structure

```
SignLanguage_ImageProcessing/
├── asl_translator/                 # Main Translation Application
│   ├── backend/                    # FastAPI Backend Server
│   │   ├── api.py                  # REST API endpoints
│   │   ├── config.py               # Model paths & system settings
│   │   ├── dip_pipeline.py         # Tasks 3-5 enhancement pipeline
│   │   ├── hand_segmentation.py    # Tasks 6-7 adaptive segmentation
│   │   ├── image_preprocessing.py  # Tasks 1-2 acquisition & normalization
│   │   ├── morphological_processing.py # Task 8 morphological cleaning & wrist pruning
│   │   ├── sign_recognition.py     # MobileNetV3 inference engine & metrics
│   │   ├── translation_service.py  # Pipeline coordinator & NLP autocorrect
│   │   ├── webcam_processor.py     # Live viewfinder processing
│   │   ├── video_processor.py      # Video temporal aggregation
│   │   ├── train_vision_model.py   # Vision model training & evaluation script
│   │   └── models/                 # Model checkpoints & metrics JSON
│   ├── frontend/                   # Modern React 18 + Vite Web App
│   │   ├── src/                    # Components (Image, Webcam, Video, Output)
│   │   └── package.json            # Frontend dependencies
│   └── start.sh                    # One-click startup script
├── dataset/                        # Benchmark ASL Alphabet Dataset (29 classes)
├── task2.py - task8.py             # Standalone reference DIP assignment tasks
├── PRESENTATION_OVERVIEW.md        # Comprehensive poster & presentation guide
└── README.md                       # Project overview & documentation
```

---

## Quick Start Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13 macOS Apple Silicon)
- Node.js 18+ and npm

### 1. One-Click Launcher
```bash
chmod +x asl_translator/start.sh
./asl_translator/start.sh
```
This automatically launches:
- **FastAPI Backend:** [http://localhost:8000](http://localhost:8000) (Interactive Swagger docs at `/docs`)
- **React Frontend:** [http://localhost:5173](http://localhost:5173)

---

### 2. Manual Startup

#### Backend
```bash
cd asl_translator/backend
python3 -m pip install -r ../../sign_language_project/requirements.txt
python3 -m pip install torch torchvision fastapi uvicorn scikit-learn
python3 run_backend.py
```

#### Frontend
```bash
cd asl_translator/frontend
npm install
npm run dev
```

---

## Poster & Presentation Documentation

For academic poster printing, presentation slide scripts, mathematical formulations, and viva defense Q&A preparation, see:
- **[Comprehensive Project & Presentation Guide (PRESENTATION_OVERVIEW.md)](PRESENTATION_OVERVIEW.md)**

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
