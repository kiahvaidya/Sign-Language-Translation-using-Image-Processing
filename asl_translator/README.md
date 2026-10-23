# SignTranslate: ASL-to-English Translator Using Digital Image Processing

An end-to-end, real-time American Sign Language (ASL) to English translation system built using classical **Digital Image Processing (DIP Tasks 1–8)** and invariant geometric feature recognition. Designed with a clean, modern "Google Translate" style interface.

---

## 1. System Architecture

```
User Input (Image / Video / Webcam)
          │
          ▼
   [Task 1 & 2: Acquisition & Representation]
   • 200x200 Normalization, Color Spaces (RGB, HSV, YCrCb)
   • Intensity Bounds & Metadata Extraction
          │
          ▼
   [Task 3, 4 & 5: Integrated Enhancement]
   • Min-Max Contrast Stretching (Task 3)
   • 5x5 Median Filter Noise Suppression (Task 4)
   • High-Boost Spatial Sharpening A=1.5 (Task 5)
          │
          ▼
   [Task 6 & 7: Hand Segmentation & Edge Detection]
   • HSV & YCrCb Dual-Channel Skin Color Fusion
   • Gradient Operators (Sobel, Prewitt, Laplacian)
   • Otsu Automatic Binarization & Region Growing
          │
          ▼
   [Task 8: Morphological Processing]
   • Morphological Opening (Salt Noise Removal)
   • Palm Center Isolation via Distance Transform
   • Boundary-Conditioned Forearm Excision
   • Morphological Closing (Cavity Filling)
          │
          ▼
   [DIP Invariant Feature Extraction (226 Dimensions)]
   • 11 Geometric Descriptors (Area, Perimeter, Aspect Ratio, Solidity, Extent, Compactness)
   • 7 Scale & Rotation Invariant Hu Moments (log10)
   • 32 Horizontal & Vertical Projection Profiles
   • 144 Spatial Zoning Block Densities (12x12)
   • 32 Directional Histogram of Oriented Gradients (HOG)
          │
          ▼
   [Recognition & Temporal Aggregation]
   • 29-Class ASL Gesture Recognition
   • Temporal Majority Voting across Video Frames
   • Fuzzy Levenshtein Auto-Correction & Sentence Formatting
   • Multilingual Translation (English, Spanish, French, German, Hindi)
          │
          ▼
   Translated English Text & Intermediate DIP Verification Modal
```

---

## 2. DIP Tasks 1–8 Integration Matrix

| Lab Task | Name | Implementation Module | Application Role |
|---|---|---|---|
| **Task 1** | Image Acquisition | `backend/image_preprocessing.py` | Ingests multi-format images (JPEG, PNG, BMP), video frames, and webcam feeds while preserving original source data. Extracts width, height, channels, and min/max intensity. |
| **Task 2** | Image Representation | `backend/image_preprocessing.py` | Handles RGB, Grayscale, HSV, and YCrCb color models; pixel matrix quantization, normalized $200 \times 200$ dimensional representation. |
| **Task 3** | Image Enhancement | `backend/dip_pipeline.py` | 5%–95% min-max dynamic range contrast stretching to normalize varied lighting conditions. |
| **Task 4** | Spatial Filtering | `backend/dip_pipeline.py` | 5×5 non-linear median filtering to eliminate camera sensor grain and salt-and-pepper noise. |
| **Task 5** | High Boost Sharpening | `backend/dip_pipeline.py` | Laplacian unsharp masking with amplification factor $A = 1.5$ to accentuate finger edges and knuckle boundaries. |
| **Task 6** | Edge Detection | `backend/hand_segmentation.py` | Applies Sobel, Prewitt, Roberts, and Laplacian operators to detect structural finger boundaries. |
| **Task 7** | Region Segmentation | `backend/hand_segmentation.py` | Fuses HSV $([0, 20, 70] \to [25, 255, 255])$ and YCrCb $([0, 133, 77] \to [255, 173, 127])$ with Otsu thresholding and seed-based region growing. |
| **Task 8** | Morphological Processing | `backend/morphological_processing.py` | 5×5 elliptical opening to strip noisy artifacts, Euclidean palm distance transform, smart boundary-checking forearm excision, and closing to seal palm cavities. |

---

## 3. UI/UX Design System

- **Warm Cream Background:** `#f2e7e8`
- **Soft Pink Accent:** `#ff79ac` (Primary actions, capture buttons)
- **Soft Teal Accent:** `#a6d5e4` (Recognized sign badges, secondary highlights)
- **Dark Readable Text:** `#2b2728`
- **Typography:** Poppins font family
- **Visual Design Rules:** Strictly **no gradients**, subtle solid borders, rounded cards (`14px`), responsive two-column workspace.
- **Examiner/Viva Tool:** Integrated *"Inspect DIP Pipeline"* modal allowing technical reviewers to view all intermediate image states and quantitative descriptors without cluttering the primary translation interface.

---

## 4. Quick Start Guide

### Option 1: One-Click Unified Launcher (Recommended)
From the project root:
```bash
./asl_translator/start.sh
```
This automatically starts:
- **FastAPI Backend:** `http://localhost:8000` (API Docs at `http://localhost:8000/docs`)
- **Vite React Frontend:** `http://localhost:5173`

---

### Option 2: Running Components Independently

#### Terminal 1 — Backend:
```bash
cd asl_translator/backend
python3 run_backend.py
```

#### Terminal 2 — Frontend:
```bash
cd asl_translator/frontend
npm run dev
```
Open `http://localhost:5173` in any modern web browser.

---

## 5. Feature Verification & Testing

1. **Preset One-Click Validation (Image Tab):**
   - Click any preset button: **HELLO**, **CAB**, **CAT**, **HELP**, **THANK YOU**, **WELCOME**.
   - Output immediately renders the translated English sentence, ASL letter chips, confidence badge, and multilingual translations.
   - Click **Inspect DIP Pipeline** to view the intermediate outputs (Acquisition $\to$ High-Boost $\to$ Raw Mask $\to$ Cleaned Morphology $\to$ Contours/Geometry) and the 226-D feature metrics.

2. **Single / Multi-Image Upload (Image Tab):**
   - Drag and drop any image from `dataset/asl-alphabet-test/` or your computer.
   - Multi-image uploads spell out words with automated Levenshtein dictionary smoothing.

3. **Webcam Real-Time Translation (Webcam Tab):**
   - Click **Start Camera** and grant browser camera permissions.
   - Position your hand inside the **Target Viewfinder Box** (eliminates background body/face noise).
   - Click **Capture & Translate** or enable **Live: ON** for continuous translation.

4. **Video Translation (Video Tab):**
   - Click **Use Sample Video** $\to$ **Translate Video to Text**.
   - Demonstrates frame extraction, per-frame DIP segmentation, temporal majority voting, and sentence synthesis.

---

## 6. Project Safety Guarantee

- Existing root files (`task1.py` through `task8.py`, `dataset/`, `outputs/`, `requirements.txt`, etc.) are completely unmodified and remain independently runnable.
- All new code is strictly self-contained within `asl_translator/`.
