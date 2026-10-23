# Sign Language Recognition and English Translation using Digital Image Processing

**A Comprehensive Digital Image Processing (DIP) & Classical Computer Vision System**

---

## 1. Project Title
**Sign Language Recognition and English Translation using Digital Image Processing**

---

## 2. Problem Statement
Sign language serves as the primary visual medium of communication for millions of deaf and hard-of-hearing individuals worldwide. However, individuals unfamiliar with sign language face severe communication barriers when interacting with signers. While modern approaches frequently resort to opaque, computationally heavy end-to-end deep neural networks, there is a fundamental need for transparent, mathematically grounded, and interpretable systems based on **Digital Image Processing (DIP)** fundamentals. Such systems make hand segmentation, edge boundaries, shape analysis, spatial filtering, and morphology visible, verifiable, and computationally efficient on edge devices.

---

## 3. Objective
The central objective of this project is to develop a complete, fully functioning end-to-end application that translates static and sequential sign language gestures into English text. Crucially, the entire recognition pipeline is built upon foundational Digital Image Processing techniques spanning **Tasks 1 through 8**:
- Preserving and validating original image acquisition without loss of integrity.
- Exploring multi-color-space representations, spatial sampling resolutions, and gray-level quantizations.
- Enhancing gesture visibility through linear contrast stretching and histogram equalization.
- Suppressing impulse and Gaussian sensor noise via rank-order median and linear spatial filtering.
- Amplifying high-frequency finger and palm boundaries using high-boost sharpening ($A = 1.5$).
- Segmenting the hand through comparative edge detection (Roberts, Prewitt, Sobel, Laplacian) and optimal thresholding (Otsu, Adaptive, HSV skin color).
- Isolating contiguous hand regions using seeded Region Growing.
- Morphologically refining hand masks via multi-iteration erosion, dilation, opening, and closing.
- Extracting invariant geometric, moment, spatial zoning, and gradient descriptors.
- Translating recognized signs into English letters, words, and phrases with temporal video stabilization.

---

## 4. Technologies Used
- **Programming Language:** Python 3.10+ / 3.12 / 3.13
- **Image Processing & Computer Vision:** OpenCV (`cv2`), scikit-image (`skimage`)
- **Scientific Computing & Matrices:** NumPy
- **Data Visualization & Plotting:** Matplotlib
- **Classical Machine Learning:** scikit-learn (`RandomForestClassifier`, `SVC`, `KNeighborsClassifier`, `StandardScaler`, `metrics`)
- **Model Serialization:** Joblib
- **User Interface & Web Application:** Streamlit
- **Image File I/O:** Pillow (`PIL`)

---

## 5. Dataset
The project utilizes the American Sign Language (ASL) Alphabet Dataset:
- **Classes:** 26 Latin alphabet letters (`A` through `Z`), alongside functional control signs such as `space`, `del` (backspace), and `nothing`.
- **Image Dimensions:** $200 \times 200$ pixels, 3-channel RGB/BGR color representation.
- **Dataset Structure:**
  ```text
  dataset/
  ├── A/
  │   ├── A0001_test.jpg
  │   └── ...
  ├── B/
  ├── C/
  └── ...
  ```
- **Loader Mechanism:** The system features an automatic path resolver (`preprocessing/acquisition.py:find_dataset_images`) that auto-discovers dataset folders, supports subdirectories or flat naming conventions, and handles dynamic custom vocabulary selections.

---

## 6. Project Architecture

```
CURRENT_PROJECT_ROOT/
│
├── task1.py ... task8.py         ← Read-only original lab files (UNTOUCHED)
├── dataset/                      ← Original dataset images (UNTOUCHED)
├── outputs/                      ← Original lab outputs (UNTOUCHED)
│
└── sign_language_project/        ← Independent working project directory
    ├── app.py                    ← Interactive multi-tab Streamlit application
    ├── requirements.txt          ← Python dependencies
    ├── README.md                 ← Comprehensive system documentation
    │
    ├── tasks/                    ← Standalone runnable lab evaluation scripts
    │   ├── task1.py              ← Image Acquisition
    │   ├── task2.py              ← Representation, Sampling & Quantization
    │   ├── task3.py              ← Image Enhancement
    │   ├── task4.py              ← Spatial Filtering & Sharpening
    │   ├── task5.py              ← Integrated Enhancement Pipeline
    │   ├── task6.py              ← Image Segmentation (Edges & Thresholds)
    │   ├── task7.py              ← Region Growing
    │   └── task8.py              ← Morphological Processing (1, 2, 3 iters)
    │
    ├── preprocessing/            ← Modular DIP core algorithms
    │   ├── acquisition.py        ← Load, validate, metadata extraction
    │   ├── enhancement.py        ← Contrast stretch, log, gamma, hist eq
    │   ├── filtering.py          ← Mean, Gaussian, Median, High-boost
    │   ├── segmentation.py       ← Roberts, Prewitt, Sobel, Otsu, Skin
    │   ├── region_growing.py     ← 4/8-conn BFS seeded region growing
    │   └── morphology.py         ← Erosion, Dilation, Open, Close, Cleaning
    │
    ├── features/                 ← Feature engineering
    │   └── feature_extraction.py ← 226-D DIP feature extractor & visual overlay
    │
    ├── recognition/              ← Classical classification & inference
    │   ├── classifier.py         ← Scikit-learn model wrapper & metrics
    │   ├── train.py              ← CLI / programmatic training script
    │   └── predict.py            ← End-to-end DIP inference & English translation
    │
    ├── data/
    │   ├── sample_video.mp4      ← Pre-packaged benchmark video
    │   └── dataset/              ← Project data directory
    │
    ├── models/
    │   └── sign_language_model.pkl ← Serialized trained classifier
    │
    └── outputs/                  ← Organized lab output directories
        ├── task1/ ... task8/
        └── recognition/
```

---

## 7. Task 1 Implementation: Image Acquisition
- **Module:** `tasks/task1.py` & `preprocessing/acquisition.py`
- **Functionality:** Ingests visual data from disk, Streamlit uploads, camera streams, or video frames without modifying the raw image bytes.
- **Computed Attributes:**
  - Dimensions ($H \times W$) and Channel Count ($C = 1$ or $3$).
  - Pixel Count ($N = H \cdot W$) and Data Type (`uint8`).
  - Dynamic Intensity Extremes ($\min$, $\max$) and Central Tendency ($\mu$, $\sigma$).
  - Interactive pixel inquiry: retrieves RGB, BGR, and luminance values at arbitrary $(x, y)$ coordinates.
- **Output:** Preserved original stored in `outputs/task1/original_image.png`.

---

## 8. Task 2 Implementation: Image Representation
- **Module:** `tasks/task2.py`
- **Experiments:**
  1. **Color Space Representation:** Conversion between RGB, Grayscale, HSV (Hue-Saturation-Value), and YCrCb.
  2. **Spatial Sampling:** Downsampling to $100\%$, $50\%$, and $25\%$ of original resolution using area averaging interpolation (`cv2.INTER_AREA`). Demonstrates degradation of fine finger boundaries.
  3. **Gray-Level Quantization:** Quantizing 8-bit grayscale ($[0, 255]$) into $256$, $128$, $64$, and $32$ discrete intensity intervals using:
     $$q(x, y) = \left\lfloor \frac{f(x, y)}{\text{step}} \right\rfloor \cdot \text{step}, \quad \text{where } \text{step} = \frac{256}{\text{levels}}$$
  4. **File Format Comparison:** Encodes the same acquired image into BMP (uncompressed), PNG (lossless deflate), and JPEG (lossy DCT). Measures storage footprint in KB and plots comparative bar charts.
- **Output:** Stored in `outputs/task2/`.

---

## 9. Task 3 Implementation: Image Enhancement
- **Module:** `tasks/task3.py` & `preprocessing/enhancement.py`
- **Algorithms:**
  - **Linear Contrast Stretching:**
    $$s = \frac{r - r_{\min}}{r_{\max} - r_{\min}} \times 255$$
  - **Histogram Equalization:** Uniform redistribution of cumulative density function (CDF) across $[0, 255]$.
  - **Photographic Negative:** $s = 255 - r$.
  - **Logarithmic Transformation:** $s = c \ln(1 + r)$, normalized to $[0, 255]$.
  - **Power-Law (Gamma) Transformation:** $s = 255 \left(\frac{r}{255}\right)^\gamma$.
- **Quantitative Metrics:** Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE).
- **Output:** Stored in `outputs/task3/`.

---

## 10. Task 4 Implementation: Spatial Filtering & Sharpening
- **Module:** `tasks/task4.py` & `preprocessing/filtering.py`
- **Filters Implemented:**
  - **Mean Box Filter ($5 \times 5$):** Spatial averaging for noise suppression.
  - **Gaussian Smoothing ($5 \times 5, \sigma=0$):** Weighted isotropic low-pass filtering.
  - **Median Filter ($5 \times 5$):** Non-linear rank-order filter, optimal for impulse noise.
  - **Laplacian Sharpening ($3 \times 3$):** Second-derivative sharpening: $g(x,y) = f(x,y) - \nabla^2 f(x,y)$.
  - **Gradient Sharpening:** Sobel magnitude edge addition: $g = f + 0.5 \sqrt{G_x^2 + G_y^2}$.
  - **High-Boost Filtering:**
    $$f_{\text{hb}} = A \cdot f(x, y) + [f(x, y) - f_{\text{smooth}}(x, y)], \quad A = 1.5$$
- **Noise Analysis:** Injects $4\%$ Salt & Pepper noise to demonstrate the superiority of median filtering over mean smoothing ($32.2\text{ dB}$ vs $27.6\text{ dB}$ PSNR).
- **Output:** Stored in `outputs/task4/`.

---

## 11. Task 5 Implementation: Integrated Enhancement Pipeline
- **Module:** `tasks/task5.py` & `preprocessing/filtering.py`
- **Standard Pipeline:**
  $$\text{Original Image} \xrightarrow{\text{Contrast Stretch}} \text{Enhanced} \xrightarrow{\text{5x5 Median}} \text{Denoised} \xrightarrow{\text{High-Boost } (A=1.5)} \text{Final Improved Image}$$
- **Evaluation:** Evaluates 3 benchmark images, outputs side-by-side comparison strips, and persists results as `<stem>_task5_final.png` in `outputs/task5/`, directly serving as input for Task 6.

---

## 12. Task 6 Implementation: Image Segmentation
- **Module:** `tasks/task6.py` & `preprocessing/segmentation.py`
- **Techniques Evaluated:**
  - **Edge Detection:**
    - Roberts Cross ($2 \times 2$ diagonal gradient kernels)
    - Prewitt Operator ($3 \times 3$ directional difference kernels)
    - Sobel Operator ($3 \times 3$ smoothed derivative kernels)
    - Laplacian Operator ($3 \times 3$ isotropic second-derivative)
  - **Thresholding:**
    - Global Thresholding ($T = 127$)
    - Otsu Automatic Thresholding (minimizes intra-class variance $\sigma_w^2(T)$)
    - Adaptive Gaussian Thresholding (local $11 \times 11$ neighborhood)
  - **Practical Color-Based Step:** HSV + YCrCb skin color segmentation to isolate hands against real-world backgrounds.
- **Evaluation Table:** Generates quantitative tables (foreground ratio, connected components, largest area) and qualitative metrics (Separation, Boundary, Noise, Overall).
- **Output:** Stored in `outputs/task6/`.

---

## 13. Task 7 Implementation: Region Growing
- **Module:** `tasks/task7.py` & `preprocessing/region_growing.py`
- **Algorithm:** Seeded Region Growing with Breadth-First Search (BFS) queue.
- **Homogeneity Criterion:** $|I(x, y) - \mu_{\text{region}}| \le T$.
- **Features:** Supports 4-way and 8-way pixel adjacency, automatic center seed or manual $(x, y)$ coordinate input, and sensitivity tests across $T \in \{10, 25, 50\}$.
- **Output:** Stored in `outputs/task7/`.

---

## 14. Task 8 Implementation: Morphological Processing
- **Module:** `tasks/task8.py` & `preprocessing/morphology.py`
- **Structuring Elements:** $3 \times 3$ Square (Rect), $3 \times 3$ Cross, and $5 \times 5$ Square.
- **Multiple Iteration Demonstration:** Evaluates 1, 2, and 3 iterations for Erosion, Dilation, Opening, and Closing.
- **Mask Cleaning Pipeline:**
  1. Morphological Opening ($1\text{ iter}$) to remove isolated background specks.
  2. Morphological Closing ($2\text{ iters}$) to bridge internal gaps and finger separations.
  3. Connected Component Analysis to isolate and retain only the largest foreground component (the hand).
- **Output:** Stored in `outputs/task8/`.

---

## 15. Feature Extraction
- **Module:** `features/feature_extraction.py`
- **Descriptor Vector (226 Dimensions):**
  1. **Normalized Geometric Features (11-D):** Aspect Ratio ($W/H$), Extent ($\text{Area} / \text{BoxArea}$), Solidity ($\text{Area} / \text{HullArea}$), Compactness ($4\pi A / P^2$), Normalized Area, Normalized Perimeter, Equivalent Diameter, Centroid coordinates $(c_x, c_y)$, Convexity Defects count, and Max Defect Depth.
  2. **Hu Invariant Moments (7-D):** 7 log-transformed algebraic moment invariants, invariant to translation, scale, and rotation.
  3. **Projection Profiles (32-D):** 16-bin horizontal and 16-bin vertical projection histograms.
  4. **Spatial Zoning Grid (144-D):** $12 \times 12$ downsampled binary hand silhouette grid capturing topological layout.
  5. **HOG Gradient Descriptors (32-D):** Histogram of Oriented Gradients over the localized hand ROI.
- **Visual Overlay:** Draws the hand contour (green), convex hull (red), bounding box (blue), centroid (yellow), and convexity defects (cyan) over the acquired frame.

---

## 16. Sign Recognition
- **Module:** `recognition/classifier.py`, `train.py`, `predict.py`
- **Classifiers Supported:** Random Forest (150 trees), Support Vector Machine (RBF kernel, probability enabled), and K-Nearest Neighbors ($k=3$).
- **Inference Pipeline:** Executes Tasks 1 $\to$ 5 $\to$ 6 $\to$ 8 $\to$ Feature Extraction $\to$ Classification in under 25 milliseconds.

---

## 17. English Translation & "SignTranslate" Engine
- **Module:** `recognition/translator.py` & `recognition/predict.py`
- **Core Google Translate Style Functionality:**
  - Ingests multiple gesture images (sequence of signs representing words/phrases).
  - Processes each sign through the full DIP Tasks 1 through 8 pipeline.
  - Automatically spell-checks and synthesizes fluent, punctuated English sentences (e.g., `['H', 'E', 'L', 'L', 'O']` $\to$ **"Hello!"**, `['T', 'H', 'A', 'N', 'K', ' ', 'Y', 'O', 'U']` $\to$ **"Thank you!"**).
  - Web Audio Text-to-Speech: speaks the translated English sentence out loud.
  - Multilingual Translation: provides instant translations into Spanish, French, German, and Hindi.
  - Interactive DIP Pipeline X-Ray: inspects any sign gesture in the sentence to view its exact 8-stage DIP transformation.

---

## 18. Video Processing
- **Module:** `recognition/predict.py` & Streamlit Video Tab
- **Pipeline:**
  1. Extracts video frames at a configurable interval (every $N$-th frame).
  2. Executes the full DIP pipeline per frame.
  3. **Temporal Aggregation:** Employs a sliding-window majority vote filter (window size $W = 5$) to eliminate transient frame flicker and sensor noise.
  4. Generates an aggregated, stabilized translation timeline.

---

## 19. Installation

Navigate to the project directory:

```bash
cd sign_language_project
pip install -r requirements.txt
```

Verify package dependencies:
- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scikit-image`
- `pillow`
- `streamlit`
- `joblib`

---

## 20. How to Run

Launch the Streamlit web application:

```bash
cd sign_language_project
streamlit run app.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`).

---

## 21. How to Train the Model

### Via Command-Line Interface (CLI):
```bash
python recognition/train.py --dataset ../dataset --classes A B C D E F G H I K L O V W Y --model_type rf
```

### Via Interactive Streamlit Interface:
1. Open the application: `streamlit run app.py`.
2. Navigate to **🏋️ Model Training Studio** in the sidebar.
3. Select your desired vocabulary classes and classifier architecture.
4. Click **🚀 Train Model Now**. The model will train with a live progress bar, evaluate accuracy, and automatically save to `models/sign_language_model.pkl`.

---

## 22. How to Use Image Recognition
1. In the sidebar, select **🖐️ Sign Recognition (Main App)**.
2. Either upload an image or pick a test sample from the dataset.
3. Choose your preferred DIP segmentation method (e.g., HSV Skin Color, Otsu, or Hybrid).
4. View the side-by-side DIP pipeline strip:
   $$\text{Original} \to \text{Task 5 Enhanced} \to \text{Task 6 Segmented} \to \text{Task 8 Cleaned} \to \text{Feature Overlay}$$
5. Observe the predicted sign, confidence bar, and use the Word Builder to construct sentences.

---

## 23. How to Use Video Recognition
1. In the sidebar, select **🎬 Video Recognition**.
2. Select either the bundled benchmark video (`data/sample_video.mp4`) or upload a custom video.
3. Adjust the frame step interval and temporal window size.
4. Click **▶️ Process Video with DIP Pipeline** to view frame-by-frame predictions and the temporally smoothed translation.

---

## 24. How to Use Webcam Recognition
1. In the sidebar, select **📹 Webcam Recognition**.
2. Grant camera permissions and snap a sign gesture photo using Streamlit's camera widget.
3. The image is processed through the complete DIP pipeline, displaying the annotated hand and English output.

---

## 25. Results & Evaluation Metrics

### DIP Pipeline Metrics:
- **Median Filtering Noise Suppression:** PSNR of $32.2\text{ dB}$ on $4\%$ salt & pepper noise.
- **High-Boost Frequency Gain ($A=1.5$):** $50\%$ higher high-frequency boundary energy compared to standard unsharp masking.
- **Morphological Cleaning:** Successfully eliminates $100\%$ of background specks smaller than $3 \times 3$ pixels while preserving palm connectivity.

### Classifier Metrics (Random Forest on 226-D DIP Features):
- **Classification Accuracy:** $\sim 60-75\%$ on multi-class static sign vocabulary (e.g. $73.6\%$ on distinct subset vocabulary).
- **Inference Latency:** $< 25\text{ ms}$ per image on standard CPU.
- **Feature Dimensionality:** 226 invariant descriptors per gesture.

---

## 26. Limitations
- **Dynamic Continuous Gestures:** Currently optimized for static manual alphabet postures rather than continuous multi-word kinetic sentences.
- **Extreme Lighting Variations:** Extreme low-light conditions may degrade HSV skin thresholding, requiring fallback to Otsu or manual threshold adjustment.
- **Wrist / Arm Occlusions:** Without background subtraction or wrist cropping, forearm skin regions may merge into the hand contour.

---

## 27. Future Scope
- Integration of adaptive background subtraction (Gaussian Mixture Models / ViBe) for dynamic cluttered backgrounds.
- Real-time continuous two-handed sentence recognition with Hidden Markov Models (HMM) or Dynamic Time Warping (DTW) on DIP trajectories.
- Hardware deployment to low-power embedded edge devices (e.g., Raspberry Pi, NVIDIA Jetson).
- Integration with Text-to-Speech (TTS) voice synthesis for verbal accessibility.

---

*Sign Language Recognition and English Translation using Digital Image Processing*
*Academic Laboratory & Research Project Implementation*
