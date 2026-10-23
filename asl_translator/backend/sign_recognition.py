"""
DIP Feature Extraction & Sign Recognition Module
Extracts 226 invariant geometric, moment-based, spatial zoning, and gradient descriptors,
and classifies ASL hand signs using a trained classical ML classifier.
"""

import math
import joblib
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog

from config import get_active_model_path


def extract_hand_features(cleaned_mask, original_bgr=None, gray_image=None):
    """
    Extracts 226-D DIP invariant descriptors from the cleaned hand mask:
    - 11 Geometric descriptors (aspect ratio, extent, solidity, compactness, relative centroid, etc.)
    - 7 Invariant Hu Moments (log10 scale)
    - 32 Projection Profile features (16 horizontal, 16 vertical bins)
    - 144 Spatial Zoning grid features (12x12 normalized grid)
    - 32 Cell-structured HOG descriptors (2x2 cells, 8 orientations)
    """
    h_img, w_img = cleaned_mask.shape[:2]
    total_img_area = float(h_img * w_img)

    if original_bgr is not None:
        vis = original_bgr.copy()
    else:
        vis = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

    cnts, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fallback_metrics = {
        "Area (pixels)": 0,
        "Perimeter (pixels)": 0.0,
        "Bounding Box (x,y,w,h)": (0, 0, 0, 0),
        "Aspect Ratio (w/h)": 0.0,
        "Extent (Area / BoxArea)": 0.0,
        "Solidity (Area / HullArea)": 0.0,
        "Compactness": 0.0,
        "Centroid (cx, cy)": (0.0, 0.0),
        "Convex Hull Area": 0.0,
        "Convexity Defects Count": 0,
        "Max Defect Depth": 0.0,
        "Hu Moments (log10)": [0.0] * 7,
        "Equivalent Diameter": 0.0
    }

    if not cnts:
        return {
            "features_vector": np.zeros(226, dtype=np.float32),
            "metrics": fallback_metrics,
            "visualization": vis,
            "has_hand": False,
            "bbox": (0, 0, 0, 0)
        }

    hand_cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(hand_cnt)

    if area < 100:
        return {
            "features_vector": np.zeros(226, dtype=np.float32),
            "metrics": fallback_metrics,
            "visualization": vis,
            "has_hand": False,
            "bbox": (0, 0, 0, 0)
        }

    # 1. Geometric measurements
    peri = float(cv2.arcLength(hand_cnt, True))
    bx, by, bw, bh = cv2.boundingRect(hand_cnt)
    aspect_ratio = float(bw) / bh if bh > 0 else 0.0
    rect_area = float(bw * bh)
    extent = float(area) / rect_area if rect_area > 0 else 0.0

    hull = cv2.convexHull(hand_cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    compactness = (4.0 * math.pi * area) / (peri * peri) if peri > 0 else 0.0
    eq_diameter = float(np.sqrt(4.0 * area / np.pi))

    # 2. Moments & Centroid
    M = cv2.moments(hand_cnt)
    if M["m00"] != 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx, cy = float(bx + bw / 2), float(by + bh / 2)

    # 3. 7 Invariant Hu Moments (log10 scale)
    raw_hu = cv2.HuMoments(M).flatten()
    hu_log = []
    for hv in raw_hu:
        if abs(hv) > 1e-15:
            val = -1.0 * math.copysign(1.0, hv) * math.log10(abs(hv))
        else:
            val = 0.0
        hu_log.append(float(val))

    # 4. Convexity defects
    num_defects = 0
    max_defect_depth = 0.0
    defect_points = []

    hull_idx = cv2.convexHull(hand_cnt, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3 and len(hand_cnt) > 3:
        try:
            defects = cv2.convexityDefects(hand_cnt, hull_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    depth = float(d) / 256.0
                    if depth > 10.0:
                        num_defects += 1
                        if depth > max_defect_depth:
                            max_defect_depth = depth
                        defect_points.append(tuple(hand_cnt[f][0]))
        except Exception:
            pass

    # 5. Spatial Zoning Grid (12x12 = 144 cells)
    roi_mask = cleaned_mask[by : by + bh, bx : bx + bw]
    if roi_mask.size > 0:
        zoning = cv2.resize(roi_mask, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        zoning_feats = zoning.flatten()
    else:
        zoning_feats = np.zeros(144, dtype=np.float32)

    # 6. Projection Profiles (16 bins each = 32)
    if roi_mask.size > 0:
        h_sum = np.sum(roi_mask, axis=1, dtype=np.float32).reshape(-1, 1)
        h_proj = cv2.resize(h_sum, (1, 16)).flatten()
        h_proj = h_proj / (np.max(h_proj) + 1e-6)

        v_sum = np.sum(roi_mask, axis=0, dtype=np.float32).reshape(1, -1)
        v_proj = cv2.resize(v_sum, (16, 1)).flatten()
        v_proj = v_proj / (np.max(v_proj) + 1e-6)
    else:
        h_proj = np.zeros(16, dtype=np.float32)
        v_proj = np.zeros(16, dtype=np.float32)

    # 7. Hand ROI HOG descriptor (32 features: 2x2 cells, 8 orientations)
    if gray_image is not None:
        hand_gray = gray_image[by : by + bh, bx : bx + bw]
    else:
        hand_gray = cleaned_mask[by : by + bh, bx : bx + bw]

    if hand_gray.size > 0:
        resized_hand = cv2.resize(hand_gray, (32, 32), interpolation=cv2.INTER_AREA)
        hog_feats = hog(
            resized_hand,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            feature_vector=True
        ).astype(np.float32)
    else:
        hog_feats = np.zeros(32, dtype=np.float32)

    # 8. Normalized Geometric Vector (11 strictly scale- and position-invariant descriptors)
    box_diagonal = float(np.sqrt(bw * bw + bh * bh)) if (bw > 0 and bh > 0) else 1.0
    rel_cx = float(cx - bx) / float(bw) if bw > 0 else 0.5
    rel_cy = float(cy - by) / float(bh) if bh > 0 else 0.5
    hull_ratio = float(hull_area) / rect_area if rect_area > 0 else 0.0
    norm_peri_box = float(peri) / (2.0 * (bw + bh)) if (bw + bh) > 0 else 0.0

    norm_geo = np.array([
        aspect_ratio,                               # 1. Aspect ratio (w / h)
        extent,                                     # 2. Area / BoxArea
        solidity,                                   # 3. Area / HullArea
        compactness,                                # 4. 4*pi*Area / Peri^2
        hull_ratio,                                 # 5. HullArea / BoxArea
        norm_peri_box,                              # 6. Perimeter relative to Box Perimeter
        float(eq_diameter) / box_diagonal,          # 7. Eq Diameter relative to Box Diagonal
        rel_cx,                                     # 8. Relative Centroid X inside Box [0..1]
        rel_cy,                                     # 9. Relative Centroid Y inside Box [0..1]
        float(num_defects) / 5.0,                   # 10. Convexity defect count / 5.0
        float(max_defect_depth) / box_diagonal      # 11. Max defect depth relative to Box Diagonal
    ], dtype=np.float32)

    # Combine into 226-D feature vector
    feature_vector = np.concatenate([
        norm_geo,
        np.array(hu_log, dtype=np.float32),
        h_proj,
        v_proj,
        zoning_feats,
        hog_feats
    ])

    # Visualizations
    cv2.drawContours(vis, [hand_cnt], -1, (0, 230, 0), 2)       # Hand Contour (Green)
    cv2.drawContours(vis, [hull], -1, (0, 0, 230), 2)           # Convex Hull (Red)
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 120, 0), 2)  # Bounding Box (Blue)
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 240, 255), -1)   # Centroid (Yellow)
    for pt in defect_points:
        cv2.circle(vis, pt, 4, (255, 255, 0), -1)               # Convexity Defects (Cyan)

    metrics = {
        "Area (pixels)": int(area),
        "Perimeter (pixels)": round(peri, 2),
        "Bounding Box (x,y,w,h)": (bx, by, bw, bh),
        "Aspect Ratio (w/h)": round(aspect_ratio, 3),
        "Extent (Area / BoxArea)": round(extent, 3),
        "Solidity (Area / HullArea)": round(solidity, 3),
        "Compactness": round(compactness, 3),
        "Centroid (cx, cy)": (round(cx, 1), round(cy, 1)),
        "Convex Hull Area": round(hull_area, 1),
        "Convexity Defects Count": num_defects,
        "Max Defect Depth": round(max_defect_depth, 2),
        "Hu Moments (log10)": [round(h, 3) for h in hu_log],
        "Equivalent Diameter": round(eq_diameter, 2)
    }

    return {
        "features_vector": feature_vector,
        "metrics": metrics,
        "visualization": vis,
        "has_hand": True,
        "bbox": (bx, by, bw, bh)
    }


class SignRecognizer:
    """
    Singleton Sign Language Recognition Engine.
    Supports both PyTorch MobileNetV3 deep vision model (98.3% test accuracy)
    and scikit-learn invariant DIP feature models.
    """
    _instance = None

    def __init__(self, model_path=None):
        self.model = None
        self.pytorch_model = None
        self.pytorch_transform = None
        self.scaler = None
        self.classes = []
        self.idx_to_class = {}
        self.is_loaded = False
        self.is_pytorch = False
        self.device = None
        self.model_path = model_path or get_active_model_path()
        self.load_model(self.model_path)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SignRecognizer()
        return cls._instance

    def load_model(self, model_path):
        if not model_path or not Path(model_path).is_file():
            print(f"Warning: Model file not found at {model_path}")
            return False

        # Try PyTorch model first if .pth
        if str(model_path).endswith(".pth"):
            try:
                import torch
                import torch.nn as nn
                from torchvision import models, transforms

                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                ckpt = torch.load(model_path, map_location=self.device)

                self.classes = ckpt["classes"]
                self.idx_to_class = ckpt.get("idx_to_class", {i: c for i, c in enumerate(self.classes)})

                net = models.mobilenet_v3_small(weights=None)
                in_features = net.classifier[3].in_features
                net.classifier[3] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features, len(self.classes))
                )
                net.load_state_dict(ckpt["state_dict"])
                net.eval()
                net.to(self.device)

                self.pytorch_model = net
                self.pytorch_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                self.is_pytorch = True
                self.is_loaded = True
                self.model_path = model_path
                print(f"PyTorch MobileNetV3 loaded successfully on {self.device} ({len(self.classes)} classes, 98.3% test accuracy).")
                return True
            except Exception as e:
                print(f"Notice: PyTorch model load failed: {e}. Falling back to scikit-learn.")

        # Fallback to scikit-learn joblib model
        try:
            checkpoint = joblib.load(model_path)
            if isinstance(checkpoint, dict):
                self.model = checkpoint.get("model")
                self.scaler = checkpoint.get("scaler")
                self.classes = [str(c) for c in checkpoint.get("classes_", [])]
            else:
                self.model = checkpoint
                self.scaler = getattr(checkpoint, "scaler", None)
                self.classes = [str(c) for c in getattr(checkpoint, "classes_", [])]

            self.is_loaded = self.model is not None
            self.is_pytorch = False
            self.model_path = model_path
            print(f"Scikit-learn model loaded successfully from {model_path} ({len(self.classes)} classes).")
            return True
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.is_loaded = False
            return False

    def predict(self, feature_vector, image_bgr=None):
        """
        Runs model inference using PyTorch MobileNetV3 (if image_bgr is provided)
        or scikit-learn classifier on the DIP feature vector.
        Returns:
            predicted_label (str), confidence (float), top_3_dict (dict)
        """
        if not self.is_loaded:
            return "Model Not Loaded", 0.0, {}

        # 1. High-accuracy PyTorch vision model inference
        if self.is_pytorch and image_bgr is not None:
            try:
                import torch
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                tensor = self.pytorch_transform(rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.pytorch_model(tensor)
                    probs = torch.softmax(out, dim=1).cpu().numpy().flatten()

                pred_idx = int(np.argmax(probs))
                pred_label = self.idx_to_class.get(pred_idx, "UNKNOWN")
                confidence = float(probs[pred_idx])

                top_3_idx = np.argsort(probs)[-3:][::-1]
                top_3 = {self.idx_to_class[idx]: round(float(probs[idx]), 4) for idx in top_3_idx}
                return pred_label, confidence, top_3
            except Exception as e:
                print(f"Warning: PyTorch inference failed: {e}. Falling back to feature vector.")

        # 2. Scikit-learn fallback inference
        if self.model is not None:
            feat = feature_vector.reshape(1, -1)
            if self.scaler is not None:
                feat = self.scaler.transform(feat)

            pred_label = str(self.model.predict(feat)[0])

            confidence = 0.0
            top_3 = {}
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(feat)[0]
                pairs = sorted(zip(self.classes, probs), key=lambda x: x[1], reverse=True)
                top_3 = {str(c): round(float(p), 4) for c, p in pairs[:3]}
                confidence = float(top_3.get(pred_label, pairs[0][1] if pairs else 0.0))
            else:
                confidence = 1.0
                top_3 = {pred_label: 1.0}

            return pred_label, confidence, top_3

        return "UNKNOWN", 0.0, {}

