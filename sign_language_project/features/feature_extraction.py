"""
Feature Extraction Module
Extracts comprehensive Digital Image Processing shape, contour, moment,
projection, and structural descriptors from segmented hand masks.
"""

import cv2
import numpy as np
import math
from skimage.feature import hog


def extract_hand_features(cleaned_mask, original_bgr=None, gray_image=None):
    """
    Extracts geometric, moment-based, spatial zoning, and gradient descriptors from the hand mask.

    Args:
        cleaned_mask (np.ndarray): 2D binary mask from Task 8 morphology.
        original_bgr (np.ndarray, optional): 3-channel image for visual overlay.
        gray_image (np.ndarray, optional): 2D grayscale image for texture/HOG extraction.

    Returns:
        dict: {
            'features_vector': np.ndarray,      # 1D numerical array for classifier
            'feature_names': list of str,       # Name of every feature in the vector
            'metrics': dict,                    # Key DIP measurements for UI display
            'visualization': np.ndarray,        # Overlay image with contour, hull, defects
            'has_hand': bool,                   # Whether a valid hand contour was detected
            'contour': np.ndarray,              # Hand contour points
            'bbox': tuple                       # (x, y, w, h)
        }
    """
    h_img, w_img = cleaned_mask.shape[:2]
    total_img_area = float(h_img * w_img)

    # Visualization canvas
    if original_bgr is not None:
        vis = original_bgr.copy()
    else:
        vis = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

    cnts, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Default fallback metrics
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
        "Equivalent Diameter": 0.0,
        "Num Connected Components": 0
    }

    # If no contour found or mask is empty
    if not cnts:
        zero_feat = np.zeros(128, dtype=np.float32)
        return {
            "features_vector": zero_feat,
            "feature_names": [f"f_{i}" for i in range(128)],
            "metrics": fallback_metrics,
            "visualization": vis,
            "has_hand": False,
            "contour": None,
            "bbox": (0, 0, 0, 0)
        }

    # Find the largest contour (the hand)
    hand_cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(hand_cnt)

    if area < 100:  # Noise threshold
        zero_feat = np.zeros(128, dtype=np.float32)
        return {
            "features_vector": zero_feat,
            "feature_names": [f"f_{i}" for i in range(128)],
            "metrics": fallback_metrics,
            "visualization": vis,
            "has_hand": False,
            "contour": None,
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

    # 3. 7 Invariant Hu Moments (log-scale)
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
                    if depth > 10.0:  # Threshold to ignore minor contour roughness
                        num_defects += 1
                        if depth > max_defect_depth:
                            max_defect_depth = depth
                        far_pt = tuple(hand_cnt[f][0])
                        defect_points.append(far_pt)
        except Exception:
            pass

    # 5. Spatial Zoning Grid (12x12 = 144 cells)
    roi_mask = cleaned_mask[by:by+bh, bx:bx+bw]
    if roi_mask.size > 0:
        zoning = cv2.resize(roi_mask, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        zoning_feats = zoning.flatten()
    else:
        zoning_feats = np.zeros(144, dtype=np.float32)

    # 6. Horizontal and Vertical Projection Profiles (16 bins each = 32)
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

    # 7. Basic Hand ROI HOG descriptor (36 features: 2x2 cells of 8 pixels, 9 orientations)
    if gray_image is not None:
        hand_gray = gray_image[by:by+bh, bx:bx+bw]
    else:
        hand_gray = cleaned_mask[by:by+bh, bx:bx+bw]

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

    # 8. Normalized Geometric Vector
    norm_geo = np.array([
        aspect_ratio,
        extent,
        solidity,
        compactness,
        area / total_img_area,
        peri / (2 * (w_img + h_img)),
        eq_diameter / max(w_img, h_img),
        cx / w_img,
        cy / h_img,
        float(num_defects) / 10.0,
        max_defect_depth / 100.0
    ], dtype=np.float32)

    # Combine all feature segments into full feature vector
    # norm_geo (11) + hu_log (7) + h_proj (16) + v_proj (16) + zoning_feats (144) + hog_feats (32) = 226 features
    feature_vector = np.concatenate([
        norm_geo,
        np.array(hu_log, dtype=np.float32),
        h_proj,
        v_proj,
        zoning_feats,
        hog_feats
    ])

    feature_names = (
        ["aspect_ratio", "extent", "solidity", "compactness", "norm_area", "norm_peri", "eq_diam", "cx", "cy", "defects_count", "max_defect"] +
        [f"hu_{i+1}" for i in range(7)] +
        [f"h_proj_{i}" for i in range(16)] +
        [f"v_proj_{i}" for i in range(16)] +
        [f"zoning_{i}" for i in range(144)] +
        [f"hog_{i}" for i in range(len(hog_feats))]
    )

    # Draw visualization overlay
    # Contour: Vibrant Green
    cv2.drawContours(vis, [hand_cnt], -1, (0, 230, 0), 2)
    # Convex Hull: Bright Red
    cv2.drawContours(vis, [hull], -1, (0, 0, 230), 2)
    # Bounding Box: Vivid Blue
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 120, 0), 2)
    # Centroid: Bright Yellow circle
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 240, 255), -1)
    # Convexity Defects: Cyan circles
    for pt in defect_points:
        cv2.circle(vis, pt, 4, (255, 255, 0), -1)

    # Label text on image
    label_txt = f"Area: {int(area)} | Peri: {int(peri)} | Defects: {num_defects}"
    cv2.putText(vis, label_txt, (max(5, bx), max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(vis, label_txt, (max(5, bx), max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

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
        "Equivalent Diameter": round(eq_diameter, 2),
        "Num Connected Components": 1
    }

    return {
        "features_vector": feature_vector,
        "feature_names": feature_names,
        "metrics": metrics,
        "visualization": vis,
        "has_hand": True,
        "contour": hand_cnt,
        "bbox": (bx, by, bw, bh)
    }
