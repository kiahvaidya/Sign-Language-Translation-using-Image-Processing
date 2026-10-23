"""
Task 6: Image Segmentation Module
Implements edge detection operators (Roberts, Prewitt, Sobel, Laplacian),
thresholding techniques (Global, Otsu, Adaptive), practical HSV/YCrCb skin segmentation,
and quantitative segmentation quality analysis.
"""

import cv2
import numpy as np


# ==========================================================
# EDGE DETECTION OPERATORS
# ==========================================================

def roberts_edge(gray):
    """
    Roberts Cross edge detector using 2x2 diagonal difference kernels.
    """
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def prewitt_edge(gray):
    """
    Prewitt edge detector using 3x3 horizontal and vertical difference kernels.
    """
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def sobel_edge(gray):
    """
    Sobel edge detector with 3x3 kernels combining Gaussian smoothing with differencing.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def laplacian_edge(gray):
    """
    Laplacian isotropic second-derivative edge detector.
    """
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return cv2.convertScaleAbs(lap)


# ==========================================================
# THRESHOLDING TECHNIQUES
# ==========================================================

def global_threshold(gray, threshold_value=127, invert=False):
    """
    Global fixed-value binary thresholding.
    """
    t_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, result = cv2.threshold(gray, threshold_value, 255, t_type)
    return result


def otsu_threshold(gray, invert=False):
    """
    Otsu thresholding: automatically minimizes intra-class intensity variance.
    """
    t_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    optimal_t, result = cv2.threshold(gray, 0, 255, t_type + cv2.THRESH_OTSU)
    return result, float(optimal_t)


def adaptive_threshold(gray, block_size=11, C=2, invert=False):
    """
    Adaptive Gaussian local window thresholding.
    """
    b = max(3, block_size if block_size % 2 == 1 else block_size + 1)
    t_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    result = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, t_type, b, C
    )
    return result


def hsv_skin_segmentation(image_bgr, lower_hsv=None, upper_hsv=None, min_sat=26):
    """
    Practical skin color segmentation combining HSV and YCrCb color spaces.
    Handles varied lighting and hand gesture foreground extraction reliably.
    Rejects beige clothing and background walls by using a calibrated saturation threshold.
    If image is grayscale, falls back to automatic Otsu segmentation.
    """
    if len(image_bgr.shape) == 2 or (len(image_bgr.shape) == 3 and image_bgr.shape[2] == 1):
        mask, _ = otsu_threshold(image_bgr)
        return mask

    if lower_hsv is None:
        lower_hsv = np.array([0, min_sat, 45], dtype=np.uint8)
    if upper_hsv is None:
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 78], dtype=np.uint8)
    upper_ycrcb = np.array([255, 175, 128], dtype=np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Fusion of both color space constraints
    combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    return combined


def prune_forearm_wrist(binary_mask, entry_side="bottom", cutoff_factor=1.2):
    """
    DIP Palm Detection & Forearm/Wrist Pruning using Distance Transform:
    Locates the palm center (maximum inscribed circle in the upper region of the mask)
    and severs the connected forearm entering from the bottom or edges.
    Leaves ONLY the palm and fingers for feature extraction.
    
    Returns:
        tuple: (pruned_binary_mask, palm_info_dict or None)
    """
    if binary_mask is None or cv2.countNonZero(binary_mask) == 0:
        return binary_mask, None

    h, w = binary_mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask, None

    # Isolate largest foreground component (the hand + arm)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Euclidean Distance Transform
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)

    # Search for palm center in the upper 75% away from the bottom entry edge
    search_dist = dist.copy()
    if entry_side == "bottom" and h > 20:
        search_dist[int(h * 0.75):, :] = 0

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(search_dist)
    if max_val < 6:
        # Fallback to entire mask
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist)
    if max_val < 6:
        return comp_mask, None

    palm_cx, palm_cy = max_loc
    palm_r = float(max_val)

    pruned = comp_mask.copy()
    wrist_y = h
    if entry_side == "bottom":
        wrist_y = int(palm_cy + cutoff_factor * palm_r)
        if wrist_y < h:
            # Sever forearm below wrist
            pruned[wrist_y:, :] = 0

    # Retain the component containing the palm center
    num_l, labs, st, _ = cv2.connectedComponentsWithStats(pruned, connectivity=8)
    if num_l > 1:
        palm_lbl = labs[min(h - 1, max(0, palm_cy)), min(w - 1, max(0, palm_cx))]
        if palm_lbl > 0:
            pruned = np.where(labs == palm_lbl, 255, 0).astype(np.uint8)
        else:
            best = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
            pruned = np.where(labs == best, 255, 0).astype(np.uint8)

    palm_info = {
        "cx": palm_cx,
        "cy": palm_cy,
        "radius": round(palm_r, 1),
        "wrist_y": wrist_y
    }
    return pruned, palm_info


def detect_hand_bounding_box(image_bgr, min_sat=26):
    """
    Automated DIP Hand Localization:
    Finds the most prominent hand candidate within the frame using skin segmentation,
    geometric aspect ratios, and palm distance transform scoring.
    Returns (x, y, w, h) bounding box.
    """
    h, w = image_bgr.shape[:2]
    total_area = float(h * w)

    skin = hsv_skin_segmentation(image_bgr, min_sat=min_sat)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, se)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, se)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if num_labels <= 1:
        # Fallback to central region
        dim = int(min(h, w) * 0.65)
        return max(0, w // 2 - dim // 2), max(0, h // 2 - dim // 2), dim, dim

    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < total_area * 0.01 or area > total_area * 0.6:
            continue
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = float(bw) / max(1, bh)

        comp_mask = (labels == i).astype(np.uint8) * 255
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        max_d = float(np.max(dist))

        # Score favors square-like hand aspect ratios and solid palm cores
        score = max_d * (1.0 / (abs(aspect - 0.9) + 0.3))
        candidates.append((score, (bx, by, bw, bh)))

    if not candidates:
        dim = int(min(h, w) * 0.65)
        return max(0, w // 2 - dim // 2), max(0, h // 2 - dim // 2), dim, dim

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_bx, best_by, best_bw, best_bh = candidates[0][1]

    # Add 15% margin and make square
    pad = int(max(best_bw, best_bh) * 0.15)
    cx = best_bx + best_bw // 2
    cy = best_by + best_bh // 2
    half_side = int(max(best_bw, best_bh) * 0.55) + pad

    x1 = max(0, cx - half_side)
    y1 = max(0, cy - half_side)
    x2 = min(w, cx + half_side)
    y2 = min(h, cy + half_side)

    return x1, y1, x2 - x1, y2 - y1


# ==========================================================
# SEGMENTATION EVALUATION & STATS
# ==========================================================

def calculate_binary_stats(binary_mask):
    """
    Calculates quantitative metrics for a binary segmentation mask.
    Returns:
        dict: foreground ratio (%), number of connected components,
              largest component area, total mask area.
    """
    total_pixels = binary_mask.size
    fg_pixels = int(np.count_nonzero(binary_mask == 255))
    fg_ratio = float(fg_pixels * 100.0 / total_pixels) if total_pixels > 0 else 0.0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Subtract background label (0)
    components = max(0, num_labels - 1)
    largest_area = 0
    if num_labels > 1:
        # stats columns: [LEFT, TOP, WIDTH, HEIGHT, AREA]
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(np.max(areas)) if len(areas) > 0 else 0

    return {
        "foreground_ratio": round(fg_ratio, 2),
        "components": components,
        "largest_area": largest_area,
        "total_pixels": total_pixels
    }


def get_task6_comparison_table():
    """
    Returns standard DIP qualitative comparison data for report generation.
    """
    return [
        {"Method": "Roberts", "Category": "Edge", "Separation": "Good", "Boundary": "Fine", "Noise Sensitivity": "Medium", "Overall": "Good"},
        {"Method": "Prewitt", "Category": "Edge", "Separation": "Good", "Boundary": "Clear", "Noise Sensitivity": "Low-Med", "Overall": "Good"},
        {"Method": "Sobel", "Category": "Edge", "Separation": "Very Good", "Boundary": "Strong & Smooth", "Noise Sensitivity": "Low", "Overall": "Very Good"},
        {"Method": "Laplacian", "Category": "Edge", "Separation": "Good", "Boundary": "Fine / Detail", "Noise Sensitivity": "High", "Overall": "Moderate"},
        {"Method": "Global Threshold", "Category": "Threshold", "Separation": "Good", "Boundary": "Moderate", "Noise Sensitivity": "Medium", "Overall": "Good"},
        {"Method": "Otsu Threshold", "Category": "Threshold", "Separation": "Very Good", "Boundary": "Good", "Noise Sensitivity": "Low", "Overall": "Very Good"},
        {"Method": "Adaptive Threshold", "Category": "Threshold", "Separation": "Good", "Boundary": "Detailed", "Noise Sensitivity": "Medium", "Overall": "Good"},
        {"Method": "HSV Skin Segmentation", "Category": "Color Segment", "Separation": "Excellent", "Boundary": "Robust", "Noise Sensitivity": "Low", "Overall": "Excellent"}
    ]
