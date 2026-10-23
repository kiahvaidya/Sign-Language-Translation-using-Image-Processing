"""
DIP Tasks 6 & 7: Hand Segmentation & Region Growing
Implements edge detection operators, global & local thresholding, calibrated skin color segmentation,
and seed-based region growing.
"""

from collections import deque
import cv2
import numpy as np

from config import MIN_SKIN_SATURATION, MAX_SKIN_HUE


# ==========================================================
# TASK 6: EDGE DETECTION OPERATORS
# ==========================================================

def roberts_edge(gray):
    """Roberts Cross 2x2 edge detector."""
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def prewitt_edge(gray):
    """Prewitt 3x3 edge detector."""
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def sobel_edge(gray):
    """Sobel 3x3 edge detector with Gaussian weight smoothing."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def laplacian_edge(gray):
    """Laplacian isotropic second-derivative edge detector."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return cv2.convertScaleAbs(lap)


# ==========================================================
# TASK 6: THRESHOLDING TECHNIQUES
# ==========================================================

def global_threshold(gray, threshold_value=127, invert=False):
    t_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, result = cv2.threshold(gray, threshold_value, 255, t_type)
    return result


def otsu_threshold(gray, invert=False):
    t_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    optimal_t, result = cv2.threshold(gray, 0, 255, t_type + cv2.THRESH_OTSU)
    return result, float(optimal_t)


def adaptive_threshold(gray, block_size=11, C=2, invert=False):
    b = max(3, block_size if block_size % 2 == 1 else block_size + 1)
    t_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, t_type, b, C)


def is_light_or_uniform_background(image_bgr):
    """
    Detects if the image has a white, near-white, or clean uniform background
    (common in educational sign charts, Google image photos, studio captures).
    """
    if len(image_bgr.shape) == 2:
        return True
    h, w = image_bgr.shape[:2]
    border_thickness = max(2, min(h, w) // 25)
    top = image_bgr[:border_thickness, :]
    bottom = image_bgr[-border_thickness:, :]
    left = image_bgr[:, :border_thickness]
    right = image_bgr[:, -border_thickness:]

    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)

    border_gray = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(border_gray))
    std_val = float(np.std(border_gray))

    return mean_val > 195 and std_val < 45


def hsv_skin_segmentation(image_bgr, min_sat=MIN_SKIN_SATURATION, max_hue=MAX_SKIN_HUE):
    """
    Calibrated Color Space Fusion:
    Combines HSV (Hue/Saturation) with YCrCb (Chrominance Red/Blue)
    to segment hand skin while rejecting white/beige clothing, walls, and ceiling lighting.
    """
    if len(image_bgr.shape) == 2:
        mask, _ = otsu_threshold(image_bgr)
        return mask

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, min_sat, 45], dtype=np.uint8)
    upper_hsv = np.array([max_hue, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 78], dtype=np.uint8)
    upper_ycrcb = np.array([255, 175, 128], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    return combined


def adaptive_hand_segmentation(image_bgr, min_sat=MIN_SKIN_SATURATION, max_hue=MAX_SKIN_HUE):
    """
    DIP Task 6 Adaptive Fusion:
    Intelligently detects whether the image is on a white/light studio background
    or a complex natural background, and selects the optimal segmentation strategy.

    - White / Light Background: Uses background thresholding (Otsu / adaptive)
      combined with chromaticity to extract a pristine, 100% complete hand mask.
    - Complex Background: Uses calibrated HSV + YCrCb color-space skin fusion.
    """
    if len(image_bgr.shape) == 2:
        mask, _ = otsu_threshold(image_bgr, invert=True)
        return mask

    if is_light_or_uniform_background(image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        otsu_mask, _ = otsu_threshold(gray, invert=True)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        skin_filter = (sat > 10) | (val < 235)

        combined = cv2.bitwise_and(otsu_mask, otsu_mask, mask=skin_filter.astype(np.uint8) * 255)
        if cv2.countNonZero(combined) > 400:
            return combined

    return hsv_skin_segmentation(image_bgr, min_sat=min_sat, max_hue=max_hue)



# ==========================================================
# TASK 7: REGION GROWING
# ==========================================================

def region_growing(gray_image, seed=None, threshold=20, connectivity=4):
    """
    Task 7: Seed-Based Region Growing Algorithm
    Grows a connected region from a seed point based on intensity homogeneity.
    """
    h, w = gray_image.shape[:2]

    # Automatic seed selection if not provided (find palm/brightest center)
    if seed is None:
        # Search in central 50%
        center_crop = gray_image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(center_crop)
        seed_x = max_l[0] + w // 4
        seed_y = max_l[1] + h // 4
    else:
        seed_x, seed_y = seed

    seed_x = max(0, min(w - 1, int(seed_x)))
    seed_y = max(0, min(h - 1, int(seed_y)))

    region = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([(seed_y, seed_x)])

    visited[seed_y, seed_x] = True
    region[seed_y, seed_x] = 255
    seed_val = float(gray_image[seed_y, seed_x])

    if connectivity == 8:
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in neighbours:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                val = float(gray_image[ny, nx])
                if abs(val - seed_val) <= threshold:
                    region[ny, nx] = 255
                    queue.append((ny, nx))

    return region, (seed_x, seed_y)


# ==========================================================
# AUTOMATED HAND BOUNDING BOX LOCALIZATION
# ==========================================================

def detect_hand_bounding_box(image_bgr, min_sat=MIN_SKIN_SATURATION):
    """
    Locates the primary hand candidate in a full camera frame
    using skin segmentation and distance-transform palm scoring.
    """
    h, w = image_bgr.shape[:2]
    total_area = float(h * w)

    skin = hsv_skin_segmentation(image_bgr, min_sat=min_sat)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, se)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, se)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if num_labels <= 1:
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

        score = max_d * (1.0 / (abs(aspect - 0.9) + 0.3))
        candidates.append((score, (bx, by, bw, bh)))

    if not candidates:
        dim = int(min(h, w) * 0.65)
        return max(0, w // 2 - dim // 2), max(0, h // 2 - dim // 2), dim, dim

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_bx, best_by, best_bw, best_bh = candidates[0][1]

    pad = int(max(best_bw, best_bh) * 0.15)
    cx = best_bx + best_bw // 2
    cy = best_by + best_bh // 2
    half_side = int(max(best_bw, best_bh) * 0.55) + pad

    x1 = max(0, cx - half_side)
    y1 = max(0, cy - half_side)
    x2 = min(w, cx + half_side)
    y2 = min(h, cy + half_side)

    return x1, y1, x2 - x1, y2 - y1
