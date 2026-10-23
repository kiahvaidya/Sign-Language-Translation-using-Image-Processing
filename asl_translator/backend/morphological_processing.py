"""
DIP Task 8: Morphological Processing & Forearm Pruning
Implements erosion, dilation, opening, closing across multiple iterations (1, 2, 3)
and distance-transform based palm detection and forearm/wrist pruning.
"""

import cv2
import numpy as np

from config import WRIST_CUTOFF_FACTOR


def get_structuring_element(se_type="rect", size=3):
    """Returns an OpenCV structuring element."""
    k = max(3, size if size % 2 == 1 else size + 1)
    if se_type.lower() == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    elif se_type.lower() == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def apply_morphology(binary_mask, op="open", se_type="rect", size=3, iterations=1):
    """Applies basic morphological transformation."""
    se = get_structuring_element(se_type, size)
    op_lower = op.lower()

    if op_lower == "erode":
        return cv2.erode(binary_mask, se, iterations=iterations)
    elif op_lower == "dilate":
        return cv2.dilate(binary_mask, se, iterations=iterations)
    elif op_lower == "open":
        return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, se, iterations=iterations)
    elif op_lower == "close":
        return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, se, iterations=iterations)
    elif op_lower == "gradient":
        return cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, se, iterations=iterations)
    else:
        raise ValueError(f"Unknown morphological operation: {op}")


def generate_iteration_series(binary_mask, op="erode", se_type="rect", size=3, max_iter=3):
    """
    Generates outputs for 1, 2, ..., max_iter iterations.
    Required by Task 8 for evaluating increasing morphological iterations.
    """
    se = get_structuring_element(se_type, size)
    series = {}
    for i in range(1, max_iter + 1):
        if op.lower() == "erode":
            series[i] = cv2.erode(binary_mask, se, iterations=i)
        elif op.lower() == "dilate":
            series[i] = cv2.dilate(binary_mask, se, iterations=i)
        elif op.lower() == "open":
            series[i] = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, se, iterations=i)
        elif op.lower() == "close":
            series[i] = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, se, iterations=i)
    return series


def prune_forearm_wrist(binary_mask, entry_side="bottom", cutoff_factor=WRIST_CUTOFF_FACTOR):
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

    # Largest foreground component (hand + connected forearm)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Euclidean Distance Transform
    dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)

    # Search for palm center in the upper 75% away from bottom entry edge
    search_dist = dist.copy()
    if entry_side == "bottom" and h > 20:
        search_dist[int(h * 0.75):, :] = 0

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(search_dist)
    if max_val < 6:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist)
    if max_val < 6:
        return comp_mask, None

    palm_cx, palm_cy = max_loc
    palm_r = float(max_val)

    pruned = comp_mask.copy()
    wrist_y = h
    if entry_side == "bottom":
        cnts_c, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_c:
            c_comp = max(cnts_c, key=cv2.contourArea)
            _, by_c, _, bh_c = cv2.boundingRect(c_comp)
            bottom_y = by_c + bh_c
            forearm_extension = bottom_y - palm_cy
            if forearm_extension > (1.85 * palm_r):
                wrist_y = int(palm_cy + max(cutoff_factor, 1.75) * palm_r)
                if wrist_y < bottom_y:
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


def clean_hand_mask(binary_mask, open_iter=1, close_iter=2, keep_largest=True, prune_wrist=True):
    """
    Task 8: Hand Mask Cleaning Pipeline:
    1. Opening (3x3 rect) to eliminate isolated background noise & specks.
    2. Closing (3x3 rect) to fill internal palm holes and connect fingers.
    3. Largest connected component selection.
    4. Optional wrist pruning to excise forearm.
    """
    rect_3 = get_structuring_element("rect", 3)

    # 1. Noise removal
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, rect_3, iterations=open_iter)

    # 2. Gap filling
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, rect_3, iterations=close_iter)

    # 3. Retain largest component
    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # 4. Forearm pruning
    if prune_wrist:
        cleaned, _ = prune_forearm_wrist(cleaned, entry_side="bottom")

    return cleaned
