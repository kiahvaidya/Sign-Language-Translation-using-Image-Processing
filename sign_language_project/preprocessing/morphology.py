"""
Task 8: Morphological Processing Module
Implements binary erosion, dilation, opening, closing across multiple iterations (1, 2, 3)
and different structuring elements (3x3 Rect, 3x3 Cross, 5x5 Rect).
Includes automated hand mask cleanup for sign language recognition.
"""

import cv2
import numpy as np


def get_structuring_element(se_type="rect", size=3):
    """
    Returns an OpenCV structuring element.
    Options: 'rect', 'cross', 'ellipse'.
    """
    k = max(3, size if size % 2 == 1 else size + 1)
    if se_type.lower() == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    elif se_type.lower() == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def apply_morphology(binary_mask, op="erode", se_type="rect", size=3, iterations=1):
    """
    Applies morphological operation to binary mask with specified structuring element and iterations.
    Operations: 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'blackhat'.
    """
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
        raise ValueError(f"Unknown morphology operation: {op}")


def generate_iteration_series(binary_mask, op="erode", se_type="rect", size=3, max_iter=3):
    """
    Generates morphological outputs for 1, 2, ..., max_iter iterations.
    Specifically required by Task 8 for evaluating increasing iterations.
    Returns:
        dict: {1: iter1_mask, 2: iter2_mask, 3: iter3_mask, ...}
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


def clean_hand_mask(binary_mask, open_iter=1, close_iter=2, keep_largest=True, prune_wrist=False):
    """
    DIP Morphological Cleaning Pipeline for Hand/Sign Recognition:
    1. Opening (erosion followed by dilation) to eliminate small salt noise & background specks.
    2. Closing (dilation followed by erosion) to fill internal holes and connect broken palm/finger contours.
    3. Connected component analysis to isolate and keep only the largest connected object (the hand).
    4. Optional DIP Wrist/Forearm Pruning via Distance Transform to sever attached forearms.
    """
    rect_3 = get_structuring_element("rect", 3)

    # 1. Noise removal via Opening
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, rect_3, iterations=open_iter)

    # 2. Gap filling via Closing
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, rect_3, iterations=close_iter)

    # 3. Retain only largest connected component (the hand)
    if keep_largest:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # 4. Optional Wrist / Forearm Pruning
    if prune_wrist:
        try:
            from preprocessing.segmentation import prune_forearm_wrist
        except ImportError:
            try:
                from .segmentation import prune_forearm_wrist
            except ImportError:
                prune_forearm_wrist = None
        if prune_forearm_wrist is not None:
            cleaned, _ = prune_forearm_wrist(cleaned, entry_side="bottom")

    return cleaned
