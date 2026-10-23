"""
DIP Tasks 3, 4, 5: Image Enhancement, Spatial Filtering, and Integrated Pipeline
Implements contrast stretching, noise reduction via median filtering, and high-boost detail sharpening.
"""

import cv2
import numpy as np

from config import HIGH_BOOST_A, MEDIAN_KERNEL


def contrast_stretch(image):
    """
    Task 3: Linear Min-Max Contrast Stretching
    Normalizes the intensity dynamic range to full scale [0, 255].
    """
    if len(image.shape) == 3:
        stretched = np.zeros_like(image)
        for i in range(3):
            ch = image[:, :, i]
            min_val = float(np.min(ch))
            max_val = float(np.max(ch))
            if max_val > min_val:
                stretched[:, :, i] = np.clip(
                    ((ch - min_val) / (max_val - min_val)) * 255.0, 0, 255
                ).astype(np.uint8)
            else:
                stretched[:, :, i] = ch
        return stretched

    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val > min_val:
        return np.clip(
            ((image - min_val) / (max_val - min_val)) * 255.0, 0, 255
        ).astype(np.uint8)
    return image.copy()


def apply_median_filter(image, kernel_size=MEDIAN_KERNEL):
    """
    Task 4: Median Spatial Filter
    Preserves hand/finger edges while eliminating salt-and-pepper noise.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.medianBlur(image, k)


def high_boost_sharpen(gray_image, A=HIGH_BOOST_A):
    """
    Task 4 & 5: High-Boost Filtering
    f_hb = A * f - f_low = (A - 1) * f + (f - f_low)
    Amplifies high-frequency boundary gradients for clear hand contour extraction.
    """
    f = gray_image.astype(np.float32)
    # Low-pass component via Gaussian smoothing
    f_low = cv2.GaussianBlur(f, (5, 5), sigmaX=1.0)
    # High-boost formula
    f_hb = (A * f) - f_low
    return np.clip(f_hb, 0, 255).astype(np.uint8)


def execute_task5_pipeline(image_bgr, A=HIGH_BOOST_A, kernel_size=MEDIAN_KERNEL):
    """
    Task 5: Integrated Image Enhancement Pipeline
    Original BGR
        ↓
    Contrast Stretching (Task 3)
        ↓
    Median Filtering (Task 4)
        ↓
    High-Boost Sharpening (Task 5)
        ↓
    Enhanced Hand Grayscale

    Returns dict of intermediate images for lab grading & inspection.
    """
    # 1. Contrast stretch
    contrast_stretched = contrast_stretch(image_bgr)

    # 2. Median filter
    median_filtered = apply_median_filter(contrast_stretched, kernel_size=kernel_size)

    # Convert to grayscale for frequency-domain / high-boost sharpening
    gray_filtered = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2GRAY)

    # 3. High-boost sharpening
    high_boost = high_boost_sharpen(gray_filtered, A=A)

    return {
        "contrast_stretched": contrast_stretched,
        "median_filtered": median_filtered,
        "high_boost": high_boost
    }
