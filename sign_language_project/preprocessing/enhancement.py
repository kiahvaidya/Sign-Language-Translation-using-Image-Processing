"""
Task 3: Image Enhancement Module
Implements spatial point processing, contrast stretching, log transform,
power-law (gamma) transform, histogram equalization, and quantitative evaluation.
"""

import cv2
import numpy as np


def contrast_stretch(img):
    """
    Applies linear contrast stretching to normalize intensities to [0, 255].
    Works on grayscale and multi-channel images.
    """
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val == min_val:
        return img.copy()

    stretched = (img.astype(np.float32) - min_val) * 255.0 / (max_val - min_val)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization(img):
    """
    Histogram equalization using cv2.equalizeHist for grayscale,
    or Y-channel equalization in YCrCb for color preservation.
    """
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(img)


def image_negative(img):
    """
    Computes photographic negative: s = (L - 1) - r
    """
    return 255 - img


def log_transform(img, c=1.0):
    """
    Logarithmic transformation: s = c * log(1 + r)
    Expands dark pixels and compresses high-intensity values.
    """
    img_float = img.astype(np.float32)
    # Use log1p to avoid log(0) and numerical instability
    log_img = c * np.log1p(img_float)
    normalized = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def gamma_transform(img, gamma=1.0):
    """
    Power-law (Gamma) transformation: s = c * r^gamma
    gamma < 1 expands dark regions (brightens)
    gamma > 1 compresses dark regions (darkens)
    """
    if gamma <= 0:
        gamma = 0.01
    normalized = img.astype(np.float32) / 255.0
    gamma_img = np.power(normalized, gamma) * 255.0
    return np.clip(gamma_img, 0, 255).astype(np.uint8)


def calculate_mse(original, processed):
    """
    Mean Squared Error between original and processed image.
    Automatically aligns channel dimensions if comparing grayscale to color.
    """
    orig = original
    proc = processed

    if orig.ndim == 3 and proc.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    elif orig.ndim == 2 and proc.ndim == 3:
        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

    orig_f = orig.astype(np.float32)
    proc_f = proc.astype(np.float32)
    return float(np.mean((orig_f - proc_f) ** 2))


def calculate_psnr(original, processed):
    """
    Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    """
    mse = calculate_mse(original, processed)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_histogram_data(img):
    """
    Computes 256-bin histogram for single or multi-channel image.
    Returns dictionary with channel names and frequency counts.
    """
    hist_data = {}
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_data["Gray"] = hist
    else:
        # Assumes BGR
        hist_data["Blue"] = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_data["Green"] = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
        hist_data["Red"] = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
    return hist_data
