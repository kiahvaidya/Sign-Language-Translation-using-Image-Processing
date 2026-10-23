"""
Task 4 & 5: Spatial Filtering and Sharpening Module
Implements neighborhood averaging, Gaussian smoothing, non-linear median filtering,
Laplacian edge enhancement, Sobel gradient sharpening, and high-boost filtering.
Also includes the integrated Task 5 enhancement pipeline.
"""

import cv2
import numpy as np
from .enhancement import contrast_stretch


def mean_filter(image, kernel_size=5):
    """
    Linear box averaging filter for spatial smoothing.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.blur(image, (k, k))


def gaussian_filter(image, kernel_size=5, sigma=0):
    """
    Gaussian smoothing filter with bell-curve weights.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.GaussianBlur(image, (k, k), sigma)


def median_filter(image, kernel_size=5):
    """
    Non-linear rank-order median filter, ideal for salt-and-pepper noise removal.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.medianBlur(image, k)


def laplacian_sharpen(image):
    """
    Second-derivative Laplacian sharpening filter:
    sharpened = gray - laplacian (or addWeighted with negative laplacian)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    sharpened = gray_f - lap
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return sharpened


def gradient_sharpen(image):
    """
    First-derivative gradient-based sharpening using horizontal & vertical Sobel operators.
    sharpened = gray + 0.5 * normalized_gradient
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    sharpened = gray_f + 0.5 * mag_norm
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return sharpened


def high_boost_filter(image, A=1.5, kernel_size=5):
    """
    High-boost filtering:
    mask = image - GaussianBlur(image)
    high_boost = A * image + mask
    When A = 1.0, this is standard unsharp masking.
    When A > 1.0, high-frequency details are emphasized while base signal is boosted.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)
    blurred = cv2.GaussianBlur(gray_f, (k, k), 0)
    high_freq_mask = gray_f - blurred

    hb = A * gray_f + high_freq_mask
    hb = np.clip(hb, 0, 255).astype(np.uint8)
    return hb


def add_salt_pepper_noise(image, amount=0.03):
    """
    Adds impulse (salt and pepper) noise to an image.
    """
    noisy = image.copy()
    h, w = noisy.shape[:2]
    num_noisy = int(amount * h * w)

    # Salt (white)
    y_salt = np.random.randint(0, h, num_noisy // 2)
    x_salt = np.random.randint(0, w, num_noisy // 2)
    noisy[y_salt, x_salt] = 255

    # Pepper (black)
    y_pep = np.random.randint(0, h, num_noisy // 2)
    x_pep = np.random.randint(0, w, num_noisy // 2)
    noisy[y_pep, x_pep] = 0

    return noisy


def add_gaussian_noise(image, mean=0, sigma=20):
    """
    Adds zero-mean Gaussian distributed noise.
    """
    img_f = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(img_f + noise, 0, 255).astype(np.uint8)
    return noisy


def task5_integrated_pipeline(image_bgr, A=1.5, median_ksize=5):
    """
    Task 5 Integrated Image Enhancement Pipeline:
    Original Image -> Contrast Stretching -> Median Filtering -> High Boost Sharpening

    Returns:
        dict: containing intermediate stage outputs:
              - 'original': immutable original BGR
              - 'contrast_stretched': enhanced image
              - 'median_filtered': filtered image
              - 'high_boost': sharpened final improved image (grayscale)
    """
    original = image_bgr.copy()
    enhanced = contrast_stretch(original)
    filtered = median_filter(enhanced, median_ksize)
    sharpened = high_boost_filter(filtered, A=A, kernel_size=5)

    return {
        "original": original,
        "contrast_stretched": enhanced,
        "median_filtered": filtered,
        "high_boost": sharpened
    }
