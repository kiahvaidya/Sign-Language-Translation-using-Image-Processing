"""
DIP Tasks 1 & 2: Image Acquisition & Representation
Implements robust visual input loading, validation, color-space representations (RGB, Grayscale, HSV),
and spatial normalization.
"""

import base64
import io
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from config import TARGET_SIZE


def load_image(source):
    """
    Task 1: Image Acquisition & Validation
    Accepts:
      - File path (str or Path)
      - Raw bytes
      - Base64 data URL string
      - NumPy ndarray
      - PIL Image

    Returns:
      tuple: (original_bgr, original_rgb, original_gray, metadata_dict)
    """
    img_bgr = None

    if isinstance(source, (str, Path)):
        s_str = str(source)
        if s_str.startswith("data:image"):
            # Base64 data URL
            header, encoded = s_str.split(",", 1)
            raw = base64.b64decode(encoded)
            file_bytes = np.asarray(bytearray(raw), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif Path(s_str).is_file():
            img_bgr = cv2.imread(s_str)
        else:
            try:
                raw = base64.b64decode(s_str)
                file_bytes = np.asarray(bytearray(raw), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception:
                raise FileNotFoundError(f"Image source file not found: {source}")

    elif isinstance(source, (bytes, bytearray)):
        file_bytes = np.asarray(bytearray(source), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif isinstance(source, np.ndarray):
        img_bgr = source.copy()

    elif isinstance(source, Image.Image):
        rgb_arr = np.array(source)
        img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    elif hasattr(source, "read"):
        # File-like object
        file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Failed to acquire valid image: file is corrupted, empty, or unreadable.")

    # Task 2: Image Representation
    h, w = img_bgr.shape[:2]
    channels = img_bgr.shape[2] if len(img_bgr.shape) == 3 else 1

    if channels == 3:
        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        original_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    else:
        original_gray = img_bgr.copy()
        original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)
        original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
        original_hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)

    metadata = {
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "dtype": str(img_bgr.dtype),
        "min_intensity": int(np.min(original_gray)),
        "max_intensity": int(np.max(original_gray)),
        "mean_intensity": round(float(np.mean(original_gray)), 2)
    }

    return img_bgr, original_rgb, original_gray, original_hsv, metadata


def normalize_spatial_resolution(image, target_size=TARGET_SIZE):
    """
    Task 2: Spatial Representation Normalization
    Rescales any input image to canonical resolution (200x200) matching ASL training dataset.
    """
    if image.shape[:2] == target_size:
        return image
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def encode_image_base64(image_rgb_or_bgr, is_bgr=False):
    """
    Encodes an image to a base64 JPEG string for JSON API transmission.
    """
    if is_bgr:
        bgr = image_rgb_or_bgr
    else:
        if len(image_rgb_or_bgr.shape) == 2:
            bgr = cv2.cvtColor(image_rgb_or_bgr, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image_rgb_or_bgr, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
