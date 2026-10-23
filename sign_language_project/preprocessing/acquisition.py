"""
Task 1: Image Acquisition Module
Handles loading, metadata extraction, validation, and safe persistence of original input images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io


def load_image(source):
    """
    Load an image safely from a file path, BytesIO/UploadedFile, or numpy array.
    Guarantees the original image is preserved as an immutable copy.

    Returns:
        tuple: (bgr_image, rgb_image, gray_image, metadata_dict)
    """
    if source is None:
        raise ValueError("Source image cannot be None.")

    bgr = None
    source_name = "Unknown"

    if isinstance(source, (str, Path)):
        path = Path(source)
        source_name = path.name
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {source}")
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not decode image from file: {source}")

    elif hasattr(source, "read"):
        # File-like object (e.g. Streamlit UploadedFile)
        source_name = getattr(source, "name", "uploaded_image.png")
        file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
        # Reset pointer for possible subsequent reads
        if hasattr(source, "seek"):
            source.seek(0)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image from uploaded bytes.")

    elif isinstance(source, np.ndarray):
        source_name = "numpy_array"
        if len(source.shape) == 2:
            bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        elif len(source.shape) == 3:
            if source.shape[2] == 4:
                bgr = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            elif source.shape[2] == 3:
                bgr = source.copy()
            else:
                raise ValueError(f"Unsupported array shape: {source.shape}")
        else:
            raise ValueError(f"Unsupported array dimension: {source.ndim}")

    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # Keep original completely untouched
    original_bgr = bgr.copy()
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    h, w = original_bgr.shape[:2]
    c = original_bgr.shape[2] if original_bgr.ndim == 3 else 1

    metadata = {
        "source_name": source_name,
        "width": int(w),
        "height": int(h),
        "channels": int(c),
        "total_pixels": int(w * h),
        "total_values": int(original_bgr.size),
        "dtype": str(original_bgr.dtype),
        "min_intensity": int(original_gray.min()),
        "max_intensity": int(original_gray.max()),
        "mean_intensity": float(round(float(original_gray.mean()), 2)),
        "std_intensity": float(round(float(original_gray.std()), 2)),
    }

    return original_bgr, original_rgb, original_gray, metadata


def get_pixel_info(img_bgr, x, y):
    """
    Extract pixel intensity details at coordinate (x, y).
    """
    h, w = img_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None

    b, g, r = [int(v) for v in img_bgr[y, x]]
    gray_val = int(round(0.299 * r + 0.587 * g + 0.114 * b))

    return {
        "x": x,
        "y": y,
        "bgr": (b, g, r),
        "rgb": (r, g, b),
        "blue": b,
        "green": g,
        "red": r,
        "gray": gray_val,
        "hex": f"#{r:02x}{g:02x}{b:02x}"
    }


def save_acquired_image(image_bgr, destination_path):
    """
    Safely save acquired image to destination path without overwriting originals.
    """
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(dest), image_bgr)
    return success, dest


def find_dataset_images(dataset_dir):
    """
    Scans dataset folder and returns a dictionary mapping labels to image file paths.
    Supports flat, nested, and subfolder structures.
    """
    base = Path(dataset_dir)
    if not base.exists() or not base.is_dir():
        return {}

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    dataset_map = {}

    for entry in sorted(base.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            images = [
                str(p) for p in sorted(entry.iterdir())
                if p.is_file() and p.suffix.lower() in extensions and not p.name.startswith(".")
            ]
            if images:
                dataset_map[entry.name] = images

    # If flat directory without subfolders
    if not dataset_map:
        flat_images = [
            str(p) for p in sorted(base.iterdir())
            if p.is_file() and p.suffix.lower() in extensions and not p.name.startswith(".")
        ]
        for img_path in flat_images:
            # Try to infer label from filename (e.g. A0001_test.jpg -> 'A')
            stem = Path(img_path).stem
            label = stem[0].upper() if stem else "UNKNOWN"
            dataset_map.setdefault(label, []).append(img_path)

    return dataset_map
