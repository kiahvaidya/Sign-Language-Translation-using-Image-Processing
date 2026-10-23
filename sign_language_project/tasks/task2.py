"""
Task 2: Image Representation and Fundamentals
Digital Image Processing Lab Evaluation Module

Experiments:
1. Color Space Representations (RGB, Grayscale, HSV)
2. Spatial Sampling (100%, 50%, 25%)
3. Gray-level Quantization (256, 128, 64, 32 levels)
4. File Format Comparison (BMP, PNG, JPEG with file sizes & bar plot)
Outputs are safely stored in outputs/task2/.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.acquisition import load_image, find_dataset_images


def quantize_image(gray_img, levels):
    """
    Quantizes grayscale image into the specified number of levels.
    """
    step = max(1, 256 // levels)
    quantized = (gray_img // step) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)


def run_task2(image_path=None, save_output=True, show_plot=False):
    out_dir = PROJECT_ROOT / "outputs" / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)

    color_dir = out_dir / "color_spaces"
    sampling_dir = out_dir / "sampling"
    quant_dir = out_dir / "quantization"
    format_dir = out_dir / "formats"
    for d in (color_dir, sampling_dir, quant_dir, format_dir):
        d.mkdir(parents=True, exist_ok=True)

    if image_path is None:
        dataset_dir = PROJECT_ROOT.parent / "dataset"
        if not dataset_dir.exists():
            dataset_dir = PROJECT_ROOT / "data" / "dataset"
        img_map = find_dataset_images(dataset_dir)
        if img_map:
            first_label = sorted(list(img_map.keys()))[0]
            image_path = img_map[first_label][0]
        else:
            raise FileNotFoundError("No input image provided.")

    print("=" * 60)
    print("TASK 2: IMAGE REPRESENTATION & FUNDAMENTALS")
    print("=" * 60)
    print(f"Input image: {image_path}")

    bgr, rgb, gray, meta = load_image(image_path)
    h, w = meta["height"], meta["width"]

    # 1. COLOR SPACES
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if save_output:
        cv2.imwrite(str(color_dir / "rgb.png"), bgr)
        cv2.imwrite(str(color_dir / "grayscale.png"), gray)
        cv2.imwrite(str(color_dir / "hsv.png"), hsv)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("Grayscale")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(hsv)
    axes[1, 0].set_title("HSV Color Space")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))
    axes[1, 1].set_title("YCrCb Color Space")
    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(str(color_dir / "color_spaces_comparison.png"), dpi=200)
    plt.close()

    # 2. SAMPLING (100%, 50%, 25%)
    img_100 = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    img_50 = cv2.resize(rgb, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    img_25 = cv2.resize(rgb, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)

    if save_output:
        cv2.imwrite(str(sampling_dir / "sampling_100.png"), cv2.cvtColor(img_100, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sampling_dir / "sampling_50.png"), cv2.cvtColor(img_50, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(sampling_dir / "sampling_25.png"), cv2.cvtColor(img_25, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_100)
    axes[0].set_title(f"100% Sampling\n({img_100.shape[1]}x{img_100.shape[0]})")
    axes[0].axis("off")
    axes[1].imshow(img_50)
    axes[1].set_title(f"50% Sampling\n({img_50.shape[1]}x{img_50.shape[0]})")
    axes[1].axis("off")
    axes[2].imshow(img_25)
    axes[2].set_title(f"25% Sampling\n({img_25.shape[1]}x{img_25.shape[0]})")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(str(sampling_dir / "sampling_comparison.png"), dpi=200)
    plt.close()

    # 3. QUANTIZATION (256, 128, 64, 32 levels)
    q_256 = quantize_image(gray, 256)
    q_128 = quantize_image(gray, 128)
    q_64 = quantize_image(gray, 64)
    q_32 = quantize_image(gray, 32)

    if save_output:
        cv2.imwrite(str(quant_dir / "quant_256.png"), q_256)
        cv2.imwrite(str(quant_dir / "quant_128.png"), q_128)
        cv2.imwrite(str(quant_dir / "quant_64.png"), q_64)
        cv2.imwrite(str(quant_dir / "quant_32.png"), q_32)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(q_256, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("256 Gray Levels")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(q_128, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("128 Gray Levels")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(q_64, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("64 Gray Levels")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(q_32, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("32 Gray Levels")
    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(str(quant_dir / "quantization_comparison.png"), dpi=200)
    plt.close()

    # 4. FILE FORMAT COMPARISON (BMP, PNG, JPEG)
    bmp_p = format_dir / "sign_image.bmp"
    png_p = format_dir / "sign_image.png"
    jpg_p = format_dir / "sign_image.jpg"

    cv2.imwrite(str(bmp_p), bgr)
    cv2.imwrite(str(png_p), bgr)
    cv2.imwrite(str(jpg_p), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    bmp_kb = os.path.getsize(bmp_p) / 1024.0
    png_kb = os.path.getsize(png_p) / 1024.0
    jpg_kb = os.path.getsize(jpg_p) / 1024.0

    print("\nFile Format Comparison:")
    print(f"  BMP  : {bmp_kb:.2f} KB (Uncompressed)")
    print(f"  PNG  : {png_kb:.2f} KB (Lossless Compression)")
    print(f"  JPEG : {jpg_kb:.2f} KB (Lossy Compression, Quality 90)")

    plt.figure(figsize=(7, 4.5))
    formats = ["BMP", "PNG", "JPEG"]
    sizes = [bmp_kb, png_kb, jpg_kb]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = plt.bar(formats, sizes, color=colors)
    plt.title("Image File Size Comparison (Storage Efficiency)", fontsize=13, fontweight="bold")
    plt.ylabel("File Size (KB)")
    for bar in bars:
        h_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h_val, f"{h_val:.2f} KB", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(format_dir / "file_size_comparison.png"), dpi=200)
    plt.close()

    print(f"\nAll Task 2 outputs saved to: {out_dir.resolve()}")
    return {
        "color_spaces": {"bgr": bgr, "gray": gray, "hsv": hsv},
        "sampling": {"100": img_100, "50": img_50, "25": img_25},
        "quantization": {"256": q_256, "128": q_128, "64": q_64, "32": q_32},
        "formats": {"bmp_kb": bmp_kb, "png_kb": png_kb, "jpg_kb": jpg_kb}
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run_task2(src)
