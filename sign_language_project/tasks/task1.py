"""
Task 1: Image Acquisition
Digital Image Processing Lab Evaluation Module

Features:
- Loads original image safely from file, dataset, or user argument.
- Preserves the original image without modification.
- Computes image dimensions, channels, total pixels, and pixel values.
- Analyzes minimum, maximum, and mean intensity.
- Saves acquired output in outputs/task1/original_image.png.
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.acquisition import load_image, get_pixel_info, find_dataset_images


def run_task1(image_path=None, save_output=True, show_plot=True):
    output_dir = PROJECT_ROOT / "outputs" / "task1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine image source
    if image_path is None:
        dataset_dir = PROJECT_ROOT.parent / "dataset"
        if not dataset_dir.exists():
            dataset_dir = PROJECT_ROOT / "data" / "dataset"
        img_map = find_dataset_images(dataset_dir)
        if img_map:
            first_label = sorted(list(img_map.keys()))[0]
            image_path = img_map[first_label][0]
        else:
            raise FileNotFoundError("No input image provided and no dataset image found.")

    print("=" * 60)
    print("TASK 1: IMAGE ACQUISITION")
    print("=" * 60)
    print(f"Acquiring Image from: {image_path}")

    # 2. Ingest and preserve original image
    bgr, rgb, gray, meta = load_image(image_path)

    print("\nImage Acquisition Information:")
    print("-" * 40)
    print(f"Source Filename     : {meta['source_name']}")
    print(f"Dimensions          : {meta['height']} x {meta['width']}")
    print(f"Number of Channels  : {meta['channels']}")
    print(f"Total Pixels        : {meta['total_pixels']}")
    print(f"Data Type           : {meta['dtype']}")
    print(f"Min Intensity       : {meta['min_intensity']}")
    print(f"Max Intensity       : {meta['max_intensity']}")
    print(f"Mean Intensity      : {meta['mean_intensity']}")
    print(f"Standard Deviation  : {meta['std_intensity']}")

    # 3. Sample pixel inquiry (center pixel)
    cx, cy = meta['width'] // 2, meta['height'] // 2
    px_info = get_pixel_info(bgr, cx, cy)
    if px_info:
        print(f"\nCenter Pixel Coordinate ({cx}, {cy}):")
        print(f"  RGB  : {px_info['rgb']}")
        print(f"  BGR  : {px_info['bgr']}")
        print(f"  Gray : {px_info['gray']}")

    # 4. Save output
    out_path = output_dir / "original_image.png"
    if save_output:
        cv2.imwrite(str(out_path), bgr)
        print(f"\nOriginal image saved safely at: {out_path.resolve()}")

    # 5. Display
    if show_plot:
        plt.figure(figsize=(7, 5))
        plt.imshow(rgb)
        plt.title(f"Acquired Sign Language Image\n{meta['width']}x{meta['height']} ({meta['channels']} channels)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(output_dir / "task1_acquisition_display.png"), dpi=200)
        plt.close()

    print("=" * 60)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return bgr, meta


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run_task1(src, show_plot=False)
