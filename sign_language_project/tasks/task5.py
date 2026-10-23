"""
Task 5: Integrated Image Enhancement Pipeline
Digital Image Processing Lab Evaluation Module

Pipeline Architecture:
Original Input
      ↓
Contrast Stretching (Min-Max Normalization)
      ↓
Median Filtering (Impulse Noise Removal)
      ↓
High-Boost Sharpening (A=1.5, High-frequency Detail Amplification)
      ↓
Final Improved Image (Saved for Task 6 Segmentation Input)

Processes at least 3 sample images and stores results in outputs/task5/.
"""

import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.acquisition import load_image, find_dataset_images
from preprocessing.filtering import task5_integrated_pipeline


def process_single_task5_image(image_path, output_dir, index=1):
    bgr, rgb, gray, meta = load_image(image_path)
    stem = Path(image_path).stem

    stages = task5_integrated_pipeline(bgr, A=1.5, median_ksize=5)
    final_output = stages["high_boost"]

    # Save output for direct use by Task 6
    out_file = output_dir / f"{stem}_task5_final.png"
    cv2.imwrite(str(out_file), final_output)

    # 4-stage pipeline plot
    plt.figure(figsize=(16, 4.5))

    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title("1. Original Input", fontsize=11, fontweight="bold")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(stages["contrast_stretched"], cv2.COLOR_BGR2RGB))
    plt.title("2. Contrast Stretching", fontsize=11, fontweight="bold")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(stages["median_filtered"], cv2.COLOR_BGR2RGB))
    plt.title("3. Median Filter (5x5)", fontsize=11, fontweight="bold")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(final_output, cmap="gray")
    plt.title("4. High-Boost Sharpened (A=1.5)", fontsize=11, fontweight="bold")
    plt.axis("off")

    plt.suptitle(f"Task 5 Pipeline - Sample {index}: {stem}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    comparison_file = output_dir / f"{stem}_pipeline_comparison.png"
    plt.savefig(str(comparison_file), dpi=200)
    plt.close()

    print(f"Sample {index} [{stem}] processed -> Saved: {out_file.name}")
    return final_output, out_file


def run_task5(sample_paths=None, save_output=True, show_plot=False):
    out_dir = PROJECT_ROOT / "outputs" / "task5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK 5: INTEGRATED ENHANCEMENT PIPELINE")
    print("=" * 60)

    # Gather at least 3 images
    if sample_paths is None or len(sample_paths) == 0:
        dataset_dir = PROJECT_ROOT.parent / "dataset"
        if not dataset_dir.exists():
            dataset_dir = PROJECT_ROOT / "data" / "dataset"
        img_map = find_dataset_images(dataset_dir)
        sample_paths = []
        for cls in sorted(list(img_map.keys()))[:3]:
            if img_map[cls]:
                sample_paths.append(img_map[cls][0])

    if len(sample_paths) < 3:
        # Fill up to 3 if fewer classes available
        for cls in img_map:
            for p in img_map[cls]:
                if p not in sample_paths and len(sample_paths) < 3:
                    sample_paths.append(p)

    print(f"Processing {len(sample_paths)} benchmark images:")
    for idx, p in enumerate(sample_paths, 1):
        print(f"  {idx}. {p}")

    results = []
    for idx, p in enumerate(sample_paths, 1):
        final_img, out_path = process_single_task5_image(p, out_dir, index=idx)
        results.append((final_img, out_path))

    print("\n" + "=" * 60)
    print(f"TASK 5 COMPLETED: {len(results)} images saved in outputs/task5/")
    print(f"Ready for input into Task 6 Segmentation.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else None
    run_task5(paths)
