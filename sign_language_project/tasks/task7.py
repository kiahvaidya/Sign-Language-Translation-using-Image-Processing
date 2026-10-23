"""
Task 7: Region-Based Segmentation Module
Digital Image Processing Lab Evaluation Module

Features:
- Seeded Region Growing with 4-way and 8-way pixel adjacency.
- Automatic (image center) or manual coordinate seed selection.
- Multi-threshold sensitivity experiment (T=10, T=25, T=50).
- Execution timing and area calculation.
- Direct side-by-side comparison with Otsu thresholding.
- Saves results in outputs/task7/.
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
from preprocessing.region_growing import (
    region_growing,
    threshold_experiment,
    draw_seed_overlay
)
from preprocessing.segmentation import otsu_threshold


def run_task7(image_path=None, seed_x=None, seed_y=None, save_output=True, show_plot=False):
    out_dir = PROJECT_ROOT / "outputs" / "task7"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print("TASK 7: REGION GROWING & REGION-BASED SEGMENTATION")
    print("=" * 60)
    print(f"Input image: {image_path}")

    bgr, rgb, gray, meta = load_image(image_path)
    h, w = meta["height"], meta["width"]

    # 1. Determine seed point
    if seed_x is None or seed_y is None:
        seed_x = w // 2
        seed_y = h // 2
    seed_point = (seed_x, seed_y)
    seed_intensity = gray[seed_y, seed_x]

    print(f"Seed Point Selected: ({seed_x}, {seed_y}) with Intensity: {seed_intensity}")

    # 2. Seed visualization
    seed_vis = draw_seed_overlay(rgb, seed_point, color=(0, 255, 0), radius=6)

    # 3. Multi-threshold experiment (T=10, T=25, T=50)
    exp_results = threshold_experiment(gray, seed_point, thresholds=(10, 25, 50), connectivity=4)

    # 4. Otsu comparison
    otsu_mask, otsu_val = otsu_threshold(gray)

    print("\nRegion Growing Threshold Experiment Results:")
    print("-" * 65)
    print(f"{'Threshold (T)':<18}{'Area (px)':<16}{'Foreground %':<16}{'Time (ms)':<15}")
    print("-" * 65)
    for res in exp_results:
        print(f"T = {res['threshold']:<14}{res['area']:<16}{res['foreground_ratio']:<16.2f}{res['time_ms']:<15.2f}")
    print("-" * 65)

    if save_output:
        cv2.imwrite(str(out_dir / "seed_marked.png"), cv2.cvtColor(seed_vis, cv2.COLOR_RGB2BGR))
        for res in exp_results:
            cv2.imwrite(str(out_dir / f"region_grown_T{res['threshold']}.png"), res["mask"])
        cv2.imwrite(str(out_dir / "otsu_comparison.png"), otsu_mask)

    # Multi-panel visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(seed_vis)
    axes[0, 0].set_title(f"Seed Point: ({seed_x}, {seed_y})", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title(f"Grayscale Input\n(Seed Val = {seed_intensity})", fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(otsu_mask, cmap="gray")
    axes[0, 2].set_title(f"Otsu Benchmark (T={otsu_val:.0f})", fontweight="bold")
    axes[0, 2].axis("off")

    for i, res in enumerate(exp_results):
        axes[1, i].imshow(res["mask"], cmap="gray")
        axes[1, i].set_title(
            f"Region Grown: T={res['threshold']}\nArea={res['area']} px ({res['time_ms']} ms)",
            fontweight="bold"
        )
        axes[1, i].axis("off")

    plt.suptitle("Task 7: Region Growing Experiment & Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_dir / "region_growing_comparison.png"), dpi=200)
    plt.close()

    print(f"\nTask 7 outputs saved successfully in: {out_dir.resolve()}")
    return {
        "seed_point": seed_point,
        "seed_vis": seed_vis,
        "experiments": exp_results,
        "otsu_mask": otsu_mask
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run_task7(src)
