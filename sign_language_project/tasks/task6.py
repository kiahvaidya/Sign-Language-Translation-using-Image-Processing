"""
Task 6: Image Segmentation Module
Digital Image Processing Lab Evaluation Module

Evaluates and compares:
1. Edge Detection Operators:
   - Roberts Cross
   - Prewitt Operator
   - Sobel Operator
   - Laplacian Operator
2. Thresholding Techniques:
   - Global Fixed Thresholding
   - Otsu Automatic Optimal Thresholding
   - Adaptive Local Thresholding
3. Practical HSV/YCrCb Skin Segmentation
Computes quantitative binary stats (foreground %, connected components, largest area)
and qualitative comparative table. Saves outputs in outputs/task6/.
"""

import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.segmentation import (
    roberts_edge,
    prewitt_edge,
    sobel_edge,
    laplacian_edge,
    global_threshold,
    otsu_threshold,
    adaptive_threshold,
    hsv_skin_segmentation,
    calculate_binary_stats,
    get_task6_comparison_table
)


def process_task6_image(input_path, output_dir):
    stem = Path(input_path).stem
    img_out_dir = output_dir / stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # 1. Edge Detection
    edges = {
        "Roberts": roberts_edge(gray),
        "Prewitt": prewitt_edge(gray),
        "Sobel": sobel_edge(gray),
        "Laplacian": laplacian_edge(gray)
    }

    # 2. Thresholding
    thresh_global = global_threshold(gray, 127)
    thresh_otsu, otsu_val = otsu_threshold(gray)
    thresh_adaptive = adaptive_threshold(gray)
    skin_mask = hsv_skin_segmentation(img) if len(img.shape) == 3 else thresh_otsu

    # Save images
    cv2.imwrite(str(img_out_dir / "input_task5.png"), gray)
    for name, edge_img in edges.items():
        cv2.imwrite(str(img_out_dir / f"{name}_edge.png"), edge_img)

    cv2.imwrite(str(img_out_dir / "Global_Threshold.png"), thresh_global)
    cv2.imwrite(str(img_out_dir / "Otsu_Threshold.png"), thresh_otsu)
    cv2.imwrite(str(img_out_dir / "Adaptive_Threshold.png"), thresh_adaptive)
    cv2.imwrite(str(img_out_dir / "HSV_Skin_Mask.png"), skin_mask)

    # 3. Statistics
    stats_g = calculate_binary_stats(thresh_global)
    stats_o = calculate_binary_stats(thresh_otsu)
    stats_a = calculate_binary_stats(thresh_adaptive)
    stats_s = calculate_binary_stats(skin_mask)

    print(f"\nSegmentation Quantitative Stats for: {stem}")
    print("-" * 65)
    print(f"{'Method':<20}{'Foreground %':<16}{'Components':<15}{'Largest Area':<15}")
    print("-" * 65)
    print(f"{'Global (T=127)':<20}{stats_g['foreground_ratio']:<16.2f}{stats_g['components']:<15}{stats_g['largest_area']:<15}")
    print(f"{f'Otsu (T={otsu_val:.0f})':<20}{stats_o['foreground_ratio']:<16.2f}{stats_o['components']:<15}{stats_o['largest_area']:<15}")
    print(f"{'Adaptive':<20}{stats_a['foreground_ratio']:<16.2f}{stats_a['components']:<15}{stats_a['largest_area']:<15}")
    print(f"{'HSV Skin Mask':<20}{stats_s['foreground_ratio']:<16.2f}{stats_s['components']:<15}{stats_s['largest_area']:<15}")
    print("-" * 65)

    # Visualization plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Task 5 Input", fontweight="bold")
    axes[0, 0].axis("off")

    for i, (name, edge_img) in enumerate(edges.items()):
        col = (i + 1) % 4
        row = (i + 1) // 4
        axes[row, col].imshow(edge_img, cmap="gray")
        axes[row, col].set_title(f"{name} Edges", fontweight="bold")
        axes[row, col].axis("off")

    axes[1, 1].imshow(thresh_global, cmap="gray")
    axes[1, 1].set_title("Global Threshold", fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(thresh_otsu, cmap="gray")
    axes[1, 2].set_title(f"Otsu (T={otsu_val:.0f})", fontweight="bold")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(skin_mask, cmap="gray")
    axes[1, 3].set_title("HSV Skin Segmentation", fontweight="bold")
    axes[1, 3].axis("off")

    plt.suptitle(f"Task 6 Segmentation Suite - {stem}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(img_out_dir / "segmentation_summary.png"), dpi=200)
    plt.close()

    return {
        "edges": edges,
        "global": thresh_global,
        "otsu": thresh_otsu,
        "otsu_val": otsu_val,
        "adaptive": thresh_adaptive,
        "skin_mask": skin_mask,
        "stats": {"global": stats_g, "otsu": stats_o, "adaptive": stats_a, "skin": stats_s}
    }


def run_task6(input_dir=None, save_output=True, show_plot=False):
    task5_out = PROJECT_ROOT / "outputs" / "task5"
    out_dir = PROJECT_ROOT / "outputs" / "task6"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK 6: IMAGE SEGMENTATION")
    print("=" * 60)

    # Look for Task 5 outputs
    files = sorted(list(task5_out.glob("*_task5_final.png")))
    if not files:
        # Fallback to dataset images if Task 5 hasn't been run yet
        from tasks.task5 import run_task5
        print("No prior Task 5 output found. Automatically running Task 5 first...")
        run_task5()
        files = sorted(list(task5_out.glob("*_task5_final.png")))

    print(f"Processing {len(files[:3])} images from Task 5:")
    results = {}
    for f in files[:3]:
        print(f"  -> {f.name}")
        results[f.stem] = process_task6_image(f, out_dir)

    print("\n" + "=" * 90)
    print("TASK 6 QUALITATIVE SEGMENTATION EVALUATION TABLE")
    print("=" * 90)
    table_rows = get_task6_comparison_table()
    print(f"{'Method':<24}{'Category':<16}{'Separation':<14}{'Boundary':<16}{'Noise':<12}{'Overall':<12}")
    print("-" * 90)
    for r in table_rows:
        print(f"{r['Method']:<24}{r['Category']:<16}{r['Separation']:<14}{r['Boundary']:<16}{r['Noise Sensitivity']:<12}{r['Overall']:<12}")
    print("=" * 90)

    return results


if __name__ == "__main__":
    run_task6()
