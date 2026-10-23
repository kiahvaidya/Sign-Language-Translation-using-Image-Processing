"""
Task 8: Morphological Processing Module
Digital Image Processing Lab Evaluation Module

Features:
- Structuring Elements: 3x3 Square, 3x3 Cross, 5x5 Square.
- Operations: Erosion, Dilation, Opening, Closing.
- Multiple Iteration Experiments: Evaluates 1, 2, and 3 iterations for each operation.
- Hand Mask Cleaning Pipeline: Noise removal, gap filling, and largest component isolation.
- Saves results in outputs/task8/.
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
from preprocessing.segmentation import otsu_threshold, hsv_skin_segmentation
from preprocessing.morphology import (
    apply_morphology,
    generate_iteration_series,
    clean_hand_mask,
    get_structuring_element
)


def run_task8(image_path=None, save_output=True, show_plot=False):
    out_dir = PROJECT_ROOT / "outputs" / "task8"
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
    print("TASK 8: BINARY MORPHOLOGICAL PROCESSING")
    print("=" * 60)
    print(f"Input image: {image_path}")

    bgr, rgb, gray, meta = load_image(image_path)

    # 1. Obtain starting binary mask (via Otsu or Skin Mask)
    raw_binary, otsu_val = otsu_threshold(gray)

    # 2. Structuring Element Variations (iterations=1)
    se_rect3 = get_structuring_element("rect", 3)
    se_cross3 = get_structuring_element("cross", 3)
    se_rect5 = get_structuring_element("rect", 5)

    erode_rect3 = cv2.erode(raw_binary, se_rect3, iterations=1)
    erode_cross3 = cv2.erode(raw_binary, se_cross3, iterations=1)
    erode_rect5 = cv2.erode(raw_binary, se_rect5, iterations=1)

    dilate_rect3 = cv2.dilate(raw_binary, se_rect3, iterations=1)
    dilate_cross3 = cv2.dilate(raw_binary, se_cross3, iterations=1)
    dilate_rect5 = cv2.dilate(raw_binary, se_rect5, iterations=1)

    # 3. Multiple Iterations Experiments (1, 2, 3 iterations)
    erosion_series = generate_iteration_series(raw_binary, op="erode", max_iter=3)
    dilation_series = generate_iteration_series(raw_binary, op="dilate", max_iter=3)
    opening_series = generate_iteration_series(raw_binary, op="open", max_iter=3)
    closing_series = generate_iteration_series(raw_binary, op="close", max_iter=3)

    # 4. Final Cleaned Hand Mask
    cleaned_mask = clean_hand_mask(raw_binary, open_iter=1, close_iter=2, keep_largest=True)

    print("\nMorphological Iterations Area Analysis (Pixels):")
    print("-" * 55)
    print(f"{'Operation':<20}{'Iter 1':<12}{'Iter 2':<12}{'Iter 3':<12}")
    print("-" * 55)
    print(f"{'Erosion':<20}{cv2.countNonZero(erosion_series[1]):<12}{cv2.countNonZero(erosion_series[2]):<12}{cv2.countNonZero(erosion_series[3]):<12}")
    print(f"{'Dilation':<20}{cv2.countNonZero(dilation_series[1]):<12}{cv2.countNonZero(dilation_series[2]):<12}{cv2.countNonZero(dilation_series[3]):<12}")
    print(f"{'Opening':<20}{cv2.countNonZero(opening_series[1]):<12}{cv2.countNonZero(opening_series[2]):<12}{cv2.countNonZero(opening_series[3]):<12}")
    print(f"{'Closing':<20}{cv2.countNonZero(closing_series[1]):<12}{cv2.countNonZero(closing_series[2]):<12}{cv2.countNonZero(closing_series[3]):<12}")
    print("-" * 55)

    if save_output:
        cv2.imwrite(str(out_dir / "raw_binary.png"), raw_binary)
        cv2.imwrite(str(out_dir / "final_cleaned_mask.png"), cleaned_mask)
        for i in (1, 2, 3):
            cv2.imwrite(str(out_dir / f"erosion_iter{i}.png"), erosion_series[i])
            cv2.imwrite(str(out_dir / f"dilation_iter{i}.png"), dilation_series[i])
            cv2.imwrite(str(out_dir / f"opening_iter{i}.png"), opening_series[i])
            cv2.imwrite(str(out_dir / f"closing_iter{i}.png"), closing_series[i])

    # 8-panel iteration comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(raw_binary, cmap="gray")
    axes[0, 0].set_title("Original Binary", fontweight="bold")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(erosion_series[1], cmap="gray")
    axes[0, 1].set_title("Erosion: 1 Iteration", fontweight="bold")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(erosion_series[2], cmap="gray")
    axes[0, 2].set_title("Erosion: 2 Iterations", fontweight="bold")
    axes[0, 2].axis("off")
    axes[0, 3].imshow(erosion_series[3], cmap="gray")
    axes[0, 3].set_title("Erosion: 3 Iterations", fontweight="bold")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(cleaned_mask, cmap="gray")
    axes[1, 0].set_title("Final Cleaned Mask", fontweight="bold")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(dilation_series[1], cmap="gray")
    axes[1, 1].set_title("Dilation: 1 Iteration", fontweight="bold")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(dilation_series[2], cmap="gray")
    axes[1, 2].set_title("Dilation: 2 Iterations", fontweight="bold")
    axes[1, 2].axis("off")
    axes[1, 3].imshow(dilation_series[3], cmap="gray")
    axes[1, 3].set_title("Dilation: 3 Iterations", fontweight="bold")
    axes[1, 3].axis("off")

    plt.suptitle("Task 8: Morphological Processing & Iteration Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_dir / "morphology_iterations_summary.png"), dpi=200)
    plt.close()

    print(f"\nTask 8 outputs saved successfully in: {out_dir.resolve()}")
    return {
        "raw_binary": raw_binary,
        "cleaned_mask": cleaned_mask,
        "erosion_series": erosion_series,
        "dilation_series": dilation_series,
        "opening_series": opening_series,
        "closing_series": closing_series
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run_task8(src)
