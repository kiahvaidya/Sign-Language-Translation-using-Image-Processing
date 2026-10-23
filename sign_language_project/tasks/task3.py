"""
Task 3: Image Enhancement Module
Digital Image Processing Lab Evaluation Module

Techniques:
1. Linear Contrast Stretching
2. Histogram Equalization
3. Image Negative Transformation
4. Logarithmic Transformation
5. Power-Law (Gamma) Transformation
Includes quantitative evaluation: Mean Squared Error (MSE) and PSNR.
Outputs saved in outputs/task3/.
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
from preprocessing.enhancement import (
    contrast_stretch,
    histogram_equalization,
    image_negative,
    log_transform,
    gamma_transform,
    calculate_mse,
    calculate_psnr
)


def run_task3(image_path=None, save_output=True, show_plot=False):
    out_dir = PROJECT_ROOT / "outputs" / "task3"
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
    print("TASK 3: IMAGE ENHANCEMENT")
    print("=" * 60)
    print(f"Input image: {image_path}")

    bgr, rgb, gray, meta = load_image(image_path)

    # 1. Transformations
    cs_img = contrast_stretch(bgr)
    hist_eq = histogram_equalization(bgr)
    neg_img = image_negative(bgr)
    log_img = log_transform(bgr, c=1.0)
    gamma_bright = gamma_transform(bgr, gamma=0.5)
    gamma_dark = gamma_transform(bgr, gamma=2.0)

    # 2. Metrics against original
    metrics = {
        "Contrast Stretching": {"MSE": calculate_mse(bgr, cs_img), "PSNR": calculate_psnr(bgr, cs_img)},
        "Histogram Equalization": {"MSE": calculate_mse(bgr, hist_eq), "PSNR": calculate_psnr(bgr, hist_eq)},
        "Image Negative": {"MSE": calculate_mse(bgr, neg_img), "PSNR": calculate_psnr(bgr, neg_img)},
        "Log Transform": {"MSE": calculate_mse(bgr, log_img), "PSNR": calculate_psnr(bgr, log_img)},
        "Gamma (0.5)": {"MSE": calculate_mse(bgr, gamma_bright), "PSNR": calculate_psnr(bgr, gamma_bright)},
        "Gamma (2.0)": {"MSE": calculate_mse(bgr, gamma_dark), "PSNR": calculate_psnr(bgr, gamma_dark)},
    }

    print("\nQuantitative Evaluation (MSE & PSNR vs Original):")
    print("-" * 55)
    print(f"{'Method':<25}{'MSE':<15}{'PSNR (dB)':<15}")
    print("-" * 55)
    for m_name, vals in metrics.items():
        print(f"{m_name:<25}{vals['MSE']:<15.2f}{vals['PSNR']:<15.2f}")
    print("-" * 55)

    if save_output:
        cv2.imwrite(str(out_dir / "contrast_stretched.png"), cs_img)
        cv2.imwrite(str(out_dir / "histogram_equalized.png"), hist_eq)
        cv2.imwrite(str(out_dir / "negative.png"), neg_img)
        cv2.imwrite(str(out_dir / "log_transform.png"), log_img)
        cv2.imwrite(str(out_dir / "gamma_0_5.png"), gamma_bright)
        cv2.imwrite(str(out_dir / "gamma_2_0.png"), gamma_dark)

    # Multi-panel visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    results = [
        ("Original Input", rgb),
        ("Contrast Stretching", cv2.cvtColor(cs_img, cv2.COLOR_BGR2RGB)),
        ("Histogram Equalization", cv2.cvtColor(hist_eq, cv2.COLOR_BGR2RGB)),
        ("Image Negative", cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)),
        ("Logarithmic Transform", cv2.cvtColor(log_img, cv2.COLOR_BGR2RGB)),
        ("Gamma Transform (γ=0.5)", cv2.cvtColor(gamma_bright, cv2.COLOR_BGR2RGB))
    ]
    for idx, (title, img) in enumerate(results):
        r, c = idx // 3, idx % 3
        axes[r, c].imshow(img)
        axes[r, c].set_title(title, fontsize=12, fontweight="bold")
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_dir / "task3_enhancement_gallery.png"), dpi=200)
    plt.close()

    print(f"\nTask 3 outputs saved successfully to: {out_dir.resolve()}")
    return {
        "contrast_stretched": cs_img,
        "histogram_equalized": hist_eq,
        "negative": neg_img,
        "log_transform": log_img,
        "gamma_bright": gamma_bright,
        "metrics": metrics
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run_task3(src)
