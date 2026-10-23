"""
Task 4: Spatial Filtering and Sharpening Module
Digital Image Processing Lab Evaluation Module

Filters Implemented:
1. Mean Smoothing Filter (Box Filter)
2. Gaussian Smoothing Filter
3. Median Filter (Rank-Order Filter)
4. Laplacian Sharpening
5. Gradient-based Sharpening (Sobel)
6. High-Boost Sharpening Filter (A=1.5)
Evaluates PSNR and MSE across filters, with noise handling demonstration.
Outputs saved in outputs/task4/.
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
from preprocessing.filtering import (
    mean_filter,
    gaussian_filter,
    median_filter,
    laplacian_sharpen,
    gradient_sharpen,
    high_boost_filter,
    add_salt_pepper_noise
)
from preprocessing.enhancement import calculate_mse, calculate_psnr


def run_task4(image_path=None, save_output=True, show_plot=False):
    out_dir = PROJECT_ROOT / "outputs" / "task4"
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
    print("TASK 4: SPATIAL FILTERING & SHARPENING")
    print("=" * 60)
    print(f"Input image: {image_path}")

    bgr, rgb, gray, meta = load_image(image_path)

    # 1. Apply Filters
    mean_res = mean_filter(bgr, 5)
    gauss_res = gaussian_filter(bgr, 5, 0)
    median_res = median_filter(bgr, 5)
    lap_res = laplacian_sharpen(bgr)
    grad_res = gradient_sharpen(bgr)
    highboost_res = high_boost_filter(bgr, A=1.5, kernel_size=5)

    # 2. Add noise test
    noisy = add_salt_pepper_noise(bgr, amount=0.04)
    denoised_median = median_filter(noisy, 5)
    denoised_mean = mean_filter(noisy, 5)

    # 3. Quantitative metrics
    metrics = {
        "Mean Filter": {"MSE": calculate_mse(bgr, mean_res), "PSNR": calculate_psnr(bgr, mean_res)},
        "Gaussian Filter": {"MSE": calculate_mse(bgr, gauss_res), "PSNR": calculate_psnr(bgr, gauss_res)},
        "Median Filter": {"MSE": calculate_mse(bgr, median_res), "PSNR": calculate_psnr(bgr, median_res)},
        "Laplacian Sharpen": {"MSE": calculate_mse(bgr, lap_res), "PSNR": calculate_psnr(bgr, lap_res)},
        "Gradient Sharpen": {"MSE": calculate_mse(bgr, grad_res), "PSNR": calculate_psnr(bgr, grad_res)},
        "High-Boost (A=1.5)": {"MSE": calculate_mse(bgr, highboost_res), "PSNR": calculate_psnr(bgr, highboost_res)},
        "Denoised (Median vs Noisy)": {"MSE": calculate_mse(bgr, denoised_median), "PSNR": calculate_psnr(bgr, denoised_median)},
        "Denoised (Mean vs Noisy)": {"MSE": calculate_mse(bgr, denoised_mean), "PSNR": calculate_psnr(bgr, denoised_mean)}
    }

    print("\nFilter Performance Metrics (MSE & PSNR vs Original):")
    print("-" * 60)
    print(f"{'Filter Technique':<28}{'MSE':<15}{'PSNR (dB)':<15}")
    print("-" * 60)
    for f_name, vals in metrics.items():
        print(f"{f_name:<28}{vals['MSE']:<15.2f}{vals['PSNR']:<15.2f}")
    print("-" * 60)

    if save_output:
        cv2.imwrite(str(out_dir / "mean_filter.png"), mean_res)
        cv2.imwrite(str(out_dir / "gaussian_filter.png"), gauss_res)
        cv2.imwrite(str(out_dir / "median_filter.png"), median_res)
        cv2.imwrite(str(out_dir / "laplacian_sharpen.png"), lap_res)
        cv2.imwrite(str(out_dir / "gradient_sharpen.png"), grad_res)
        cv2.imwrite(str(out_dir / "high_boost.png"), highboost_res)
        cv2.imwrite(str(out_dir / "noisy_salt_pepper.png"), noisy)
        cv2.imwrite(str(out_dir / "denoised_median.png"), denoised_median)

    # 6-panel comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    results = [
        ("Original Input", rgb),
        ("Mean Filter (5x5)", cv2.cvtColor(mean_res, cv2.COLOR_BGR2RGB)),
        ("Gaussian Filter (5x5)", cv2.cvtColor(gauss_res, cv2.COLOR_BGR2RGB)),
        ("Median Filter (5x5)", cv2.cvtColor(median_res, cv2.COLOR_BGR2RGB)),
        ("Laplacian Sharpen", cv2.cvtColor(lap_res, cv2.COLOR_BGR2RGB) if len(lap_res.shape) == 3 else lap_res),
        ("High-Boost Sharpening", highboost_res if len(highboost_res.shape) == 2 else cv2.cvtColor(highboost_res, cv2.COLOR_BGR2RGB))
    ]
    for idx, (title, img) in enumerate(results):
        r, c = idx // 3, idx % 3
        if len(img.shape) == 2:
            axes[r, c].imshow(img, cmap="gray")
        else:
            axes[r, c].imshow(img)
        axes[r, c].set_title(title, fontsize=12, fontweight="bold")
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_dir / "task4_filter_comparison.png"), dpi=200)
    plt.close()

    print(f"\nTask 4 outputs saved successfully to: {out_dir.resolve()}")
    return {
        "mean": mean_res,
        "gaussian": gauss_res,
        "median": median_res,
        "high_boost": highboost_res,
        "metrics": metrics
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    run_task4(src)
