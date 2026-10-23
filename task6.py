import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# SIGN LANGUAGE IMAGE PROCESSING - TASK 6
# IMAGE SEGMENTATION
#
# INPUT:
# Best outputs obtained from Task 5
#
# Edge Detection:
# 1. Roberts
# 2. Prewitt
# 3. Sobel
# 4. Laplacian
#
# Thresholding:
# 1. Global Thresholding
# 2. Otsu Thresholding
# 3. Adaptive Thresholding
#
# Input images are automatically read from:
# outputs/task5/
#
# Results are automatically saved in:
# outputs/task6/
# ==========================================================

INPUT_DIR = Path("outputs") / "task5"
OUTPUT_DIR = Path("outputs") / "task6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# EDGE DETECTION
# ==========================================================

def roberts_operator(gray):
    kernel_x = np.array([[1, 0],
                         [0, -1]], dtype=np.float32)

    kernel_y = np.array([[0, 1],
                         [-1, 0]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

    magnitude = cv2.magnitude(gx, gy)

    return cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def prewitt_operator(gray):
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)

    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

    magnitude = cv2.magnitude(gx, gy)

    return cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def sobel_operator(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)

    return cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)


def laplacian_operator(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    absolute = cv2.convertScaleAbs(lap)

    return absolute


# ==========================================================
# THRESHOLDING
# ==========================================================

def global_threshold(gray, threshold_value=127):
    _, result = cv2.threshold(
        gray, threshold_value, 255, cv2.THRESH_BINARY
    )
    return result


def otsu_threshold(gray):
    threshold_value, result = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return result, threshold_value


def adaptive_threshold(gray):
    result = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return result


# ==========================================================
# DISPLAY EDGE RESULTS
# ==========================================================

def display_edges(gray, results, image_name):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Task 5 Input")
    plt.axis("off")

    titles = ["Roberts", "Prewitt", "Sobel", "Laplacian"]

    for i, title in enumerate(titles):
        plt.subplot(2, 3, i + 2)
        plt.imshow(results[title], cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.suptitle(f"Edge Detection - {image_name}")
    plt.tight_layout()
    plt.show()


# ==========================================================
# DISPLAY THRESHOLD RESULTS
# ==========================================================

def display_thresholds(gray, global_result, otsu_result,
                       adaptive_result, otsu_value, image_name):
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Task 5 Input")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(global_result, cmap="gray")
    plt.title("Global Threshold")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(otsu_result, cmap="gray")
    plt.title(f"Otsu (T={otsu_value:.0f})")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(adaptive_result, cmap="gray")
    plt.title("Adaptive Threshold")
    plt.axis("off")

    plt.suptitle(f"Thresholding - {image_name}")
    plt.tight_layout()
    plt.show()


# ==========================================================
# SIMPLE ANALYSIS
# ==========================================================

def calculate_binary_stats(binary):
    foreground_ratio = np.mean(binary == 255) * 100

    # Number of connected components
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    components = max(0, num_labels - 1)

    return foreground_ratio, components


def analyze_results(global_result, otsu_result, adaptive_result):
    g_ratio, g_components = calculate_binary_stats(global_result)
    o_ratio, o_components = calculate_binary_stats(otsu_result)
    a_ratio, a_components = calculate_binary_stats(adaptive_result)

    print("\nThresholding statistics:")
    print("-" * 65)
    print(f"{'Method':<22}{'Foreground %':<18}{'Components':<15}")
    print("-" * 65)
    print(f"{'Global':<22}{g_ratio:<18.2f}{g_components:<15}")
    print(f"{'Otsu':<22}{o_ratio:<18.2f}{o_components:<15}")
    print(f"{'Adaptive':<22}{a_ratio:<18.2f}{a_components:<15}")


# ==========================================================
# PROCESS ONE IMAGE
# ==========================================================

def process_image(input_path):
    image_name = input_path.stem

    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not read: {input_path}")
        return

    image_output_dir = OUTPUT_DIR / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # EDGE DETECTION
    # ------------------------------------------------------

    edges = {
        "Roberts": roberts_operator(img),
        "Prewitt": prewitt_operator(img),
        "Sobel": sobel_operator(img),
        "Laplacian": laplacian_operator(img)
    }

    # ------------------------------------------------------
    # THRESHOLDING
    # ------------------------------------------------------

    global_result = global_threshold(img, 127)

    otsu_result, otsu_value = otsu_threshold(img)

    adaptive_result = adaptive_threshold(img)

    # ------------------------------------------------------
    # SAVE INPUT COPY
    # ------------------------------------------------------

    cv2.imwrite(
        str(image_output_dir / "task5_input.png"),
        img
    )

    # ------------------------------------------------------
    # SAVE EDGE RESULTS
    # ------------------------------------------------------

    for name, result in edges.items():
        cv2.imwrite(
            str(image_output_dir / f"{name}.png"),
            result
        )

    # ------------------------------------------------------
    # SAVE THRESHOLD RESULTS
    # ------------------------------------------------------

    cv2.imwrite(
        str(image_output_dir / "Global_Threshold.png"),
        global_result
    )

    cv2.imwrite(
        str(image_output_dir / "Otsu.png"),
        otsu_result
    )

    cv2.imwrite(
        str(image_output_dir / "Adaptive_Threshold.png"),
        adaptive_result
    )

    # ------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------

    display_edges(img, edges, image_name)

    display_thresholds(
        img,
        global_result,
        otsu_result,
        adaptive_result,
        otsu_value,
        image_name
    )

    # ------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print(f"ANALYSIS: {image_name}")
    print("=" * 65)

    print("\nEdge Detection:")
    print("Roberts    : Detects fine boundaries but may be sensitive to noise.")
    print("Prewitt    : Gives clear basic horizontal and vertical edges.")
    print("Sobel      : Produces stronger and smoother hand boundaries.")
    print("Laplacian  : Detects fine details but may emphasize noise.")

    print("\nThresholding:")
    print("Global     : Simple but depends strongly on the chosen threshold.")
    print("Otsu       : Automatically selects the threshold from image histogram.")
    print("Adaptive   : Handles non-uniform illumination better.")

    analyze_results(
        global_result,
        otsu_result,
        adaptive_result
    )

    print(f"\nResults saved in: {image_output_dir.resolve()}")


# ==========================================================
# COMPARISON TABLE
# ==========================================================

def print_comparison_table():
    print("\n")
    print("=" * 105)
    print("TASK 6 - SEGMENTATION COMPARISON TABLE")
    print("=" * 105)

    print(f"{'Method':<22}"
          f"{'Separation':<16}"
          f"{'Boundary':<16}"
          f"{'Noise':<14}"
          f"{'Overall':<16}")

    print("-" * 105)

    rows = [
        ("Roberts", "Good", "Fine", "Medium", "Good"),
        ("Prewitt", "Good", "Clear", "Low-Med", "Good"),
        ("Sobel", "Very Good", "Strong", "Low", "Very Good"),
        ("Laplacian", "Good", "Fine", "Medium", "Good"),
        ("Global Threshold", "Good", "Moderate", "Medium", "Good"),
        ("Otsu", "Very Good", "Good", "Low", "Very Good"),
        ("Adaptive Threshold", "Good", "Good", "Medium", "Good")
    ]

    for row in rows:
        print(f"{row[0]:<22}"
              f"{row[1]:<16}"
              f"{row[2]:<16}"
              f"{row[3]:<14}"
              f"{row[4]:<16}")

    print("-" * 105)


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 65)
    print("SIGN LANGUAGE IMAGE PROCESSING - TASK 6")
    print("IMAGE SEGMENTATION: EDGE DETECTION + THRESHOLDING")
    print("=" * 65)

    if not INPUT_DIR.exists():
        print("\nERROR: outputs/task5/ does not exist.")
        print("Run task5.py first.")
        return

    input_files = sorted(
        list(INPUT_DIR.glob("*.png")) +
        list(INPUT_DIR.glob("*.jpg")) +
        list(INPUT_DIR.glob("*.jpeg"))
    )

    if len(input_files) == 0:
        print("\nNo Task 5 outputs found.")
        print("Run task5.py first and generate the 3 outputs.")
        return

    print(f"\nFound {len(input_files)} Task 5 output image(s).")

    for input_path in input_files:
        print(f"\nProcessing: {input_path.name}")
        process_image(input_path)

    print_comparison_table()

    print("\n" + "=" * 65)
    print("TASK 6 COMPLETED")
    print("=" * 65)
    print(f"All Task 6 results are saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nRecommended method for this project:")
    print("Sobel for edge detection.")
    print("Otsu for automatic threshold-based segmentation.")
    print("\nFinal selection should be based on the actual output images.")


if __name__ == "__main__":
    main()
