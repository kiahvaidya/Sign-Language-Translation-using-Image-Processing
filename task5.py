import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# SIGN LANGUAGE IMAGE PROCESSING - TASK 5
# Spatial Domain Filtering / Sharpening
#
# Pipeline:
# ORIGINAL -> CONTRAST STRETCHING -> MEDIAN FILTER
#          -> HIGH-BOOST FILTERING
#
# The ORIGINAL image is never replaced.
# The final Task 5 output is automatically saved in:
# outputs/task5/
#
# Enter 3 images when the program starts.
# ==========================================================

OUTPUT_DIR = Path("outputs") / "task5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def contrast_stretch(img):
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val == min_val:
        return img.copy()

    stretched = (img.astype(np.float32) - min_val) * 255.0 / (max_val - min_val)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def median_filter(img):
    return cv2.medianBlur(img, 5)


def high_boost_filter(img, A=1.5):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = gray - blurred

    high_boost = A * gray + mask
    high_boost = np.clip(high_boost, 0, 255)

    return high_boost.astype(np.uint8)


def process_image(path, index):
    original = cv2.imread(str(path))

    if original is None:
        print(f"ERROR: Could not load {path}")
        return None

    # STEP 1: Contrast enhancement
    enhanced = contrast_stretch(original)

    # STEP 2: Noise reduction
    filtered = median_filter(enhanced)

    # STEP 3: High-boost sharpening
    final = high_boost_filter(filtered, 1.5)

    # Use the original filename for easy identification
    stem = Path(path).stem
    output_path = OUTPUT_DIR / f"{stem}_task5_final.png"

    cv2.imwrite(str(output_path), final)

    # Display the complete Task 5 pipeline
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Input")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title("Contrast Stretching")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    plt.title("Median Filter")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(final, cmap="gray")
    plt.title("Task 5 Final Output")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Image {index} completed.")
    print(f"Task 5 output saved: {output_path}")

    return final


def main():
    print("=" * 60)
    print("SIGN LANGUAGE IMAGE PROCESSING - TASK 5")
    print("=" * 60)

    print("\nEnter the paths of 3 project dataset images.")
    print("Example: dataset/sign1.jpg")

    image_paths = []

    for i in range(1, 4):
        while True:
            path = input(f"\nEnter path for Image {i}: ").strip()

            if not Path(path).is_file():
                print("File not found. Please enter a valid image path.")
                continue

            image_paths.append(path)
            break

    print("\n" + "=" * 60)
    print("PROCESSING 3 IMAGES")
    print("=" * 60)

    successful = 0

    for i, path in enumerate(image_paths, start=1):
        result = process_image(path, i)
        if result is not None:
            successful += 1

    print("\n" + "=" * 60)
    print("TASK 5 COMPLETED")
    print("=" * 60)
    print(f"Successfully processed: {successful}/3 images")
    print(f"\nAll final outputs are stored in:")
    print(OUTPUT_DIR.resolve())
    print("\nThese saved images are now ready to be used as")
    print("INPUT images for task6.py.")


if __name__ == "__main__":
    main()