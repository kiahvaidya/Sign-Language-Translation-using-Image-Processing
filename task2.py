# ============================================================
# TASK 2: IMAGE ACQUISITION, REPRESENTATION & IMAGE FUNDAMENTALS
# Mini Project: Sign Language Detection
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 1. PROJECT FOLDER SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

ACQUISITION_DIR = os.path.join(OUTPUT_DIR, "acquisition")
COLOR_DIR = os.path.join(OUTPUT_DIR, "color_spaces")
SAMPLING_DIR = os.path.join(OUTPUT_DIR, "sampling")
QUANTIZATION_DIR = os.path.join(OUTPUT_DIR, "quantization")
FORMAT_DIR = os.path.join(OUTPUT_DIR, "formats")

folders = [
    DATASET_DIR,
    OUTPUT_DIR,
    ACQUISITION_DIR,
    COLOR_DIR,
    SAMPLING_DIR,
    QUANTIZATION_DIR,
    FORMAT_DIR
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# 2. SELECT IMAGE FROM DATASET
# ============================================================

def find_first_image(folder):
    """
    Finds the first supported image inside the dataset folder.
    Searches through subfolders as well.
    """

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                return os.path.join(root, file)

    return None


image_path = find_first_image(DATASET_DIR)

if image_path is None:
    print("\nERROR: No image found in the dataset folder.")
    print("Please put your sign-language images inside:")
    print(DATASET_DIR)
    print("\nExample:")
    print("dataset/A/A_001.jpg")
    exit()


print("=" * 60)
print("TASK 2 - IMAGE PROCESSING")
print("=" * 60)
print("\nSelected image:")
print(image_path)


# ============================================================
# TASK 2.3 - IMAGE ACQUISITION
# ============================================================

print("\n" + "=" * 60)
print("TASK 2.3 - IMAGE ACQUISITION")
print("=" * 60)

# Read image using OpenCV
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Unable to read image.")
    exit()

# Convert BGR to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(7, 5))
plt.imshow(image_rgb)
plt.title("Original Sign Language Image")
plt.axis("off")
plt.tight_layout()
plt.show()

# Save acquired image
acquisition_path = os.path.join(
    ACQUISITION_DIR,
    "original_image.png"
)

cv2.imwrite(acquisition_path, image)

print("Image successfully acquired.")
print("Original image saved at:")
print(acquisition_path)


# ============================================================
# TASK 2.4 - IMAGE REPRESENTATION
# ============================================================

print("\n" + "=" * 60)
print("TASK 2.4 - IMAGE REPRESENTATION")
print("=" * 60)

height, width = image.shape[:2]

# Number of channels
if len(image.shape) == 3:
    channels = image.shape[2]
else:
    channels = 1

# Number of pixels
total_pixels = height * width

# Total number of values/elements
total_values = image.size

# Data type
data_type = image.dtype

print("\nImage Information:")
print("-" * 40)
print(f"Height              : {height} pixels")
print(f"Width               : {width} pixels")
print(f"Dimensions          : {height} x {width}")
print(f"Number of channels  : {channels}")
print(f"Total pixels        : {total_pixels}")
print(f"Total image values  : {total_values}")
print(f"Data type           : {data_type}")
print(f"File name           : {os.path.basename(image_path)}")

# User-specified coordinate
print("\nEnter a pixel coordinate.")
print(f"Valid X range: 0 - {width - 1}")
print(f"Valid Y range: 0 - {height - 1}")

try:
    x = int(input("Enter X coordinate: "))
    y = int(input("Enter Y coordinate: "))

    if 0 <= x < width and 0 <= y < height:

        pixel_value = image[y, x]

        print("\nPixel Information:")
        print(f"Coordinate (X, Y) : ({x}, {y})")
        print(f"Pixel value      : {pixel_value}")

        if channels == 3:
            print("Channel order    : BGR")
            print(f"Blue             : {pixel_value[0]}")
            print(f"Green            : {pixel_value[1]}")
            print(f"Red              : {pixel_value[2]}")

    else:
        print("Invalid coordinate.")

except ValueError:
    print("Invalid input. Please enter integer coordinates.")


# ============================================================
# TASK 2.5 - COLOR SPACE CONVERSION
# ============================================================

print("\n" + "=" * 60)
print("TASK 2.5 - COLOR SPACE CONVERSION")
print("=" * 60)

# OpenCV image is BGR
# Convert to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("Successfully converted image into:")
print("1. RGB")
print("2. Grayscale")
print("3. HSV")

# Display all colour spaces
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original RGB Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(hsv_image)
plt.title("HSV Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(rgb_image)
plt.title("RGB Image")
plt.axis("off")

plt.tight_layout()

color_comparison_path = os.path.join(
    COLOR_DIR,
    "color_space_comparison.png"
)

plt.savefig(color_comparison_path, dpi=300, bbox_inches="tight")
plt.show()

# Save individual images
cv2.imwrite(
    os.path.join(COLOR_DIR, "grayscale.png"),
    gray_image
)

cv2.imwrite(
    os.path.join(COLOR_DIR, "hsv.png"),
    hsv_image
)

cv2.imwrite(
    os.path.join(COLOR_DIR, "rgb.png"),
    cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
)

print("\nColor space outputs saved to:")
print(COLOR_DIR)


# ============================================================
# TASK 2.6 - SAMPLING
# ============================================================

print("\n" + "=" * 60)
print("TASK 2.6 - SAMPLING")
print("=" * 60)

# Original dimensions
original_height, original_width = image.shape[:2]

# Resize to different sampling levels
image_100 = cv2.resize(
    image_rgb,
    (original_width, original_height),
    interpolation=cv2.INTER_AREA
)

image_50 = cv2.resize(
    image_rgb,
    (original_width // 2, original_height // 2),
    interpolation=cv2.INTER_AREA
)

image_25 = cv2.resize(
    image_rgb,
    (original_width // 4, original_height // 4),
    interpolation=cv2.INTER_AREA
)

print("\nSampling dimensions:")
print(
    f"100% : {image_100.shape[1]} x {image_100.shape[0]}"
)
print(
    f"50%  : {image_50.shape[1]} x {image_50.shape[0]}"
)
print(
    f"25%  : {image_25.shape[1]} x {image_25.shape[0]}"
)

# Display sampling results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_100)
plt.title("100% Sampling")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_50)
plt.title("50% Sampling")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_25)
plt.title("25% Sampling")
plt.axis("off")

plt.tight_layout()

sampling_path = os.path.join(
    SAMPLING_DIR,
    "sampling_comparison.png"
)

plt.savefig(sampling_path, dpi=300, bbox_inches="tight")
plt.show()

# Save individual sampling images
cv2.imwrite(
    os.path.join(SAMPLING_DIR, "sampling_100_percent.png"),
    cv2.cvtColor(image_100, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(SAMPLING_DIR, "sampling_50_percent.png"),
    cv2.cvtColor(image_50, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(SAMPLING_DIR, "sampling_25_percent.png"),
    cv2.cvtColor(image_25, cv2.COLOR_RGB2BGR)
)


# ============================================================
# GRAY-LEVEL QUANTIZATION
# ============================================================

print("\n" + "=" * 60)
print("GRAY-LEVEL QUANTIZATION")
print("=" * 60)

# Convert image to grayscale
gray = gray_image.astype(np.uint8)


def quantize_image(image, levels):
    """
    Quantize an 8-bit grayscale image
    into the specified number of gray levels.
    """

    step = 256 // levels

    quantized = (image // step) * step

    # Prevent values from exceeding 255
    quantized = np.clip(quantized, 0, 255)

    return quantized.astype(np.uint8)


# Create different quantized versions
quantized_256 = quantize_image(gray, 256)
quantized_128 = quantize_image(gray, 128)
quantized_64 = quantize_image(gray, 64)
quantized_32 = quantize_image(gray, 32)

print("Generated:")
print("256 gray levels")
print("128 gray levels")
print("64 gray levels")
print("32 gray levels")

# Display quantization
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(quantized_256, cmap="gray", vmin=0, vmax=255)
plt.title("256 Gray Levels")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(quantized_128, cmap="gray", vmin=0, vmax=255)
plt.title("128 Gray Levels")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(quantized_64, cmap="gray", vmin=0, vmax=255)
plt.title("64 Gray Levels")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(quantized_32, cmap="gray", vmin=0, vmax=255)
plt.title("32 Gray Levels")
plt.axis("off")

plt.tight_layout()

quantization_path = os.path.join(
    QUANTIZATION_DIR,
    "quantization_comparison.png"
)

plt.savefig(quantization_path, dpi=300, bbox_inches="tight")
plt.show()

# Save quantized images
cv2.imwrite(
    os.path.join(QUANTIZATION_DIR, "256_levels.png"),
    quantized_256
)

cv2.imwrite(
    os.path.join(QUANTIZATION_DIR, "128_levels.png"),
    quantized_128
)

cv2.imwrite(
    os.path.join(QUANTIZATION_DIR, "64_levels.png"),
    quantized_64
)

cv2.imwrite(
    os.path.join(QUANTIZATION_DIR, "32_levels.png"),
    quantized_32
)


# ============================================================
# TASK 2.7 - IMAGE FILE FORMATS
# ============================================================

print("\n" + "=" * 60)
print("TASK 2.7 - IMAGE FILE FORMATS")
print("=" * 60)

# Save same image in BMP, PNG and JPEG
bmp_path = os.path.join(
    FORMAT_DIR,
    "sign_image.bmp"
)

png_path = os.path.join(
    FORMAT_DIR,
    "sign_image.png"
)

jpeg_path = os.path.join(
    FORMAT_DIR,
    "sign_image.jpg"
)

# Save using OpenCV
cv2.imwrite(bmp_path, image)
cv2.imwrite(png_path, image)
cv2.imwrite(
    jpeg_path,
    image,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)

# Get file sizes
bmp_size = os.path.getsize(bmp_path)
png_size = os.path.getsize(png_path)
jpeg_size = os.path.getsize(jpeg_path)


def convert_size(size_bytes):
    """
    Convert bytes into KB.
    """
    return size_bytes / 1024


bmp_kb = convert_size(bmp_size)
png_kb = convert_size(png_size)
jpeg_kb = convert_size(jpeg_size)

print("\nFile Format Comparison:")
print("-" * 50)

print(
    f"BMP  : {bmp_kb:.2f} KB"
)

print(
    f"PNG  : {png_kb:.2f} KB"
)

print(
    f"JPEG : {jpeg_kb:.2f} KB"
)

# Create format comparison figure
bmp_display = cv2.cvtColor(
    cv2.imread(bmp_path),
    cv2.COLOR_BGR2RGB
)

png_display = cv2.cvtColor(
    cv2.imread(png_path),
    cv2.COLOR_BGR2RGB
)

jpeg_display = cv2.cvtColor(
    cv2.imread(jpeg_path),
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(bmp_display)
plt.title(f"BMP\n{bmp_kb:.2f} KB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(png_display)
plt.title(f"PNG\n{png_kb:.2f} KB")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(jpeg_display)
plt.title(f"JPEG\n{jpeg_kb:.2f} KB")
plt.axis("off")

plt.tight_layout()

format_comparison_path = os.path.join(
    FORMAT_DIR,
    "format_comparison.png"
)

plt.savefig(
    format_comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FILE SIZE BAR GRAPH
# ============================================================

formats = ["BMP", "PNG", "JPEG"]
sizes = [bmp_kb, png_kb, jpeg_kb]

plt.figure(figsize=(8, 5))

plt.bar(formats, sizes)

plt.title("Image File Size Comparison")
plt.xlabel("Image Format")
plt.ylabel("File Size (KB)")

for i, size in enumerate(sizes):
    plt.text(
        i,
        size,
        f"{size:.2f} KB",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

file_size_graph_path = os.path.join(
    FORMAT_DIR,
    "file_size_comparison.png"
)

plt.savefig(
    file_size_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# TASK 2 SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutputs generated in:")
print(OUTPUT_DIR)

print("\nFolders created:")
print("1. acquisition/")
print("2. color_spaces/")
print("3. sampling/")
print("4. quantization/")
print("5. formats/")

print("\nKey observations:")
print("- Sampling reduces spatial resolution and image detail.")
print("- Lower sampling levels make fine hand/finger details less visible.")
print("- Quantization reduces the number of available intensity levels.")
print("- Lower gray-level quantization produces visible intensity banding.")
print("- BMP generally requires more storage because it has little/no compression.")
print("- PNG uses lossless compression and preserves image information.")
print("- JPEG uses lossy compression and may introduce compression artifacts.")

print("\nAll Task 2 operations have been completed.")