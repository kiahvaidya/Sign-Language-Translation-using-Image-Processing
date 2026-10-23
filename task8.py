import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TASK 8: BINARY MORPHOLOGICAL PROCESSING
# EROSION AND DILATION
# ============================================================

# ------------------------------------------------------------
# TASK 1: LOAD AND CONVERT SEGMENTED IMAGE TO BINARY
# ------------------------------------------------------------

# Enter the path of your segmented image from Task 6
image_path = input("Enter the path of the segmented image from Task 6: ")

image = cv2.imread(image_path)

if image is None:
    print("Error: Image could not be loaded.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("\nImage Dimensions:")
print("Width :", image.shape[1])
print("Height:", image.shape[0])

print("\nMinimum intensity:", gray.min())
print("Maximum intensity:", gray.max())


# ------------------------------------------------------------
# TRY TWO DIFFERENT THRESHOLD VALUES
# ------------------------------------------------------------

threshold1 = 100
threshold2 = 150

_, binary1 = cv2.threshold(
    gray, threshold1, 255, cv2.THRESH_BINARY
)

_, binary2 = cv2.threshold(
    gray, threshold2, 255, cv2.THRESH_BINARY
)


# ------------------------------------------------------------
# DISPLAY ORIGINAL, GRAYSCALE AND BINARY IMAGES
# ------------------------------------------------------------

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Segmented Input Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(binary1, cmap="gray")
plt.title(f"Binary Image (Threshold = {threshold1})")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(binary2, cmap="gray")
plt.title(f"Binary Image (Threshold = {threshold2})")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# TASK 2: EROSION
# ============================================================

# Use threshold 1 binary image for further processing
binary = binary1


# ------------------------------------------------------------
# STRUCTURING ELEMENTS
# ------------------------------------------------------------

# 3 x 3 Square
square_3 = cv2.getStructuringElement(
    cv2.MORPH_RECT, (3, 3)
)

# Cross-shaped
cross = cv2.getStructuringElement(
    cv2.MORPH_CROSS, (3, 3)
)

# 5 x 5 Square
square_5 = cv2.getStructuringElement(
    cv2.MORPH_RECT, (5, 5)
)


# ------------------------------------------------------------
# EROSION USING DIFFERENT STRUCTURING ELEMENTS
# ------------------------------------------------------------

erosion_3 = cv2.erode(binary, square_3, iterations=1)
erosion_cross = cv2.erode(binary, cross, iterations=1)
erosion_5 = cv2.erode(binary, square_5, iterations=1)


# ------------------------------------------------------------
# DISPLAY EROSION RESULTS
# ------------------------------------------------------------

plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(erosion_3, cmap="gray")
plt.title("Erosion - 3×3 Square")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(erosion_cross, cmap="gray")
plt.title("Erosion - Cross SE")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(erosion_5, cmap="gray")
plt.title("Erosion - 5×5 Square")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# TASK 3: DILATION
# ============================================================

dilation_3 = cv2.dilate(binary, square_3, iterations=1)
dilation_cross = cv2.dilate(binary, cross, iterations=1)
dilation_5 = cv2.dilate(binary, square_5, iterations=1)


# ------------------------------------------------------------
# DISPLAY DILATION RESULTS
# ------------------------------------------------------------

plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(dilation_3, cmap="gray")
plt.title("Dilation - 3×3 Square")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(dilation_cross, cmap="gray")
plt.title("Dilation - Cross SE")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(dilation_5, cmap="gray")
plt.title("Dilation - 5×5 Square")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# TASK 4: COMPARE EROSION AND DILATION
# ============================================================

plt.figure(figsize=(15, 8))

plt.subplot(1, 3, 1)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(erosion_3, cmap="gray")
plt.title("Erosion - 3×3")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(dilation_3, cmap="gray")
plt.title("Dilation - 3×3")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# TASK 5: EFFECT OF NUMBER OF ITERATIONS
# ============================================================

# ------------------------------------------------------------
# EROSION: 1, 2 AND 3 ITERATIONS
# ------------------------------------------------------------

erosion_iter1 = cv2.erode(
    binary, square_3, iterations=1
)

erosion_iter2 = cv2.erode(
    binary, square_3, iterations=2
)

erosion_iter3 = cv2.erode(
    binary, square_3, iterations=3
)


# ------------------------------------------------------------
# DILATION: 1, 2 AND 3 ITERATIONS
# ------------------------------------------------------------

dilation_iter1 = cv2.dilate(
    binary, square_3, iterations=1
)

dilation_iter2 = cv2.dilate(
    binary, square_3, iterations=2
)

dilation_iter3 = cv2.dilate(
    binary, square_3, iterations=3
)


# ------------------------------------------------------------
# DISPLAY ITERATION RESULTS
# ------------------------------------------------------------

plt.figure(figsize=(15, 10))

# Erosion
plt.subplot(2, 4, 1)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(erosion_iter1, cmap="gray")
plt.title("Erosion - 1 Iteration")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(erosion_iter2, cmap="gray")
plt.title("Erosion - 2 Iterations")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(erosion_iter3, cmap="gray")
plt.title("Erosion - 3 Iterations")
plt.axis("off")

# Dilation
plt.subplot(2, 4, 5)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(dilation_iter1, cmap="gray")
plt.title("Dilation - 1 Iteration")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(dilation_iter2, cmap="gray")
plt.title("Dilation - 2 Iterations")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(dilation_iter3, cmap="gray")
plt.title("Dilation - 3 Iterations")
plt.axis("off")

plt.tight_layout()
plt.show()


