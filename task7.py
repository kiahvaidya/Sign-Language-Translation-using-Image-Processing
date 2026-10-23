# ============================================================
# DIGITAL IMAGE PROCESSING - TASK 7
# REGION GROWING, REGION SPLITTING AND REGION MERGING
# VS CODE VERSION
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque


# ============================================================
# TASK 1 - INPUT IMAGE
# ============================================================

print("=" * 65)
print("DIGITAL IMAGE PROCESSING - TASK 7")
print("REGION GROWING, SPLITTING AND MERGING")
print("=" * 65)

print("\nEnter the path of the ORIGINAL image.")
print("Example: images/original.jpg")
print("The original image must be used, not a filtered image.\n")

image_path = input("Enter image path: ").strip()

# Read image
original = cv2.imread(image_path)

if original is None:
    raise FileNotFoundError(
        "\nERROR: Image could not be loaded.\n"
        "Check that the path and filename are correct."
    )

# Convert BGR to RGB
original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

# Convert ORIGINAL image directly to grayscale
gray = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2GRAY
)

height, width = gray.shape


# ============================================================
# IMAGE INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("INPUT IMAGE INFORMATION")
print("=" * 65)

print("Image dimensions :", width, "x", height)
print("Minimum intensity:", gray.min())
print("Maximum intensity:", gray.max())
print("Mean intensity   :", round(gray.mean(), 2))


# ============================================================
# DISPLAY ORIGINAL AND GRAYSCALE
# ============================================================

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title(
    "Original Input Image",
    fontsize=14,
    fontweight="bold"
)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title(
    "Grayscale Image",
    fontsize=14,
    fontweight="bold"
)
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# SELECT SEED POINT
# ============================================================

print("\n" + "=" * 65)
print("SEED POINT SELECTION")
print("=" * 65)

print(
    "\nYou can enter a seed point manually."
    "\nPress ENTER to automatically select the centre."
)

x_input = input(
    f"Enter X coordinate (0 - {width - 1}) [centre]: "
).strip()

y_input = input(
    f"Enter Y coordinate (0 - {height - 1}) [centre]: "
).strip()


if x_input == "" or y_input == "":
    seed_x = width // 2
    seed_y = height // 2
else:
    seed_x = int(x_input)
    seed_y = int(y_input)


# Validate seed
if not (
    0 <= seed_x < width
    and
    0 <= seed_y < height
):
    raise ValueError(
        "Seed point is outside the image."
    )


print("\nSelected Seed Point")
print("X =", seed_x)
print("Y =", seed_y)
print(
    "Seed intensity =",
    gray[seed_y, seed_x]
)


# ============================================================
# DISPLAY SEED
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    gray,
    cmap="gray"
)

plt.scatter(
    seed_x,
    seed_y,
    s=120,
    marker="x"
)

plt.title(
    f"Selected Seed Point ({seed_x}, {seed_y})",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")
plt.show()


# ============================================================
# REGION GROWING FUNCTION
# ============================================================

def region_growing(
    image,
    seed,
    threshold,
    connectivity=4
):

    h, w = image.shape

    sx, sy = seed

    # Region mask
    region = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    # Visited pixels
    visited = np.zeros(
        (h, w),
        dtype=bool
    )

    # Queue
    queue = deque()

    # Add seed
    queue.append((sy, sx))

    visited[sy, sx] = True
    region[sy, sx] = 255

    # Initial region statistics
    region_sum = float(
        image[sy, sx]
    )

    region_count = 1

    # --------------------------------------------------------
    # 4-CONNECTED NEIGHBOURS
    # --------------------------------------------------------

    neighbours_4 = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    # --------------------------------------------------------
    # 8-CONNECTED NEIGHBOURS
    # --------------------------------------------------------

    neighbours_8 = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1)
    ]

    if connectivity == 4:
        neighbours = neighbours_4
    else:
        neighbours = neighbours_8

    # --------------------------------------------------------
    # REGION GROWING
    # --------------------------------------------------------

    while queue:

        y, x = queue.popleft()

        # Update region mean
        region_mean = (
            region_sum /
            region_count
        )

        for dy, dx in neighbours:

            ny = y + dy
            nx = x + dx

            # Check image boundaries
            if (
                ny < 0
                or ny >= h
                or nx < 0
                or nx >= w
            ):
                continue

            # Skip visited pixels
            if visited[ny, nx]:
                continue

            visited[ny, nx] = True

            pixel_value = float(
                image[ny, nx]
            )

            # Similarity criterion
            if abs(
                pixel_value -
                region_mean
            ) <= threshold:

                region[ny, nx] = 255

                queue.append(
                    (ny, nx)
                )

                # Update region statistics
                region_sum += pixel_value
                region_count += 1

    return region


# ============================================================
# THRESHOLD EXPERIMENT
# ============================================================

print("\n" + "=" * 65)
print("THRESHOLD EXPERIMENT")
print("=" * 65)

print("\nThree threshold values will be tested:")
print("Low    = 10")
print("Medium = 25")
print("High   = 50")

thresholds = [10, 25, 50]

threshold_results = []
threshold_times = []


for T in thresholds:

    start_time = time.perf_counter()

    mask = region_growing(
        gray,
        (seed_x, seed_y),
        T,
        connectivity=4
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    threshold_results.append(mask)
    threshold_times.append(elapsed)

    pixels = np.sum(mask > 0)

    percentage = (
        pixels /
        (height * width)
    ) * 100

    print(
        f"\nThreshold: {T}"
        f"\nRegion pixels: {pixels}"
        f"\nRegion percentage: {percentage:.2f}%"
        f"\nProcessing time: {elapsed:.6f} seconds"
    )


# ============================================================
# DISPLAY THRESHOLD RESULTS
# ============================================================

plt.figure(figsize=(16, 5))

for i in range(3):

    plt.subplot(
        1,
        3,
        i + 1
    )

    plt.imshow(
        threshold_results[i],
        cmap="gray"
    )

    plt.title(
        f"Threshold = {thresholds[i]}",
        fontsize=14,
        fontweight="bold"
    )

    plt.axis("off")

plt.suptitle(
    "Region Growing with Different Thresholds",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# ============================================================
# FINAL SEGMENTED REGION
# ============================================================

# Medium threshold
final_mask = threshold_results[1]

# Apply mask to ORIGINAL image
segmented = cv2.bitwise_and(
    original_rgb,
    original_rgb,
    mask=final_mask
)


# ============================================================
# DISPLAY FINAL SEGMENTATION
# ============================================================

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)

plt.imshow(
    original_rgb
)

plt.title(
    "Original Image",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    final_mask,
    cmap="gray"
)

plt.title(
    "Region Growing Mask\nThreshold = 25",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    segmented
)

plt.title(
    "Final Segmented Region",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# TASK 2
# 4-CONNECTED VS 8-CONNECTED
# ============================================================

print("\n" + "=" * 65)
print("TASK 2: 4-CONNECTED VS 8-CONNECTED")
print("=" * 65)

comparison_threshold = 25


# ------------------------------------------------------------
# 4-CONNECTED
# ------------------------------------------------------------

start = time.perf_counter()

region_4 = region_growing(
    gray,
    (seed_x, seed_y),
    comparison_threshold,
    connectivity=4
)

time_4 = (
    time.perf_counter()
    - start
)


# ------------------------------------------------------------
# 8-CONNECTED
# ------------------------------------------------------------

start = time.perf_counter()

region_8 = region_growing(
    gray,
    (seed_x, seed_y),
    comparison_threshold,
    connectivity=8
)

time_8 = (
    time.perf_counter()
    - start
)


pixels_4 = np.sum(
    region_4 > 0
)

pixels_8 = np.sum(
    region_8 > 0
)


print("\nThreshold =", comparison_threshold)

print("\n4-CONNECTED")
print("Region pixels =", pixels_4)
print(
    "Processing time =",
    round(time_4, 6),
    "seconds"
)

print("\n8-CONNECTED")
print("Region pixels =", pixels_8)
print(
    "Processing time =",
    round(time_8, 6),
    "seconds"
)


# ============================================================
# DISPLAY CONNECTIVITY COMPARISON
# ============================================================

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)

plt.imshow(
    gray,
    cmap="gray"
)

plt.scatter(
    seed_x,
    seed_y,
    s=100,
    marker="x"
)

plt.title(
    "Grayscale + Seed",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    region_4,
    cmap="gray"
)

plt.title(
    "4-Connected Region",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    region_8,
    cmap="gray"
)

plt.title(
    "8-Connected Region",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# TASK 3 - REGION SPLITTING
# ============================================================

print("\n" + "=" * 65)
print("TASK 3: REGION SPLITTING")
print("=" * 65)


def is_homogeneous(
    region,
    threshold
):

    if region.size == 0:
        return True

    intensity_range = (
        int(region.max())
        -
        int(region.min())
    )

    return (
        intensity_range <= threshold
    )


def region_splitting(
    image,
    threshold=30,
    min_size=16
):

    h, w = image.shape

    output = np.zeros_like(
        image
    )

    def split(
        x,
        y,
        region_width,
        region_height
    ):

        current = image[
            y:y + region_height,
            x:x + region_width
        ]

        # ----------------------------------------------------
        # Homogeneity test
        # ----------------------------------------------------

        if is_homogeneous(
            current,
            threshold
        ):

            mean_value = int(
                np.mean(current)
            )

            output[
                y:y + region_height,
                x:x + region_width
            ] = mean_value

            return

        # ----------------------------------------------------
        # Minimum size condition
        # ----------------------------------------------------

        if (
            region_width <= min_size
            or
            region_height <= min_size
        ):

            mean_value = int(
                np.mean(current)
            )

            output[
                y:y + region_height,
                x:x + region_width
            ] = mean_value

            return

        # ----------------------------------------------------
        # Split into four regions
        # ----------------------------------------------------

        half_width = (
            region_width // 2
        )

        half_height = (
            region_height // 2
        )

        # Top-left
        split(
            x,
            y,
            half_width,
            half_height
        )

        # Top-right
        split(
            x + half_width,
            y,
            region_width - half_width,
            half_height
        )

        # Bottom-left
        split(
            x,
            y + half_height,
            half_width,
            region_height - half_height
        )

        # Bottom-right
        split(
            x + half_width,
            y + half_height,
            region_width - half_width,
            region_height - half_height
        )

    split(
        0,
        0,
        w,
        h
    )

    return output


split_threshold = 30

split_result = region_splitting(
    gray,
    threshold=split_threshold,
    min_size=16
)


# ============================================================
# DISPLAY REGION SPLITTING
# ============================================================

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)

plt.imshow(
    gray,
    cmap="gray"
)

plt.title(
    "Original Grayscale Image",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    split_result,
    cmap="gray"
)

plt.title(
    "Region Splitting Result",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# REGION MERGING
# ============================================================

print("\n" + "=" * 65)
print("REGION MERGING")
print("=" * 65)


def region_merging(
    image,
    block_size=16,
    merge_threshold=20
):

    h, w = image.shape

    rows = (
        h + block_size - 1
    ) // block_size

    cols = (
        w + block_size - 1
    ) // block_size

    # --------------------------------------------------------
    # Calculate block means
    # --------------------------------------------------------

    means = np.zeros(
        (rows, cols),
        dtype=float
    )

    for r in range(rows):

        for c in range(cols):

            y1 = (
                r * block_size
            )

            y2 = min(
                (r + 1) * block_size,
                h
            )

            x1 = (
                c * block_size
            )

            x2 = min(
                (c + 1) * block_size,
                w
            )

            means[r, c] = np.mean(
                image[
                    y1:y2,
                    x1:x2
                ]
            )

    # --------------------------------------------------------
    # Union-Find structure
    # --------------------------------------------------------

    parent = np.arange(
        rows * cols
    )

    def find(x):

        while parent[x] != x:

            parent[x] = parent[
                parent[x]
            ]

            x = parent[x]

        return x

    def union(a, b):

        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:

            parent[root_b] = root_a

    # --------------------------------------------------------
    # Merge adjacent similar regions
    # --------------------------------------------------------

    for r in range(rows):

        for c in range(cols):

            current = (
                r * cols + c
            )

            # Right neighbour
            if c + 1 < cols:

                right = (
                    r * cols
                    +
                    c + 1
                )

                difference = abs(
                    means[r, c]
                    -
                    means[r, c + 1]
                )

                if (
                    difference
                    <= merge_threshold
                ):

                    union(
                        current,
                        right
                    )

            # Bottom neighbour
            if r + 1 < rows:

                bottom = (
                    (r + 1) * cols
                    +
                    c
                )

                difference = abs(
                    means[r, c]
                    -
                    means[r + 1, c]
                )

                if (
                    difference
                    <= merge_threshold
                ):

                    union(
                        current,
                        bottom
                    )

    # --------------------------------------------------------
    # Create merged image
    # --------------------------------------------------------

    merged = np.zeros_like(
        image
    )

    region_values = {}

    for r in range(rows):

        for c in range(cols):

            index = (
                r * cols + c
            )

            root = find(index)

            if root not in region_values:
                region_values[root] = []

            region_values[root].append(
                means[r, c]
            )

    # Assign merged values
    for r in range(rows):

        for c in range(cols):

            index = (
                r * cols + c
            )

            root = find(index)

            mean_value = int(
                np.mean(
                    region_values[root]
                )
            )

            y1 = (
                r * block_size
            )

            y2 = min(
                (r + 1) * block_size,
                h
            )

            x1 = (
                c * block_size
            )

            x2 = min(
                (c + 1) * block_size,
                w
            )

            merged[
                y1:y2,
                x1:x2
            ] = mean_value

    return merged


merge_threshold = 20

merge_result = region_merging(
    gray,
    block_size=16,
    merge_threshold=merge_threshold
)


# ============================================================
# DISPLAY REGION MERGING
# ============================================================

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)

plt.imshow(
    gray,
    cmap="gray"
)

plt.title(
    "Original Grayscale Image",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    merge_result,
    cmap="gray"
)

plt.title(
    "Region Merging Result",
    fontsize=14,
    fontweight="bold"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)

print("\nREGION GROWING")

print(
    f"{'Threshold':<15}"
    f"{'Pixels':<15}"
    f"{'Region %':<15}"
    f"{'Time (sec)':<15}"
)

print("-" * 60)

for T, mask, t in zip(
    thresholds,
    threshold_results,
    threshold_times
):

    pixels = np.sum(
        mask > 0
    )

    percentage = (
        pixels /
        (height * width)
    ) * 100

    print(
        f"{T:<15}"
        f"{pixels:<15}"
        f"{percentage:.2f}%{'':<10}"
        f"{t:.6f}"
    )


print("\n4-CONNECTED VS 8-CONNECTED")

print(
    f"{'Neighbourhood':<20}"
    f"{'Pixels':<15}"
    f"{'Time (sec)':<15}"
)

print("-" * 50)

print(
    f"{'4-connected':<20}"
    f"{pixels_4:<15}"
    f"{time_4:.6f}"
)

print(
    f"{'8-connected':<20}"
    f"{pixels_8:<15}"
    f"{time_8:.6f}"
)


print("\n" + "=" * 65)
print("TASK 7 COMPLETED SUCCESSFULLY")
print("=" * 65)