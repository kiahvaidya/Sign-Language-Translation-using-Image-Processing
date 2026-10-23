"""
Task 7: Region-Based Processing Module
Implements seeded Region Growing segmentation, 4-way and 8-way neighborhood expansion,
seed selection, and multi-threshold experiments.
"""

import cv2
import numpy as np
import time
from collections import deque


def region_growing(image_gray, seed_point, threshold=25, connectivity=4):
    """
    Seeded Region Growing algorithm.
    Recursively/iteratively aggregates connected pixels whose intensity
    is within `threshold` of the running region mean.

    Args:
        image_gray (np.ndarray): 2D grayscale image.
        seed_point (tuple): (x, y) coordinates of the starting seed.
        threshold (float): Homogeneity criterion tolerance |pixel - mean| <= threshold.
        connectivity (int): 4 or 8 neighbor connectivity.

    Returns:
        np.ndarray: Binary mask (255 for grown region, 0 elsewhere).
    """
    h, w = image_gray.shape[:2]
    sx, sy = int(seed_point[0]), int(seed_point[1])

    if not (0 <= sx < w and 0 <= sy < h):
        raise ValueError(f"Seed point ({sx}, {sy}) is outside image boundaries ({w}x{h}).")

    region = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)

    queue = deque()
    queue.append((sy, sx))
    visited[sy, sx] = True
    region[sy, sx] = 255

    region_sum = float(image_gray[sy, sx])
    region_count = 1

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    # Limit maximum iterations to prevent infinite freeze on very large flat images
    max_pixels = h * w

    while queue and region_count < max_pixels:
        y, x = queue.popleft()
        region_mean = region_sum / region_count

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx]:
                    visited[ny, nx] = True
                    pixel_val = float(image_gray[ny, nx])

                    if abs(pixel_val - region_mean) <= threshold:
                        region[ny, nx] = 255
                        queue.append((ny, nx))
                        region_sum += pixel_val
                        region_count += 1

    return region


def threshold_experiment(image_gray, seed_point, thresholds=(10, 25, 50), connectivity=4):
    """
    Runs region growing across multiple threshold levels to demonstrate
    under-segmentation vs over-segmentation.

    Returns:
        list of dict: [{'threshold': T, 'mask': mask, 'time_ms': ms, 'area': area, 'fg_pct': pct}, ...]
    """
    results = []
    total_pixels = image_gray.size

    for t in thresholds:
        start_t = time.perf_counter()
        mask = region_growing(image_gray, seed_point, threshold=t, connectivity=connectivity)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        area = int(np.count_nonzero(mask == 255))
        fg_pct = float(area * 100.0 / total_pixels) if total_pixels > 0 else 0.0

        results.append({
            "threshold": t,
            "mask": mask,
            "time_ms": round(elapsed_ms, 2),
            "area": area,
            "foreground_ratio": round(fg_pct, 2)
        })

    return results


def draw_seed_overlay(image, seed_point, color=(0, 255, 0), radius=5):
    """
    Draws a crosshair and circle over the seed point for clear visual feedback.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    sx, sy = int(seed_point[0]), int(seed_point[1])
    cv2.circle(vis, (sx, sy), radius, color, -1)
    cv2.line(vis, (sx - 10, sy), (sx + 10, sy), (0, 0, 255), 2)
    cv2.line(vis, (sx, sy - 10), (sx, sy + 10), (0, 0, 255), 2)
    return vis
