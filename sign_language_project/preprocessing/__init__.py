"""
Preprocessing package for Sign Language DIP System.
Contains fundamental image processing operations structured by tasks.
"""

from .acquisition import load_image, get_pixel_info, save_acquired_image
from .enhancement import (
    contrast_stretch,
    histogram_equalization,
    image_negative,
    log_transform,
    gamma_transform,
    calculate_psnr,
    calculate_mse
)
from .filtering import (
    mean_filter,
    gaussian_filter,
    median_filter,
    laplacian_sharpen,
    gradient_sharpen,
    high_boost_filter,
    add_salt_pepper_noise,
    add_gaussian_noise
)
from .segmentation import (
    roberts_edge,
    prewitt_edge,
    sobel_edge,
    laplacian_edge,
    global_threshold,
    otsu_threshold,
    adaptive_threshold,
    hsv_skin_segmentation,
    calculate_binary_stats
)
from .region_growing import region_growing, threshold_experiment
from .morphology import (
    apply_morphology,
    generate_iteration_series,
    clean_hand_mask
)
