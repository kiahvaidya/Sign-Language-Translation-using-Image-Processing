import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# SIGN LANGUAGE IMAGE PROCESSING
# WEEK 4 + WEEK 5
#
# IMPORTANT:
# Every individual operation uses ORIGINAL IMAGE as input.
# Processed images do NOT replace the original image.
#
# Task 5 is the only intentional sequence:
# Original -> Enhancement -> Median -> High-Boost
# ==========================================================


# ==========================================================
# DISPLAY FUNCTION
# ==========================================================

def display_comparison(original, result, title):

    plt.figure(figsize=(12, 6))

    # ORIGINAL INPUT
    plt.subplot(1, 2, 1)

    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap="gray")

    plt.title("Original Input Image")
    plt.axis("off")

    # PROCESSED OUTPUT
    plt.subplot(1, 2, 2)

    if len(result.shape) == 3:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(result, cmap="gray")

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================================
# WEEK 4 - IMAGE ENHANCEMENT
# ==========================================================


# ----------------------------------------------------------
# IMAGE NEGATIVE
# ----------------------------------------------------------

def image_negative(img):

    return 255 - img


# ----------------------------------------------------------
# LOG TRANSFORMATION
# ----------------------------------------------------------

def log_transform(img, c=1):

    img_float = img.astype(np.float32)

    log_img = c * np.log1p(img_float)

    log_img = cv2.normalize(
        log_img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return log_img.astype(np.uint8)


# ----------------------------------------------------------
# GAMMA TRANSFORMATION
# ----------------------------------------------------------

def gamma_transform(img, gamma=1.0):

    normalized = img / 255.0

    gamma_img = np.power(
        normalized,
        gamma
    )

    return np.uint8(
        gamma_img * 255
    )


# ----------------------------------------------------------
# CONTRAST STRETCHING
# ----------------------------------------------------------

def contrast_stretch(img):

    min_val = np.min(img)
    max_val = np.max(img)

    if max_val == min_val:
        return img.copy()

    stretched = (
        (img - min_val)
        * 255
        / (max_val - min_val)
    )

    return stretched.astype(np.uint8)


# ----------------------------------------------------------
# HISTOGRAM
# ----------------------------------------------------------

def show_histogram(img):

    plt.figure(figsize=(8, 5))

    if len(img.shape) == 2:

        plt.hist(
            img.ravel(),
            256,
            [0, 256]
        )

    else:

        colors = ("b", "g", "r")

        for i, color in enumerate(colors):

            histogram = cv2.calcHist(
                [img],
                [i],
                None,
                [256],
                [0, 256]
            )

            plt.plot(
                histogram,
                color=color
            )

    plt.title("Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])

    plt.show()


# ----------------------------------------------------------
# HISTOGRAM EQUALIZATION
# ----------------------------------------------------------

def histogram_equalization(img):

    if len(img.shape) == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = img

    return cv2.equalizeHist(gray)


# ==========================================================
# WEEK 5 - NOISE REDUCTION FILTERS
# ==========================================================


# ----------------------------------------------------------
# MEAN FILTER
# ----------------------------------------------------------

def mean_filter(img):

    return cv2.blur(
        img,
        (5, 5)
    )


# ----------------------------------------------------------
# MEDIAN FILTER
# ----------------------------------------------------------

def median_filter(img):

    return cv2.medianBlur(
        img,
        5
    )


# ==========================================================
# WEEK 5 - SHARPENING FILTERS
# ==========================================================


# ----------------------------------------------------------
# LAPLACIAN SHARPENING
# ----------------------------------------------------------

def laplacian_sharpening(img):

    if len(img.shape) == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = img.copy()

    gray = gray.astype(np.float32)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=3
    )

    sharpened = gray - laplacian

    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened.astype(np.uint8)


# ----------------------------------------------------------
# GRADIENT-BASED SHARPENING
# ----------------------------------------------------------

def gradient_sharpening(img):

    if len(img.shape) == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = img.copy()

    gray = gray.astype(np.float32)

    # Horizontal gradient
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    # Vertical gradient
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Gradient magnitude
    magnitude = cv2.magnitude(
        gx,
        gy
    )

    # Normalize gradient
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Sharpen
    sharpened = (
        gray
        + 0.5 * magnitude
    )

    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened.astype(np.uint8)


# ----------------------------------------------------------
# HIGH-BOOST FILTERING
# ----------------------------------------------------------

def high_boost_filter(img, A=1.5):

    if len(img.shape) == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = img.copy()

    gray = gray.astype(np.float32)

    # Smooth image
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # High-frequency mask
    mask = gray - blurred

    # High-boost equation
    high_boost = (
        A * gray
        + mask
    )

    high_boost = np.clip(
        high_boost,
        0,
        255
    )

    return high_boost.astype(np.uint8)


# ==========================================================
# TASK 2 - FILTER COMPARISON
# ==========================================================

def compare_filters(original):

    mean_result = mean_filter(original)

    median_result = median_filter(original)

    laplacian_result = laplacian_sharpening(
        original
    )

    gradient_result = gradient_sharpening(
        original
    )

    highboost_result = high_boost_filter(
        original,
        1.5
    )

    results = [
        mean_result,
        median_result,
        laplacian_result,
        gradient_result,
        highboost_result
    ]

    titles = [
        "Mean Filter",
        "Median Filter",
        "Laplacian",
        "Gradient Sharpening",
        "High-Boost"
    ]

    # ------------------------------------------------------
    # DISPLAY ALL RESULTS
    # ------------------------------------------------------

    plt.figure(figsize=(15, 8))

    # ORIGINAL
    plt.subplot(2, 3, 1)

    plt.imshow(
        cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Original Input")
    plt.axis("off")

    # FILTER RESULTS
    for i in range(len(results)):

        result = results[i]

        plt.subplot(
            2,
            3,
            i + 2
        )

        if len(result.shape) == 3:

            plt.imshow(
                cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                )
            )

        else:

            plt.imshow(
                result,
                cmap="gray"
            )

        plt.title(
            titles[i]
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------
    # MSE + PSNR
    # ------------------------------------------------------

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    print("\n==============================================")
    print("FILTER COMPARISON - MSE AND PSNR")
    print("==============================================")

    print(
        f"{'Filter':<25}"
        f"{'MSE':<15}"
        f"{'PSNR':<15}"
    )

    print("-" * 55)

    for title, result in zip(
        titles,
        results
    ):

        if len(result.shape) == 3:

            result_gray = cv2.cvtColor(
                result,
                cv2.COLOR_BGR2GRAY
            )

        else:

            result_gray = result

        mse = np.mean(
            (
                original_gray.astype(
                    np.float32
                )
                -
                result_gray.astype(
                    np.float32
                )
            ) ** 2
        )

        if mse == 0:

            psnr = float("inf")

        else:

            psnr = 10 * np.log10(
                (255 ** 2) / mse
            )

        print(
            f"{title:<25}"
            f"{mse:<15.2f}"
            f"{psnr:<15.2f}"
        )

    print("\nNote:")
    print(
        "MSE/PSNR show difference from the original image."
    )
    print(
        "Visual quality should also be considered."
    )


# ==========================================================
# TASK 3 - SELECT BEST FILTER
# ==========================================================

def select_best_filter(original):

    print("\n==============================================")
    print("TASK 3 - BEST FILTER SELECTION")
    print("==============================================")

    print("\nSelected filter sequence:")
    print("Median Filtering + High-Boost Sharpening")

    print("\nReason:")
    print(
        "Median filtering reduces unwanted noise "
        "while preserving hand boundaries."
    )

    print(
        "High-Boost sharpening enhances hand edges "
        "and fine finger details."
    )

    print(
        "\nThis combination is suitable for "
        "sign-language image processing."
    )

    # Show the selected filters against original
    median_result = median_filter(
        original
    )

    highboost_result = high_boost_filter(
        original,
        1.5
    )

    plt.figure(figsize=(15, 5))

    # ORIGINAL
    plt.subplot(1, 3, 1)

    plt.imshow(
        cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Original Input")
    plt.axis("off")

    # MEDIAN
    plt.subplot(1, 3, 2)

    plt.imshow(
        cv2.cvtColor(
            median_result,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Selected: Median")
    plt.axis("off")

    # HIGH BOOST
    plt.subplot(1, 3, 3)

    plt.imshow(
        highboost_result,
        cmap="gray"
    )

    plt.title("Selected: High-Boost")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================================
# TASK 5 - ENHANCEMENT + FILTERING
# ==========================================================

def enhancement_and_filtering(original):

    # ------------------------------------------------------
    # STEP 1 - ENHANCEMENT
    # ------------------------------------------------------

    enhanced = contrast_stretch(
        original
    )

    # ------------------------------------------------------
    # STEP 2 - NOISE REDUCTION
    # ------------------------------------------------------

    filtered = median_filter(
        enhanced
    )

    # ------------------------------------------------------
    # STEP 3 - SHARPENING
    # ------------------------------------------------------

    final = high_boost_filter(
        filtered,
        1.5
    )

    # ------------------------------------------------------
    # DISPLAY COMPLETE PIPELINE
    # ------------------------------------------------------

    plt.figure(figsize=(16, 5))

    # ORIGINAL
    plt.subplot(1, 4, 1)

    plt.imshow(
        cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Original Input")
    plt.axis("off")

    # ENHANCED
    plt.subplot(1, 4, 2)

    plt.imshow(
        cv2.cvtColor(
            enhanced,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Enhanced Image")
    plt.axis("off")

    # MEDIAN
    plt.subplot(1, 4, 3)

    plt.imshow(
        cv2.cvtColor(
            filtered,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Median Filter")
    plt.axis("off")

    # FINAL
    plt.subplot(1, 4, 4)

    plt.imshow(
        final,
        cmap="gray"
    )

    plt.title("Final Improved Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return final


# ==========================================================
# IMAGE ADDITION
# ==========================================================

def image_addition(img1, img2):

    img2 = cv2.resize(
        img2,
        (img1.shape[1], img1.shape[0])
    )

    return cv2.add(
        img1,
        img2
    )


# ==========================================================
# IMAGE SUBTRACTION
# ==========================================================

def image_subtraction(img1, img2):

    img2 = cv2.resize(
        img2,
        (img1.shape[1], img1.shape[0])
    )

    return cv2.absdiff(
        img1,
        img2
    )


# ==========================================================
# IMAGE AVERAGING
# ==========================================================

def image_averaging(img1, img2):

    img2 = cv2.resize(
        img2,
        (img1.shape[1], img1.shape[0])
    )

    return cv2.addWeighted(
        img1,
        0.5,
        img2,
        0.5,
        0
    )


# ==========================================================
# SAVE IMAGE
# ==========================================================

def save_image(img):

    filename = input(
        "Enter filename to save: "
    )

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        filename += ".jpg"

    cv2.imwrite(
        filename,
        img
    )

    print(
        "Image saved as:",
        filename
    )


# ==========================================================
# MAIN MENU
# ==========================================================

def main():

    print("\n==============================================")
    print("       SIGN LANGUAGE IMAGE PROCESSING")
    print("       ENHANCEMENT + FILTERING")
    print("==============================================")

    path = input(
        "Enter image path: "
    )

    # ------------------------------------------------------
    # LOAD ORIGINAL IMAGE ONCE
    # ------------------------------------------------------

    original_image = cv2.imread(
        path
    )

    if original_image is None:

        print(
            "Error: Image could not be loaded."
        )

        return

    print("\nImage loaded successfully.")

    while True:

        print("\n==============================================")
        print("                    MENU")
        print("==============================================")

        print("\n--- IMAGE ENHANCEMENT ---")
        print("1. Image Negative")
        print("2. Log Transformation")
        print("3. Gamma Transformation")
        print("4. Contrast Stretching")
        print("5. Histogram")
        print("6. Histogram Equalization")

        print("\n--- IMAGE OPERATIONS ---")
        print("7. Image Addition")
        print("8. Image Subtraction")
        print("9. Image Averaging")

        print("\n--- FILTERING ---")
        print("10. Mean Filter")
        print("11. Median Filter")
        print("12. Laplacian Sharpening")
        print("13. Gradient-Based Sharpening")
        print("14. High-Boost Filtering")
        print("15. Compare All Filters")
        print("16. Select Best Filter")
        print("17. Enhancement + Filtering")
      

        print("\n--- OTHER ---")
        print("18. Save Final Improved Image")
        print("19. Reset / Show Original")
        print("0. Exit")

        choice = input(
            "\nEnter your choice: "
        )


        # ==================================================
        # 1. IMAGE NEGATIVE
        # INPUT = ORIGINAL
        # ==================================================

        if choice == "1":

            result = image_negative(
                original_image
            )

            display_comparison(
                original_image,
                result,
                "Image Negative"
            )


        # ==================================================
        # 2. LOG
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "2":

            c = float(
                input(
                    "Enter log constant c: "
                )
            )

            result = log_transform(
                original_image,
                c
            )

            display_comparison(
                original_image,
                result,
                "Log Transformation"
            )


        # ==================================================
        # 3. GAMMA
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "3":

            gamma = float(
                input(
                    "Enter gamma value "
                    "(<1 brighter, >1 darker): "
                )
            )

            result = gamma_transform(
                original_image,
                gamma
            )

            display_comparison(
                original_image,
                result,
                "Gamma Transformation"
            )


        # ==================================================
        # 4. CONTRAST STRETCH
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "4":

            result = contrast_stretch(
                original_image
            )

            display_comparison(
                original_image,
                result,
                "Contrast Stretching"
            )


        # ==================================================
        # 5. HISTOGRAM
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "5":

            show_histogram(
                original_image
            )


        # ==================================================
        # 6. HISTOGRAM EQUALIZATION
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "6":

            result = histogram_equalization(
                original_image
            )

            display_comparison(
                original_image,
                result,
                "Histogram Equalization"
            )


        # ==================================================
        # 7. IMAGE ADDITION
        # ORIGINAL USED AS FIRST INPUT
        # ==================================================

        elif choice == "7":

            path2 = input(
                "Enter second image path: "
            )

            img2 = cv2.imread(
                path2
            )

            if img2 is None:

                print(
                    "Could not load second image."
                )

                continue

            result = image_addition(
                original_image,
                img2
            )

            display_comparison(
                original_image,
                result,
                "Image Addition"
            )


        # ==================================================
        # 8. IMAGE SUBTRACTION
        # ORIGINAL USED AS FIRST INPUT
        # ==================================================

        elif choice == "8":

            path2 = input(
                "Enter second image path: "
            )

            img2 = cv2.imread(
                path2
            )

            if img2 is None:

                print(
                    "Could not load second image."
                )

                continue

            result = image_subtraction(
                original_image,
                img2
            )

            display_comparison(
                original_image,
                result,
                "Image Subtraction"
            )


        # ==================================================
        # 9. IMAGE AVERAGING
        # ORIGINAL USED AS FIRST INPUT
        # ==================================================

        elif choice == "9":

            path2 = input(
                "Enter second image path: "
            )

            img2 = cv2.imread(
                path2
            )

            if img2 is None:

                print(
                    "Could not load second image."
                )

                continue

            result = image_averaging(
                original_image,
                img2
            )

            display_comparison(
                original_image,
                result,
                "Image Averaging"
            )


        # ==================================================
        # 10. MEAN FILTER
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "10":

            result = mean_filter(
                original_image
            )

            display_comparison(
                original_image,
                result,
                "Mean Filter"
            )


        # ==================================================
        # 11. MEDIAN FILTER
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "11":

            result = median_filter(
                original_image
            )

            display_comparison(
                original_image,
                result,
                "Median Filter"
            )


        # ==================================================
        # 12. LAPLACIAN
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "12":

            result = laplacian_sharpening(
                original_image
            )

            display_comparison(
                original_image,
                result,
                "Laplacian Sharpening"
            )


        # ==================================================
        # 13. GRADIENT
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "13":

            result = gradient_sharpening(
                original_image
            )

            display_comparison(
                original_image,
                result,
                "Gradient-Based Sharpening"
            )


        # ==================================================
        # 14. HIGH-BOOST
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "14":

            A = float(
                input(
                    "Enter high-boost value A "
                    "(recommended 1.5): "
                )
            )

            result = high_boost_filter(
                original_image,
                A
            )

            display_comparison(
                original_image,
                result,
                "High-Boost Filtering"
            )


        # ==================================================
        # 15. TASK 2 - COMPARE
        # ALL FILTERS USE ORIGINAL
        # ==================================================

        elif choice == "15":

            compare_filters(
                original_image
            )


        # ==================================================
        # 16. TASK 3 - BEST FILTER
        # INPUT = ORIGINAL
        # ==================================================

        elif choice == "16":

            select_best_filter(
                original_image
            )


        # ==================================================
        # 17. TASK 5
        #
        # ONLY HERE DO WE INTENTIONALLY CHAIN
        # ENHANCEMENT -> FILTERING
        # ==================================================

        elif choice == "17":

            final_image = enhancement_and_filtering(
                original_image
            )

            print("\n======================================")
            print("TASK 5 COMPLETED")
            print("======================================")

            print(
                "Original -> Contrast Enhancement "
                "-> Median Filtering -> High-Boost"
            )


        # ==================================================
        # 18. SAVE FINAL IMAGE
        # ==================================================

        elif choice == "18":

            final_image = enhancement_and_filtering(
                original_image
            )

            save_image(
                final_image
            )


        # ==================================================
        # 19. RESET / SHOW ORIGINAL
        # ==================================================

        elif choice == "19":

            plt.figure(figsize=(7, 5))

            plt.imshow(
                cv2.cvtColor(
                    original_image,
                    cv2.COLOR_BGR2RGB
                )
            )

            plt.title(
                "Original Input Image"
            )

            plt.axis("off")
            plt.tight_layout()
            plt.show()

            print(
                "Original image remains unchanged."
            )


        # ==================================================
        # EXIT
        # ==================================================

        elif choice == "0":

            print(
                "\nExiting program..."
            )

            break


        else:

            print(
                "\nInvalid choice. Please try again."
            )


# ==========================================================
# RUN PROGRAM
# ==========================================================

if __name__ == "__main__":
    main()