import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


# ==========================================================
# NOISE FUNCTIONS
# ==========================================================

def add_salt_pepper_noise(image, amount=0.03):
    noisy = image.copy()

    height, width = image.shape[:2]
    number_of_pixels = int(amount * height * width)

    # Salt noise - white pixels
    y = np.random.randint(0, height, number_of_pixels)
    x = np.random.randint(0, width, number_of_pixels)
    noisy[y, x] = 255

    # Pepper noise - black pixels
    y = np.random.randint(0, height, number_of_pixels)
    x = np.random.randint(0, width, number_of_pixels)
    noisy[y, x] = 0

    return noisy


def add_gaussian_noise(image, mean=0, sigma=20):
    image_float = image.astype(np.float32)

    noise = np.random.normal(
        mean,
        sigma,
        image.shape
    )

    noisy = image_float + noise
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


# ==========================================================
# SMOOTHING FILTERS
# ==========================================================

def mean_filter(image, kernel_size):
    return cv2.blur(
        image,
        (kernel_size, kernel_size)
    )


def gaussian_filter(image, kernel_size):
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


def median_filter(image, kernel_size):
    return cv2.medianBlur(
        image,
        kernel_size
    )


# ==========================================================
# SHARPENING FILTERS
# ==========================================================

def laplacian_sharpen(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    sharpened = cv2.addWeighted(
        gray,
        1.5,
        laplacian,
        -0.5,
        0
    )

    return cv2.cvtColor(
        sharpened,
        cv2.COLOR_GRAY2BGR
    )


def gradient_sharpen(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        sobel_x.astype(np.float32),
        sobel_y.astype(np.float32)
    )

    magnitude = cv2.convertScaleAbs(
        magnitude
    )

    sharpened = cv2.addWeighted(
        gray,
        1.0,
        magnitude,
        0.7,
        0
    )

    return cv2.cvtColor(
        sharpened,
        cv2.COLOR_GRAY2BGR
    )


def high_boost_filter(image, A=1.5):

    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    mask = cv2.subtract(
        image,
        blurred
    )

    high_boost = cv2.addWeighted(
        image,
        A,
        mask,
        1,
        0
    )

    return high_boost


# ==========================================================
# MSE AND PSNR
# ==========================================================

def calculate_mse(original, processed):

    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    mse = np.mean(
        (original - processed) ** 2
    )

    return mse


def calculate_psnr(original, processed):

    mse = calculate_mse(
        original,
        processed
    )

    if mse == 0:
        return float("inf")

    return 10 * np.log10(
        (255 ** 2) / mse
    )


# ==========================================================
# MAIN GUI
# ==========================================================

class FilteringApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Sign Language - Spatial Domain Filtering"
        )

        self.root.geometry(
            "1250x800"
        )

        self.root.configure(
            bg="#f2e7e8"
        )

        self.original = None
        self.processed = None

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        title = tk.Label(
            root,
            text="Spatial Domain Filtering Module",
            font=("Arial", 24, "bold"),
            bg="#f2e7e8",
            fg="black"
        )

        title.pack(
            pady=(20, 15)
        )

        # --------------------------------------------------
        # CONTROL FRAME
        # --------------------------------------------------

        control_frame = tk.Frame(
            root,
            bg="#f2e7e8"
        )

        control_frame.pack(
            pady=10
        )

        # Load button
        load_button = tk.Button(
            control_frame,
            text="Load Image",
            command=self.load_image,
            width=16,
            height=2
        )

        load_button.grid(
            row=0,
            column=0,
            padx=8
        )

        # Filter dropdown
        self.filter_var = tk.StringVar()

        filters = [
            "Salt & Pepper Noise",
            "Gaussian Noise",
            "Mean 3x3",
            "Mean 5x5",
            "Gaussian 3x3",
            "Gaussian 5x5",
            "Median 3x3",
            "Median 5x5",
            "Laplacian",
            "Gradient Sharpening",
            "High-Boost"
        ]

        self.filter_var.set(
            filters[0]
        )

        filter_menu = tk.OptionMenu(
            control_frame,
            self.filter_var,
            *filters
        )

        filter_menu.config(
            width=22,
            height=2
        )

        filter_menu.grid(
            row=0,
            column=1,
            padx=8
        )

        # Apply button
        apply_button = tk.Button(
            control_frame,
            text="Apply Filter",
            command=self.apply_filter,
            width=16,
            height=2
        )

        apply_button.grid(
            row=0,
            column=2,
            padx=8
        )

        # Save button
        save_button = tk.Button(
            control_frame,
            text="Save Result",
            command=self.save_result,
            width=16,
            height=2
        )

        save_button.grid(
            row=0,
            column=3,
            padx=8
        )

        # --------------------------------------------------
        # IMAGE DISPLAY FRAME
        # --------------------------------------------------

        image_frame = tk.Frame(
            root,
            bg="#f2e7e8"
        )

        image_frame.pack(
            pady=20
        )

        # Original heading
        original_heading = tk.Label(
            image_frame,
            text="Original Image",
            font=("Arial", 16, "bold"),
            bg="#f2e7e8"
        )

        original_heading.grid(
            row=0,
            column=0,
            pady=(0, 8)
        )

        # Processed heading
        processed_heading = tk.Label(
            image_frame,
            text="Processed Image",
            font=("Arial", 16, "bold"),
            bg="#f2e7e8"
        )

        processed_heading.grid(
            row=0,
            column=1,
            pady=(0, 8)
        )

        # Original image display
        self.original_label = tk.Label(
            image_frame,
            text="Load an image",
            bg="white",
            width=600,
            height=500,
            relief="solid",
            borderwidth=1
        )

        self.original_label.grid(
            row=1,
            column=0,
            padx=15
        )

        # Processed image display
        self.processed_label = tk.Label(
            image_frame,
            text="Processed image will appear here",
            bg="white",
            width=600,
            height=500,
            relief="solid",
            borderwidth=1
        )

        self.processed_label.grid(
            row=1,
            column=1,
            padx=15
        )

        # --------------------------------------------------
        # RESULT INFORMATION
        # --------------------------------------------------

        self.result_label = tk.Label(
            root,
            text="MSE: -     PSNR: -",
            font=("Arial", 15, "bold"),
            bg="#f2e7e8"
        )

        self.result_label.pack(
            pady=10
        )

    # ======================================================
    # LOAD IMAGE
    # ======================================================

    def load_image(self):

        path = filedialog.askopenfilename(
            title="Select Sign Language Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                )
            ]
        )

        if not path:
            return

        self.original = cv2.imread(
            path
        )

        if self.original is None:

            messagebox.showerror(
                "Error",
                "Unable to load the selected image."
            )

            return

        # Display original
        self.show_image(
            self.original,
            self.original_label
        )

        # Reset processed image
        self.processed = None

        self.processed_label.config(
            image="",
            text="Processed image will appear here"
        )

        self.processed_label.image = None

        self.result_label.config(
            text="MSE: -     PSNR: -"
        )

    # ======================================================
    # APPLY SELECTED FILTER
    # ======================================================

    def apply_filter(self):

        if self.original is None:

            messagebox.showwarning(
                "No Image",
                "Please load an image first."
            )

            return

        selected = self.filter_var.get()

        # ----------------------------------------------
        # NOISE
        # ----------------------------------------------

        if selected == "Salt & Pepper Noise":

            result = add_salt_pepper_noise(
                self.original
            )

        elif selected == "Gaussian Noise":

            result = add_gaussian_noise(
                self.original
            )

        # ----------------------------------------------
        # MEAN
        # ----------------------------------------------

        elif selected == "Mean 3x3":

            result = mean_filter(
                self.original,
                3
            )

        elif selected == "Mean 5x5":

            result = mean_filter(
                self.original,
                5
            )

        # ----------------------------------------------
        # GAUSSIAN
        # ----------------------------------------------

        elif selected == "Gaussian 3x3":

            result = gaussian_filter(
                self.original,
                3
            )

        elif selected == "Gaussian 5x5":

            result = gaussian_filter(
                self.original,
                5
            )

        # ----------------------------------------------
        # MEDIAN
        # ----------------------------------------------

        elif selected == "Median 3x3":

            result = median_filter(
                self.original,
                3
            )

        elif selected == "Median 5x5":

            result = median_filter(
                self.original,
                5
            )

        # ----------------------------------------------
        # LAPLACIAN
        # ----------------------------------------------

        elif selected == "Laplacian":

            result = laplacian_sharpen(
                self.original
            )

        # ----------------------------------------------
        # GRADIENT
        # ----------------------------------------------

        elif selected == "Gradient Sharpening":

            result = gradient_sharpen(
                self.original
            )

        # ----------------------------------------------
        # HIGH BOOST
        # ----------------------------------------------

        elif selected == "High-Boost":

            result = high_boost_filter(
                self.original
            )

        else:

            return

        # Store result
        self.processed = result

        # Display result
        self.show_image(
            result,
            self.processed_label
        )

        # Calculate metrics
        mse = calculate_mse(
            self.original,
            result
        )

        psnr = calculate_psnr(
            self.original,
            result
        )

        if np.isinf(psnr):

            psnr_text = "∞"

        else:

            psnr_text = f"{psnr:.2f} dB"

        self.result_label.config(
            text=f"MSE: {mse:.2f}     PSNR: {psnr_text}"
        )

    # ======================================================
    # DISPLAY IMAGE
    # ======================================================

    def show_image(self, image, label):

        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            rgb
        )

        # ----------------------------------------------
        # Resize while preserving aspect ratio
        # ----------------------------------------------

        max_width = 560
        max_height = 450

        width, height = pil_image.size

        scale = min(
            max_width / width,
            max_height / height
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        pil_image = pil_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        # Convert to Tkinter image
        photo = ImageTk.PhotoImage(
            pil_image
        )

        # Display
        label.config(
            image=photo,
            text=""
        )

        # IMPORTANT:
        # Keep a reference so image doesn't disappear
        label.image = photo

    # ======================================================
    # SAVE RESULT
    # ======================================================

    def save_result(self):

        if self.processed is None:

            messagebox.showwarning(
                "No Result",
                "Please apply a filter first."
            )

            return

        path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Image", "*.jpg"),
                ("PNG Image", "*.png")
            ]
        )

        if not path:
            return

        cv2.imwrite(
            path,
            self.processed
        )

        messagebox.showinfo(
            "Saved",
            "Processed image saved successfully."
        )


# ==========================================================
# START APPLICATION
# ==========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = FilteringApp(
        root
    )

    root.mainloop()