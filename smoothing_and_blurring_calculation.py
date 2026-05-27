import cv2
import numpy as np


def add_label(image, text):
    labeled_img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 10, labeled_img.shape[0] - 20
    cv2.rectangle(labeled_img, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), 0, -1)
    cv2.putText(labeled_img, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)

    return labeled_img

# 1. Read a grayscale image
image_path = 'flog_original.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not read image from {image_path}. Please check the path.")
    exit()

# Define a function to normalize box kernels
def get_box_kernel(size):
    kernel = np.ones((size, size), np.float32) / (size * size)
    return kernel

# --- Task 2: Normalized Box Filters ---
# Applying a 3x3 box filter
box_3X3= cv2.filter2D(img, -1, get_box_kernel(3))
box_3X3=add_label(box_3X3, "3x3")
# Applying a 5x5 box filter
box_5x5 = cv2.filter2D(img, -1, get_box_kernel(5))
box_5x5=add_label(box_5x5, "5x5")
# --- Task 3: Gaussian Filter ---
# Using standard parameters: sigma=1.0, and letting the kernel be derived.
gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0, sigmaY=1.0) # (0,0) allows auto kernel calc
gaussian=add_label(gaussian, "Gaussian")
# --- Task 4: Bilateral Filter ---
# Using typical parameters for smoothing while preserving edges.
# The values are d=9 (diameter of pixel neighborhood),
# sigmaColor=50 (filter sigma in color space),
# sigmaSpace=50 (filter sigma in coordinate space).
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50)
bilateral=add_label(bilateral, "Bilateral")
img=add_label(img, "Original Image")
# --- Task 5: Display the four images side by side ---
# Horizontal stack
# Original, Box, Gaussian, Bilateral
combined_imgs = np.hstack((img, box_3X3,box_5x5, gaussian, bilateral))

# Display the result locally
cv2.imshow('Smoothing Comparison (Original, box_3X3, Box 5x5, Gaussian, Bilateral)', combined_imgs)
cv2.imwrite('smoothing_comparison.png', combined_imgs)
cv2.waitKey(0) # Wait for a key press to close
cv2.destroyAllWindows()