import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_test_image(filename="test_rect.jpg"):
    """Helper: Creates a test image with a grid and a distinct rectangular box."""
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200
    # Draw a grid for visual reference
    for i in range(0, 400, 50):
        cv2.line(img, (i, 0), (i, 400), (150, 150, 150), 1)
        cv2.line(img, (0, i), (400, i), (150, 150, 150), 1)

    # Draw a distinct black rectangular box
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), -1)

    # Add colored corners to the box for easy tracking
    cv2.circle(img, (100, 100), 10, (255, 0, 0), -1)  # Top-Left (Blue)
    cv2.circle(img, (300, 100), 10, (0, 255, 0), -1)  # Top-Right (Green)
    cv2.circle(img, (300, 300), 10, (0, 0, 255), -1)  # Bottom-Right (Red)
    cv2.circle(img, (100, 300), 10, (0, 255, 255), -1)  # Bottom-Left (Yellow)

    cv2.imwrite(filename, img)
    print(f"Created test image: {filename}")


def compute_homography(src_pts, dst_pts):
    """
    Computes the 3x3 homography matrix using Direct Linear Transformation (DLT).
    Does NOT use OpenCV.
    """
    A = []
    for i in range(4):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]

        # Create the two rows for this point correspondence
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    A = np.array(A)

    # Solve Ah = 0 using SVD
    _, _, Vh = np.linalg.svd(A)

    # The solution is the last row of Vh (which corresponds to the smallest singular value)
    L = Vh[-1, :]

    # Reshape into 3x3 matrix
    H = L.reshape(3, 3)

    # Normalize the matrix so H[2,2] = 1
    H = H / H[2, 2]
    return H


def warp_image_manual(img, H, out_shape):
    """
    Applies homography transformation to an image using inverse mapping.
    Does NOT use OpenCV.
    """
    h_out, w_out = out_shape[:2]
    H_inv = np.linalg.inv(H)

    # 1. Create a grid of all (x, y) coordinates in the output image
    y_out, x_out = np.indices((h_out, w_out))
    ones = np.ones_like(x_out)

    # 2. Stack into homogeneous coordinates: shape (3, N)
    coords_out = np.stack([x_out.flatten(), y_out.flatten(), ones.flatten()])

    # 3. Transform output coordinates back to input image space
    coords_in = H_inv @ coords_out

    # 4. Normalize by the 3rd coordinate (z) to convert from homogeneous back to 2D
    coords_in = coords_in / coords_in[2, :]

    # Extract nearest neighbor pixel indices
    x_in = np.round(coords_in[0, :]).astype(int)
    y_in = np.round(coords_in[1, :]).astype(int)

    # Create empty output image
    out_img = np.zeros((h_out, w_out, img.shape[2]), dtype=img.dtype)

    # 5. Filter out coordinates that map outside the original image boundaries
    h_in, w_in = img.shape[:2]
    valid_mask = (x_in >= 0) & (x_in < w_in) & (y_in >= 0) & (y_in < h_in)

    # 6. Map the valid pixels (Nearest Neighbor interpolation)
    valid_y_out = y_out.flatten()[valid_mask]
    valid_x_out = x_out.flatten()[valid_mask]
    valid_y_in = y_in[valid_mask]
    valid_x_in = x_in[valid_mask]

    out_img[valid_y_out, valid_x_out] = img[valid_y_in, valid_x_in]

    return out_img


def main():
    # 1. Prepare and read the image
    img_path = "test_rect.jpg"
    create_test_image(img_path)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # For correct matplotlib display

    # 2. Define 4 points (Original corners of the inner box)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    src_pts = np.array([
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300]
    ])

    # --- TRANSFORMATION 1: Perspective Warp (Simulating viewing the box from an angle) ---
    dst_pts_1 = np.array([
        [150, 150],  # Top-left moves in and down
        [350, 100],  # Top-right moves out
        [250, 350],  # Bottom-right moves in and down
        [50, 300]  # Bottom-left moves out
    ])

    # 3. Compute Homography Matrix
    H1 = compute_homography(src_pts, dst_pts_1)
    print("Homography Matrix 1:\n", H1)

    # 4. Transform Image
    out_img_1 = warp_image_manual(img_rgb, H1, (400, 400))

    # --- TRANSFORMATION 2: Shearing / Affine stretch ---
    dst_pts_2 = np.array([
        [100, 100],
        [400, 100],  # Stretch right side
        [350, 300],  # Shear bottom right
        [150, 300]  # Shear bottom left
    ])

    H2 = compute_homography(src_pts, dst_pts_2)
    print("\nHomography Matrix 2:\n", H2)
    out_img_2 = warp_image_manual(img_rgb, H2, (400, 500))  # Made output wider to fit

    # --- DISPLAY RESULTS ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    # Plot original points
    for pt in src_pts:
        plt.plot(pt[0], pt[1], 'ro', markersize=8)

    plt.subplot(1, 3, 2)
    plt.title("Transformation 1 (Perspective)")
    plt.imshow(out_img_1)
    for pt in dst_pts_1:
        plt.plot(pt[0], pt[1], 'go', markersize=8)

    plt.subplot(1, 3, 3)
    plt.title("Transformation 2 (Stretch/Shear)")
    plt.imshow(out_img_2)
    for pt in dst_pts_2:
        plt.plot(pt[0], pt[1], 'bo', markersize=8)

    plt.tight_layout()
    plt.savefig("test_result.png")
    plt.show()



if __name__ == "__main__":
    main()