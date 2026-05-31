import cv2
import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt

def normalize_points(pts):
    centroid = np.mean(pts, axis=0)
    shifted_pts = pts - centroid
    distances = np.sqrt(np.sum(shifted_pts ** 2, axis=1))
    mean_distance = np.mean(distances)
    scale = np.sqrt(2) / mean_distance

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    pts_homo = np.column_stack((pts, np.ones(len(pts))))
    normalized_pts_homo = (T @ pts_homo.T).T
    return normalized_pts_homo[:, :2], T

def compute_fundamental_matrix_8pt(pts1, pts2):
    assert len(pts1) == len(pts2)
    assert len(pts1) >= 8

    N = len(pts1)
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)

    A = np.zeros((N, 9))
    for i in range(N):
        u1, v1 = norm_pts1[i]
        u2, v2 = norm_pts2[i]
        A[i] = [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1]

    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1]
    F_est = f.reshape(3, 3)

    U_F, S_F, Vt_F = np.linalg.svd(F_est)
    S_F[2] = 0
    F_norm = U_F @ np.diag(S_F) @ Vt_F

    F = T2.T @ F_norm @ T1
    if F[2, 2] != 0:
        F = F / F[2, 2]

    return F

def draw_epipolar_lines(img1_path, img2_path, F, pts1, pts2, savepath, title="Epipolar Lines"):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    h, w = img2.shape[:2]

    colors = plt.cm.get_cmap('hsv', len(pts1))
    pts1_homo = np.column_stack((pts1, np.ones(len(pts1))))
    lines = (F @ pts1_homo.T).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i, (r, p1, p2) in enumerate(zip(lines, pts1, pts2)):
        color = colors(i)[:3]

        ax1.scatter(p1[0], p1[1], color=color, s=100, edgecolors='white', label=f'Pt {i}')
        ax1.text(p1[0], p1[1], str(i), color='white', fontsize=12)

        a, b, c = r
        if abs(b) > abs(a):
            x0, x1 = 0, w
            y0 = -(a * x0 + c) / b
            y1 = -(a * x1 + c) / b
        else:
            y0, y1 = 0, h
            x0 = -(b * y0 + c) / a
            x1 = -(b * y1 + c) / a

        ax2.plot([x0, x1], [y0, y1], color=color, linewidth=2)

        ax2.scatter(p2[0], p2[1], color=color, s=100, edgecolors='white')
        ax2.text(p2[0], p2[1], str(i), color='white', fontsize=12)

    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)

    ax1.set_title(f"Left Image: Selected Points ({title})")
    ax2.set_title(f"Right Image: Epipolar Lines ({title})")

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    img1_path = os.path.join("..", "secen1", "Scene1left.jpg")
    img2_path = os.path.join("..", "secen1", "Scene1right.jpg")
    points_file = os.path.join("..", "secen1", "saved_points.json")

    with open(points_file, 'r') as f:
        data = json.load(f)
        pts_left = np.array(data['pts_left'], dtype=np.float32)
        pts_right = np.array(data['pts_right'], dtype=np.float32)

    print(f"Loaded {len(pts_left)} point pairs.")

    # 1. 8-Point Algorithm
    start_8pt = time.time()
    F_8pt = compute_fundamental_matrix_8pt(pts_left, pts_right)
    time_8pt = time.time() - start_8pt
    print("\n--- 8-Point Algorithm ---")
    print(f"Time taken: {time_8pt:.6f} seconds")
    print("Fundamental Matrix F:")
    print(F_8pt)
    
    # 2. 5-Point Algorithm
    # For 5-point algorithm, we need the camera matrix K. 
    # The assignment states we can use arbitrary values (f=1, center=(0,0))
    K = np.eye(3, dtype=np.float64)
    start_5pt = time.time()
    
    # findEssentialMat requires at least 5 points. cv2.RANSAC or LMEDS can be used.
    # Here we just use the points.
    E, mask = cv2.findEssentialMat(pts_left, pts_right, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is not None and E.shape == (3, 3):
        # Compute F from E: F = K^-T * E * K^-1
        K_inv = np.linalg.inv(K)
        F_5pt = K_inv.T @ E @ K_inv
        
        # Normalize F to have F[2,2] = 1 for better comparison
        if F_5pt[2, 2] != 0:
            F_5pt = F_5pt / F_5pt[2, 2]
    else:
        # If multiple matrices returned, take the first 3x3 block
        E_first = E[0:3, 0:3]
        K_inv = np.linalg.inv(K)
        F_5pt = K_inv.T @ E_first @ K_inv
        if F_5pt[2, 2] != 0:
            F_5pt = F_5pt / F_5pt[2, 2]
            
    time_5pt = time.time() - start_5pt
    inliers = np.sum(mask) if mask is not None else 0
    print("\n--- 5-Point Algorithm ---")
    print(f"Time taken: {time_5pt:.6f} seconds")
    print(f"Inliers: {inliers} / {len(pts_left)}")
    print("Essential Matrix E (and Fundamental Matrix F since K=I):")
    print(F_5pt)

    # 3. Draw Epipolar Lines
    draw_epipolar_lines(img1_path, img2_path, F_8pt, pts_left, pts_right, "epipolar_lines_8pt.png", "8-Point Algorithm")
    print("\n[+] Epipolar lines for 8-point saved to: epipolar_lines_8pt.png")
    
    draw_epipolar_lines(img1_path, img2_path, F_5pt, pts_left, pts_right, "epipolar_lines_5pt.png", "5-Point Algorithm")
    print("[+] Epipolar lines for 5-point saved to: epipolar_lines_5pt.png")

if __name__ == "__main__":
    main()
