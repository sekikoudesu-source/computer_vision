import cv2
import numpy as np
from scipy.optimize import least_squares
from gooey import Gooey, GooeyParser
import matplotlib.pyplot as plt
import sys
import os
import json


def pick_matching_points(img_path1, img_path2, num_pairs=8, save_path=None):
    # 1. 安全读取图像
    img1 = cv2.imdecode(np.fromfile(img_path1, dtype=np.uint8), -1)
    img2 = cv2.imdecode(np.fromfile(img_path2, dtype=np.uint8), -1)

    if img1 is None or img2 is None:
        raise ValueError(f"[Error] Failed to read images.")

    # 2. 缩放用于交互显示
    h1_orig, w1_orig = img1.shape[:2]
    h2_orig, w2_orig = img2.shape[:2]
    target_h = max(h1_orig, h2_orig)
    scale1 = target_h / h1_orig
    scale2 = target_h / h2_orig

    img1_resized = cv2.resize(img1, (int(w1_orig * scale1), target_h))
    img2_resized = cv2.resize(img2, (int(w2_orig * scale2), target_h))
    img_combined = cv2.hconcat([img1_resized, img2_resized])

    state = {
        'pts1': [], 'pts2': [],
        'w1_display': img1_resized.shape[1],
        'expected_pairs': num_pairs,
        'img_display': img_combined.copy(),
        'window_name': f"Interactive Point Picker (Target: {num_pairs} pairs)"
    }

    def mouse_callback(event, x, y, flags, param):
        s = param
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = len(s['pts2']) + 1
            if len(s['pts2']) >= s['expected_pairs']: return

            if len(s['pts1']) == len(s['pts2']):  # 点左图
                if x < s['w1_display']:
                    s['pts1'].append((x, y))
                    cv2.circle(s['img_display'], (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(s['img_display'], str(idx), (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(s['window_name'], s['img_display'])
            elif len(s['pts1']) > len(s['pts2']):  # 点右图
                if x >= s['w1_display']:
                    s['pts2'].append((x, y))
                    cv2.circle(s['img_display'], (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(s['img_display'], str(idx), (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.line(s['img_display'], s['pts1'][-1], (x, y), (255, 0, 0), 1)
                    cv2.imshow(s['window_name'], s['img_display'])

    cv2.namedWindow(state['window_name'], cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(state['window_name'], mouse_callback, param=state)
    cv2.imshow(state['window_name'], state['img_display'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 7. 坐标转换与保存逻辑 ---
    final_pts1, final_pts2 = [], []
    valid_pairs = min(len(state['pts1']), len(state['pts2']))

    # 重新拼接原始图用于保存高质量结果
    vis_h = max(h1_orig, h2_orig)
    vis_w = w1_orig + w2_orig
    vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    vis_img[:h1_orig, :w1_orig] = img1
    vis_img[:h2_orig, w1_orig:] = img2

    for i in range(valid_pairs):
        # 还原左图坐标
        x1_raw, y1_raw = state['pts1'][i][0] / scale1, state['pts1'][i][1] / scale1
        final_pts1.append([x1_raw, y1_raw])

        # 还原右图坐标
        x2_raw = (state['pts2'][i][0] - state['w1_display']) / scale2
        y2_raw = state['pts2'][i][1] / scale2
        final_pts2.append([x2_raw, y2_raw])

        # 在 vis_img 上绘制连线
        p1 = (int(x1_raw), int(y1_raw))
        p2 = (int(x2_raw) + w1_orig, int(y2_raw))
        color = (0, 255, 255)  # 黄色连线
        cv2.line(vis_img, p1, p2, color, 2)
        cv2.circle(vis_img, p1, 10, (0, 0, 255), -1)  # 左点红色
        cv2.circle(vis_img, p2, 10, (0, 255, 0), -1)  # 右点绿色
        cv2.putText(vis_img, str(i + 1), (p1[0] + 15, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"\n[+] Visual confirmation image saved to: {save_path}")

    return np.array(final_pts1, dtype=np.float32), np.array(final_pts2, dtype=np.float32)


def draw_epipolar_lines(img1_path, img2_path, F, pts1, pts2, savepath):
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

    ax1.set_title("Left Image: Selected Points")
    ax2.set_title("Right Image: Epipolar Lines")

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()


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


def check_matrix_rank_by_eigenvalues(F, threshold=1e-13):
    print("=" * 50)
    print("A-3: Verify the Rank of the F Matrix")
    print("=" * 50)

    FtF = F.T @ F
    eigenvalues, eigenvectors = np.linalg.eig(FtF)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    print("\n--- Method 1: Based on Eigenvalues of F^T * F ---")
    print("Calculated Eigenvalues (descending order):")
    for i, val in enumerate(sorted_eigenvalues):
        print(f"  Eigenvalue {i + 1}: {val:.6e}")

    rank_eig = np.sum(sorted_eigenvalues > threshold)
    print(f"  -> Inferred rank from eigenvalues: {rank_eig}")

    U, singular_values, Vt = np.linalg.svd(F)

    print("\n--- Method 2: Based on Singular Values of F ---")
    print("Calculated Singular Values (descending order):")
    for i, val in enumerate(singular_values):
        print(f"  Singular value {i + 1}: {val:.6e}")

    rank_svd = np.sum(singular_values > threshold)
    print(f"  -> Inferred rank from singular values: {rank_svd}")
    print("\nConclusion:")

    if rank_svd == 2:
        print("✅ Verification Successful: The 3rd eigenvalue/singular value is close to 0.")
    else:
        print(f"❌ Warning: The rank of the matrix is {rank_svd}.")


def build_omega_star(f, cu, cv):
    K = np.array([
        [f, 0, cu],
        [0, f, cv],
        [0, 0, 1]
    ], dtype=float)
    return K @ K.T


def kruppa_equations(F, w, h):
    print("\n" + "=" * 50)
    print(" B-7: Intrinsic Parameters Estimation (Kruppa)")
    print("=" * 50)

    U, S, Vt = np.linalg.svd(F)
    s1, s2 = S[0], S[1]
    V = Vt.T

    def residuals_f(params):
        f1, f2 = params
        cu, cv = w / 2.0, h / 2.0

        w1_star = build_omega_star(f1, cu, cv)
        w2_star = build_omega_star(f2, cu, cv)

        A = V.T @ w1_star @ V
        B = U.T @ w2_star @ U

        a11, a12, a22 = A[0, 0], A[0, 1], A[1, 1]
        b11, b12, b22 = B[0, 0], B[0, 1], B[1, 1]

        e1 = (s1 ** 2) * a11 * b12 + s1 * s2 * a12 * b22
        e2 = s1 * s2 * a12 * b11 + (s2 ** 2) * a22 * b12
        e3 = (s1 ** 2) * a11 * b11 - (s2 ** 2) * a22 * b22

        scale = max(abs(e1), abs(e2), abs(e3)) + 1e-10
        return [e1 / scale, e2 / scale, e3 / scale]

    f_init = max(w, h)
    res_f = least_squares(residuals_f, [f_init, f_init], bounds=(1.0, np.inf))
    f1_est, f2_est = res_f.x

    print("[B-7-i] Estimated Focal Lengths:")
    print(f"  -> f1 = {f1_est:.2f} px, f2 = {f2_est:.2f} px")

    f_avg = (f1_est + f2_est) / 2.0

    def residuals_c(params):
        cu, cv = params
        w1_star = build_omega_star(f_avg, cu, cv)
        w2_star = build_omega_star(f_avg, cu, cv)

        A = V.T @ w1_star @ V
        B = U.T @ w2_star @ U

        a11, a12, a22 = A[0, 0], A[0, 1], A[1, 1]
        b11, b12, b22 = B[0, 0], B[0, 1], B[1, 1]

        e1 = (s1 ** 2) * a11 * b12 + s1 * s2 * a12 * b22
        e2 = s1 * s2 * a12 * b11 + (s2 ** 2) * a22 * b12
        e3 = (s1 ** 2) * a11 * b11 - (s2 ** 2) * a22 * b22

        scale = max(abs(e1), abs(e2), abs(e3)) + 1e-10
        return [e1 / scale, e2 / scale, e3 / scale]

    res_c = least_squares(residuals_c, [w / 2.0, h / 2.0], bounds=(0, max(w, h)))
    cu_est, cv_est = res_c.x

    print("\n[B-7-ii] Estimated Image Center:")
    print(f"  -> (cu, cv) = ({cu_est:.2f}, {cv_est:.2f})")

    return f1_est, f2_est, cu_est, cv_est


def verify_parameters(f1, f2, cu, cv, w, h):
    print("\n" + "=" * 50)
    print(" B-8: Verification of Estimated Parameters")
    print("=" * 50)

    f_ratio = min(f1, f2) / max(f1, f2)
    print("1. Focal Length Check:")
    print(f"   Ratio (min/max): {f_ratio:.3f} (Ideal: ~1.0)")
    if f_ratio > 0.85:
        print("   [SUCCESS] f1 and f2 are approximately equal.")
    else:
        print("   [WARNING] Significant variance in estimated focal lengths.")

    true_cu, true_cv = w / 2.0, h / 2.0
    dist = np.sqrt((cu - true_cu) ** 2 + (cv - true_cv) ** 2)
    diag = np.sqrt(w ** 2 + h ** 2)
    dist_percent = (dist / diag) * 100

    print(f"\n2. Image Center Check:")
    print(f"   Distance from True Center: {dist:.2f} px ({dist_percent:.2f}% of diagonal)")
    if dist_percent < 15.0:
        print("   [SUCCESS] The estimated center is reasonably close to the image center.")
    else:
        print("   [WARNING] The center has drifted too far. (Typical for noisy F matrices).")


@Gooey(
    program_name="8-Point Algorithm - Epipolar Line Tool",
    language='english',
    default_size=(700, 780),
    header_bg_color='#f0f0f0'
)
def main():
    parser = GooeyParser(
        description="Select left/right images and an output folder to compute F and draw epipolar lines.")

    file_group = parser.add_argument_group("Image Configuration")
    file_group.add_argument(
        'IMAGE_LEFT',
        metavar='Left Image (Image 1)',
        widget='FileChooser',
        help="Select the left view image"
    )
    file_group.add_argument(
        'IMAGE_RIGHT',
        metavar='Right Image (Image 2)',
        widget='FileChooser',
        help="Select the right view image"
    )
    file_group.add_argument(
        '--output_dir',
        metavar='Output Directory (Save Path)',
        widget='DirChooser',
        default=os.getcwd(),
        help="Select the folder where result images will be saved"
    )
    file_group.add_argument(
        '--load_points',
        metavar='Load Saved Points (Optional)',
        widget='FileChooser',
        default="",
        help="[Optional] Select a previously saved .json file to skip manual point picking."
    )

    action_group = parser.add_argument_group("Additional Features")

    # [新增] 选取点数量输入
    action_group.add_argument(
        '--num_points',
        metavar='Number of Points to Pick',
        type=int,
        default=8,
        help="Specify the number of matching points to select (default is 8)."
    )

    action_group.add_argument(
        '--verify_extra',
        metavar='Enable Extra Point Verification (A-5)',
        help="Pick 3 NEW pairs after initial computation to verify Epipolar Lines.",
        action='store_true'
    )
    action_group.add_argument(
        '--run_kruppa', metavar='Run Kruppa Intrinsics Estimation (B-7 & B-8)',
        help="Estimate camera focal length and image center using Kruppa equations.",
        action='store_true',
        default=True
    )
    args = parser.parse_args()

    try:
        out_dir = args.output_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        img_temp = cv2.imdecode(np.fromfile(args.IMAGE_LEFT, dtype=np.uint8), -1)
        if img_temp is None:
            raise ValueError("Could not read left image. Please check path.")
        img_height, img_width = img_temp.shape[:2]

        print(f">>> Processing started...\nOutput Directory: {out_dir}")
        sys.stdout.flush()

        points_file = os.path.join(out_dir, "saved_points.json")

        if args.load_points and os.path.isfile(args.load_points):
            print(f">>> Loading points from: {args.load_points}")
            with open(args.load_points, 'r') as f:
                data = json.load(f)
                pts_left = np.array(data['pts_left'], dtype=np.float32)
                pts_right = np.array(data['pts_right'], dtype=np.float32)
            print(f">>> Successfully loaded {len(pts_left)} point pairs.")
        else:
            print(f">>> No points file provided. Starting manual point selection for {args.num_points} pairs...")
            matches_save_path = os.path.join(out_dir, "point_matches.png")
            # [修改] 传入 args.num_points
            pts_left, pts_right = pick_matching_points(args.IMAGE_LEFT, args.IMAGE_RIGHT, num_pairs=args.num_points,
                                                       save_path=matches_save_path)

            with open(points_file, 'w') as f:
                json.dump({'pts_left': pts_left.tolist(), 'pts_right': pts_right.tolist()}, f)
            print(f"[+] Matching points saved to JSON: {points_file}")

        F_matrix = compute_fundamental_matrix_8pt(pts_left, pts_right)

        print("\n>>> 8-Point Algorithm output F:")
        np.set_printoptions(precision=6, suppress=True)
        print(F_matrix)
        check_matrix_rank_by_eigenvalues(F_matrix)
        sys.stdout.flush()

        epipolar_save_path = os.path.join(out_dir, "epipolar_lines.png")
        draw_epipolar_lines(args.IMAGE_LEFT, args.IMAGE_RIGHT, F_matrix, pts_left, pts_right, epipolar_save_path)
        print(f"\n[+] Epipolar lines saved to: {epipolar_save_path}")

        if args.verify_extra:
            print("\n" + "=" * 50)
            print(" PHASE 2: A-5 Extra Point Verification")
            print("=" * 50)
            print(">>> Please select 3 completely NEW point pairs in the image window.")
            sys.stdout.flush()

            extra_pts_left, extra_pts_right = pick_matching_points(args.IMAGE_LEFT, args.IMAGE_RIGHT, num_pairs=3,
                                                                   save_path=None)

            print("\n>>> Drawing epipolar lines for the EXTRA points...")
            sys.stdout.flush()

            verify_save_path = os.path.join(out_dir, "epipolar_lines_verify.png")
            draw_epipolar_lines(args.IMAGE_LEFT, args.IMAGE_RIGHT, F_matrix, extra_pts_left, extra_pts_right,
                                verify_save_path)
            print(f"\n[+] Extra verification lines saved to: {verify_save_path}")

        if args.run_kruppa:
            f1, f2, cu, cv = kruppa_equations(F_matrix, img_width, img_height)
            verify_parameters(f1, f2, cu, cv, img_width, img_height)
            sys.stdout.flush()

        print("\n>>> All processing complete!")
    except Exception as e:
        print(f"\n[!] Error: {e}")


if __name__ == "__main__":
    main()