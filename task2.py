import cv2
from gooey import Gooey, GooeyParser

@Gooey(
    program_name="task2",      # 窗口标题
    default_size=(800, 600),                  # 初始窗口大小
    language='english',                       # 界面语言
    header_height=80,                         # 顶部横幅高度
    navigation='sidebar',                     # 导航栏模式（sidebar 或 tabbed）
    show_sidebar=False,                        # 是否显示侧边栏
    body_bg_color='#f3f3f3',                  # 主体背景颜色
    terminal_panel_color='#ffffff',           # 终端输出框颜色
)
def main():
    parser = GooeyParser(description="Feature matching comparison script.")
    parser.add_argument("num_points", type=int, nargs='?', default=20, help="Number of corresponding points to draw.",)
    parser.add_argument("--img1", default="1.jpg", help="Path to first image.",widget="FileChooser")
    parser.add_argument("--img2", default="2.jpg", help="Path to second image.",widget="FileChooser")
    args = parser.parse_args()

    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)

    if img1 is None or img2 is None:
        print(f"Error: Could not load images. Check paths: {args.img1}, {args.img2}")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    methods = [
        ("SIFT", cv2.SIFT_create(), cv2.NORM_L2),
        ("ORB", cv2.ORB_create(), cv2.NORM_HAMMING),
        ("AKAZE", cv2.AKAZE_create(), cv2.NORM_HAMMING)
    ]

    for name, detector, norm_type in methods:
        print(f"Processing using {name}...")

        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)


        bf = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)


        n_matches = min(len(matches), args.num_points)
        final_matches = matches[:n_matches]

        res_img = cv2.drawMatches(img1, kp1, img2, kp2, final_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        window_name = f"{name} Matching (Top {n_matches} points)"
        cv2.imshow(window_name, res_img)
        cv2.imwrite(f"{name}.jpg", res_img)
        print(f"{name}: Found {len(matches)} total matches. Drawing {n_matches}.")

    print("\nPress any key in an image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()