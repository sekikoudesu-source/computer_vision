import cv2
import os


def process_video(video_path, output_name):
    # Check if the file exists
    if not os.path.exists(video_path):
        print(
            f"Error: File not found {video_path}. Please ensure the video is in the same directory as the code, or provide an absolute path.")
        return

    print(f"Processing: {video_path}")

    # 2. Read the video using OpenCV function
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # 3. Extract single frame from the video
    # Read the first frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Unable to read frame from {video_path}")
        cap.release()
        return

    # 4. Show the captured image using imshow
    window_name_orig = f"Original Frame - {output_name}"
    cv2.imshow(window_name_orig, frame)
    print("Press any key to continue...")
    cv2.waitKey(0)  # Wait for user keystroke

    # 5. Extract feature points from the captured image and show it
    # Convert the image to grayscale (feature extraction is typically done on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect feature points (Keypoints)
    keypoints = orb.detect(gray_frame, None)

    # Draw feature points on the original image (green circles)
    frame_with_features = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

    window_name_feat = f"Extracted Features - {output_name}"
    cv2.imshow(window_name_feat, frame_with_features)
    print("Press any key to save and process the next one (or exit)...")
    cv2.waitKey(0)  # Wait for user keystroke

    # 6. Save the captured image and the image with the extracted feature points
    orig_filename = f"{output_name}_original.jpg"
    feat_filename = f"{output_name}_features.jpg"

    cv2.imwrite(orig_filename, frame)
    cv2.imwrite(feat_filename, frame_with_features)

    print(f"Saved: {orig_filename}")
    print(f"Saved: {feat_filename}\n")

    # Release resources and close the windows for this video
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Please ensure these two video files are in the same folder as your Python script
    # If they are in a different folder, change this to the absolute path of the video (e.g., 'C:/Downloads/flog.m1v')
    videos = [
        {"path": "flog.m1v", "name": "flog"},
        {"path": "navisys.wmv", "name": "navisys"}
    ]

    for v in videos:
        process_video(v["path"], v["name"])

    print("All videos processed successfully!")