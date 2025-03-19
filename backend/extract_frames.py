import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    count = 0
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f'frame_{count}.jpg')
            cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    print(f"Frames extracted for {video_path} and saved in {output_folder}")

# Example usage (Run this script separately)
extract_frames("videos/inside_out_2015.mkv", "dataset/inside_out")

