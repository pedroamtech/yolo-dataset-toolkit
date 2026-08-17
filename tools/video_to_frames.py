import cv2
import os
import sys
import tkinter as tk
from tkinter import filedialog

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')


def extract_frames_from_video(video_path: str) -> int:
    """Extract all frames from a single video into a subfolder beside it."""
    name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
    output_dir  = os.path.join(os.path.dirname(video_path), name_no_ext)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"  [ERROR] Cannot create directory {output_dir}: {e}")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_{count:05d}.jpg"), frame)
        count += 1
        if count % 50 == 0:
            print(f"  {count}/{total if total > 0 else '?'} frames", end='\r')

    cap.release()
    print(f"  {count} frames → {output_dir}{' ' * 20}")
    return count


def pick_folder() -> str:
    """Open a folder-picker dialog and return the selected path."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the folder containing the videos")
    root.destroy()
    return folder


def process_folder(folder: str):
    # Filter out hidden files and macOS resource forks (._filename)
    videos = sorted(
        f for f in os.listdir(folder)
        if not f.startswith('.')
        and os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )

    if not videos:
        print(f"No videos found in: {folder}")
        return

    print(f"\nFolder  : {folder}")
    print(f"Videos  : {len(videos)}\n")

    total_frames = 0
    for i, name in enumerate(videos, 1):
        path = os.path.join(folder, name)
        print(f"[{i}/{len(videos)}] {name}")
        total_frames += extract_frames_from_video(path)

    print(f"\nDone. {total_frames} frames extracted from {len(videos)} videos.")


if __name__ == "__main__":
    # Accept folder as CLI argument or fall back to dialog
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        if not os.path.isdir(folder):
            print(f"Error: '{folder}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the folder containing the videos in the pop-up window...")
        folder = pick_folder()
        if not folder:
            print("No folder selected.")
            sys.exit(0)

    process_folder(folder)
