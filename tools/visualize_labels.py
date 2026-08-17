"""
visualize_labels.py — Interactive YOLO bounding-box viewer

Draws YOLO annotations on each image and lets you navigate with the keyboard.

Controls:
    →  /  D  /  Space   : next image
    ←  /  A              : previous image
    Q  /  Esc            : quit

Usage:
    python tools/visualize_labels.py                        # folder dialog
    python tools/visualize_labels.py path/to/dataset        # CLI path
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')

# One color per class (BGR). Cycles if more classes than colors.
PALETTE = [
    (0,   255,  57),   # class 0 — green   (person)
    (255,  57,  57),   # class 1 — red
    (57,  138, 255),   # class 2 — blue
    (255, 200,   0),   # class 3 — yellow
    (200,   0, 255),   # class 4 — purple
    (0,   220, 255),   # class 5 — cyan
    (255, 128,   0),   # class 6 — orange
]

# Optional: map class IDs to readable names
CLASS_NAMES: dict[int, str] = {
    0: 'person',
}

# Max display size — image is downscaled to fit while keeping aspect ratio
MAX_W, MAX_H = 1280, 720


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the dataset folder")
    root.destroy()
    return folder


def draw_boxes(img: np.ndarray, label_path: str) -> tuple[np.ndarray, int]:
    """Draw YOLO bounding boxes on img. Returns (annotated_img, box_count)."""
    canvas = img.copy()
    h, w   = canvas.shape[:2]
    count  = 0

    if not os.path.isfile(label_path):
        return canvas, 0

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls  = int(parts[0])
            xc   = float(parts[1]) * w
            yc   = float(parts[2]) * h
            bw   = float(parts[3]) * w
            bh   = float(parts[4]) * h
            x1   = int(xc - bw / 2)
            y1   = int(yc - bh / 2)
            x2   = int(xc + bw / 2)
            y2   = int(yc + bh / 2)

            color = PALETTE[cls % len(PALETTE)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label = CLASS_NAMES.get(cls, str(cls))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            count += 1

    return canvas, count


def fit_to_screen(img: np.ndarray) -> np.ndarray:
    """Downscale image to fit MAX_W × MAX_H while preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def overlay_info(img: np.ndarray, text: str) -> np.ndarray:
    """Burn info text into the top-left corner."""
    for i, line in enumerate(text.split('\n')):
        y = 22 + i * 22
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _resolve_images_labels(dataset_dir: str) -> tuple[str, str]:
    """
    Return (images_dir, labels_dir) from the selected folder.

    Accepts three layouts:
      1. dataset/images/  + dataset/labels/       → select dataset/
      2. dataset/train/images/ + dataset/train/labels/  → select dataset/ or dataset/train/
      3. Any subfolder that directly contains images/   → auto-detected
    """
    direct = os.path.join(dataset_dir, 'images')
    if os.path.isdir(direct):
        return direct, os.path.join(dataset_dir, 'labels')

    # Look one level deeper for a partition subfolder (train, val, test, …)
    for sub in sorted(os.listdir(dataset_dir)):
        sub_path = os.path.join(dataset_dir, sub)
        if os.path.isdir(sub_path) and os.path.isdir(os.path.join(sub_path, 'images')):
            print(f"[INFO] Auto-detected partition: {sub}/")
            return (os.path.join(sub_path, 'images'),
                    os.path.join(sub_path, 'labels'))

    return direct, os.path.join(dataset_dir, 'labels')   # fallback — error reported below


def run_viewer(dataset_dir: str):
    images_dir, labels_dir = _resolve_images_labels(dataset_dir)

    if not os.path.isdir(images_dir):
        print(f"[ERROR] 'images' folder not found in: {dataset_dir}")
        print("        Select the folder that contains 'images/' and 'labels/',")
        print("        or the dataset root folder (e.g. VisDrone/) if it has train/val/… subfolders.")
        sys.exit(1)

    images = sorted(
        f for f in os.listdir(images_dir)
        if not f.startswith('.')
        and os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )

    if not images:
        print(f"No images found in: {images_dir}")
        sys.exit(0)

    print(f"{len(images)} images found.")
    print("Controls: → / Space = next  |  ← = previous  |  Q / Esc = quit")

    cv2.namedWindow('Viewer', cv2.WINDOW_NORMAL)
    idx = 0

    while True:
        img_name   = images[idx]
        img_path   = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            idx = (idx + 1) % len(images)
            continue

        canvas, count = draw_boxes(img, label_path)
        canvas = fit_to_screen(canvas)
        info   = f"{idx + 1}/{len(images)}  {img_name}\nPersons: {count}"
        overlay_info(canvas, info)

        cv2.imshow('Viewer', canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):          # Q or Esc — quit
            break
        elif key in (83, ord('d'), 32):    # → or D or Space — next
            idx = (idx + 1) % len(images)
        elif key in (81, ord('a')):        # ← or A — previous
            idx = (idx - 1) % len(images)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if args:
        dataset_dir = args[0]
        if not os.path.isdir(dataset_dir):
            print(f"Error: '{dataset_dir}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the dataset folder in the pop-up window...")
        dataset_dir = pick_folder()
        if not dataset_dir:
            print("No folder selected.")
            sys.exit(0)

    run_viewer(dataset_dir)
