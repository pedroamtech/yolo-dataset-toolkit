"""
clean_dataset.py — Remove images with no person annotations

Keeps only images that have at least one YOLO label with class 0 (person).
Rejected images and their label files are moved to a _removed/ subfolder
by default; use --delete to permanently delete them instead.

Expected dataset structure:
    dataset/
        images/   ← source images
        labels/   ← YOLO .txt files (one per image)

Usage:
    python tools/clean_dataset.py                        # folder dialog
    python tools/clean_dataset.py path/to/dataset        # CLI path
    python tools/clean_dataset.py path/to/dataset --delete
"""

import os
import sys
import shutil
import tkinter as tk
from tkinter import filedialog

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')


def has_person(label_path: str) -> bool:
    """Return True if the label file contains at least one class-0 annotation."""
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == '0':
                    return True
    except OSError:
        pass
    return False


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the dataset folder")
    root.destroy()
    return folder


def clean_dataset(dataset_dir: str, delete: bool = False):
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    if not os.path.isdir(images_dir):
        print(f"[ERROR] 'images' folder not found in: {dataset_dir}")
        sys.exit(1)
    if not os.path.isdir(labels_dir):
        print(f"[ERROR] 'labels' folder not found in: {dataset_dir}")
        sys.exit(1)

    # Destination folders for rejected files (only used when not deleting)
    removed_images = os.path.join(dataset_dir, '_removed', 'images')
    removed_labels = os.path.join(dataset_dir, '_removed', 'labels')
    if not delete:
        os.makedirs(removed_images, exist_ok=True)
        os.makedirs(removed_labels, exist_ok=True)

    images = sorted(
        f for f in os.listdir(images_dir)
        if not f.startswith('.')
        and os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )

    n_kept = n_no_label = n_no_person = 0

    for img_name in images:
        stem       = os.path.splitext(img_name)[0]
        img_path   = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, stem + '.txt')

        # No label file → no annotations at all
        if not os.path.isfile(label_path):
            n_no_label += 1
            _discard(img_path, None, removed_images, removed_labels, delete)
            continue

        # Label exists but has no class-0 row
        if not has_person(label_path):
            n_no_person += 1
            _discard(img_path, label_path, removed_images, removed_labels, delete)
            continue

        n_kept += 1

    n_removed = n_no_label + n_no_person
    action    = 'deleted' if delete else 'moved to _removed/'

    print(f"\nDataset : {dataset_dir}")
    print(f"  Kept          : {n_kept}")
    print(f"  {action}:")
    print(f"    No label     : {n_no_label}")
    print(f"    No person    : {n_no_person}")
    print(f"    Total        : {n_removed}")


def _discard(img_path, label_path, removed_images, removed_labels, delete):
    if delete:
        os.remove(img_path)
        if label_path and os.path.isfile(label_path):
            os.remove(label_path)
    else:
        shutil.move(img_path, os.path.join(removed_images, os.path.basename(img_path)))
        if label_path and os.path.isfile(label_path):
            shutil.move(label_path, os.path.join(removed_labels, os.path.basename(label_path)))


if __name__ == '__main__':
    delete = '--delete' in sys.argv
    args   = [a for a in sys.argv[1:] if not a.startswith('--')]

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

    if delete:
        confirm = input(
            "\n⚠️  --delete mode: files will be permanently deleted.\n"
            "Continue? [y/N]: "
        ).strip().lower()
        if confirm not in ('y', 'yes'):
            print("Cancelled.")
            sys.exit(0)

    clean_dataset(dataset_dir, delete=delete)
