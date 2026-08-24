"""
tools/remove_unlabeled_images.py — Remove images with no .txt label file

Unlike clean_dataset.py (which also drops images whose label file exists but
has no class-0/person row), this only checks whether a matching .txt label
file exists at all. Images missing a label file are moved to a _removed/
subfolder (never permanently deleted).

Expected dataset structure:
    dataset/
        images/   ← source images
        labels/   ← YOLO .txt files (one per image)

Usage:
    python tools/remove_unlabeled_images.py                        # folder dialog
    python tools/remove_unlabeled_images.py path/to/dataset
"""

import argparse
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the dataset folder")
    root.destroy()
    return folder


def remove_unlabeled(dataset_dir: Path):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        print(f"[ERROR] 'images' folder not found in: {dataset_dir}")
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"[ERROR] 'labels' folder not found in: {dataset_dir}")
        sys.exit(1)

    removed_images_dir = dataset_dir / "_removed" / "images"
    removed_images_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in IMAGE_EXTS
    )

    n_kept = n_removed = 0

    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.is_file():
            n_kept += 1
            continue

        n_removed += 1
        shutil.move(str(img_path), str(removed_images_dir / img_path.name))

    print(f"\nDataset : {dataset_dir}")
    print(f"  Kept               : {n_kept}")
    print(f"  Removed (no label) : {n_removed} (moved to {removed_images_dir})")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", type=Path, default=None,
                    help="Dataset folder (contains images/ and labels/). "
                         "Falls back to a folder-picker dialog if omitted.")
    args = p.parse_args()

    if args.dataset:
        dataset_dir = args.dataset
        if not dataset_dir.is_dir():
            print(f"Error: '{dataset_dir}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the dataset folder in the pop-up window...")
        picked = pick_folder()
        if not picked:
            print("No folder selected.")
            sys.exit(0)
        dataset_dir = Path(picked)

    remove_unlabeled(dataset_dir)


if __name__ == "__main__":
    main()
