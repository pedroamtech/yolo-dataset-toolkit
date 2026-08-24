"""
tools/remove_unlabeled_images.py — Remove images with no .txt label file

Unlike clean_dataset.py (which also drops images whose label file exists but
has no class-0/person row), this only checks whether a matching .txt label
file exists at all. Images missing a label file are moved to a _removed/
subfolder by default; use --delete to permanently delete them instead.

Expected dataset structure:
    dataset/
        images/   ← source images
        labels/   ← YOLO .txt files (one per image)

Usage:
    python tools/remove_unlabeled_images.py                        # folder dialog
    python tools/remove_unlabeled_images.py path/to/dataset
    python tools/remove_unlabeled_images.py path/to/dataset --delete
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


def remove_unlabeled(dataset_dir: Path, delete: bool = False):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        print(f"[ERROR] 'images' folder not found in: {dataset_dir}")
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"[ERROR] 'labels' folder not found in: {dataset_dir}")
        sys.exit(1)

    removed_images_dir = dataset_dir / "_removed" / "images"
    if not delete:
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
        if delete:
            img_path.unlink()
        else:
            shutil.move(str(img_path), str(removed_images_dir / img_path.name))

    action = "deleted" if delete else f"moved to {removed_images_dir}"
    print(f"\nDataset : {dataset_dir}")
    print(f"  Kept               : {n_kept}")
    print(f"  Removed (no label) : {n_removed} ({action})")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", type=Path, default=None,
                    help="Dataset folder (contains images/ and labels/). "
                         "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--delete", action="store_true",
                    help="Permanently delete instead of moving to _removed/.")
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

    if args.delete:
        confirm = input(
            "\n[WARNING] --delete mode: files will be permanently deleted.\n"
            "Continue? [y/N]: "
        ).strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(0)

    remove_unlabeled(dataset_dir, delete=args.delete)


if __name__ == "__main__":
    main()
