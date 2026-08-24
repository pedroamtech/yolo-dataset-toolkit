"""
tools/split_dataset.py — Split a labeled dataset into train/val

Uses sklearn's train_test_split with a fixed random_state (reproducible) to
split the dataset by ratio (default 80/20). Each image is copied together
with its matching YOLO label file, so pairs always stay together. Images
without a label file are still included in the split (treated as
background/negatives).

Expected dataset structure:
    dataset/
        images/   ← source images
        labels/   ← YOLO .txt files (one per image)

Output structure:
    dataset/
        train/
            images/
            labels/
        val/
            images/
            labels/

Usage:
    python tools/split_dataset.py path/to/dataset
    python tools/split_dataset.py path/to/dataset --ratio 0.8 --seed 42
    python tools/split_dataset.py path/to/dataset --move
"""

import argparse
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from sklearn.model_selection import train_test_split

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the dataset folder")
    root.destroy()
    return folder


def split_dataset(dataset_dir: Path, ratio: float, seed: int, move: bool):
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        print(f"[ERROR] 'images' folder not found in: {dataset_dir}")
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"[ERROR] 'labels' folder not found in: {dataset_dir}")
        sys.exit(1)

    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"[ERROR] no images found in: {images_dir}")
        sys.exit(1)

    train_images, val_images = train_test_split(
        images, train_size=ratio, random_state=seed)
    splits = {"train": train_images, "val": val_images}

    n_no_label = 0
    transfer = shutil.move if move else shutil.copy2

    for split_name, split_images in splits.items():
        split_images_dir = dataset_dir / split_name / "images"
        split_labels_dir = dataset_dir / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            transfer(str(img_path), str(split_images_dir / img_path.name))

            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.is_file():
                transfer(str(label_path), str(split_labels_dir / label_path.name))
            else:
                n_no_label += 1

    action = "moved" if move else "copied"
    print(f"\nDataset : {dataset_dir}")
    print(f"  Seed          : {seed}")
    print(f"  Ratio         : {ratio:.2f} / {1 - ratio:.2f}")
    print(f"  Train         : {len(splits['train'])} images -> {dataset_dir / 'train'}")
    print(f"  Val           : {len(splits['val'])} images -> {dataset_dir / 'val'}")
    print(f"  Without label : {n_no_label}")
    print(f"  Files {action} (originals left in place: {not move})")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", type=Path, default=None,
                    help="Dataset folder (contains images/ and labels/). "
                         "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--ratio", type=float, default=0.8,
                    help="Fraction of images assigned to train (default: 0.8).")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for the shuffle, for a reproducible split (default: 42).")
    p.add_argument("--move", action="store_true",
                    help="Move files instead of copying them (default: copy, leaves "
                         "images/ and labels/ untouched).")
    args = p.parse_args()

    if not 0.0 < args.ratio < 1.0:
        print(f"[ERROR] --ratio must be between 0 and 1, got {args.ratio}")
        sys.exit(1)

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

    split_dataset(dataset_dir, ratio=args.ratio, seed=args.seed, move=args.move)


if __name__ == "__main__":
    main()
