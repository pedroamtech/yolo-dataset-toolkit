"""
tools/rename_images.py — Batch-rename images with a fixed prefix

Prepends a fixed prefix to every image in a folder. With --labels, the
matching YOLO label file (same stem, .txt) in that folder is renamed with the
same prefix, so image/label pairs stay together.

Usage:
    python tools/rename_images.py                                  # folder dialog, no prefix
    python tools/rename_images.py path/to/images --prefix "cam1_"
    python tools/rename_images.py path/to/images --prefix "cam1_" --labels path/to/labels
"""

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from tqdm import tqdm

EXTENSIONS = {".jpg", ".JPG", ".jpeg", ".png", ".PNG"}


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the folder containing the images")
    root.destroy()
    return folder


def rename_images(img_dir: Path, prefix: str, labels_dir: Path | None) -> int:
    images = [p for p in img_dir.iterdir() if p.suffix in EXTENSIONS]
    for img_path in tqdm(images, desc="Renaming"):
        renames = [(img_path, img_path.with_name(prefix + img_path.name))]
        if labels_dir is not None:
            old_label = labels_dir / (img_path.stem + ".txt")
            if old_label.is_file():
                renames.append((old_label, labels_dir / (prefix + old_label.name)))

        clash = next((dst for _, dst in renames if dst.exists()), None)
        if clash is not None:
            print(f"[SKIP] target already exists: {clash.name}")
            continue

        for src, dst in renames:
            src.rename(dst)
    return len(images)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="?", default=None,
                   help="Folder containing the images to rename. "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--prefix", default="", help="Prefix prepended to every filename.")
    p.add_argument("--labels", type=Path, default=None,
                   help="Optional labels folder; matching .txt files get the same prefix.")
    args = p.parse_args()

    if args.images:
        img_dir = args.images
        if not os.path.isdir(img_dir):
            print(f"Error: '{img_dir}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the images folder in the pop-up window...")
        img_dir = pick_folder()
        if not img_dir:
            print("No folder selected.")
            sys.exit(0)

    if args.labels is not None and not args.labels.is_dir():
        print(f"Error: '{args.labels}' is not a valid folder.")
        sys.exit(1)

    count = rename_images(Path(img_dir), args.prefix, args.labels)
    suffix = " (labels renamed too)" if args.labels is not None else ""
    print(f"Done — {count} images renamed with prefix '{args.prefix}'{suffix}")


if __name__ == "__main__":
    main()
