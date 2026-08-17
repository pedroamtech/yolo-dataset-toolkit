"""
tools/rename_images.py — Batch-rename images with a fixed prefix

Usage:
    python tools/rename_images.py                                  # folder dialog, no prefix
    python tools/rename_images.py path/to/images --prefix "cam1_"
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


def rename_images(img_dir: Path, prefix: str) -> int:
    images = [p for p in img_dir.iterdir() if p.suffix in EXTENSIONS]
    for img_path in tqdm(images, desc="Renaming"):
        new_name = img_path.parent / (prefix + img_path.name)
        img_path.rename(new_name)
    return len(images)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="?", default=None,
                   help="Folder containing the images to rename. "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--prefix", default="", help="Prefix prepended to every filename.")
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

    count = rename_images(Path(img_dir), args.prefix)
    print(f"Done — {count} images renamed with prefix '{args.prefix}'")


if __name__ == "__main__":
    main()
