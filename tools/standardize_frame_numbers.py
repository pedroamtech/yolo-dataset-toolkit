"""
tools/standardize_frame_numbers.py — Zero-pad the trailing frame number in filenames

Takes the block of digits after the LAST underscore in each filename stem and
left-pads it with zeros to 6 characters:

    clip_40.jpg      -> clip_000040.jpg
    cam1_20230101_300.jpg -> cam1_20230101_000300.jpg
    frame_000040.jpg -> frame_000040.jpg   (already 6, unchanged)
    frame_1234567.jpg -> frame_1234567.jpg (already >6, never truncated)

Files whose stem has no underscore, or whose trailing block is not purely
numeric, are left untouched. Matching YOLO label files (same stem, .txt) in an
optional labels folder are renamed in lockstep.

Usage:
    python tools/standardize_frame_numbers.py                         # folder dialog
    python tools/standardize_frame_numbers.py path/to/images
    python tools/standardize_frame_numbers.py path/to/images --labels path/to/labels
    python tools/standardize_frame_numbers.py path/to/images --dry-run
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PAD_WIDTH = 6


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the folder containing the images")
    root.destroy()
    return folder


def standardized_stem(stem: str) -> str:
    """Return `stem` with its trailing post-underscore number zero-padded to
    PAD_WIDTH. Returns the stem unchanged when there is nothing to pad."""
    if "_" not in stem:
        return stem
    head, tail = stem.rsplit("_", 1)
    if not tail.isdigit():
        return stem
    # `:0{PAD_WIDTH}d` only ever pads — numbers already >= PAD_WIDTH digits pass
    # through untouched, so the frame number is never truncated.
    return f"{head}_{int(tail):0{PAD_WIDTH}d}"


def standardize(img_dir: Path, labels_dir: Path | None, dry_run: bool) -> int:
    images = [
        p for p in sorted(img_dir.iterdir())
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS
    ]

    n_changed = 0
    for img_path in tqdm(images, desc="Standardizing"):
        new_stem = standardized_stem(img_path.stem)
        if new_stem == img_path.stem:
            continue

        renames = [(img_path, img_path.with_name(new_stem + img_path.suffix))]
        if labels_dir is not None:
            old_label = labels_dir / (img_path.stem + ".txt")
            if old_label.is_file():
                renames.append((old_label, labels_dir / (new_stem + ".txt")))

        clash = next((dst for _, dst in renames if dst.exists()), None)
        if clash is not None:
            print(f"[SKIP] target already exists: {clash.name}")
            continue

        for src, dst in renames:
            if dry_run:
                print(f"[DRY-RUN] {src.name} -> {dst.name}")
            else:
                src.rename(dst)
        n_changed += 1

    return n_changed


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="?", type=Path, default=None,
                   help="Folder containing the images to rename. "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--labels", type=Path, default=None,
                   help="Optional labels folder; matching .txt files are renamed too.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned renames without touching any file.")
    args = p.parse_args()

    if args.images:
        img_dir = args.images
        if not img_dir.is_dir():
            print(f"Error: '{img_dir}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the images folder in the pop-up window...")
        picked = pick_folder()
        if not picked:
            print("No folder selected.")
            sys.exit(0)
        img_dir = Path(picked)

    if args.labels is not None and not args.labels.is_dir():
        print(f"Error: '{args.labels}' is not a valid folder.")
        sys.exit(1)

    n_changed = standardize(img_dir, args.labels, args.dry_run)
    verb = "would be renamed" if args.dry_run else "renamed"
    print(f"Done — {n_changed} files {verb}")


if __name__ == "__main__":
    main()
