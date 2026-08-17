"""
tools/normalize_manipal_labels.py — Clamp YOLO labels into [0, 1]

Clamps normalized YOLO coordinates into [0, 1] and drops degenerate
(zero-area) boxes. Scans train/val/test partitions under the dataset root,
skipping any that don't exist.

Usage:
    python tools/normalize_manipal_labels.py                  # folder dialog
    python tools/normalize_manipal_labels.py path/to/dataset  # CLI path
"""

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from tqdm import tqdm

PARTITIONS = ["train", "val", "test"]   # folders to process; skip missing ones


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the dataset root folder")
    root.destroy()
    return folder


def normalize_dataset(base_dir: Path) -> tuple[int, int]:
    total_files = total_boxes = 0

    for partition in PARTITIONS:
        label_dir = base_dir / partition / "labels"
        if not label_dir.exists():
            continue

        txt_files = list(label_dir.glob("*.txt"))
        for src in tqdm(txt_files, desc=partition):
            lines_out = []
            for line in src.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, w, h = (float(p) for p in parts[1:])
                xc, yc, w, h = (max(0.0, min(1.0, v)) for v in (xc, yc, w, h))
                if w > 0 and h > 0:
                    lines_out.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            src.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
            total_boxes += len(lines_out)

        total_files += len(txt_files)

    return total_files, total_boxes


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default=None,
                   help="Dataset root containing train/val/test partitions. "
                        "Falls back to a folder-picker dialog if omitted.")
    args = p.parse_args()

    if args.dataset:
        base_dir = args.dataset
        if not os.path.isdir(base_dir):
            print(f"Error: '{base_dir}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the dataset root folder in the pop-up window...")
        base_dir = pick_folder()
        if not base_dir:
            print("No folder selected.")
            sys.exit(0)

    total_files, total_boxes = normalize_dataset(Path(base_dir))
    print(f"Done — {total_files} files, {total_boxes} boxes normalized")


if __name__ == "__main__":
    main()
