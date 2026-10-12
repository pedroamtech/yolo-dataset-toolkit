"""
tools/normalize_manipal_labels.py — Clamp YOLO labels into [0, 1]

Clamps normalized YOLO coordinates into [0, 1] and drops degenerate
(zero-area) boxes. Processes every partition folder under the dataset root
that has a labels/ subfolder (e.g. training/, validation/, test/ or
train/, val/, test/), keeping its name, and writes the result to a separate
output folder (images are copied, labels are normalized) so the source
dataset is left untouched:

    <output>/<partition>/images/...   <output>/<partition>/labels/...

Usage:
    python tools/normalize_manipal_labels.py                                # folder dialog
    python tools/normalize_manipal_labels.py path/to/dataset                # output: <dataset>_normalized
    python tools/normalize_manipal_labels.py path/to/dataset -o path/to/out
"""

import argparse
import os
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from tqdm import tqdm


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the dataset root folder")
    root.destroy()
    return folder


def normalize_label(src: Path) -> list[str]:
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
    return lines_out


def find_partitions(base_dir: Path) -> list[str]:
    """Every subfolder of the dataset root that has a labels/ folder."""
    return sorted(d.name for d in base_dir.iterdir() if d.is_dir() and (d / "labels").is_dir())


def normalize_dataset(base_dir: Path, out_dir: Path, partitions: list[str]) -> tuple[int, int, int]:
    total_files = total_boxes = total_images = 0

    for partition in partitions:
        label_dir = base_dir / partition / "labels"

        label_out = out_dir / partition / "labels"
        label_out.mkdir(parents=True, exist_ok=True)

        txt_files = list(label_dir.glob("*.txt"))
        for src in tqdm(txt_files, desc=f"{partition} labels"):
            lines_out = normalize_label(src)
            (label_out / src.name).write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
            total_boxes += len(lines_out)
        total_files += len(txt_files)

        image_dir = base_dir / partition / "images"
        if image_dir.exists():
            image_out = out_dir / partition / "images"
            image_out.mkdir(parents=True, exist_ok=True)
            images = [p for p in image_dir.iterdir() if p.is_file()]
            for img in tqdm(images, desc=f"{partition} images"):
                shutil.copy2(img, image_out / img.name)
            total_images += len(images)

    return total_files, total_boxes, total_images


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default=None,
                   help="Dataset root containing train/val/test partitions. "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("-o", "--output", default=None,
                   help="Output folder (default: '<dataset>_normalized' next to the dataset).")
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

    base_dir = Path(base_dir).resolve()
    out_dir = Path(args.output).resolve() if args.output else base_dir.with_name(base_dir.name + "_normalized")
    if out_dir == base_dir:
        print("Error: the output folder must be different from the dataset folder.")
        sys.exit(1)
    partitions = find_partitions(base_dir)
    if not partitions:
        print(f"Error: no partition folders with a labels/ subfolder found in '{base_dir}'.")
        sys.exit(1)
    print(f"Partitions: {', '.join(partitions)}")
    print(f"Output: {out_dir}")

    total_files, total_boxes, total_images = normalize_dataset(base_dir, out_dir, partitions)
    print(f"Done — {total_files} label files, {total_boxes} boxes normalized, "
          f"{total_images} images copied to {out_dir}")


if __name__ == "__main__":
    main()
