"""
tools/rename_images.py — Batch-rename files with a fixed prefix

Prepends a fixed prefix to every file in a folder. By default only common
image extensions are touched; use --ext to target other extensions, or
--all-files to rename everything. With --labels, the matching YOLO label file
(same stem, .txt) in that folder is renamed with the same prefix, so
image/label pairs stay together.

Usage:
    python tools/rename_images.py                                  # folder dialog, no prefix
    python tools/rename_images.py path/to/images --prefix "cam1_"
    python tools/rename_images.py path/to/labels --prefix "cam1_" --ext .txt
    python tools/rename_images.py path/to/folder --prefix "cam1_" --all-files
    python tools/rename_images.py path/to/images --prefix "cam1_" --labels path/to/labels
    python tools/rename_images.py path/to/images --prefix "cam1_" --dry-run
"""

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from tqdm import tqdm

# Default whitelist when neither --ext nor --all-files is given. Matched
# case-insensitively (so .JPG / .JPEG / .PNG are covered too).
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the folder containing the files")
    root.destroy()
    return folder


def normalize_exts(raw: str) -> set[str]:
    """Parse a comma/space separated extension list into a lowercase set with
    leading dots, e.g. 'txt, .JSON' -> {'.txt', '.json'}."""
    exts = set()
    for token in raw.replace(",", " ").split():
        token = token.lower()
        exts.add(token if token.startswith(".") else "." + token)
    return exts


def rename_files(folder: Path, prefix: str, exts: set[str] | None,
                 labels_dir: Path | None, dry_run: bool) -> int:
    # exts is None -> rename every file, whatever its extension.
    targets = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and not p.name.startswith(".")
        and (exts is None or p.suffix.lower() in exts)
    ]

    scope = "file(s)" if exts is None else "matching file(s)"
    print(f"Folder : {folder}")
    print(f"Found  : {len(targets)} {scope}")
    if labels_dir is not None:
        print(f"Labels : {labels_dir}")
    if not targets:
        print("Nothing to rename — check the folder path and the --ext filter.")
        return 0

    renamed = skipped = errors = labels_hit = 0
    for path in tqdm(targets, desc="Renaming", disable=dry_run):
        renames = [(path, path.with_name(prefix + path.name))]
        has_label = False
        if labels_dir is not None:
            old_label = labels_dir / (path.stem + ".txt")
            if old_label.is_file():
                has_label = True
                renames.append((old_label, labels_dir / (prefix + old_label.name)))

        clash = next((dst for _, dst in renames if dst.exists()), None)
        if clash is not None:
            print(f"[SKIP] target already exists: {clash.name}")
            skipped += 1
            continue

        try:
            for src, dst in renames:
                if dry_run:
                    print(f"[DRY-RUN] {src.name} -> {dst.name}")
                else:
                    src.rename(dst)
        except OSError as e:
            print(f"[ERROR] {path.name}: {e}")
            errors += 1
            continue

        renamed += 1
        if has_label:
            labels_hit += 1

    verb = "would be renamed" if dry_run else "renamed"
    print(f"\n{renamed} {verb}"
          + (f", {labels_hit} with a matching label" if labels_dir is not None else "")
          + (f", {skipped} skipped (name clash)" if skipped else "")
          + (f", {errors} error(s)" if errors else ""))
    return renamed


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default=None,
                   help="Folder containing the files to rename. "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--prefix", default="", help="Prefix prepended to every filename.")
    p.add_argument("--ext", default=None,
                   help="Comma/space separated extensions to rename (e.g. '.txt,.json'). "
                        "Default: common image extensions.")
    p.add_argument("--all-files", action="store_true",
                   help="Rename every file in the folder regardless of extension.")
    p.add_argument("--labels", type=Path, default=None,
                   help="Optional labels folder; matching .txt files get the same prefix.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned renames without touching any file.")
    args = p.parse_args()

    if args.folder:
        folder = args.folder
        if not os.path.isdir(folder):
            print(f"Error: '{folder}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the folder in the pop-up window...")
        folder = pick_folder()
        if not folder:
            print("No folder selected.")
            sys.exit(0)

    if args.all_files:
        exts = None
    elif args.ext:
        exts = normalize_exts(args.ext)
    else:
        exts = IMAGE_EXTS

    if args.labels is not None and not args.labels.is_dir():
        print(f"Error: '{args.labels}' is not a valid folder.")
        sys.exit(1)

    rename_files(Path(folder), args.prefix, exts, args.labels, args.dry_run)


if __name__ == "__main__":
    main()
