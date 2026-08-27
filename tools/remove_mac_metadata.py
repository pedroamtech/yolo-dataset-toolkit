"""
tools/remove_mac_metadata.py — Delete macOS metadata files

Recursively removes the junk files macOS sprinkles across folders when they
are browsed or copied from a Mac:

    .DS_Store       ← Finder folder-view settings
    ._<name>        ← AppleDouble sidecar files (resource forks / xattrs)

These files can confuse dataset loaders (an "image" like ._photo.jpg has no
pixels) and inflate file counts, so it is safe to strip them.

On Windows these files are usually flagged Hidden (H) / System (S) / read-only
(R), so a plain delete fails. This is the Python equivalent of running

    del /s /f /q /a:h .DS_Store ._*

from CMD: os.walk visits every sub-folder (including hidden ones), the H/S/R
attributes are cleared with `attrib`, and a `del` fallback is used if
os.remove still refuses.

Usage:
    python tools/remove_mac_metadata.py                       # folder dialog
    python tools/remove_mac_metadata.py path/to/folder
    python tools/remove_mac_metadata.py path/to/folder --dry-run
"""

import argparse
import os
import stat
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

IS_WINDOWS = os.name == "nt"


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the folder to clean")
    root.destroy()
    return folder


def is_mac_metadata(name: str) -> bool:
    return name == ".DS_Store" or name.startswith("._")


def force_delete(path: str) -> None:
    """Delete a file even if flagged read-only / hidden / system.

    Mirrors `del /f /a:h`: clear the attributes, then remove. Falls back to
    CMD's own `del` if Python's os.remove is still denied.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass

    if IS_WINDOWS:
        subprocess.run(["attrib", "-H", "-S", "-R", path],
                       capture_output=True, check=False)

    try:
        os.remove(path)
        return
    except OSError:
        if not IS_WINDOWS:
            raise

    # Last resort on Windows: let the shell delete it.
    result = subprocess.run(["cmd", "/c", "del", "/f", "/q", "/a", path],
                            capture_output=True, check=False)
    if os.path.exists(path):
        raise OSError(result.stderr.decode(errors="replace").strip()
                      or "del command failed")


def remove_mac_metadata(target_dir: Path, dry_run: bool = False) -> None:
    # os.walk("E:") walks E:'s *current* directory, not its root — normalise.
    root_dir = os.path.abspath(str(target_dir))

    n_removed = n_failed = 0
    freed_bytes = 0

    for root, _dirs, files in os.walk(root_dir):
        for name in files:
            if not is_mac_metadata(name):
                continue

            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0

            if dry_run:
                print(f"  would remove : {path}")
                n_removed += 1
                freed_bytes += size
                continue

            try:
                force_delete(path)
                print(f"  removed : {path}")
                n_removed += 1
                freed_bytes += size
            except OSError as exc:
                print(f"  [FAILED] {path} — {exc}")
                n_failed += 1

    verb = "Would remove" if dry_run else "Removed"
    print(f"\nFolder : {root_dir}")
    print(f"  {verb} : {n_removed} file(s), {freed_bytes / 1024:.1f} KiB")
    if n_failed:
        print(f"  Failed : {n_failed} file(s)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", type=Path, default=None,
                   help="Folder to scan recursively. "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the files that would be removed without deleting them.")
    args = p.parse_args()

    if args.folder:
        target_dir = args.folder
        if not target_dir.is_dir():
            print(f"Error: '{target_dir}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the folder in the pop-up window...")
        picked = pick_folder()
        if not picked:
            print("No folder selected.")
            sys.exit(0)
        target_dir = Path(picked)

    remove_mac_metadata(target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
