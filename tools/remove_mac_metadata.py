"""
tools/remove_mac_metadata.py — Delete macOS metadata files

Recursively removes the junk files macOS sprinkles across folders when they
are browsed or copied from a Mac:

    .DS_Store       ← Finder folder-view settings
    ._<name>        ← AppleDouble sidecar files (resource forks / xattrs)

These files can confuse dataset loaders (an "image" like ._photo.jpg has no
pixels) and inflate file counts, so it is safe to strip them.

On Windows these files are usually flagged Hidden (H) / System (S) / read-only
(R), so a plain delete fails. This is the Python equivalent of

    del /s /f /q /a:h .DS_Store ._*

plus a pass for the same names when they are *not* hidden: os.walk visits
every sub-folder (hidden ones included), the H/S/R attributes are cleared via
the Win32 API, and CMD's own `del` is used as a fallback. If the recursive
walk finds nothing on Windows, the raw `del` commands are run as a safety net.

Usage:
    python tools/remove_mac_metadata.py                       # folder dialog
    python tools/remove_mac_metadata.py path/to/folder
    python tools/remove_mac_metadata.py path/to/folder --dry-run
"""

import argparse
import os
import re
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


def normalize_dir(raw: str) -> str:
    """Resolve the folder to an absolute path.

    A bare drive letter like "E:" resolves to that drive's *current*
    directory, not its root, so turn "E:" into "E:\\" first.
    """
    raw = raw.strip().strip('"')
    if re.fullmatch(r"[A-Za-z]:", raw):
        raw += "\\"
    return os.path.abspath(raw)


def is_mac_metadata(name: str) -> bool:
    return name == ".DS_Store" or name.startswith("._")


def clear_attributes(path: str) -> None:
    """Strip read-only / hidden / system so the file can be deleted."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    if IS_WINDOWS:
        FILE_ATTRIBUTE_NORMAL = 0x80
        try:
            import ctypes
            ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_NORMAL)
        except Exception:
            subprocess.run(["attrib", "-H", "-S", "-R", path],
                           capture_output=True, check=False)


def force_delete(path: str) -> None:
    """Delete a file even if flagged read-only / hidden / system."""
    clear_attributes(path)
    try:
        os.remove(path)
        return
    except OSError:
        if not IS_WINDOWS:
            raise

    # Last resort on Windows: let the shell delete it.
    subprocess.run(["cmd", "/c", "del", "/f", "/q", "/a", path],
                   capture_output=True, check=False)
    if os.path.exists(path):
        raise OSError("delete denied by the OS")


def cmd_del_sweep(root_dir: str) -> None:
    """Run the raw CMD delete as a safety net (hidden + non-hidden passes)."""
    for extra in (["/a:h"], []):
        subprocess.run(
            ["cmd", "/c", "del", "/s", "/f", "/q", *extra, ".DS_Store", "._*"],
            cwd=root_dir, capture_output=True, check=False)


def remove_mac_metadata(target_dir: str, dry_run: bool = False) -> None:
    root_dir = normalize_dir(target_dir)
    print(f"Scanning : {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"[ERROR] Not a folder: {root_dir}")
        sys.exit(1)

    def on_walk_error(exc: OSError) -> None:
        print(f"  [skipped] {exc}")

    matches = []
    for root, _dirs, files in os.walk(root_dir, onerror=on_walk_error):
        for name in files:
            if is_mac_metadata(name):
                matches.append(os.path.join(root, name))

    print(f"Found    : {len(matches)} macOS metadata file(s)")

    n_removed = n_failed = 0
    freed_bytes = 0

    for path in matches:
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

    if not dry_run and IS_WINDOWS and n_removed == 0 and n_failed == 0:
        print("  walk found nothing — running raw CMD 'del' sweep...")
        cmd_del_sweep(root_dir)

    verb = "Would remove" if dry_run else "Removed"
    print(f"\nFolder : {root_dir}")
    print(f"  {verb} : {n_removed} file(s), {freed_bytes / 1024:.1f} KiB")
    if n_failed:
        print(f"  Failed : {n_failed} file(s)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default=None,
                   help="Folder to scan recursively. "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--dry-run", action="store_true",
                   help="List the files that would be removed without deleting them.")
    args = p.parse_args()

    if args.folder:
        target_dir = args.folder
    else:
        print("Select the folder in the pop-up window...")
        picked = pick_folder()
        if not picked:
            print("No folder selected.")
            sys.exit(0)
        target_dir = picked

    remove_mac_metadata(target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
