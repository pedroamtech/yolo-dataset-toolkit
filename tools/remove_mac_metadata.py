"""
tools/remove_mac_metadata.py — Find and delete macOS metadata files

Recursively finds the junk files macOS sprinkles across folders when they are
browsed or copied from a Mac, lists them, and (after a confirmation) deletes
them:

    .DS_Store       ← Finder folder-view settings
    ._<name>        ← AppleDouble sidecar files (resource forks / xattrs)

These files can confuse dataset loaders (an "image" like ._photo.jpg has no
pixels) and inflate file counts, so it is safe to strip them.

Flow: the folder is always scanned and the matches listed first. Then, unless
--dry-run was given, you are asked to confirm before anything is deleted
(--yes skips the prompt, e.g. for scripting).

On Windows these files are usually flagged Hidden (H) / System (S) / read-only
(R), so a plain delete fails. The script clears those attributes via the
Win32 API and falls back to CMD's own `del` (`del /s /f /q /a:h .DS_Store
._*`) if needed.

Usage:
    python tools/remove_mac_metadata.py                       # folder dialog
    python tools/remove_mac_metadata.py path/to/folder        # list, then ask
    python tools/remove_mac_metadata.py path/to/folder --dry-run   # list only
    python tools/remove_mac_metadata.py path/to/folder --yes       # no prompt
"""

import argparse
import os
import re
import stat
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog

IS_WINDOWS = os.name == "nt"
LIST_LIMIT = 40  # show every path up to this many, then summarise


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


def _win_long(path: str) -> str:
    """Prefix with \\\\?\\ so Win32 calls survive paths longer than 260 chars."""
    abspath = os.path.abspath(path)
    if abspath.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abspath[2:]
    return "\\\\?\\" + abspath


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
            ctypes.windll.kernel32.SetFileAttributesW(_win_long(path), FILE_ATTRIBUTE_NORMAL)
        except Exception:
            subprocess.run(["attrib", "-H", "-S", "-R", path],
                           capture_output=True, check=False)


def _still_there(path: str) -> bool:
    try:
        os.stat(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True  # can't stat but not "not found" — assume still present


def force_delete(path: str) -> None:
    """Delete a file even if flagged read-only / hidden / system."""
    clear_attributes(path)

    try:
        os.remove(path)
    except OSError:
        if not IS_WINDOWS:
            raise

    if not _still_there(path):
        return

    # Win32 DeleteFileW with the long-path prefix.
    if IS_WINDOWS:
        try:
            import ctypes
            ok = ctypes.windll.kernel32.DeleteFileW(_win_long(path))
            if ok and not _still_there(path):
                return
        except Exception:
            pass

    # Last resort: let the shell delete it.
    subprocess.run(["cmd", "/c", "del", "/f", "/q", "/a", path],
                   capture_output=True, check=False)
    if _still_there(path):
        raise OSError("delete denied by the OS")


def cmd_del_sweep(root_dir: str) -> None:
    """Run the raw CMD delete as a safety net (hidden + non-hidden passes)."""
    for extra in (["/a:h"], []):
        subprocess.run(
            ["cmd", "/c", "del", "/s", "/f", "/q", *extra, ".DS_Store", "._*"],
            cwd=root_dir, capture_output=True, check=False)


def scan(root_dir: str) -> list:
    def on_walk_error(exc: OSError) -> None:
        print(f"  [skipped] {exc}")

    found = []
    for root, _dirs, files in os.walk(root_dir, onerror=on_walk_error):
        for name in files:
            if is_mac_metadata(name):
                found.append(os.path.join(root, name))
    return found


def print_list(paths: list) -> None:
    shown = paths if len(paths) <= LIST_LIMIT else paths[:LIST_LIMIT]
    for path in shown:
        print(f"  {path}")
    if len(paths) > LIST_LIMIT:
        print(f"  ... and {len(paths) - LIST_LIMIT} more")


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def remove_mac_metadata(target_dir: str, dry_run: bool = False,
                        assume_yes: bool = False) -> None:
    root_dir = normalize_dir(target_dir)
    print(f"Scanning : {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"[ERROR] Not a folder: {root_dir}")
        sys.exit(1)

    matches = scan(root_dir)
    print(f"Found    : {len(matches)} macOS metadata file(s)")
    if matches:
        print_list(matches)

    if not matches:
        print("Nothing to clean.")
        return

    if dry_run:
        print("\n--dry-run: nothing was deleted.")
        return

    if not assume_yes:
        if not sys.stdin.isatty():
            print("\nNot an interactive terminal — re-run with --yes to delete, "
                  "or --dry-run to only list.")
            return
        if not confirm(f"\nDelete these {len(matches)} file(s)?"):
            print("Aborted — nothing was deleted.")
            return

    print()
    n_removed = n_failed = 0
    freed_bytes = 0
    for path in matches:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        try:
            force_delete(path)
            print(f"  removed : {path}")
            n_removed += 1
            freed_bytes += size
        except OSError as exc:
            print(f"  [FAILED] {path} — {exc}")
            n_failed += 1

    if IS_WINDOWS and (n_failed or n_removed == 0):
        print("  running raw CMD 'del' sweep as a fallback...")
        cmd_del_sweep(root_dir)

    print(f"\nFolder : {root_dir}")
    print(f"  Removed : {n_removed} file(s), {freed_bytes / 1024:.1f} KiB")
    if n_failed:
        print(f"  Failed  : {n_failed} file(s)")

    remaining = scan(root_dir)
    if remaining:
        print(f"  Still present : {len(remaining)} file(s) — first few:")
        for path in remaining[:5]:
            print(f"    {path}")
    else:
        print("  Verified : no macOS metadata files remain.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default=None,
                   help="Folder to scan recursively. "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only list the files found; never delete or prompt.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Delete without asking for confirmation.")
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

    remove_mac_metadata(target_dir, dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
