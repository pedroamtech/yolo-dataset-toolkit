# YOLO Dataset Toolkit: Cleaning, Validation, and Size-Distribution Analysis for YOLO Datasets

Standalone utilities for preparing, cleaning, validating, and inspecting
YOLO-format object detection datasets (aerial/UAV person-detection focused,
but generic to any single- or multi-class YOLO dataset).

Extracted from the [Aerial Person Aug](https://github.com/pedroamtech/aerial-person-aug) project's
internal `tools/` folder.

## Tools

| Script | Purpose |
|---|---|
| `tools/clean_dataset.py` | Removes images with no person (class 0) annotations. Rejected files are moved to `_removed/` (or deleted with `--delete`). |
| `tools/remove_unlabeled_images.py` | Moves images with no matching `.txt` label file to `_removed/`. |
| `tools/split_dataset.py` | 80/20 train/val split with a reproducible random seed (uses `sklearn.train_test_split`). |
| `tools/validate_labels.py` | Scans label files for issues that can crash training: out-of-range classes, malformed rows, NaN/out-of-bounds coordinates, orphan files. Writes a CSV report. |
| `tools/normalize_manipal_labels.py` | Clamps YOLO-normalized coordinates into `[0, 1]` and drops degenerate (zero-area) boxes. |
| `tools/yolo_person_labeler.py` | All-in-one labeling/editing/viewing tool — HOG auto-detection, manual box drawing, zoom/pan, click-to-select-and-delete, per-class color rendering. |
| `tools/analyze_size_distribution.py` | Object size distribution analysis (Absolute/Relative Size, log-normal fit, CCDF heavy-tail diagnostic) following the TinyPerson Benchmark methodology (Yu et al., 2019). Runs on synthetic data if no `--labels`/`--images` are given. |
| `tools/rename_images.py` | Batch-renames images in a folder with a fixed prefix. |
| `tools/standardize_frame_numbers.py` | Zero-pads the trailing `_<number>` (the digits after the last underscore) in each filename to 6 digits, e.g. `clip_40.jpg` → `clip_000040.jpg`, `cam1_20230101_300.jpg` → `cam1_20230101_000300.jpg`. Numbers already ≥ 6 digits are never truncated. Optionally renames matching `labels/*.txt` in lockstep; `--dry-run` previews. |
| `tools/video_to_frames.py` | Extracts every frame from all videos in a folder into per-video subfolders. |
| `tools/remove_mac_metadata.py` | Recursively finds macOS junk files (`.DS_Store` and `._*` AppleDouble sidecars), lists them, then asks before deleting (`--dry-run` to only list, `--yes` to skip the prompt). |

`clean_labels.py` and `visualize_labels.py` were merged into
`yolo_person_labeler.py` (zoom/pan box editing + per-class color rendering
now live there) and have been removed.

### Tool reference

Every script also prints this detail in its `--help` / module docstring. All
of them accept the target path as the first positional argument and fall back
to a folder-picker dialog when run with no arguments.

#### `clean_dataset.py` — drop images with no person

Keeps only images whose YOLO label file has at least one **class 0** (person)
row. Images that are missing a label, have an empty label, or have labels for
other classes only are rejected: the image **and** its `.txt` are moved to
`_removed/` (use `--delete` to erase them instead). Needs `images/` and
`labels/` under the dataset folder.

```bash
python tools/clean_dataset.py path/to/dataset          # move rejects to _removed/
python tools/clean_dataset.py path/to/dataset --delete # permanently delete rejects
```

#### `remove_unlabeled_images.py` — drop images with no label file

Lighter check than `clean_dataset.py`: it only asks whether a matching `.txt`
exists at all (contents are not inspected). Images with no label file are
moved to `_removed/images/` and never deleted. Needs `images/` and `labels/`
under the dataset folder.

```bash
python tools/remove_unlabeled_images.py path/to/dataset
```

#### `split_dataset.py` — reproducible train/val split

Splits an `images/` + `labels/` dataset into `train/` and `val/`
subfolders (each with its own `images/` and `labels/`) using
`sklearn.train_test_split` with a fixed seed. Every image is kept together
with its label file; images without a label are still placed in the split as
background/negatives. Files are copied by default; `--move` relocates them.

```bash
python tools/split_dataset.py path/to/dataset
python tools/split_dataset.py path/to/dataset --ratio 0.8 --seed 42 --move
```

#### `validate_labels.py` — pre-training label sanity check

Recursively finds every folder named `labels/` and scans each `.txt` for
issues that crash training runs: class ids `>= --num-classes` or negative,
rows without exactly 5 fields, non-finite (`NaN`/`inf`) or out-of-`[0,1]`
coordinates, and orphan label/image files. Writes a CSV report (default
`label_validation_report.csv`; `--report PATH` to rename, `--no-report` to
skip). Built after an `EdgeYOLO` CUDA assert traced back to a single
out-of-range class id.

```bash
python tools/validate_labels.py path/to/dataset --num-classes 2
python tools/validate_labels.py path/to/dataset --num-classes 2 --report out.csv
```

#### `normalize_manipal_labels.py` — clamp coordinates into `[0, 1]`

Walks the `train/`, `val/`, `test/` partitions under the dataset root
(skipping any that are absent), clamps every YOLO coordinate into `[0, 1]`,
and drops degenerate boxes that collapse to zero area after clamping. Rewrites
the label files in place.

```bash
python tools/normalize_manipal_labels.py path/to/dataset
```

#### `yolo_person_labeler.py` — label, edit and review boxes

All-in-one GUI (OpenCV window) that replaced the old `clean_labels.py` +
`visualize_labels.py`. Fits each image to the window (never upscales),
supports scroll-wheel zoom and right-drag pan, draws new class-0 boxes by
dragging, selects/deletes existing boxes by clicking, runs HOG person
detection on the current image (`Space`) or across the whole dataset (`B`),
and renders boxes per-class in distinct colors. `S` saves and advances;
`D`/`A` move without saving. Press `H` in-window for the full key list.

```bash
python tools/yolo_person_labeler.py path/to/dataset
```

#### `analyze_size_distribution.py` — object size statistics

Computes Absolute Size `AS = sqrt(w·h)` and Relative Size
`RS = sqrt(w·h / (W·H))` per object (TinyPerson Benchmark, Yu et al. 2019),
fits a log-normal distribution, and plots the CCDF heavy-tail diagnostic.
With no `--labels`/`--images` it runs on synthetic data so you can see the
output; `--images` is required whenever `--labels` is given (needed for `RS`).
`--save` writes the figure to PNG, `--compare` overlays multiple datasets.

```bash
python tools/analyze_size_distribution.py                              # synthetic demo
python tools/analyze_size_distribution.py --labels L --images I --save
```

#### `rename_images.py` — prepend a fixed prefix

Renames every image in a folder to `<prefix><original name>`. With no
`--prefix` it is a no-op pass over the folder.

```bash
python tools/rename_images.py path/to/images --prefix "cam1_"
```

#### `standardize_frame_numbers.py` — zero-pad the trailing frame number

Takes the digit block after the **last** underscore in each filename stem and
left-pads it with zeros to 6 characters, so frame numbers sort correctly:

| Before | After |
|---|---|
| `clip_40.jpg` | `clip_000040.jpg` |
| `cam1_20230101_300.jpg` | `cam1_20230101_000300.jpg` |
| `frame_000040.jpg` | `frame_000040.jpg` (already 6, unchanged) |
| `frame_1234567.jpg` | `frame_1234567.jpg` (already > 6, never truncated) |

Stems with no underscore or a non-numeric tail are left alone. Renames stop
if the target name already exists. `--labels DIR` renames the matching
`DIR/<stem>.txt` label in lockstep; `--dry-run` prints the plan without
touching anything.

```bash
python tools/standardize_frame_numbers.py path/to/images
python tools/standardize_frame_numbers.py path/to/images --labels path/to/labels
python tools/standardize_frame_numbers.py path/to/images --dry-run
```

#### `video_to_frames.py` — explode videos into frames

For every video in a folder (`.mp4 .avi .mov .mkv .flv .wmv`) it writes all
frames to a sibling subfolder named after the video, as
`<video>_<index:05d>.jpg`.

```bash
python tools/video_to_frames.py path/to/videos
```

#### `remove_mac_metadata.py` — strip `.DS_Store` and `._*`

Recursively lists the macOS junk files that break loaders (`._photo.jpg` has
no pixels) and inflate file counts, then asks before deleting. On Windows it
first clears the Hidden/System/read-only attributes and falls back to CMD's
`del` if needed. `--dry-run` only lists; `--yes` skips the confirmation.

```bash
python tools/remove_mac_metadata.py path/to/folder            # list, then ask
python tools/remove_mac_metadata.py path/to/folder --dry-run  # list only
python tools/remove_mac_metadata.py path/to/folder --yes      # delete, no prompt
```

## Installation

Developed and tested with **Python 3.13** (Anaconda). Set up the
environment manually, step by step:

1. **Create the Conda environment** with a pinned Python version:

   ```bash
   conda create -n yolo-toolkit python=3.13
   ```

2. **Activate it:**

   ```bash
   conda activate yolo-toolkit
   ```

3. **Install the dependencies** — either the major-version-capped ranges:

   ```bash
   pip install -r requirements.txt
   ```

   or the fully pinned versions for an exact, reproducible environment:

   ```bash
   pip install -r requirements.lock
   ```

4. **Verify the install:**

   ```bash
   python -c "import cv2, numpy, pandas, matplotlib, scipy, sklearn; print('OK')"
   ```

After changing `requirements.txt`, regenerate the lock file:

```bash
pip install -r requirements.txt && pip freeze > requirements.lock
```

`tkinter` (used for folder-picker dialogs) ships with standard CPython and
with Anaconda on Windows/macOS; on Linux install it via your package
manager (e.g. `sudo apt install python3-tk`).

## Usage

### Interactive launcher

```bash
python main.py
```

Shows a numbered menu of tasks; pick one and it prompts for whatever that
tool needs (dataset path, number of classes, etc.) before running it.

### Direct script / non-interactive

Every script accepts its path as a CLI argument, or falls back to a
folder-picker dialog when run with no arguments. `main.py` also accepts a
tool id followed by args passed straight through, for scripting:

```bash
python tools/clean_dataset.py path/to/dataset
python tools/video_to_frames.py path/to/videos
python main.py validate_labels path/to/dataset --num-classes 2
python main.py --list   # show all tool ids
```

See each script's module docstring for exact usage and keyboard controls.

### Removing macOS metadata (`.DS_Store`, `._*`)

Datasets copied from a Mac carry hidden `.DS_Store` and `._*` (AppleDouble)
files that can break loaders — a `._photo.jpg` has no pixels.

The tool always **scans and lists** every match first, then asks for
confirmation before deleting:

```bash
python main.py remove_mac_metadata "D:\Dataset\Cenidet-UAV\images"
#   Scanning : D:\Dataset\Cenidet-UAV\images
#   Found    : 3 macOS metadata file(s)
#     ...list...
#   Delete these 3 file(s)? [y/N]:      <- press Enter/n to keep them, y to delete
```

Flags:

```bash
python main.py remove_mac_metadata "D:\" --dry-run   # only list, never delete or ask
python main.py remove_mac_metadata "D:\" --yes       # delete without the prompt
```

It clears the Hidden/System/read-only attributes before deleting and re-scans
afterwards, printing `Verified: no macOS metadata files remain` or the files
still present.

#### Emergency fallback (Windows CMD)

If the script cannot be run, delete the files directly from **CMD** (`Win`+`R`
→ `cmd`). Switch to the target drive and run:

```bat
D:
del /s /f /q /a:h .DS_Store ._*
```

`/s` recurses into every sub-folder, `/f` forces read-only files, `/q` is
quiet (no per-file prompt), `/a:h` includes hidden files. This only removes
hidden entries; if some `._*` files are not hidden, repeat without `/a:h`.

## Expected dataset layout

```
dataset/
    images/   ← .jpg / .jpeg / .png / .bmp / ...
    labels/   ← YOLO .txt files (one per image, "class cx cy w h" normalized)
```

## Related projects

- [UAV Auto Labeler](https://github.com/pedroamtech/uav-auto-labeler) — semi-automatic pre-labeling tool that runs a YOLOv8 model fine-tuned on VisDrone over a folder of images and generates YOLO-format labels. A natural first step before cleaning/validating with this toolkit.

## License

MIT
