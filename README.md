# YOLO Dataset Toolkit

Standalone utilities for preparing, cleaning, validating, and inspecting
YOLO-format object detection datasets (aerial/UAV person-detection focused,
but generic to any single- or multi-class YOLO dataset).

Extracted from the [aerial-person-aug](https://github.com/pedroamtech/aerial-person-aug) project's
internal `tools/` folder.

## Tools

| Script | Purpose |
|---|---|
| `tools/clean_dataset.py` | Removes images with no person (class 0) annotations. Rejected files are moved to `_removed/` (or deleted with `--delete`). |
| `tools/clean_labels.py` | Interactive bounding-box editor (OpenCV GUI) — zoom, pan, draw, select/delete boxes, save. |
| `tools/visualize_labels.py` | Interactive YOLO label viewer — draws boxes per class and lets you page through a dataset. |
| `tools/validate_labels.py` | Scans label files for issues that can crash training: out-of-range classes, malformed rows, NaN/out-of-bounds coordinates, orphan files. Writes a CSV report. |
| `tools/normalize_manipal_labels.py` | Clamps YOLO-normalized coordinates into `[0, 1]` and drops degenerate (zero-area) boxes. |
| `tools/yolo_person_labeler.py` | Semi-automatic labeler — HOG person detector for auto-suggestions plus manual box drawing/editing. |
| `tools/analyze_size_distribution.py` | Object size distribution analysis (Absolute/Relative Size, log-normal fit, CCDF heavy-tail diagnostic) following the TinyPerson Benchmark methodology (Yu et al., 2019). |
| `tools/rename_images.py` | Batch-renames images in a folder with a fixed prefix. |
| `tools/video_to_frames.py` | Extracts every frame from all videos in a folder into per-video subfolders. |

## Installation

```bash
pip install -r requirements.txt
```

`tkinter` (used for folder-picker dialogs) ships with standard CPython on
Windows/macOS; on Linux install it via your package manager
(e.g. `sudo apt install python3-tk`).

## Usage

Most scripts accept a dataset path as a CLI argument, or fall back to a
folder-picker dialog when run with no arguments:

```bash
python tools/clean_dataset.py path/to/dataset
python tools/visualize_labels.py path/to/dataset
python tools/video_to_frames.py path/to/videos
```

A few scripts (`clean_labels.py`, `normalize_manipal_labels.py`,
`rename_images.py`, `validate_labels.py`) configure their input paths via
constants at the top of the file — edit those before running. See each
script's module docstring for exact usage and keyboard controls.

## Expected dataset layout

```
dataset/
    images/   ← .jpg / .jpeg / .png / .bmp / ...
    labels/   ← YOLO .txt files (one per image, "class cx cy w h" normalized)
```

## Related projects

- [uav-auto-labeler](https://github.com/pedroamtech/uav-auto-labeler) — semi-automatic pre-labeling tool that runs a YOLOv8 model fine-tuned on VisDrone over a folder of images and generates YOLO-format labels. A natural first step before cleaning/validating with this toolkit.

## License

MIT
