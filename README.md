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
| `tools/validate_labels.py` | Scans label files for issues that can crash training: out-of-range classes, malformed rows, NaN/out-of-bounds coordinates, orphan files. Writes a CSV report. |
| `tools/normalize_manipal_labels.py` | Clamps YOLO-normalized coordinates into `[0, 1]` and drops degenerate (zero-area) boxes. |
| `tools/yolo_person_labeler.py` | All-in-one labeling/editing/viewing tool — HOG auto-detection, manual box drawing, zoom/pan, click-to-select-and-delete, per-class color rendering. |
| `tools/analyze_size_distribution.py` | Object size distribution analysis (Absolute/Relative Size, log-normal fit, CCDF heavy-tail diagnostic) following the TinyPerson Benchmark methodology (Yu et al., 2019). Runs on synthetic data if no `--labels`/`--images` are given. |
| `tools/rename_images.py` | Batch-renames images in a folder with a fixed prefix. |
| `tools/video_to_frames.py` | Extracts every frame from all videos in a folder into per-video subfolders. |

`clean_labels.py` and `visualize_labels.py` were merged into
`yolo_person_labeler.py` (zoom/pan box editing + per-class color rendering
now live there) and are kept only as deprecated stubs pointing to it.

## Installation

```bash
pip install -r requirements.txt
```

`tkinter` (used for folder-picker dialogs) ships with standard CPython on
Windows/macOS; on Linux install it via your package manager
(e.g. `sudo apt install python3-tk`).

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
