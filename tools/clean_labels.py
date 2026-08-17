"""
tools/clean_labels.py — DEPRECATED

Merged into tools/yolo_person_labeler.py, which now includes this script's
zoom/pan viewport and click-to-select-box deletion, plus HOG auto-detection
and multi-class visualization.

Use tools/yolo_person_labeler.py instead. This file is kept as a stub
pointing to the new location; delete it once you've confirmed the merge
covers your workflow.
"""

if __name__ == "__main__":
    raise SystemExit(
        "clean_labels.py was merged into tools/yolo_person_labeler.py.\n"
        "Run: python tools/yolo_person_labeler.py <dataset_dir>"
    )
