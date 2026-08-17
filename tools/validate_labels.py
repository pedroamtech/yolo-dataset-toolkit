"""
tools/validate_labels.py — YOLO label validator

Scans every label .txt file under a dataset and flags anything that can
crash training: out-of-range class indices, malformed rows, non-finite/
out-of-bounds coordinates, and orphan label/image files.

Root cause this was built for: EdgeYOLO crashed with a CUDA assert inside
F.one_hot(gt_classes, num_classes=N) at a deterministic iter/epoch — some
label line held a class id >= N (or negative), which only blew up once
mosaic augmentation combined it into that specific batch.

Usage:
    python tools/validate_labels.py path/to/dataset --num-classes 2
    python tools/validate_labels.py path/to/dataset --num-classes 2 --report out.csv
    python tools/validate_labels.py path/to/dataset --num-classes 2 --no-report
"""

import argparse
import csv
from pathlib import Path


def find_label_dirs(root: Path) -> list[Path]:
    """Every folder literally named 'labels' under root (e.g. train/labels, validation/labels)."""
    if root.name == "labels":
        return [root]
    return sorted(d for d in root.rglob("labels") if d.is_dir())


def _parse_class_id(cls_raw: str) -> tuple[int | None, str | None]:
    """Returns (class_id, error). Tolerates '1.0'-style floats but rejects '1.5'."""
    try:
        return int(cls_raw), None
    except ValueError:
        pass
    try:
        cls_f = float(cls_raw)
    except ValueError:
        return None, f"class id '{cls_raw}' is not a number"
    if not cls_f.is_integer():
        return None, f"class id '{cls_raw}' is not a whole number"
    return int(cls_f), None


def validate_file(label_path: Path, num_classes: int) -> list[dict]:
    issues = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as e:
        return [{"file": str(label_path), "line": 0, "content": "", "reason": f"encoding error: {e}"}]

    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            issues.append({"file": str(label_path), "line": i, "content": line,
                            "reason": f"expected 5 fields, got {len(parts)}"})
            continue

        cls_raw, cx_raw, cy_raw, w_raw, h_raw = parts

        cls, err = _parse_class_id(cls_raw)
        if err:
            issues.append({"file": str(label_path), "line": i, "content": line, "reason": err})
            continue
        if cls < 0 or cls >= num_classes:
            issues.append({"file": str(label_path), "line": i, "content": line,
                            "reason": f"class id {cls} out of range [0, {num_classes - 1}]"})

        try:
            cx, cy, w, h = (float(v) for v in (cx_raw, cy_raw, w_raw, h_raw))
        except ValueError:
            issues.append({"file": str(label_path), "line": i, "content": line,
                            "reason": "non-numeric coordinate"})
            continue

        for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if v != v or v in (float("inf"), float("-inf")):
                issues.append({"file": str(label_path), "line": i, "content": line,
                                "reason": f"{name} is NaN/Inf"})

        if w <= 0 or h <= 0:
            issues.append({"file": str(label_path), "line": i, "content": line,
                            "reason": f"non-positive box size (w={w}, h={h})"})

        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            issues.append({"file": str(label_path), "line": i, "content": line,
                            "reason": f"center out of [0,1] range (cx={cx}, cy={cy})"})

        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
            issues.append({"file": str(label_path), "line": i, "content": line,
                            "reason": f"box extends outside image bounds "
                                      f"(x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f})"})

    return issues


def check_orphans(labels_dir: Path) -> list[dict]:
    """Flags label files with no matching image file in the sibling images/ folder."""
    images_dir = labels_dir.parent / "images"
    if not images_dir.is_dir():
        return []
    img_stems = {p.stem for p in images_dir.iterdir() if p.is_file()}
    return [
        {"file": str(lf), "line": 0, "content": "", "reason": "label has no matching image"}
        for lf in labels_dir.glob("*.txt")
        if lf.name != "classes.txt" and lf.stem not in img_stems
    ]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path,
                   help="Dataset root — scans every 'labels' folder found under it.")
    p.add_argument("--num-classes", type=int, required=True,
                   help="Number of classes; class ids must fall in [0, num_classes-1].")
    p.add_argument("--report", type=Path, default=Path("bad_labels.csv"),
                   help="Where to write the CSV issue report (default: bad_labels.csv).")
    p.add_argument("--no-report", action="store_true",
                   help="Skip writing the CSV report.")
    args = p.parse_args()

    root = args.dataset
    num_classes = args.num_classes
    report = None if args.no_report else args.report

    if not root.exists():
        print(f"[ERROR] path not found: {root}")
        return

    label_dirs = [d for d in find_label_dirs(root) if d.is_dir()]
    if not label_dirs:
        print(f"[ERROR] no 'labels' folders found under: {root}")
        return

    all_issues: list[dict] = []
    total_files = 0
    total_boxes = 0
    class_counts: dict[int, int] = {}

    for labels_dir in label_dirs:
        print(f"\nScanning {labels_dir} ...")
        for lf in sorted(labels_dir.glob("*.txt")):
            if lf.name == "classes.txt":
                continue
            total_files += 1
            all_issues.extend(validate_file(lf, num_classes))

            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, err = _parse_class_id(parts[0])
                    if not err:
                        total_boxes += 1
                        class_counts[cls] = class_counts.get(cls, 0) + 1

        all_issues.extend(check_orphans(labels_dir))

    print("\n" + "=" * 70)
    print("  LABEL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Label folders scanned : {len(label_dirs)}")
    print(f"  Label files scanned   : {total_files}")
    print(f"  Boxes parsed          : {total_boxes}")
    print(f"  Class distribution    : {dict(sorted(class_counts.items()))}")
    print(f"  Issues found          : {len(all_issues)}")
    print("=" * 70)

    if all_issues:
        print("\nFirst 30 issues:")
        for issue in all_issues[:30]:
            print(f"  {issue['file']}:{issue['line']}  [{issue['reason']}]  {issue['content']}")
        if len(all_issues) > 30:
            print(f"  ... and {len(all_issues) - 30} more")

    if report:
        with open(report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "line", "content", "reason"])
            writer.writeheader()
            writer.writerows(all_issues)
        print(f"\n[Report saved] -> {report}")


if __name__ == "__main__":
    main()
