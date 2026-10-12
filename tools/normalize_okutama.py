"""
tools/normalize_okutama.py — Convert Okutama-Action into a clean YOLO dataset

Okutama-Action ships frames per video and VATIC-style annotations in 4K
pixel space:

    <root>/TrainSetFrames/Drone{1,2}/{Morning,Noon}/Extracted-Frames-1280x720/<video>/<frame>.jpg
    <root>/TrainSetFrames/Labels/MultiActionLabels/3840x2160/<video>.txt
    (same layout under TestSetFrames)

Each annotation row is:
    track_id xmin ymin xmax ymax frame lost occluded generated "Person" "action"...

This script converts it to YOLO format (single class 0 = person):
    <output>/train/images/<video>_<frame>.jpg   <output>/train/labels/<video>_<frame>.txt
    <output>/test/images/...                    <output>/test/labels/...

Cleaning applied:
  - only 1 in 10 frames was annotated by hand; VATIC linearly interpolated
    the rest, so those boxes drift off the person when the drone turns fast.
    Interpolated boxes are rebuilt by estimating the camera motion (optical
    flow + homography) and warping the surrounding hand-annotated boxes into
    the frame. Frames where this fails are skipped (see --max-interp and
    --no-motion-comp)
  - skips frames outside any annotated track (never labeled), and frames
    right after/before a track that ends/starts while the person is still
    visible (that person would be in the image without a box)
  - drops rows flagged lost=1 (person outside the frame)
  - clamps boxes to the image and drops boxes smaller than --min-size px
  - drops near-duplicate boxes in the same frame (IoU >= --dedup-iou)
  - skips frames with no visible person unless --keep-empty is given

Usage:
    python tools/normalize_okutama.py                                  # folder dialog
    python tools/normalize_okutama.py "path/to/Okutama Action"         # output: <root>_YOLO
    python tools/normalize_okutama.py "path/to/Okutama Action" -o out --keep-empty
    python tools/normalize_okutama.py "path/to/Okutama Action" --max-interp 0   # keyframes only
"""

import argparse
import bisect
import os
import shutil
import sys
import tkinter as tk
from collections import Counter, defaultdict
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from tqdm import tqdm

SPLITS = {"TrainSetFrames": "train", "TestSetFrames": "test"}   # source -> output partition
TIME_OF_DAY = {"1": "Morning", "2": "Noon"}                     # 2nd field of the video id
LABELS_SUBDIR = Path("Labels") / "MultiActionLabels" / "3840x2160"
LABEL_W, LABEL_H = 3840, 2160                                   # annotation pixel space
FLOW_W, FLOW_H = 640, 360                                       # resolution for camera motion
FLOW_SCALE = LABEL_W / FLOW_W
MIN_FLOW_POINTS = 20
KEYFRAME_GAP = 10                                               # frames between hand annotations


def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the Okutama-Action root folder")
    root.destroy()
    return folder


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def frames_dir(split_dir: Path, video: str) -> Path:
    drone, tod = video.split(".")[:2]
    return split_dir / f"Drone{drone}" / TIME_OF_DAY[tod] / "Extracted-Frames-1280x720" / video


class CameraMotion:
    """Frame-to-frame homographies of a video, estimated from sparse optical flow.

    People are few and small, so RANSAC fits the ground plane and the
    transforms describe how the drone camera moved.
    """

    def __init__(self, src_dir: Path, n_frames: int):
        self.steps = [None] * n_frames          # steps[f] maps frame f-1 -> f
        prev = self._load(src_dir, 0)
        for f in tqdm(range(1, n_frames), desc="  camera motion", unit="frame", leave=False):
            cur = self._load(src_dir, f)
            if prev is not None and cur is not None:
                self.steps[f] = self._estimate(prev, cur)
            prev = cur

    @staticmethod
    def _load(src_dir: Path, frame: int):
        img = cv2.imread(str(src_dir / f"{frame}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return cv2.resize(img, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _estimate(g0, g1):
        pts = cv2.goodFeaturesToTrack(g0, 400, 0.01, 8)
        if pts is None or len(pts) < MIN_FLOW_POINTS:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=4)
        ok = status.ravel() == 1
        if ok.sum() < MIN_FLOW_POINTS:
            return None
        H, inliers = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
        if H is None or inliers.sum() < MIN_FLOW_POINTS:
            return None
        return H

    def between(self, src: int, dst: int):
        """Homography mapping frame src -> dst, or None if any step failed."""
        lo, hi = min(src, dst), max(src, dst)
        H = np.eye(3)
        for f in range(lo + 1, hi + 1):
            if f >= len(self.steps) or self.steps[f] is None:
                return None
            H = self.steps[f] @ H
        return H if src <= dst else np.linalg.inv(H)


def warp_box(H, box) -> np.ndarray:
    """Move a 4K box with a homography estimated at FLOW_W x FLOW_H, keeping it axis-aligned."""
    x1, y1, x2, y2 = (v / FLOW_SCALE for v in box)
    corners = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]).reshape(-1, 1, 2)
    c = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    cx, cy = c[:, 0].mean(), c[:, 1].mean()
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[3] - c[2])) / 2
    h = (np.linalg.norm(c[2] - c[0]) + np.linalg.norm(c[3] - c[1])) / 2
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * FLOW_SCALE


def parse_annotations(txt: Path, stats: Counter) -> dict:
    """Return {track: {frame: (box, lost, generated)}}."""
    tracks = defaultdict(dict)
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 9:
            stats["malformed"] += 1
            continue
        track, x1, y1, x2, y2, frame, lost, _, generated = (int(p) for p in parts[:9])
        tracks[track][frame] = (np.array([x1, y1, x2, y2], dtype=float), lost, generated)
    return tracks


def keyframes_of(tracks: dict) -> dict:
    return {t: sorted(f for f, (_, _, gen) in rows.items() if not gen) for t, rows in tracks.items()}


def open_track_frames(tracks: dict) -> set:
    """Frames where a person is likely visible but has no row.

    Some tracks start or end while the person is still in view (the annotator
    closed the track instead of following it). The frames between that visible
    endpoint and the neighbouring keyframe have no box for that person.
    """
    frames = set()
    for rows in tracks.values():
        first, last = min(rows), max(rows)
        if not rows[first][1]:
            frames.update(range(first - KEYFRAME_GAP + 1, first))
        if not rows[last][1]:
            frames.update(range(last + 1, last + KEYFRAME_GAP))
    return frames


def keyframe_distance(keyframes: list, frame: int) -> float:
    if not keyframes:
        return float("inf")
    i = bisect.bisect_left(keyframes, frame)
    near = keyframes[max(0, i - 1):i + 1]
    return min(abs(frame - k) for k in near)


def frame_boxes(frame: int, tracks: dict, keyframes: dict, motion):
    """Return (boxes, reliable, max keyframe distance) for one frame.

    Only ~1 in 10 frames was annotated by hand (generated=0); VATIC filled the
    rest by linear interpolation, which drifts off the person when the drone
    turns fast. With camera motion available, each interpolated box is rebuilt
    from the two surrounding keyframes warped into this frame and blended by
    distance. A frame is unreliable when that cannot be done for every track
    (camera motion lost, or the person enters/leaves the image between keyframes).
    """
    boxes, dist = [], 0
    for track, rows in tracks.items():
        if frame not in rows:
            continue
        box, lost, generated = rows[frame]
        kf = keyframes[track]
        dist = max(dist, keyframe_distance(kf, frame))

        if not generated or motion is None:
            if not lost:
                boxes.append(box)
            continue

        i = bisect.bisect_left(kf, frame)
        if i == 0 or i == len(kf):
            return [], False, dist
        a, b = kf[i - 1], kf[i]
        box_a, lost_a, _ = rows[a]
        box_b, lost_b, _ = rows[b]
        if lost_a and lost_b:
            continue
        if lost_a != lost_b:
            return [], False, dist
        H_a, H_b = motion.between(a, frame), motion.between(b, frame)
        if H_a is None or H_b is None:
            return [], False, dist
        t = (frame - a) / (b - a)
        boxes.append((1 - t) * warp_box(H_a, box_a) + t * warp_box(H_b, box_b))
    return boxes, True, dist


def clean_boxes(boxes: list, min_size: float, stats: Counter) -> list:
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, min(LABEL_W, x1)), max(0.0, min(LABEL_W, x2))
        y1, y2 = max(0.0, min(LABEL_H, y1)), max(0.0, min(LABEL_H, y2))
        if x2 - x1 < min_size or y2 - y1 < min_size:
            stats["too_small"] += 1
            continue
        out.append((x1, y1, x2, y2))
    return out


def dedup(boxes: list, thr: float) -> list:
    if thr <= 0:
        return boxes
    kept = []
    for b in boxes:
        if all(iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept


def to_yolo(box) -> str:
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 / LABEL_W
    yc = (y1 + y2) / 2 / LABEL_H
    w = (x2 - x1) / LABEL_W
    h = (y2 - y1) / LABEL_H
    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def convert_split(split_dir: Path, out_dir: Path, args, stats: Counter):
    label_dir = split_dir / LABELS_SUBDIR
    if not label_dir.exists():
        print(f"[WARN] Missing {label_dir}, skipping.")
        return

    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    transfer = shutil.move if args.move else shutil.copy2

    for txt in tqdm(sorted(label_dir.glob("*.txt")), desc=out_dir.name, unit="video"):
        video = txt.stem
        src_dir = frames_dir(split_dir, video)
        if not src_dir.exists():
            print(f"[WARN] No frames folder for video {video}: {src_dir}")
            stats["missing_video"] += 1
            continue

        tracks = parse_annotations(txt, stats)
        keyframes = keyframes_of(tracks)
        annotated = {f for rows in tracks.values() for f in rows}
        open_frames = open_track_frames(tracks)
        frames = sorted(int(p.stem) for p in src_dir.glob("*.jpg") if p.stem.isdigit())

        motion = None
        if not args.no_motion_comp and args.max_interp != 0 and frames:
            motion = CameraMotion(src_dir, frames[-1] + 1)

        for frame in frames[::args.step]:
            if frame not in annotated:
                stats["skipped_unannotated"] += 1
                continue
            raw, reliable, dist = frame_boxes(frame, tracks, keyframes, motion)
            if args.max_interp >= 0 and dist > args.max_interp:
                stats["skipped_interpolated"] += 1
                continue
            if not reliable or frame in open_frames:
                stats["skipped_unreliable"] += 1
                continue
            cleaned = clean_boxes(raw, args.min_size, stats)
            kept = dedup(cleaned, args.dedup_iou)
            stats["duplicates"] += len(cleaned) - len(kept)
            if not kept and not args.keep_empty:
                stats["skipped_empty"] += 1
                continue

            name = f"{video.replace('.', '_')}_{frame:06d}"
            transfer(src_dir / f"{frame}.jpg", img_out / f"{name}.jpg")
            lines = [to_yolo(b) for b in kept]
            (lbl_out / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

            stats[f"{out_dir.name}_images"] += 1
            stats["boxes"] += len(lines)
            if not lines:
                stats["empty_kept"] += 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default=None,
                   help="Okutama-Action root (contains TrainSetFrames/ and TestSetFrames/). "
                        "Falls back to a folder-picker dialog if omitted.")
    p.add_argument("-o", "--output", default=None,
                   help="Output folder (default: '<dataset>_YOLO' next to the dataset).")
    p.add_argument("--max-interp", type=int, default=-1,
                   help="Keep a frame only if every box is within N frames of a hand-annotated "
                        "keyframe (keyframes are every 10 frames). 0 = keyframes only, "
                        "-1 = all frames. Default: -1.")
    p.add_argument("--no-motion-comp", action="store_true",
                   help="Keep VATIC's linear interpolation instead of rebuilding interpolated "
                        "boxes with camera motion (faster, but boxes drift during fast turns).")
    p.add_argument("--step", type=int, default=1,
                   help="Keep one of every N frames (consecutive frames are near-identical). Default: 1.")
    p.add_argument("--min-size", type=float, default=3,
                   help="Minimum box width/height in 3840x2160 pixels. Default: 3.")
    p.add_argument("--dedup-iou", type=float, default=0.9,
                   help="Drop boxes overlapping a kept box in the same frame with IoU >= this. "
                        "0 disables. Default: 0.9.")
    p.add_argument("--keep-empty", action="store_true",
                   help="Keep frames with no visible person as negatives (empty .txt).")
    p.add_argument("--move", action="store_true",
                   help="Move images instead of copying them (modifies the source dataset).")
    args = p.parse_args()

    if args.step < 1:
        p.error("--step must be >= 1")

    if args.dataset:
        base_dir = args.dataset
        if not os.path.isdir(base_dir):
            print(f"Error: '{base_dir}' is not a valid folder.")
            sys.exit(1)
    else:
        print("Select the Okutama-Action root folder in the pop-up window...")
        base_dir = pick_folder()
        if not base_dir:
            print("No folder selected.")
            sys.exit(0)

    base_dir = Path(base_dir).resolve()
    out_root = Path(args.output) if args.output else base_dir.with_name(base_dir.name + "_YOLO")
    print(f"Output: {out_root}")

    stats = Counter()
    for src_name, part in SPLITS.items():
        split_dir = base_dir / src_name
        if split_dir.exists():
            convert_split(split_dir, out_root / part, args, stats)
        else:
            print(f"[WARN] Missing {split_dir}, skipping.")

    (out_root / "data.yaml").write_text(
        f"path: {out_root.as_posix()}\n"
        "train: train/images\n"
        "val: test/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['person']\n")

    print("\nDone")
    print(f"  train images:          {stats['train_images']}")
    print(f"  test images:           {stats['test_images']}")
    print(f"  boxes written:         {stats['boxes']}")
    print(f"  empty frames kept:     {stats['empty_kept']}")
    print(f"  empty frames skipped:  {stats['skipped_empty']}")
    print(f"  interpolated skipped:  {stats['skipped_interpolated']}")
    print(f"  unannotated skipped:   {stats['skipped_unannotated']}")
    print(f"  unreliable skipped:    {stats['skipped_unreliable']}")
    print(f"  small boxes dropped:   {stats['too_small']}")
    print(f"  duplicate boxes:       {stats['duplicates']}")
    print(f"  malformed rows:        {stats['malformed']}")
    print(f"  missing video folders: {stats['missing_video']}")


if __name__ == "__main__":
    main()
