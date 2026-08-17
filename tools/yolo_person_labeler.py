"""
tools/yolo_person_labeler.py — Semi-automatic YOLO person labeler & editor

All-in-one labeling tool. Merges what used to be three separate scripts:
    - HOG-based automatic person detection (this script, original)
    - Zoom/pan viewport + click-to-select-box deletion (former clean_labels.py)
    - Per-class color-coded box rendering (former visualize_labels.py)

Controls:
  Mouse
    Scroll wheel / trackpad — zoom in / out (centered on cursor)
    Right click + drag      — pan (when zoomed in)
    Left click               — select / deselect an existing box (red = marked for deletion)
    Left click + drag        — draw a new box (class 0 = person)

  Keyboard
    + / =   zoom in   (centered on cursor)
    -       zoom out  (centered on cursor)
    Space   auto-detect persons in current image (HOG) — adds boxes
    B       batch auto-detect + save across ALL images in the dataset
    S       delete selected (red) boxes, save remaining, next image
    D       advance without saving
    A       go back to previous image (no save)
    Z       undo last added box
    C       clear all boxes on current image (in-memory only — press S to persist)
    R       show / hide boxes
    G       open "go to image #" field
    H       print controls
    Q       quit

Usage:
    python tools/yolo_person_labeler.py                      # folder dialog
    python tools/yolo_person_labeler.py path/to/dataset      # CLI path
"""

import argparse
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
from tqdm import tqdm

WINDOW_NAME = "YOLO Person Labeler"

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
NEW_CLS = 0   # class written for manually-drawn and auto-detected boxes

WIN_W, WIN_H   = 1280, 720
ZOOM_STEP      = 1.25
MIN_ZOOM       = 0.2
MAX_ZOOM       = 12.0
DRAG_THRESHOLD = 6

# One color per class (BGR). Cycles if more classes than colors.
PALETTE = [
    (0,   255,  57),   # class 0 — green   (person)
    (255,  57,  57),   # class 1 — red
    (57,  138, 255),   # class 2 — blue
    (255, 200,   0),   # class 3 — yellow
    (200,   0, 255),   # class 4 — purple
    (0,   220, 255),   # class 5 — cyan
    (255, 128,   0),   # class 6 — orange
]
CLASS_NAMES: dict[int, str] = {0: 'person'}

COLOR_SELECTED = (0,   0, 255)    # red   — box marked for deletion
COLOR_NEW      = (0, 165, 255)    # orange — box being drawn

# HOG detector (classic pedestrian detector — good enough for a first pass,
# weak on tiny/aerial persons; always review its output before trusting it).
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# ─── HOG auto-detection ───────────────────────────────────────────────────────
def detect_persons_hog(img, min_area=1500, min_confidence=0.8):
    """Returns a list of (x1, y1, x2, y2) pixel boxes likely containing a person."""
    rects, weights = _HOG.detectMultiScale(
        img, winStride=(8, 8), padding=(16, 16),
        scale=1.05, hitThreshold=0.5, groupThreshold=2,
    )
    boxes = []
    for i, (x, y, w, h) in enumerate(rects):
        if i >= len(weights):
            continue
        confidence  = weights[i]
        area        = w * h
        aspect_ratio = h / w if w > 0 else 0
        if confidence > min_confidence and area > min_area and 1.2 < aspect_ratio < 4.0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def batch_auto_detect(images: list[Path], labels_dir: Path) -> int:
    """Runs HOG detection over every image and appends+saves the results. Returns total boxes added."""
    total_added = 0
    for img_path in tqdm(images, desc="Auto-detecting"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = load_boxes(label_path, iw, ih)
        detections = detect_persons_hog(img)
        for x1, y1, x2, y2 in detections:
            boxes.append([NEW_CLS, x1, y1, x2, y2])
        save_boxes(label_path, boxes, iw, ih)
        total_added += len(detections)
    return total_added


# ─── Coordinate helpers (zoom/pan viewport) ───────────────────────────────────
def view_size(zoom, img_w, img_h):
    return min(img_w, int(WIN_W / zoom)), min(img_h, int(WIN_H / zoom))


def clamp_pan(px, py, zoom, img_w, img_h):
    vw, vh = view_size(zoom, img_w, img_h)
    return max(0, min(img_w - vw, px)), max(0, min(img_h - vh, py))


def s2i(sx, sy, state):
    """Screen (window) coords → image (pixel) coords."""
    vw, vh = view_size(state["zoom"], state["iw"], state["ih"])
    return (int(sx / WIN_W * vw + state["px"]),
            int(sy / WIN_H * vh + state["py"]))


def i2s(ix, iy, state):
    """Image (pixel) coords → screen (window) coords."""
    vw, vh = view_size(state["zoom"], state["iw"], state["ih"])
    return (int((ix - state["px"]) / vw * WIN_W),
            int((iy - state["py"]) / vh * WIN_H))


def do_zoom(state, factor, sx, sy):
    new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, state["zoom"] * factor))
    vw,  vh  = view_size(state["zoom"], state["iw"], state["ih"])
    img_cx   = sx / WIN_W * vw + state["px"]
    img_cy   = sy / WIN_H * vh + state["py"]
    nvw, nvh = view_size(new_zoom, state["iw"], state["ih"])
    state["px"] = int(img_cx - sx / WIN_W * nvw)
    state["py"] = int(img_cy - sy / WIN_H * nvh)
    state["px"], state["py"] = clamp_pan(
        state["px"], state["py"], new_zoom, state["iw"], state["ih"])
    state["zoom"] = new_zoom


# ─── Render ───────────────────────────────────────────────────────────────────
def render(img, boxes, selected, state, preview=None):
    vw, vh = view_size(state["zoom"], state["iw"], state["ih"])
    crop = img[state["py"]:state["py"] + vh, state["px"]:state["px"] + vw]
    out  = cv2.resize(crop, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

    if state["show_boxes"]:
        for i, (cls, x1, y1, x2, y2) in enumerate(boxes):
            sx1, sy1 = i2s(x1, y1, state)
            sx2, sy2 = i2s(x2, y2, state)
            color = COLOR_SELECTED if i in selected else PALETTE[cls % len(PALETTE)]
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), color, 2, cv2.LINE_AA)
            label = f"{i}:{CLASS_NAMES.get(cls, str(cls))}"
            cv2.putText(out, label, (sx1 + 2, max(12, sy1 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if preview:
        cv2.rectangle(out, preview[:2], preview[2:], COLOR_NEW, 2, cv2.LINE_AA)

    cv2.putText(out, f"{state['zoom']:.1f}x", (8, WIN_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if state["input_mode"]:
        prompt = f"Go to image (1-{state['n_images']}):  {state['input_text']}_"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        (tw, th), _ = cv2.getTextSize(prompt, font, scale, thick)
        bx = WIN_W // 2 - tw // 2 - 16
        by = WIN_H // 2 - th // 2 - 16
        overlay = out.copy()
        cv2.rectangle(overlay, (bx, by), (bx + tw + 32, by + th + 32), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
        cv2.putText(out, prompt, (bx + 16, by + th + 12),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    return out


# ─── Label I/O ────────────────────────────────────────────────────────────────
def load_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = int(float(parts[0])), *map(float, parts[1:])
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        boxes.append([cls, x1, y1, x2, y2])
    return boxes


def save_boxes(label_path: Path, boxes: list, img_w: int, img_h: int) -> None:
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w  = abs(x2 - x1) / img_w
        h  = abs(y2 - y1) / img_h
        if w > 0 and h > 0:
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ─── Mouse callback ───────────────────────────────────────────────────────────
def make_mouse_cb(state):
    def cb(event, x, y, flags, _):
        state["mx"], state["my"] = x, y

        if event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            if flags == 0:
                return
            factor = ZOOM_STEP if flags > 0 else 1 / ZOOM_STEP
            do_zoom(state, factor, x, y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            state["panning"]  = True
            state["pan_s0"]   = (x, y)
            state["pan_off0"] = (state["px"], state["py"])

        elif event == cv2.EVENT_MOUSEMOVE and state["panning"]:
            vw, vh = view_size(state["zoom"], state["iw"], state["ih"])
            dx = int((state["pan_s0"][0] - x) / WIN_W * vw)
            dy = int((state["pan_s0"][1] - y) / WIN_H * vh)
            state["px"] = state["pan_off0"][0] + dx
            state["py"] = state["pan_off0"][1] + dy
            state["px"], state["py"] = clamp_pan(
                state["px"], state["py"], state["zoom"], state["iw"], state["ih"])

        elif event == cv2.EVENT_RBUTTONUP:
            state["panning"] = False

        elif event == cv2.EVENT_LBUTTONDOWN:
            if state["input_mode"]:
                return
            state["down"]  = True
            state["start"] = (x, y)
            state["cur"]   = (x, y)
            state["drag"]  = False

        elif event == cv2.EVENT_MOUSEMOVE and state["down"]:
            state["cur"] = (x, y)
            if (abs(x - state["start"][0]) > DRAG_THRESHOLD or
                    abs(y - state["start"][1]) > DRAG_THRESHOLD):
                state["drag"] = True

        elif event == cv2.EVENT_LBUTTONUP:
            state["down"] = False
            if state["drag"]:
                ix1, iy1 = s2i(min(state["start"][0], x), min(state["start"][1], y), state)
                ix2, iy2 = s2i(max(state["start"][0], x), max(state["start"][1], y), state)
                if (ix2 - ix1) > 2 and (iy2 - iy1) > 2:
                    state["boxes"].append([NEW_CLS, ix1, iy1, ix2, iy2])
                state["drag"] = False
            else:
                cx, cy = s2i(state["start"][0], state["start"][1], state)
                for i, (_, x1, y1, x2, y2) in enumerate(state["boxes"]):
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        if i in state["selected"]:
                            state["selected"].discard(i)
                        else:
                            state["selected"].add(i)
                        break
    return cb


# ─── Dataset resolution ────────────────────────────────────────────────────────
def pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the dataset folder")
    root.destroy()
    return folder


def resolve_images_labels(dataset_dir: str) -> tuple[str, str]:
    """
    Returns (images_dir, labels_dir). Accepts:
      1. dataset/images/  + dataset/labels/
      2. dataset/train/images/ + dataset/train/labels/  (auto-detects the partition)
    """
    direct = os.path.join(dataset_dir, 'images')
    if os.path.isdir(direct):
        return direct, os.path.join(dataset_dir, 'labels')

    for sub in sorted(os.listdir(dataset_dir)):
        sub_path = os.path.join(dataset_dir, sub)
        if os.path.isdir(sub_path) and os.path.isdir(os.path.join(sub_path, 'images')):
            print(f"[INFO] Auto-detected partition: {sub}/")
            return (os.path.join(sub_path, 'images'),
                    os.path.join(sub_path, 'labels'))

    return direct, os.path.join(dataset_dir, 'labels')   # fallback — error reported by caller


def print_controls():
    print(__doc__.split("Controls:\n")[1].split("Usage:")[0].strip())


# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default=None,
                   help="Dataset folder (with images/ and labels/, or train/val/test "
                        "partitions). Falls back to a folder-picker dialog if omitted.")
    args = p.parse_args()

    if args.dataset:
        dataset_dir = args.dataset
        if not os.path.isdir(dataset_dir):
            print(f"Error: '{dataset_dir}' is not a valid folder.")
            raise SystemExit(1)
    else:
        print("Select the dataset folder in the pop-up window...")
        dataset_dir = pick_folder()
        if not dataset_dir:
            print("No folder selected.")
            raise SystemExit(0)

    images_dir_s, labels_dir_s = resolve_images_labels(dataset_dir)
    images_dir, labels_dir = Path(images_dir_s), Path(labels_dir_s)

    if not images_dir.is_dir():
        print(f"[ERROR] 'images' folder not found in: {dataset_dir}")
        raise SystemExit(1)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in: {images_dir}")
        raise SystemExit(0)

    print(f"Found {len(images)} images in {images_dir}")
    print_controls()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)

    i = 0
    while i < len(images):
        img_path   = images[i]
        label_path = labels_dir / (img_path.stem + ".txt")
        img        = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read {img_path.name}, skipping")
            i += 1
            continue
        ih, iw = img.shape[:2]

        state = {
            "boxes":    load_boxes(label_path, iw, ih),
            "selected": set(),
            "iw": iw, "ih": ih,
            "zoom": 1.0, "px": 0, "py": 0,
            "down": False, "drag": False, "start": (0, 0), "cur": (0, 0),
            "panning": False, "pan_s0": (0, 0), "pan_off0": (0, 0),
            "mx": WIN_W // 2, "my": WIN_H // 2,
            "input_mode": False, "input_text": "",
            "n_images": len(images),
            "show_boxes": True,
        }
        cv2.setMouseCallback(WINDOW_NAME, make_mouse_cb(state))

        while True:
            preview = (state["start"][0], state["start"][1],
                       state["cur"][0],   state["cur"][1]) if state["drag"] else None
            frame = render(img, state["boxes"], state["selected"], state, preview)

            title = (f"[{i+1}/{len(images)}] {img_path.name}  |  "
                     f"boxes:{len(state['boxes'])}  sel:{len(state['selected'])}  |  "
                     f"scroll/+- zoom  RMB pan  Space detect  B batch  S save  D next  "
                     f"A back  Z undo  C clear  R hide  G goto  Q quit")
            cv2.setWindowTitle(WINDOW_NAME, title)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(20) & 0xFF

            if state["input_mode"]:
                if key == 13:                                # Enter — jump
                    if state["input_text"]:
                        n = int(state["input_text"]) - 1
                        if 0 <= n < len(images):
                            i = n
                            state["input_mode"] = False
                            state["input_text"] = ""
                            break
                    state["input_mode"] = False
                    state["input_text"] = ""
                elif key == 27:                              # Escape — cancel
                    state["input_mode"] = False
                    state["input_text"] = ""
                elif key == 8 and state["input_text"]:       # Backspace
                    state["input_text"] = state["input_text"][:-1]
                elif 48 <= key <= 57:                         # 0-9
                    state["input_text"] += chr(key)
            else:
                if key in (ord("+"), ord("=")):
                    do_zoom(state, ZOOM_STEP, state["mx"], state["my"])
                elif key == ord("-"):
                    do_zoom(state, 1 / ZOOM_STEP, state["mx"], state["my"])
                elif key == ord("g"):
                    state["input_mode"] = True
                    state["input_text"] = ""
                elif key == ord(" "):                         # auto-detect current image
                    detections = detect_persons_hog(img)
                    for x1, y1, x2, y2 in detections:
                        state["boxes"].append([NEW_CLS, x1, y1, x2, y2])
                    print(f"Auto-detected {len(detections)} person(s) in {img_path.name}")
                elif key == ord("b"):                         # batch auto-detect + save all
                    print(f"\nBatch auto-detecting across {len(images)} images...")
                    added = batch_auto_detect(images, labels_dir)
                    print(f"Batch done — {added} box(es) added.\n")
                    state["boxes"]    = load_boxes(label_path, iw, ih)
                    state["selected"] = set()
                elif key == ord("c"):                         # clear all (in-memory)
                    state["boxes"]    = []
                    state["selected"] = set()
                elif key == ord("r"):                         # show/hide boxes
                    state["show_boxes"] = not state["show_boxes"]
                elif key == ord("h"):
                    print_controls()
                elif key == ord("s"):                         # save + next
                    state["boxes"] = [b for j, b in enumerate(state["boxes"])
                                       if j not in state["selected"]]
                    save_boxes(label_path, state["boxes"], iw, ih)
                    print(f"Saved {img_path.name} — {len(state['boxes'])} boxes")
                    i += 1; break
                elif key == ord("d"):                         # advance without saving
                    i += 1; break
                elif key == ord("a") and i > 0:               # back without saving
                    i -= 1; break
                elif key == ord("z") and state["boxes"]:
                    state["boxes"].pop()
                    state["selected"].discard(len(state["boxes"]))
                elif key == ord("q"):
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
