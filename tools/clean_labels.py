"""
clean_labels.py — Interactive bounding-box editor for aerial person labels.

Controls:
  Scroll wheel / trackpad — zoom in / out (centered on cursor)
  + / =                   — zoom in  (keyboard, centered on cursor)
  -                       — zoom out (keyboard, centered on cursor)
  Right click + drag      — pan when zoomed in
  Left click              — select / deselect an existing box (turns red)
  Left click + drag       — draw a new bounding box (class 0 = person)
  S                       — save changes (deletes red boxes, keeps new), next image
  D                       — advance without saving
  A                       — go back to previous image (no save)
  Z                       — undo last drawn box
  G                       — open "go to image #" field
  Q                       — quit
"""

import cv2
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_DIR   = Path("/Volumes/KINGSTON/Dataset/Okutama-Action/train/images")
LABEL_DIR = Path("/Volumes/KINGSTON/Dataset/Okutama-Action/train/labels")
NEW_CLS   = 0

WIN_W, WIN_H   = 1280, 720
ZOOM_STEP      = 1.25
MIN_ZOOM       = 0.2
MAX_ZOOM       = 12.0
DRAG_THRESHOLD = 6

COLOR_OK  = (0, 255,   0)
COLOR_DEL = (0,   0, 255)
COLOR_NEW = (0, 165, 255)

# ─── Coordinate helpers ───────────────────────────────────────────────────────
def view_size(zoom, img_w, img_h):
    return min(img_w, int(WIN_W / zoom)), min(img_h, int(WIN_H / zoom))

def clamp_pan(px, py, zoom, img_w, img_h):
    vw, vh = view_size(zoom, img_w, img_h)
    return max(0, min(img_w - vw, px)), max(0, min(img_h - vh, py))

def s2i(sx, sy, state):
    vw, vh = view_size(state["zoom"], state["iw"], state["ih"])
    return (int(sx / WIN_W * vw + state["px"]),
            int(sy / WIN_H * vh + state["py"]))

def i2s(ix, iy, state):
    vw, vh = view_size(state["zoom"], state["iw"], state["ih"])
    return (int((ix - state["px"]) / vw * WIN_W),
            int((iy - state["py"]) / vh * WIN_H))

# ─── Render ───────────────────────────────────────────────────────────────────
def render(img, boxes, selected, state, preview=None):
    vw, vh = view_size(state["zoom"], state["iw"], state["ih"])
    crop = img[state["py"]:state["py"]+vh, state["px"]:state["px"]+vw]
    out  = cv2.resize(crop, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

    for i, (_, x1, y1, x2, y2) in enumerate(boxes):
        sx1, sy1 = i2s(x1, y1, state)
        sx2, sy2 = i2s(x2, y2, state)
        color = COLOR_DEL if i in selected else COLOR_OK
        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), color, 2, cv2.LINE_AA)
        cv2.putText(out, str(i), (sx1 + 2, sy1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if preview:
        cv2.rectangle(out, preview[:2], preview[2:], COLOR_NEW, 2, cv2.LINE_AA)

    cv2.putText(out, f"{state['zoom']:.1f}x", (8, WIN_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if state["input_mode"]:
        prompt = f"Ir a imagen (1-{state['n_images']}):  {state['input_text']}_"
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

# ─── I/O ──────────────────────────────────────────────────────────────────────
def load_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        boxes.append([cls, x1, y1, x2, y2])
    return boxes

def save_boxes(label_path, boxes, img_w, img_h):
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w  = abs(x2 - x1) / img_w
        h  = abs(y2 - y1) / img_h
        if w > 0 and h > 0:
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

# ─── Zoom helper ──────────────────────────────────────────────────────────────
def do_zoom(state, factor, sx, sy):
    new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, state["zoom"] * factor))
    vw,  vh  = view_size(state["zoom"], state["iw"], state["ih"])
    img_cx   = sx / WIN_W * vw + state["px"]
    img_cy   = sy / WIN_H * vh + state["py"]
    nvw, nvh = view_size(new_zoom, state["iw"], state["ih"])
    state["px"]   = int(img_cx - sx / WIN_W * nvw)
    state["py"]   = int(img_cy - sy / WIN_H * nvh)
    state["px"], state["py"] = clamp_pan(
        state["px"], state["py"], new_zoom, state["iw"], state["ih"])
    state["zoom"] = new_zoom

# ─── Mouse callback ───────────────────────────────────────────────────────────
def make_mouse_cb(state):
    def cb(event, x, y, flags, _):
        state["mx"], state["my"] = x, y

        if event == cv2.EVENT_MOUSEWHEEL or event == cv2.EVENT_MOUSEHWHEEL:
            if flags == 0:
                return
            factor = ZOOM_STEP if flags > 0 else 1 / ZOOM_STEP
            do_zoom(state, factor, x, y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            state["panning"]   = True
            state["pan_s0"]    = (x, y)
            state["pan_off0"]  = (state["px"], state["py"])

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

# ─── Main loop ────────────────────────────────────────────────────────────────
if not IMG_DIR.exists():
    raise FileNotFoundError(f"IMG_DIR not found: {IMG_DIR}")
if not LABEL_DIR.exists():
    raise FileNotFoundError(f"LABEL_DIR not found: {LABEL_DIR}")

images = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.JPG")) + sorted(IMG_DIR.glob("*.png"))
images = [p for p in images if (LABEL_DIR / (p.stem + ".txt")).exists()]

print(f"Found {len(images)} images with labels in {IMG_DIR}")
if not images:
    print("Nothing to review — check that IMG_DIR and LABEL_DIR are correct.")
    exit()

cv2.namedWindow("Label Editor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Label Editor", WIN_W, WIN_H)

i = 0
while i < len(images):
    img_path   = images[i]
    label_path = LABEL_DIR / (img_path.stem + ".txt")
    img        = cv2.imread(str(img_path))
    ih, iw     = img.shape[:2]

    state = {
        "boxes":    load_boxes(label_path, iw, ih),
        "selected": set(),
        "iw": iw, "ih": ih,
        "zoom": 1.0, "px": 0, "py": 0,
        "down": False, "drag": False, "start": (0,0), "cur": (0,0),
        "panning": False, "pan_s0": (0,0), "pan_off0": (0,0),
        "mx": WIN_W // 2, "my": WIN_H // 2,
        "input_mode": False, "input_text": "",
        "n_images": len(images),
    }

    cv2.setMouseCallback("Label Editor", make_mouse_cb(state))

    while True:
        preview = (state["start"][0], state["start"][1],
                   state["cur"][0],   state["cur"][1]) if state["drag"] else None
        frame   = render(img, state["boxes"], state["selected"], state, preview)

        title = (f"[{i+1}/{len(images)}] {img_path.name}  |  "
                 f"boxes:{len(state['boxes'])}  sel:{len(state['selected'])}  |  "
                 f"scroll/+- =zoom  RMB=pan  G=goto  S=save  D=next  A=back  Z=undo  Q=quit")
        cv2.setWindowTitle("Label Editor", title)
        cv2.imshow("Label Editor", frame)
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
            elif key == 8 and state["input_text"]:      # Backspace
                state["input_text"] = state["input_text"][:-1]
            elif 48 <= key <= 57:                       # 0-9
                state["input_text"] += chr(key)
        else:
            if key in (ord("+"), ord("=")):
                do_zoom(state, ZOOM_STEP, state["mx"], state["my"])
            elif key == ord("-"):
                do_zoom(state, 1 / ZOOM_STEP, state["mx"], state["my"])
            elif key == ord("g"):                        # open goto field
                state["input_mode"] = True
                state["input_text"] = ""
            elif key == ord("s"):                        # save + next
                state["boxes"] = [b for j, b in enumerate(state["boxes"])
                                  if j not in state["selected"]]
                save_boxes(label_path, state["boxes"], iw, ih)
                print(f"Saved {img_path.name} — {len(state['boxes'])} boxes")
                i += 1; break
            elif key == ord("d"):                        # advance without saving
                i += 1; break
            elif key == ord("a") and i > 0:             # back without saving
                i -= 1; break
            elif key == ord("z") and state["boxes"]:
                state["boxes"].pop()
                state["selected"].discard(len(state["boxes"]))
            elif key == ord("q"):
                cv2.destroyAllWindows(); exit()

cv2.destroyAllWindows()
print("Done")
