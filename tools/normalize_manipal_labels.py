from pathlib import Path
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR  = Path("C:/Users/pedroam/Documents/Data-Augmentation/Datasets/Manipal-UAV")
PARTITIONS = ["train", "val", "test"]   # folders to process; skip missing ones

# ─── Processing ───────────────────────────────────────────────────────────────
total_files = total_boxes = 0

for partition in PARTITIONS:
    label_dir = BASE_DIR / partition / "labels"
    if not label_dir.exists():
        continue

    txt_files = list(label_dir.glob("*.txt"))
    for src in tqdm(txt_files, desc=partition):
        lines_out = []
        for line in src.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls        = int(float(parts[0]))
            xc, yc, w, h = (float(p) for p in parts[1:])
            xc, yc, w, h = (max(0.0, min(1.0, v)) for v in (xc, yc, w, h))
            if w > 0 and h > 0:
                lines_out.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        src.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
        total_boxes += len(lines_out)

    total_files += len(txt_files)

print(f"Done — {total_files} files, {total_boxes} boxes normalized")
