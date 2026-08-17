from pathlib import Path
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_DIR    = Path("C:/Users/pedroam/Documents/Datasets/Okutama Action/img/train/Drone1/Morning/Extracted-Frames-1280x720/1.1.1")
PREFIX     = "1.1.1_"          # identifier prepended to every filename
EXTENSIONS = {".jpg", ".JPG", ".jpeg", ".png", ".PNG"}

# ─── Rename ───────────────────────────────────────────────────────────────────
images = [p for p in IMG_DIR.iterdir() if p.suffix in EXTENSIONS]

for img_path in tqdm(images, desc="Renaming"):
    new_name = img_path.parent / (PREFIX + img_path.name)
    img_path.rename(new_name)

print(f"Done — {len(images)} images renamed with prefix '{PREFIX}'")
