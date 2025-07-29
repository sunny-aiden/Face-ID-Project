"""
clean_lfw.py
============
End‑to‑end cleaning pipeline for LFW (Labeled Faces in the Wild).

Steps
-----
1. Download via KaggleHub  ·  caches zip or extracted directory
2. Remove corrupted JPEGs
3. Convert to RGB + resize 160×160 px
4. Person‑wise train / val / test split 70 / 15 / 15
5. Save cleaned PNGs and a JSON log of stats
6. Provide sample_grid() helper for manual mis‑label inspection

Run:
$ python scripts/clean_lfw.py
"""

import random, zipfile, shutil, json
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm

# --------------------- CONFIG ---------------------------------------------
KAGGLE_DS   = "jessicali9530/lfw-dataset"
RAW_DIR     = Path("data/raw/lfw")
PROC_DIR    = Path("data/processed")
IMG_SIZE    = (160, 160)
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
MIN_CLASS   = 2          # drop identities with <2 images
SEED        = 42
LOG_PATH    = Path("data/cleaning_log.json")
# --------------------------------------------------------------------------

def download_dataset():
    """Download via kagglehub; handle zip OR already‑extracted folder."""
    import kagglehub
    path = Path(kagglehub.dataset_download(KAGGLE_DS))

    if path.is_dir():
        # already extracted
        if not RAW_DIR.exists():
            shutil.copytree(path, RAW_DIR)
            print("[✓] Copied cached dataset to", RAW_DIR)
        else:
            print("[✓] Raw folder exists – skipping copy")
        return

    # otherwise path is a zip file
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(RAW_DIR.parent)
    print("[✓] Extracted ZIP to", RAW_DIR)

def remove_corrupted():
    """Delete images that Pillow can't open/verify."""
    bad = 0
    for fp in tqdm(list(RAW_DIR.rglob("*.jpg")), desc="Corruption check"):
        try:
            Image.open(fp).verify()
        except (UnidentifiedImageError, OSError):
            fp.unlink(missing_ok=True)
            bad += 1
    return bad

def build_class_dict():
    """Return {label: [image_paths]} filtered by MIN_CLASS."""
    class2imgs = {}
    for fp in RAW_DIR.rglob("*.jpg"):
        label = fp.parent.name
        class2imgs.setdefault(label, []).append(fp)
    before = len(class2imgs)
    class2imgs = {k:v for k,v in class2imgs.items() if len(v) >= MIN_CLASS}
    print(f"[i] Kept {len(class2imgs)}/{before} classes (≥{MIN_CLASS} images)")
    return class2imgs

def save_clean_split(class2imgs):
    random.seed(SEED)
    for split in ("train","val","test"):
        (PROC_DIR/split).mkdir(parents=True, exist_ok=True)

    for label, paths in tqdm(class2imgs.items(), desc="Saving cleaned PNGs"):
        random.shuffle(paths)
        n = len(paths)
        n_val, n_test = int(n*VAL_RATIO), int(n*TEST_RATIO)
        parts = {"train": paths[:n-n_val-n_test],
                 "val"  : paths[n-n_val-n_test:n-n_test],
                 "test" : paths[n-n_test:]}

        for split, files in parts.items():
            out_dir = PROC_DIR/split/label
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                img = Image.open(src).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
                arr = (np.asarray(img, dtype=np.float32)/255.0*255).astype(np.uint8)
                Image.fromarray(arr).save(out_dir/f"{src.stem}.png", optimize=True)

def log_stats(bad, class2imgs):
    stats = {
        "bad_files_removed": bad,
        "classes_retained": len(class2imgs),
        "images_retained": sum(len(v) for v in class2imgs.values()),
        "train_count": sum(1 for _ in (PROC_DIR/"train").rglob("*.png")),
        "val_count":   sum(1 for _ in (PROC_DIR/"val").rglob("*.png")),
        "test_count":  sum(1 for _ in (PROC_DIR/"test").rglob("*.png")),
        "img_size": IMG_SIZE,
        "pixel_norm": "[0,1] applied later in DataLoader",
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(stats, indent=2))
    print("[✓] Cleaning log saved →", LOG_PATH)

# --------------------- helper for manual QA -------------------------------
def sample_grid(n_classes=5, n_per=5):
    """
    Display a grid of faces for quick mis‑label spotting.
    Run this function in a Jupyter/VSCode notebook.
    """
    import matplotlib.pyplot as plt, random
    labels = list((PROC_DIR/"train").iterdir())
    if len(labels) < n_classes:
        n_classes = len(labels)
    chosen = random.sample(labels, n_classes)
    fig, axes = plt.subplots(n_classes, n_per, figsize=(n_per*2, n_classes*2))

    for r, lab_dir in enumerate(chosen):
        imgs = list(lab_dir.glob("*.png"))[:n_per]
        for c in range(n_per):
            ax = axes[r, c]
            if c < len(imgs):
                ax.imshow(Image.open(imgs[c]))
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(lab_dir.name[:12], rotation=0, labelpad=25)
    plt.tight_layout(); plt.show()

# --------------------- main ------------------------------------------------
if __name__ == "__main__":
    download_dataset()
    bad = remove_corrupted()
    classes = build_class_dict()
    save_clean_split(classes)
    log_stats(bad, classes)
    print("\nRun sample_grid() in a notebook to eyeball for mis‑labels.")

