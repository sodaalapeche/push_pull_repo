import os
import shutil
import numpy as np
import cv2
from glob import glob

# ==========================================================
# PARAMETERS
# ==========================================================

SRC_BASE = "/home/chorus/HETEROGENE2/"
DST_BASE = "/home/chorus/HETEROGENE_binned/"
BIN_FACTOR = 4
SYSTEM_TYPE = "heterogeneous"  # ou "homogeneous"

os.makedirs(DST_BASE, exist_ok=True)

# ==========================================================
# UTILITIES
# ==========================================================

def find_tiff_subfolder(exp_dir):
    for d in os.listdir(exp_dir):
        path = os.path.join(exp_dir, d)
        if os.path.isdir(path) and glob(os.path.join(path, "*.tif")):
            return d
    return None


def bin_image_mean(img, factor):
    h, w = img.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError("Image size not divisible by binning factor")

    img = img.reshape(
        h // factor, factor,
        w // factor, factor
    )
    return img.mean(axis=(1, 3))


def create_label_file(dst_exp, size_folder, sand_folder):
    label_path = os.path.join(dst_exp, "label.txt")

    with open(label_path, "w") as f:
        f.write(f"{SYSTEM_TYPE}\n")
        f.write(f"{size_folder}\n")
        f.write(f"{sand_folder}\n")


# ==========================================================
# PROCESS ONE EXPERIMENT
# ==========================================================
def process_experiment(size_folder, sand_folder, exp_name):

    src_exp = os.path.join(SRC_BASE, size_folder, sand_folder, exp_name)
    dst_exp = os.path.join(DST_BASE, size_folder, sand_folder, exp_name)

    print(f"\n[INFO] Processing: {size_folder}/{sand_folder}/{exp_name}")

    if not os.path.exists(src_exp):
        print("  [SKIP] Source folder does not exist")
        return

    # --- find TIFF files directly in experiment folder ---
    tif_files = sorted(
        glob(os.path.join(src_exp, "*.tif")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    if len(tif_files) == 0:
        print("  [SKIP] No TIFF files found")
        return

    # --- Create destination structure ---
    dst_serie = os.path.join(dst_exp, "serie1")
    os.makedirs(dst_serie, exist_ok=True)
    os.makedirs(dst_exp, exist_ok=True)

    # --- Copy metadata (non-tif files) ---
    for f in os.listdir(src_exp):
        src_f = os.path.join(src_exp, f)
        dst_f = os.path.join(dst_exp, f)

        if os.path.isfile(src_f) and not f.lower().endswith(".tif"):
            shutil.copy2(src_f, dst_f)

    # --- Create label.txt ---
    create_label_file(dst_exp, size_folder, sand_folder)

    print(f"  Found {len(tif_files)} images")

    # --- Process images ---
    for f in tif_files:

        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"  [WARN] Could not read {f}")
            continue

        binned = bin_image_mean(img.astype(np.float32), BIN_FACTOR)
        binned = np.clip(binned, 0, 65535).astype(np.uint16)

        out_path = os.path.join(dst_serie, os.path.basename(f))
        cv2.imwrite(out_path, binned)

    print("  [DONE]")


# ==========================================================
# MAIN LOOP (scan full NAS structure)
# ==========================================================

sizes = [
    d for d in os.listdir(SRC_BASE)
    if os.path.isdir(os.path.join(SRC_BASE, d)) and d.lower().endswith("mm")
]

print(f"Found {len(sizes)} size folders")

for size_folder in sizes:

    size_path = os.path.join(SRC_BASE, size_folder)

    sands = [
        d for d in os.listdir(size_path)
        if os.path.isdir(os.path.join(size_path, d))
        and d.lower() in ["fine", "coarse"]
    ]

    for sand_folder in sands:

        sand_path = os.path.join(size_path, sand_folder)

        experiments = [
            d for d in os.listdir(sand_path)
            if os.path.isdir(os.path.join(sand_path, d))
        ]

        for exp in experiments:
            process_experiment(size_folder, sand_folder, exp)

print("\nAll experiments converted successfully.")