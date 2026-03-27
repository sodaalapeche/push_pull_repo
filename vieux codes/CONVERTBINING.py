import os
import shutil
import numpy as np
import cv2
from glob import glob

# ==========================================================
# PARAMETERS
# ==========================================================

SRC_BASE = "/home/chorus/EXP_convert/"
DST_BASE = "/home/chorus/EXP_convert_binned/"
BIN_FACTOR = 4

os.makedirs(DST_BASE, exist_ok=True)

# ==========================================================
# UTILITIES
# ==========================================================

def find_tiff_subfolder(exp_dir):
    """
    Return the name of the first subfolder containing .tif files
    """
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


# ==========================================================
# PROCESS ONE EXPERIMENT
# ==========================================================

def process_experiment_folder(exp_name):
    src_exp = os.path.join(SRC_BASE, exp_name)
    dst_exp = os.path.join(DST_BASE, exp_name)

    print(f"\n[INFO] Processing experiment: {exp_name}")

    # --- find image subfolder ---
    img_sub = find_tiff_subfolder(src_exp)
    if img_sub is None:
        print("  [SKIP] No TIFF subfolder found")
        return

    src_img_dir = os.path.join(src_exp, img_sub)
    dst_img_dir = os.path.join(dst_exp, img_sub)

    os.makedirs(dst_img_dir, exist_ok=True)

    # --- copy metadata files ---
    os.makedirs(dst_exp, exist_ok=True)

    for f in os.listdir(src_exp):
        src_f = os.path.join(src_exp, f)
        dst_f = os.path.join(dst_exp, f)

        if os.path.isfile(src_f) and not f.lower().endswith(".tif"):
            shutil.copy2(src_f, dst_f)

    # --- process images ---
    tif_files = sorted(
        glob(os.path.join(src_img_dir, "*.tif")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    print(f"  Found {len(tif_files)} images in '{img_sub}'")

    for f in tif_files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"  [WARN] Could not read {f}")
            continue

        if img.dtype != np.uint16:
            print(f"  [WARN] {f} not uint16")

        binned = bin_image_mean(img.astype(np.float32), BIN_FACTOR)
        binned = np.clip(binned, 0, 65535).astype(np.uint16)

        out_path = os.path.join(dst_img_dir, os.path.basename(f))
        cv2.imwrite(out_path, binned)

    print(f"  [DONE] Converted images saved to {dst_img_dir}")


# ==========================================================
# MAIN
# ==========================================================

experiments = [
    d for d in sorted(os.listdir(SRC_BASE))
    if os.path.isdir(os.path.join(SRC_BASE, d))
]

print(f"Found {len(experiments)} experiments")

for exp in experiments:
    process_experiment_folder(exp)

print("\nAll experiments converted successfully.")
