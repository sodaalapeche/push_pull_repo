import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scienceplots
from glob import glob

plt.style.use('science')  # style scientifique

# ----------------- Fonctions utilitaires -----------------

def find_image_folder(root):
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and glob(os.path.join(p, "*.tif")):
            return p
    raise RuntimeError(f"No TIFF folder found under: {root}")

def list_tifs(folder):
    files = sorted(
        glob(os.path.join(folder, "*.tif")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    if not files:
        raise RuntimeError(f"No .tif found in {folder}")
    return files
def compute_col_profile(frame0):
    col_profile = np.mean(frame0, axis=0)
    col_profile /= np.mean(col_profile)
    return np.maximum(col_profile, 1e-6)


def apply_col_profile(frame, col_profile):
    return frame / col_profile[None, :]
def build_t_img_from_csv(root_folder, n_images):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)
    t = df["Timestamp"].values.astype(float)
    if len(t) < 2:
        raise ValueError("Not enough timestamps in weight_data.csv")
    dt = np.median(np.diff(t))
    t_img = t[0] + dt * np.arange(n_images)
    return t_img, dt
def roi_stats(frame, Xpix, Ypix, xc, yc, r_eval_pix):
    r2 = (Xpix - xc) ** 2 + (Ypix - yc) ** 2
    roi = r2 <= r_eval_pix ** 2
    vals = frame[roi]

    if vals.size == 0:
        return np.nan, np.nan

    return float(np.mean(vals)), float(np.var(vals))
# ----------------- Fonctions pour le Ta robuste -----------------
def process_experiment(root_folder, R_EVAL_PIX=200):
    """
    Charge les images, calcule le profil vertical et la variance dans le ROI,
    puis calcule un Ta robuste et les temps réduits.
    """
    img_folder = find_image_folder(root_folder)
    files = list_tifs(img_folder)
    n = len(files)

    frame0 = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    col_profile = compute_col_profile(frame0)  # fonction fournie ailleurs

    H, W = frame0.shape
    Ypix, Xpix = np.indices((H, W))
    xc, yc = W / 2, H / 2

    t_img, _ = build_t_img_from_csv(root_folder, n)

    mean_vals = np.full(n, np.nan)
    var_vals = np.full(n, np.nan)
    images = []

    for i, f in enumerate(files):
        frame = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        frame = apply_col_profile(frame, col_profile)  # fonction fournie ailleurs
        m, v = roi_stats(frame, Xpix, Ypix, xc, yc, R_EVAL_PIX)  # fonction fournie ailleurs
        mean_vals[i] = m
        var_vals[i] = v
        images.append(frame)

    i0 = int(np.nanargmax(var_vals))
    dmean_dt = np.gradient(mean_vals, t_img)
    i_start = min(i0 + 30, len(mean_vals) - 1)
    if i_start < len(mean_vals) - 1:
        i_min = i_start + int(np.nanargmin(dmean_dt[i_start:]))
    else:
        i_min = i0

    Ta = (t_img[i_min] - t_img[i0]) / 30
    times_reduced = (t_img - t_img[i0]) / Ta

    return images, times_reduced, i0, Ta

# ----------------- Recadrage en disque -----------------
def crop_to_disk(img, radius_ratio=0.45):
    H, W = img.shape
    Y, X = np.ogrid[:H, :W]
    xc, yc = W/2, H/2
    R = radius_ratio * min(H, W)
    mask = (X - xc)**2 + (Y - yc)**2 <= R**2
    return np.where(mask, img, np.nan)

# ----------------- Paramètres des expériences -----------------
BASE_PATH = "/home/chorus/HETEROGENE_binned/"
REQUEST = [(0, "fine"), (10, "fine"), (0, "coarse"), (10, "coarse")]
target_times = [0, 3, 9, 18]  # en Ta

# ----------------- Chargement des expériences -----------------
root_folders = []
experiment_labels = []

for L_mm, sand in REQUEST:
    size_folder = f"{L_mm}mm" if L_mm != 0 else "0mm"
    sand_folder = sand.lower()
    path = os.path.join(BASE_PATH, size_folder, sand_folder)
    if not os.path.exists(path):
        print(f"[WARN] Folder not found: {path}")
        continue
    for d in os.listdir(path):
        exp_path = os.path.join(path, d)
        if os.path.isdir(exp_path):
            tif_files = glob(os.path.join(exp_path, "**", "*.tif"), recursive=True)
            csv_path = os.path.join(exp_path, "weight_data.csv")
            if tif_files and os.path.isfile(csv_path):
                root_folders.append(exp_path)
                experiment_labels.append((L_mm, sand))
                break

# ----------------- Traitement des données -----------------
all_images = []
all_times = []
all_Ta = []

for folder in root_folders:
    images, times, i0, Ta = process_experiment(folder)
    all_images.append(images)
    all_times.append(times)
    all_Ta.append(Ta)

# ----------------- Création de la figure -----------------
fig = plt.figure(figsize=(14, 10))
n_rows = len(all_images)
n_cols = len(target_times)
height_ratios = [1,1,0.1,1,1]  # séparation fine/coarse
gs = gridspec.GridSpec(n_rows+1, n_cols+1, width_ratios=[0.01]+[0.02]*n_cols,
                       height_ratios=height_ratios, wspace=0.001, hspace=0.05)
for i, (images, times, (L_mm, sand), Ta) in enumerate(zip(all_images, all_times, experiment_labels, all_Ta)):
    row_idx = i if i < 2 else i + 1  # ligne vide entre fine et coarse
    gs = gridspec.GridSpec(n_rows + 1, n_cols + 1, width_ratios=[0.01] + [0.02] * n_cols,
                           height_ratios=height_ratios, wspace=0.00001, hspace=0.05)
    # Sélection des images les plus proches des temps cibles
    selected_images = [images[np.argmin(np.abs(times - t))] for t in target_times]
    selected_images = [np.log(img) for img in selected_images]

    # Normalisation basée sur la 3ème image
    ref_cropped = crop_to_disk(selected_images[1])
    vmin, vmax = np.nanmin(ref_cropped), np.nanmax(ref_cropped)
    normalized_images = [(crop_to_disk(img) - vmin)/(vmax - vmin) for img in selected_images]

    # Label de ligne
    ax_label = fig.add_subplot(gs[row_idx, 0])
    label_name = r"(a)" if L_mm == 0 else "(b)"
        # f"bidisperse,\n $d_M =${L_mm} mm"
    ax_label.text(1, 0.5, f"{label_name}", rotation=90,
                  va='center', ha='center',fontsize=18,weight='bold')
    ax_label.axis('off')

    # Tracé des images
    for j, img in enumerate(normalized_images):
        ax = fig.add_subplot(gs[row_idx, j+1])
        im = ax.imshow(img, cmap='plasma', vmin=0, vmax=1)
        ax.axis('off')
        if i == 0:
            ax.set_title(f'$2t \cdot u =$ {2*int(target_times[j])} cm', fontsize=18,weight='bold')
        if j == n_cols-1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
    # Add a 10 mm scale bar on the bottom-right image
    last_ax = ax  # 'ax' is the last subplot in the loop
    bar_length_m = 0.01  # 10 mm
    pixel_size_m = 0.06 / 512  # m/pixel
    bar_length_px = int(bar_length_m / pixel_size_m)  # number of pixels

    # Position: bottom-left of the image, 5% padding
    H, W = normalized_images[-1].shape
    x_start = int(W * 0.03)
    y_start = int(H * 0.95)
    x_end = x_start + bar_length_px

    # Draw white line
    last_ax.hlines(y=y_start, xmin=x_start, xmax=x_end, colors='black', linewidth=3)
    # Optional: add text "10 mm" above the bar
    last_ax.text(x_start + bar_length_px / 2, y_start - H * 0.03, "10 mm",
                 color='black', ha='center', va='bottom', fontsize=11, weight='bold')
fig.tight_layout()
plt.savefig('/home/chorus/strip.pdf')
plt.show()