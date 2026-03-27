import os
import re
from glob import glob
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Define the required experiment types - using coarse instead of fine for 0mm
REQUEST = [(0, "coarse"), (10, "fine"),(10, "coarse"), (3, "fine") ]

# Base path for experiments
BASE_PATH = "/home/chorus/HETEROGENE_binned/"

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

def build_t_img_from_csv(root_folder, n_images):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)
    t = df["Timestamp"].values.astype(float)
    if len(t) < 2:
        raise ValueError("Pas assez de timestamps dans weight_data.csv")
    dt = np.median(np.diff(t))
    t_img = t[0] + dt * np.arange(n_images)
    return t_img, dt

def extract_Ta_from_csv(root_folder, colonne="grande"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)
    D = 0.055 if colonne == "grande" else 0.027
    L = 0.01  # Default L, can be extracted from label if needed
    # Compute Ta using weight data
    timestamps = df["Timestamp"].values.astype(float)
    weights_g = df["Weight"].values.astype(float)
    # Fit line to weight data to get flow rate
    coeffs = np.polyfit(timestamps, weights_g, 1)
    dMdt_g_s = coeffs[0]
    Q_m3_s = dMdt_g_s * 1e-6
    A_tot = np.pi * (D / 2)**2
    eps_sable = 0.5
    f_billes = 0.3  # Placeholder
    A_pore = A_tot * eps_sable * (1 - f_billes)
    vp = Q_m3_s / A_pore
    Ta = L / vp
    return Ta

def compute_variance(frame):
    """Compute variance of the frame, ignoring NaN values"""
    valid_mask = ~np.isnan(frame)
    if not np.any(valid_mask):
        return np.nan
    return np.var(frame[valid_mask])

def load_images_and_times(root_folder):
    img_folder = find_image_folder(root_folder)
    files = list_tifs(img_folder)
    n = len(files)

    # Compute image times
    t_img, _ = build_t_img_from_csv(root_folder, n)
    Ta = extract_Ta_from_csv(root_folder)
    times = t_img / Ta

    # Load images
    images = []
    for f in files:
        img_u16 = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img_u16 is None:
            continue
        frame = img_u16.astype(np.float32) / 65535.0
        images.append(frame)

    # Compute variances
    variances = [compute_variance(img) for img in images]

    # Find time of maximum variance
    i0 = np.nanargmax(variances)

    # Rescale times so that t0 is at maximum variance
    times0 = times - times[i0]

    return images, times0, i0, Ta

# Select experiments in the order of REQUEST
root_folders = []
experiment_labels = []
experiment_Ta = []
for L_mm, sand in REQUEST:
    size_folder = f"{L_mm}mm" if L_mm != 0 else "0mm"  # Handle 0mm case
    sand_folder = sand.lower()
    path = os.path.join(BASE_PATH, size_folder, sand_folder)

    if not os.path.exists(path):
        print(f"[WARN] Folder not found: {path}")
        continue

    # Find the first valid experiment subfolder
    for d in os.listdir(path):
        exp_path = os.path.join(path, d)
        if os.path.isdir(exp_path):
            tif_files = glob(os.path.join(exp_path, "**", "*.tif"), recursive=True)
            csv_path = os.path.join(exp_path, "weight_data.csv")
            if tif_files and os.path.isfile(csv_path):
                root_folders.append(exp_path)
                experiment_labels.append((L_mm, sand))
                break
    else:
        print(f"[WARN] No valid experiment folder found for {L_mm}mm {sand}")
        continue

# If we didn't find enough experiments, exit
if len(root_folders) < len(REQUEST):
    print(f"Error: Found {len(root_folders)} experiments, expected {len(REQUEST)}")
    exit()

# Load all data and compute Ta values
all_images = []
all_times = []
all_Ta = []
for i, root_folder in enumerate(root_folders):
    images, times, i0, Ta = load_images_and_times(root_folder)
    all_images.append(images)
    all_times.append(times)
    all_Ta.append(Ta)

# Target times relative to t0 (time of maximum variance)
target_times = [-1.3, 3, 7, 15, 24]  # Times before and after t0

# Create plot with space for labels on the left
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(4, 6, width_ratios=[0.1, 1, 1, 1, 1, 1], wspace=0.1, hspace=0.3)

# Add labels for each row with Ta values (only once)
for i, ((L_mm, sand), Ta) in enumerate(zip(experiment_labels, all_Ta)):
    label_text = f"{L_mm} mm, {sand}\nTa ≈ {int(round(Ta))}s"
    ax_label = fig.add_subplot(gs[i, 0])
    ax_label.text(0.5, 0.5, label_text, rotation=90, va='center', ha='center', fontsize=12)
    ax_label.axis('off')

# Create custom ice colormap similar to Fiji
# Définition plus précise de la colormap Green Fire Blue (sans rouge)
gfb_colors = [
    (0, 0, 0.5),    # Bleu foncé
    (0, 0, 1),      # Bleu
    (0, 0.5, 1),    # Bleu-vert
    (0, 1, 1),      # Cyan
    (0.5, 1, 0.5),  # Vert clair
    (1, 1, 0),      # Jaune
    (1, 1, 0.5),    # Jaune pâle
]
from cmap import Colormap
gfb_cmap = Colormap('imagej:GreenFireBlue')  # case insensitive

# Plot images
for i, (images, times) in enumerate(zip(all_images, all_times)):
    # Find closest images to target times
    selected_images = []
    selected_times = []
    for target in target_times:
        idx = np.argmin(np.abs(times - target))
        selected_images.append(np.log(images[idx]))
        selected_times.append(times[idx])

    # Normalize images within this line based on the 4th time point (index 3)
    reference_image = selected_images[2]
    img_min = reference_image.min()
    img_max = reference_image.max()
    # Avoid division by zero if all values are the same
    if img_max - img_min < 1e-10:
        img_max = img_min + 1e-10
    normalized_images = [(img - img_min) / (img_max - img_min) for img in selected_images]

    for j, img in enumerate(normalized_images):
        ax = fig.add_subplot(gs[i, j+1])
        im = ax.imshow(img, cmap='plasma', vmin=0, vmax=1)
        ax.axis('off')
        # Only show time labels on the first row to avoid clutter
        if i == 0:
            ax.set_title(f't ≈ {int(selected_times[j])} Ta')

plt.tight_layout()
plt.show()