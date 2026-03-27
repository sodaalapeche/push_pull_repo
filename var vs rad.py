


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import pandas as pd
import math

# "/home/chorus/exp_12_01_2/",
# "/home/chorus/exp_13_01_2/",
# "/home/chorus/exp_18_12_2/",
# "/home/chorus/exp_18_12_2/",

root_folders = [    "/home/chorus/16_10_4/"



]

dt = 2
dx = 0.06 / 2048
idx0 = 30
alpha = 0.4               


def Ta_for_bead(timestamps, weights_g, D=0.055, L=0.01, f_billes=0.6):
    timestamps = np.array(timestamps, dtype=float)
    weights_g = np.array(weights_g, dtype=float)

    valid = weights_g > 1000
    ts = timestamps[valid]
    ws = weights_g[valid]

    coeffs = np.polyfit(ts, ws, 1)
    dMdt_g_s = coeffs[0]

    Q_m3_s = dMdt_g_s * 1e-6
    A_tot = np.pi * (D / 2)**2
    A_pore = A_tot * (1 - f_billes)**2
    vp = Q_m3_s / A_pore
    Ta = L / vp

    return Ta, vp


def extract_Ta_from_csv(root_folder, colonne="grande"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)

    D = 0.055 if colonne == "grande" else 0.027
    Ta, vp = Ta_for_bead(df["Timestamp"], df["Weight"], D=D)
    return Ta, vp

# ==========================================================
# IMAGE UTILITIES
# ==========================================================

def find_image_folder(root):
    for d in os.listdir(root):
        path = os.path.join(root, d)
        if os.path.isdir(path) and glob(os.path.join(path, "*.tif")):
            return path
    raise RuntimeError("No TIFF folder found")


def load_full_sequence_16bit(folder):
    files = sorted(
        glob(os.path.join(folder, "*.tif")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    img0 = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    h, w = img0.shape
    stack = np.empty((len(files), h, w), dtype=np.float32)

    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        stack[i] = img.astype(np.float32) / 65535.0

    return stack


def select_roi_from_frame(frame):
    img8 = np.uint8(255 * (frame - frame.min()) / frame.ptp())
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", 1200, 800)
    roi = cv2.selectROI("Select ROI", img8, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi

# ==========================================================
# DISPERSION RADIUS
# ==========================================================

def compute_dispersion_radius(I, dx, alpha):
    nt, ny, nx = I.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)

    R = np.full(nt, np.nan)

    for t in range(nt):
        C = I[t]
        if np.all(np.isnan(C)):
            continue

        cmax = np.nanmax(C)
        mask = (~np.isnan(C)) & (C > alpha * cmax)
        if np.sum(mask) < 50:
            continue

        Cw = C[mask]
        Xw = X[mask]
        Yw = Y[mask]

        M = np.sum(Cw)
        xc = np.sum(Xw * Cw) / M
        yc = np.sum(Yw * Cw) / M

        r2 = np.sum(((Xw - xc)**2 + (Yw - yc)**2) * Cw) / M
        R[t] = np.sqrt(r2)

    return R

# ==========================================================
# MAIN PROCESSING PIPELINE
# ==========================================================

def process_experiment(root_folder, colonne="grande"):
    img_folder = find_image_folder(root_folder)
    I_full = load_full_sequence_16bit(img_folder)

    best_frame = I_full[np.argmax(I_full.mean(axis=(1,2)))]
    #x, y, w, h = select_roi_from_frame(best_frame)
    I = I_full

    # --- masks (same logic as original scripts) ---
    mask1 = np.swapaxes(
        np.swapaxes(
            np.tile((I[0] < 0.8*np.nanmedian(I[0])).reshape(I.shape[1], I.shape[2], 1), I.shape[0]),
            0, 2),
        1, 2
    )
    I[mask1] = np.nan

    mask2 = np.swapaxes(
        np.swapaxes(
            np.tile((I[idx0 + 10] < np.nanquantile(I[idx0], 0.6)).reshape(I.shape[1], I.shape[2], 1), I.shape[0]),
            0, 2),
        1, 2
    )
    I[mask2] = np.nan

    # --- statistics ---
    mean = np.nanmean(I, axis=(1,2))
    var = np.nanvar(I, axis=(1,2))

    A = (np.nansum(mask1[0]) + np.nansum(mask2[0])) * dx**2

    Ta, vp = extract_Ta_from_csv(root_folder, colonne)
    time = np.arange(I.shape[0]) * dt / Ta

    R = compute_dispersion_radius(I, dx, alpha)

    t_debug = idx0 + 20  # choisis un instant représentatif

    C = I[t_debug]

    # même masque que pour le rayon
    cmax = np.nanmax(C)
    mask = (~np.isnan(C)) & (C > alpha * cmax)

    x = np.arange(C.shape[1]) * dx
    y = np.arange(C.shape[0]) * dx
    X, Y = np.meshgrid(x, y)

    Cw = C[mask]
    Xw = X[mask]
    Yw = Y[mask]

    xc = np.sum(Xw * Cw) / np.sum(Cw)
    yc = np.sum(Yw * Cw) / np.sum(Cw)

    plt.figure(figsize=(5, 5))
    plt.imshow(
        C,
        origin="lower",
        cmap="gray",
        extent=[x.min(), x.max(), y.min(), y.max()]
    )
    plt.colorbar(label="Concentration")

    circle = plt.Circle((xc, yc), R[t_debug],
                        color="red", lw=2, fill=False)

    plt.gca().add_patch(circle)
    plt.title(f"t/t_a = {time[t_debug]:.2f}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.show()

    return time, mean, var, A, R

# ==========================================================
# EXECUTION & VISUALIZATION
# ==========================================================

plt.figure(figsize=(6,5))

for root in root_folders:
    time, mean, var, A, R = process_experiment(root)

    Sigma = var / (A * mean**2)

    mask = (
        ~np.isnan(R) &
        ~np.isnan(Sigma) &
        (R > 0) &
        (Sigma > 0)
    )

    plt.plot(time[mask]
        ,
        Sigma[mask]*R[mask]**2,
        'o',
        ms=4,
        label=os.path.basename(os.path.normpath(root))
    )
    plt.plot(time,mean,"gx",label="mass")


plt.xlabel("time")
plt.ylabel(r"$R^2 \cdot \sigma_c^2 / (A\mu_c^2)$")

plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# RAYON DE DISPERSION vs TEMPS
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(time, R, 'o', ms=4)
plt.xlabel(r"$t/t_a$")
plt.ylabel("Dispersion radius $R$ (m)")
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()
# ==========================================================
# VARIANCE SCALAIRE vs TEMPS
# ==========================================================

plt.figure(figsize=(6,4))
plt.plot(time, var, 'o', ms=4)
plt.xlabel(r"$t/t_a$")
plt.ylabel(r"Scalar variance $\sigma_c^2$")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, alpha=alpha)
plt.tight_layout()
plt.show()

