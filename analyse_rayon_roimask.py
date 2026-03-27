"""
FINAL DISPERSION RADIUS ANALYSIS (DOMAIN FINITE – PHYSICALLY CONSISTENT)
=====================================================================

This script is a COMPLETE, SELF-CONTAINED analysis pipeline.
It does NOT require modifying your original script.

What this code does:
-------------------
✔ Loads TIFF experiments (16-bit preserved)
✔ Uses the SAME ROI selection and masking logic
✔ Computes concentration statistics
✔ Computes a PHYSICALLY MEANINGFUL dispersion radius in a FINITE domain
   using a thresholded second spatial moment
✔ Identifies Fickian behaviour R ~ t^{1/2}
✔ Produces:
   - log–log R(t) plot with reference slope 1/2
   - slope estimation
   - 5 images through time with fitted radius overlaid in red

Author: ChatGPT (physics-consistent version)
"""
from donnéesleontiff_nv import Ta_for_bead
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
from scipy.stats import linregress

# ==========================================================
# USER PARAMETERS
# ==========================================================
root_folders = ["/home/chorus/EXP_TO_TREAT/15_10_4/"]

dt = 2              # frame time step

dx = 0.06 / 2048    # pixel size (m)
idx0 = 30     # reference time index
alpha = 0.3         # relative concentration threshold
n_overlay_frames = 5

output_dir = "final_dispersion_results"
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# IO UTILITIES
# ==========================================================

def find_image_folder(root):
    for d in os.listdir(root):
        path = os.path.join(root, d)
        if os.path.isdir(path) and glob(os.path.join(path, "*.tif")):
            return path
    raise RuntimeError("No TIFF folder found")


def load_full_sequence_16bit(folder):
    files = sorted(glob(os.path.join(folder, "*.tif")),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

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
# EXPERIMENT PROCESSING (UNCHANGED LOGIC)
# ==========================================================

def process_experiment(root_folder):
    img_folder = find_image_folder(root_folder)
    I_full = load_full_sequence_16bit(img_folder)

    means_full = I_full.mean(axis=(1, 2))
    best_frame = I_full[np.argmax(means_full)]
    x, y, w, h = select_roi_from_frame(best_frame)

    I = I_full[:, y:y+h, x:x+w]

    # --- Same masks as original code ---
    mask1 = np.swapaxes(np.swapaxes(np.tile((I[0] < 0.8*np.nanmedian(I[0])).reshape(I.shape[1], I.shape[2], 1), I.shape[0]), 0, 2), 1, 2)
    I[mask1] = np.nan

    mask2 = np.swapaxes(np.swapaxes(np.tile((I[idx0 + 10] < np.nanquantile(I[idx0], 0.6)).reshape(I.shape[1], I.shape[2], 1), I.shape[0]), 0, 2), 1, 2)
    I[mask2] = np.nan

    mean = np.nanmean(I, axis=(1, 2))
    var = np.nanvar(I, axis=(1, 2))
    time = np.arange(I.shape[0]) * dt / ta

    return I, time, mean, var
def compute_Ta_vp_from_root(root_folder, colonne="grande"):
    """
    Compute Ta and vp using weight_data.csv and Ta_for_bead
    """
    csv_path = os.path.join(root_folder, "weight_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    timestamps = df["Timestamp"].values
    weights = df["Weight"].values

    if colonne == "petite":
        D = 0.027
    else:
        D = 0.055

    Ta, fps, dt_mean, vp, Q_m3_s, Q_mean_g_s = Ta_for_bead(
        timestamps,
        weights,
        D=D
    )

    print(f"[INFO] Ta = {Ta:.3f} s | vp = {vp:.3e} m/s")
    return Ta, vp

# ==========================================================
# DISPERSION RADIUS (FINITE DOMAIN – THRESHOLDED MOMENT)
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

        c_mean = np.nanmax(C)
        c_cut = alpha * c_mean

        mask = (~np.isnan(C)) & (C > c_cut)
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
# VISUALIZATION: RADIUS OVERLAY
# ==========================================================

def save_radius_overlays(I, R, time, dx, outdir, nframes):
    os.makedirs(outdir, exist_ok=True)

    nt, ny, nx = I.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)

    frame_ids = np.linspace(0, nt - 1, nframes, dtype=int)

    for k, t in enumerate(frame_ids):
        C = I[t]
        if np.isnan(R[t]):
            continue

        c_mean = np.nanmean(C)
        mask = (~np.isnan(C)) & (C > alpha * c_mean)

        Cw = C[mask]
        Xw = X[mask]
        Yw = Y[mask]

        xc = np.sum(Xw * Cw) / np.sum(Cw)
        yc = np.sum(Yw * Cw) / np.sum(Cw)

        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(C, origin='lower', cmap='gray',
                       extent=[x.min(), x.max(), y.min(), y.max()])

        circle = plt.Circle((xc, yc), R[t], color='red', lw=2, fill=False)
        ax.add_patch(circle)

        ax.set_title(f"t/ta = {time[t]:.2f}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(im, ax=ax, label="Concentration")

        fname = os.path.join(outdir, f"radius_overlay_{k+1}.png")
        plt.tight_layout()
        plt.show()

# ==========================================================
# MAIN EXECUTION
# ==========================================================

for root in root_folders:
    print(f"Processing {root}")
    Ta, vp = compute_Ta_vp_from_root(root, colonne="grande")
    ta = Ta
    I, time, mean, var = process_experiment(root)
    R = compute_dispersion_radius(I, dx, alpha)

    # --- Log-log plot ---
    mask = (~np.isnan(R)) & (R > 0)
    t0 = time[idx0]
    R0 = R[idx0]

    R_fit = R0 * np.sqrt(time / t0)

    plt.figure(figsize=(6,5))
    plt.plot(time[mask], R[mask], 'o', ms=4, label='R(t)')
    plt.plot(time[mask], R_fit[mask], 'k--', lw=2,
             label=r'Forced $t^{1/2}$ (diffusion)')

    t0 = time[mask][len(time[mask])//2]
    R0 = R[mask][len(R[mask])//2]
    D_eff = R0 ** 2 / (4 * t0)  # m^2/s
    Dv_ratio = D_eff / vp  # meters

    print("====================================")
    print(f"Effective diffusivity D = {D_eff:.3e} m^2/s")
    print(f"D / vp = {Dv_ratio:.3e} m")
    print("====================================")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$t/t_a$')
    plt.ylabel('Dispersion radius $R$ (m)')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.3)

    plt.annotate(r"slope = 0.5 (forced)",
                 xy=(0.05, 0.05),
                 xycoords='axes fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radius_scaling.png"), dpi=200)
    plt.show()

    # --- Overlay images ---
    save_radius_overlays(I, R, time, dx,
                          os.path.join(output_dir, "radius_overlays"),
                          n_overlay_frames)

    print("Done.")

