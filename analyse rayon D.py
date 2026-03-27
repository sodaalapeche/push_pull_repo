
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import pandas as pd


root_folders = [
    "/home/chorus/exp_3_11_3/"
]

dt = 2.0                    # frame interval (s)
dx = 0.06 / 2048            # pixel size (m)
idx0 = 40                   # reference frame
save_figures = True
out_folder = "dispersion_results"

os.makedirs(out_folder, exist_ok=True)


def Ta_for_bead(timestamps, weights_g, D=0.055, L=0.01, f_billes=0.6):
    timestamps = np.array(timestamps, dtype=float)
    weights_g = np.array(weights_g, dtype=float)

    valid_mask = weights_g > 1000
    ts = timestamps[valid_mask]
    ws = weights_g[valid_mask]

    diff_w = np.diff(ws)
    jump_indices = np.where(diff_w < -0.5 * np.max(ws))[0]

    if len(jump_indices) > 0:
        segments = []
        start = 0
        for j in jump_indices:
            end = j + 1
            segments.append((ts[start:end], ws[start:end]))
            start = end
        if start < len(ts):
            segments.append((ts[start:], ws[start:]))
    else:
        segments = [(ts, ws)]

    best_seg = None
    best_score = -np.inf

    for seg_ts, seg_ws in segments:
        if len(seg_ws) < 3:
            continue
        coeffs = np.polyfit(seg_ts, seg_ws, 1)
        slope = coeffs[0]
        fit = np.polyval(coeffs, seg_ts)
        r2 = 1 - np.sum((seg_ws - fit) ** 2) / np.sum((seg_ws - np.mean(seg_ws)) ** 2)
        if slope > 0 and r2 > best_score:
            best_score = r2
            best_seg = (seg_ts, seg_ws, slope)

    seg_ts, seg_ws, dMdt_g_s = best_seg

    Q_m3_s = dMdt_g_s * 1e-6
    A_tot = np.pi * (D / 2.0) ** 2
    A_pore = A_tot * (1 - f_billes) ** 2
    vp = Q_m3_s / A_pore
    Ta = L / vp

    return Ta, vp


def extract_Ta_from_csv(root_folder, colonne="grande"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)

    timestamps = df["Timestamp"].values
    weights = df["Weight"].values

    D = 0.055 if colonne == "grande" else 0.027
    Ta, vp = Ta_for_bead(timestamps, weights, D=D)

    return Ta, vp

# ============================================================
# IMAGE UTILITIES
# ============================================================

def find_image_folder(root):
    for d in os.listdir(root):
        path = os.path.join(root, d)
        if os.path.isdir(path) and glob(os.path.join(path, "*.tif")):
            return path
    raise RuntimeError("No image folder containing .tif found")


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

# ============================================================
# ORIGINAL PROCESSING PIPELINE
# ============================================================

def process_experiment(root_folder):
    img_folder = find_image_folder(root_folder)
    I_full = load_full_sequence_16bit(img_folder)

    means = I_full.mean(axis=(1, 2))
    best_frame = I_full[np.argmax(means)]
    x, y, w, h = select_roi_from_frame(best_frame)
    I = I_full[:, y:y+h, x:x+w]

    mask1 = np.swapaxes(
        np.swapaxes(
            np.tile((I[0] < 0.9 * np.nanmedian(I[0])).reshape(I.shape[1], I.shape[2], 1), I.shape[0]),
            0, 2),
        1, 2)
    I[mask1] = np.nan

    mask2 = np.swapaxes(
        np.swapaxes(
            np.tile((I[idx0 + 10] < np.nanquantile(I[idx0], [0.5])).reshape(I.shape[1], I.shape[2], 1), I.shape[0]),
            0, 2),
        1, 2)
    I[mask2] = np.nan

    A = (np.nansum(mask1[0]) + np.nansum(mask2[0])) * dx**2

    mean = np.nanmean(I, axis=(1, 2))
    var = np.nanvar(I, axis=(1, 2))

    return I, mean, var, A

# ============================================================
# DISPERSION RADIUS
# ============================================================

def compute_dispersion_radius(I, dx, alpha=0.3):
    nt, ny, nx = I.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)

    R = np.full(nt, np.nan)
    r2 = np.full(nt, np.nan)

    for t in range(nt):
        C = I[t]
        if np.all(np.isnan(C)):
            continue

        mask = (~np.isnan(C)) & (C > alpha * np.nanmean(C))
        if np.sum(mask) < 50:
            continue

        Cw = C[mask]
        Xw = X[mask]
        Yw = Y[mask]

        M = np.sum(Cw)
        xc = np.sum(Xw * Cw) / M
        yc = np.sum(Yw * Cw) / M

        r2_t = np.sum(((Xw - xc)**2 + (Yw - yc)**2) * Cw) / M

        r2[t] = r2_t
        R[t] = np.sqrt(r2_t)

    return R, r2

# ============================================================
# DIFFUSIVE FIT (FORCED SLOPE 1/2)
# ============================================================

def fit_diffusion_fixed_slope(t_star, R, idx0):
    """
    Diffusive law R^2 = 4 D_eff t
    Fit is STRICTLY anchored at idx0
    """
    if np.isnan(R[idx0]) or t_star[idx0] <= 0:
        raise ValueError("Invalid anchor point for diffusion fit")

    t0 = t_star[idx0]
    R0 = R[idx0]

    D_eff = R0**2 / (4 * t0)
    return D_eff


# ============================================================
# PLOT
# ============================================================
def plot_radius_with_fit(t_star, R, D_eff, idx0, title, savepath=None):
    mask = (~np.isnan(R)) & (R > 0) & (t_star > 0)

    plt.figure(figsize=(6,5))
    plt.loglog(t_star[mask], R[mask], 'o', ms=4, label="R(t)")

    t_fit = np.linspace(t_star[mask].min(), t_star[mask].max(), 300)
    R_fit = np.sqrt(4 * D_eff * t_fit)

    plt.loglog(
        t_fit, R_fit, 'k--', lw=2,
        label=rf"$R=\sqrt{{4Dt}}$ (anchored at $t_0$)"
    )

    # Anchor point
    plt.plot(t_star[idx0], R[idx0], 'ro', ms=7, label="anchor (idx0)")

    plt.xlabel(r"$t/t_a$")
    plt.ylabel("Radius (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()


# ============================================================
# MAIN
# ============================================================

def run_all(folders):
    for root in folders:
        print(f"\n===== Processing {root} =====")

        # --- TA FROM YOUR FUNCTION ---
        Ta, vp = extract_Ta_from_csv(root)
        print(f"Ta = {Ta:.2f} s")

        # --- IMAGE PIPELINE ---
        I, mean, var, A = process_experiment(root)

        nt = I.shape[0]
        t_star = np.arange(nt) * dt / Ta

        R, r2 = compute_dispersion_radius(I, dx)

        # --- DIFFUSION FIT ---
        D_eff = fit_diffusion_fixed_slope(t_star, R,idx0)


        base = os.path.basename(os.path.normpath(root))
        figpath = os.path.join(out_folder, f"radius_{base}.png")

        plot_radius_with_fit(t_star, R, D_eff,idx0, base, figpath if save_figures else None)
        D = D_eff / Ta

        D = D/vp
        print(f"Diffusvité  alpha = {D:.3e} m")

        save_npz = os.path.join(out_folder, f"dispersion_{base}.npz")
        np.savez(save_npz, t_star=t_star, R=R, r2=r2, D=D, Ta=Ta)

        print(f"Saved results to {save_npz}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_all(root_folders)

# EOF
