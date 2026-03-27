import os
import re
from glob import glob
from multiprocessing import Pool, cpu_count

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.filters as sk
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.signal import fftconvolve

# ==========================================================
# GLOBAL PARAMETERS
# ==========================================================
BASE_PATH = "/home/chorus/HETEROGENE_binned/"
MASK_REF_PATH = "/home/chorus/test/masque.jpg"

dx = (0.06/ 2048) * 4  # m/pixel

# Recalage masque hex (pipeline workflow_new)
ANGLE_MAX = 2.0
ANGLE_STEP = 0.1
TRANS_MAX = 30

# RMS radius
RMS_BLUR_CENTER_SIGMA = 1.0
RMS_BLUR_MOMENTS_SIGMA = 2.0
RMS_RING_RMIN_FRAC = 0.42
RMS_RING_RMAX_FRAC = 0.5
RMS_THRESH_K_SIGMA = 1.0
RMS_ROI_K = 2.5
RMS_ROI_RMIN_PIX = 25.0
RMS_ROI_RMAX_FRAC = 0.5


# ==========================================================
# DISCOVERY / IO
# ==========================================================
def select_exact_combinations(base_path, combinations):
    selected = []
    for L_mm, sand in combinations:
        size_folder = f"{L_mm}mm"
        sand_folder = sand.lower()
        path = os.path.join(base_path, size_folder, sand_folder)

        if not os.path.exists(path):
            print(f"[WARN] Folder not found: {path}")
            continue

        for d in os.listdir(path):
            exp_path = os.path.join(path, d)
            if not os.path.isdir(exp_path):
                continue

            tif_files = glob(os.path.join(exp_path, "**", "*.tif"), recursive=True)
            csv_path = os.path.join(exp_path, "weight_data.csv")
            if tif_files and os.path.isfile(csv_path):
                selected.append(exp_path)
            else:
                if not tif_files:
                    print(f"[SKIP] No TIFF: {exp_path}")
                if not os.path.isfile(csv_path):
                    print(f"[SKIP] Missing weight_data.csv: {exp_path}")
    return selected


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


def read_label(root):
    p = os.path.join(root, "label.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    return os.path.basename(root)


def extract_L_from_label(root, default_L=0.01):
    p = os.path.join(root, "label.txt")
    if not os.path.exists(p):
        return default_L
    txt = open(p).read().lower()
    m1 = re.search(r'(\d+(?:\.\d+)?)\s*mm', txt)
    return float(m1.group(1)) * 1e-3 if m1 else default_L


# ==========================================================
# TIME / Ta utilities (workflow_new)
# ==========================================================
def build_t_img_from_csv(root_folder, n_images):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)
    t = df["Timestamp"].values.astype(float)

    if len(t) < 2:
        raise ValueError("Pas assez de timestamps dans weight_data.csv")

    dt = np.median(np.diff(t))
    t_img = t[0] + dt * np.arange(n_images)
    return t_img, dt


def Ta_for_bead(timestamps, weights_g, D=0.055, L=0.01, f_billes=0.3, Tip="homo"):
    weights_g = np.array(weights_g, dtype=float)
    eps_sable = 0.5
    valid = weights_g > 1000
    ts = timestamps[valid]
    ws = weights_g[valid]

    coeffs = np.polyfit(ts, ws, 1)
    dMdt_g_s = coeffs[0]

    Q_m3_s = dMdt_g_s * 1e-6
    A_tot = np.pi * (D / 2) ** 2
    if Tip == "homo coarse":
        A_pore = A_tot * eps_sable * 0.75
    elif Tip == "homo fine":
        A_pore = A_tot * eps_sable * 0.67
    elif Tip == "0.1":
        A_pore = A_tot * eps_sable * (1 - f_billes) * 1.7
    elif Tip == "0.6":
        A_pore = A_tot * eps_sable * (1 - f_billes) * 0.8
    elif Tip == "0.3 coarse":
        A_pore = A_tot * eps_sable * (1 - f_billes) * 0.9
    elif Tip == "0.6 fine":
        A_pore = A_tot * eps_sable * (1 - f_billes) * 1.23
    elif Tip == "Hetero":
        A_pore = A_tot * eps_sable * (1 - f_billes)
    elif Tip == "0.1 coarse":
        A_pore = A_tot * eps_sable * (1 - f_billes) * 0.95
    elif Tip == "stokes 0.1":
        A_pore = A_tot * 0.4
    else:
        A_pore = A_tot * eps_sable * (1 - f_billes)

    vp = Q_m3_s / A_pore
    Ta = L / vp
    return Ta, vp


def extract_Ta_from_csv(root_folder, colonne="grande", Tip="homo"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)
    label = read_label(root_folder)

    D = 0.055 if colonne == "grande" else 0.027
    L = extract_L_from_label(root_folder)

    if L == 0.001:
        Tip = "0.1"
    if L == 0.01 and "fine" in label:
        Tip = "Hetero"
    if L == 0.01 and "coarse" in label:
        Tip = "0.1 coarse"
    if L == 0.006 and "coarse" in label:
        Tip = "0.6"
    if L == 0.006 and "fine" in label:
        Tip = "0.6 fine"
    if L == 0.003:
        Tip = "0.3 coarse"
    if L == 0.0 and "fine" in label:
        Tip = "homo fine"
        L = 0.01
    if L == 0.0 and "coarse" in label:
        Tip = "homo coarse"
        L = 0.01
    if L == 0.01 and "stokes" in label:
        Tip = "stokes 0.1"

    L = 0.01
    Ta, vp = Ta_for_bead(df["Timestamp"], df["Weight"], D=D, L=L, Tip=Tip)
    return Ta, vp


# ==========================================================
# Stripe correction / mask pipeline (workflow_new)
# ==========================================================
def compute_col_profile_from_frame0(frame0, smooth=2, eps=1e-6):
    col_profile = np.nanmean(frame0, axis=0)
    if smooth and smooth > 3:
        col_profile = ndimage.median_filter(col_profile, size=smooth)
    col_profile /= np.nanmean(col_profile)
    col_profile = np.maximum(col_profile, eps)
    return col_profile


def apply_col_profile(frame, col_profile):
    return frame / col_profile[None, :]


def extract_structure_sato(img):
    I_sato = sk.sato(img, sigmas=range(4, 10, 1))
    I_sato = ndimage.median_filter(I_sato, size=13)
    mask_sato = I_sato > 1e-5

    H, W = img.shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask_circle = (X - W / 2) ** 2 + (Y - H / 2) ** 2 > 213 ** 2
    structure = mask_sato & (~mask_circle)
    return structure.astype(np.uint8)


def compute_hex_mask_from_sato(img0, mask_ref_u8):
    H, W = img0.shape
    structure = extract_structure_sato(img0)

    edges_mask = cv2.Canny(mask_ref_u8 * 255, 50, 150)
    edges_struct = cv2.Canny(structure * 255, 50, 150)

    center = (W / 2, H / 2)
    angles = np.arange(-ANGLE_MAX, ANGLE_MAX + ANGLE_STEP, ANGLE_STEP)

    best_angle, best_score = 0.0, -np.inf
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(edges_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        score = np.sum(rot * edges_struct)
        if score > best_score:
            best_score, best_angle = score, angle

    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    edges_mask_rot = cv2.warpAffine(edges_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

    corr = fftconvolve(edges_struct.astype(float), edges_mask_rot[::-1, ::-1].astype(float), mode="same")
    cy, cx = H // 2, W // 2
    corr_window = corr[cy - TRANS_MAX: cy + TRANS_MAX + 1, cx - TRANS_MAX: cx + TRANS_MAX + 1]
    dy, dxw = np.unravel_index(np.argmax(corr_window), corr_window.shape)
    ty = dy - TRANS_MAX
    tx = dxw - TRANS_MAX

    M[0, 2] += tx
    M[1, 2] += ty

    mask_aligned = cv2.warpAffine(mask_ref_u8, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    mask_aligned = ndimage.binary_erosion(mask_aligned.astype(bool), iterations=4)
    return mask_aligned


# ==========================================================
# RMS radius (ton analyse rayon, appelée dans la pipeline workflow_new)
# ==========================================================
def compute_rms_radius(img, dx,
                       blur_center_sigma=RMS_BLUR_CENTER_SIGMA,
                       blur_moments_sigma=RMS_BLUR_MOMENTS_SIGMA,
                       ring_rmin_frac=RMS_RING_RMIN_FRAC,
                       ring_rmax_frac=RMS_RING_RMAX_FRAC,
                       thresh_k_sigma=RMS_THRESH_K_SIGMA,
                       roi_k=RMS_ROI_K,
                       roi_rmin_pix=RMS_ROI_RMIN_PIX,
                       roi_rmax_frac=RMS_ROI_RMAX_FRAC,
                       eps=1e-12):
    img = img.astype(np.float32)
    H, W = img.shape
    yy, xx = np.indices((H, W))

    env_c = cv2.GaussianBlur(img, (0, 0), blur_center_sigma)

    cx0, cy0 = W / 2.0, H / 2.0
    r0 = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)

    rmax_global = roi_rmax_frac * min(H, W)
    rmin_ring = ring_rmin_frac * min(H, W)
    rmax_ring = min(ring_rmax_frac * min(H, W), rmax_global)

    ring = (r0 >= rmin_ring) & (r0 <= rmax_ring)
    if not np.any(ring):
        bg = float(np.percentile(env_c, 5.0))
        sigma_bg = float(np.std(env_c))
    else:
        vals = env_c[ring]
        bg = float(np.median(vals))
        sigma_bg = float(1.4826 * np.median(np.abs(vals - bg)))

    w_c = env_c - bg
    w_c[w_c < 0] = 0.0
    Mc = w_c.sum()
    if Mc <= eps:
        return np.nan, (np.nan, np.nan), (np.nan, np.nan), {"reason": "no_signal_center"}

    xc = float((w_c * xx).sum() / Mc)
    yc = float((w_c * yy).sum() / Mc)

    env_m = cv2.GaussianBlur(img, (0, 0), blur_moments_sigma)
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)

    ring2 = (r >= rmin_ring) & (r <= rmax_ring)
    if np.any(ring2):
        vals2 = env_m[ring2]
        bg2 = float(np.median(vals2))
        sigma_bg2 = float(1.4826 * np.median(np.abs(vals2 - bg2)))
    else:
        bg2, sigma_bg2 = bg, sigma_bg

    w = env_m - bg2
    w[w < 0] = 0.0

    if thresh_k_sigma and thresh_k_sigma > 0:
        thr = bg2 + thresh_k_sigma * sigma_bg2
        w = np.where(env_m >= thr, w, 0.0)

    wpos = w.copy()
    wpos[r > rmax_global] = 0.0
    M = wpos.sum()
    if M <= eps:
        return np.nan, (xc, yc), (np.nan, np.nan), {"reason": "no_signal_moments"}

    rbin = np.floor(r).astype(np.int32)
    nbins = int(np.floor(rmax_global)) + 1
    nbins = max(nbins, 10)
    E = np.bincount(rbin.ravel(), weights=wpos.ravel(), minlength=nbins).astype(np.float64)
    Ecum = np.cumsum(E)
    Etot = Ecum[-1]
    if Etot <= eps:
        return np.nan, (xc, yc), (np.nan, np.nan), {"reason": "Etot_zero"}

    idx80 = int(np.searchsorted(Ecum, 0.80 * Etot))
    r_eff = float(np.clip(idx80, 1, rmax_global))

    roi_r = float(np.clip(roi_k * r_eff, roi_rmin_pix, rmax_global))
    roi = r <= roi_r

    w_roi = np.where(roi, w, 0.0)
    M2 = w_roi.sum()
    if M2 <= eps:
        return np.nan, (xc, yc), (np.nan, np.nan), {"reason": "no_signal_in_roi"}

    xc2 = float((w_roi * xx).sum() / M2)
    yc2 = float((w_roi * yy).sum() / M2)

    dx2 = (xx - xc2) ** 2
    dy2 = (yy - yc2) ** 2
    var_x = float((w_roi * dx2).sum() / M2)
    var_y = float((w_roi * dy2).sum() / M2)

    sigma_x_pix = np.sqrt(max(var_x, 0.0))
    sigma_y_pix = np.sqrt(max(var_y, 0.0))

    sigma_x_m = sigma_x_pix * dx
    sigma_y_m = sigma_y_pix * dx
    sigma_m = np.sqrt(0.5 * (sigma_x_m ** 2 + sigma_y_m ** 2))

    debug = {
        "bg": bg2,
        "sigma_bg": sigma_bg2,
        "thr_k": thresh_k_sigma,
        "roi_r_pix": roi_r,
        "r_eff_pix": r_eff,
    }
    return sigma_m, (xc2, yc2), (sigma_x_m, sigma_y_m), debug
def process_experiment_rms(root_folder, colonne="grande"):
    img_folder = find_image_folder(root_folder)
    files = list_tifs(img_folder)
    n = len(files)
    if n == 0:
        raise RuntimeError(f"No tif files found in {img_folder}")

    img0_u16 = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if img0_u16 is None:
        raise RuntimeError(f"Cannot read {files[0]}")
    frame0 = img0_u16.astype(np.float32) / 65535.0

    # pipeline workflow_new
    col_profile = compute_col_profile_from_frame0(frame0)

    mask_ref = cv2.imread(MASK_REF_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_ref is None:
        raise RuntimeError(f"Cannot read mask ref: {MASK_REF_PATH}")
    mask_ref_u8 = (mask_ref > 128).astype(np.uint8)
    mask_ref_u8 = cv2.resize(mask_ref_u8, frame0.shape[::-1], interpolation=cv2.INTER_NEAREST)

    frame0c = apply_col_profile(frame0, col_profile)
    hex_mask = compute_hex_mask_from_sato(frame0c, mask_ref_u8)

    t_img, _ = build_t_img_from_csv(root_folder, n)
    Ta, _ = extract_Ta_from_csv(root_folder, colonne=colonne)
    time = np.asarray(t_img, dtype=float) / float(Ta)

    sigma_m = np.full(n, np.nan, dtype=float)
    sigma_x = np.full(n, np.nan, dtype=float)
    sigma_y = np.full(n, np.nan, dtype=float)
    xc_pix = np.full(n, np.nan, dtype=float)
    yc_pix = np.full(n, np.nan, dtype=float)
    vars_I = np.full(n, np.nan, dtype=float)

    for i, f in enumerate(files):
        img_u16 = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img_u16 is None:
            continue

        frame = img_u16.astype(np.float32) / 65535.0
        frame = apply_col_profile(frame, col_profile)

        # t0 : variance sur la zone utile uniquement
        frame_var = frame.copy()
        frame_var[~hex_mask] = np.nan
        vars_I[i] = np.nanvar(frame_var)

        # RMS : garder le traitement rayon, donc PAS de masque hex ici
        s, (xc, yc), (sx, sy), _ = compute_rms_radius(frame, dx)
        sigma_m[i] = s
        sigma_x[i] = sx
        sigma_y[i] = sy
        xc_pix[i] = xc
        yc_pix[i] = yc

    valid_for_t0 = np.isfinite(vars_I) & np.isfinite(time)
    if not np.any(valid_for_t0):
        raise RuntimeError(f"No valid frame found to compute t0 in {root_folder}")

    idx_valid = np.where(valid_for_t0)[0]
    i0_local = int(np.nanargmax(vars_I[valid_for_t0]))
    i0 = idx_valid[i0_local]

    time0 = time - time[i0]

    return {
        "root": os.path.basename(root_folder),
        "label": read_label(root_folder),
        "time": time0,
        "Ta": Ta,
        "t0_index": i0,
        "t0_time": time[i0],
        "var_I": vars_I,
        "sigma_m": sigma_m,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "hex_mask": hex_mask,
        "xc_pix": xc_pix,
        "yc_pix": yc_pix,
    }
def run_one_experiment(root):
    try:
        return process_experiment_rms(root)
    except Exception as e:
        print(f"[ERROR] {root}: {e}")
        return None


# ==========================================================
# Plot comparaison façon workflow_new, mais rayon seulement
# ==========================================================
def plot_rms_comparison(results, normalize=False):
    style_map = {
        (10, "fine"): dict(marker="o", color="tab:blue", mfc="tab:blue", ms=4, alpha=0.8),
        (10, "coarse"): dict(marker="D", color="tab:blue", mfc="none", mew=1.2, ms=5, alpha=0.8),
        (10, "stokes"): dict(marker="^", color="purple", mfc="none", mew=1.2, ms=5, alpha=0.8),
        (6, "fine"): dict(marker="o", color="tab:orange", mfc="tab:orange", ms=4, alpha=0.8),
        (6, "coarse"): dict(marker="D", color="tab:orange", mfc="none", mew=1.2, ms=5, alpha=0.8),
        (3, "fine"): dict(marker="o", color="tab:red", mfc="tab:red", ms=4, alpha=0.8),
        (3, "coarse"): dict(marker="D", color="tab:red", mfc="none", mew=1.2, ms=5, alpha=0.8),
        (0, "fine"): dict(marker="o", color="tab:green", mfc="tab:green", ms=4, alpha=0.8),
        (0, "coarse"): dict(marker="D", color="tab:green", mfc="none", mew=1.2, ms=5, alpha=0.8),
    }

    def parse_label(label_text, root_name=""):
        txt = (label_text or "").lower().strip()

        # 1) cas standard : "10 mm", "6mm", etc.
        m = re.search(r'(?P<L>\d+(?:\.\d+)?)\s*mm\b', txt)

        # 2) secours : nombre seul parmi 0, 3, 6, 10
        if not m:
            m = re.search(r'(?<!\d)(?P<L>0|3|6|10)(?!\d)', txt)

        if not m:
            print(f"[WARN] Impossible de lire L dans label: {root_name} -> {label_text!r}")
            return None

        L_mm = int(float(m.group("L")))

        if "stokes" in txt:
            sand = "stokes"
        elif "coarse" in txt:
            sand = "coarse"
        elif "fine" in txt:
            sand = "fine"
        else:
            print(f"[WARN] Impossible de lire le type de sable: {root_name} -> {label_text!r}")
            return None

        return (L_mm, sand)

    fig, ax = plt.subplots(figsize=(8, 6))
    n_plotted = 0

    for res in results:
        key = parse_label(res.get("label", ""), res.get("root", ""))
        if key is None or key not in style_map:
            continue

        style = style_map[key]
        time = np.asarray(res["time"], dtype=float)
        rad = np.asarray(res["sigma_m"], dtype=float)
        i0= res["t0_index"]
        rad = rad/rad[i0]
        var = np.asarray(res["var_I"], dtype=float)
        var = var/np.max(var)

        valid = np.isfinite(time) & np.isfinite(rad) & (rad > 0)
        if np.any(valid):
            y = rad[valid] if normalize else 1e3 * rad[valid]
            ax.plot(time[valid][::2],var[::2], linestyle="None", **style)
            ax.plot(time[valid][::2],1e6*y[::2]**-2, linestyle="--", **style)
            n_plotted += 1
        else:
            print(f"[WARN] Aucun point valide pour {res.get('root', 'unknown')}")

    if n_plotted == 0:
        print("[ERROR] Aucune courbe tracée.")
        return

    legend_elements = []
    for (L_mm, sand), style in style_map.items():
        legend_elements.append(
            Line2D(
                [0], [0],
                marker=style.get("marker", "o"),
                linestyle="None",
                color=style.get("color", "k"),
                markerfacecolor=style.get("mfc", style.get("color", "k")),
                markeredgewidth=style.get("mew", 1.0),
                markersize=style.get("ms", 6),
                alpha=style.get("alpha", 1.0),
                label=f"{L_mm} mm, {sand}",
            )
        )

    ax.legend(handles=legend_elements)
    ax.axvline(0.0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel(r"$(t - t_0)/t_a$", fontsize=16)
    ax.set_ylabel(r"$\sigma_{RMS}$ / $\sigma_{RMS,max}$" if normalize else r"$R(t)^2 \cdot Var(C) /(R_0^2 \cdot Var_0) $", fontsize=16)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlim(0.1,42)
    fig.tight_layout()
    ax.set_yscale("log")
    plt.show()

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    REQUEST = [
        # (0, "coarse"),
        # (10, "fine"),
        # (10, "coarse"),
        # (10, "stokes"),
        # (6, "fine"),
        # (6, "coarse"),
        # (3, "coarse"),
        (0, "fine"),
         # (3,'fine')
    ]

    root_folders = select_exact_combinations(BASE_PATH, REQUEST)
    print("Experiments sélectionnées :", len(root_folders))
    if not root_folders:
        raise RuntimeError("Aucune expérience sélectionnée.")

    nproc = max(1, min(cpu_count() - 1, len(root_folders)))
    with Pool(nproc) as pool:
        results = pool.map(run_one_experiment, root_folders)

    results = [r for r in results if r is not None]
    print("Experiments OK :", len(results))

    plot_rms_comparison(results, normalize=False)
    plt.show()
    # plot_rms_comparison(results, normalize=True)
