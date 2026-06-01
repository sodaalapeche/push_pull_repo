import os
import re
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

width = 18
height = width * 0.5
font_size = 11
inches = 2.54

# ==========================================================
# PARAMÈTRES GLOBAUX
# ==========================================================
BASE_PATH     = "/home/chorus/HETEROGENE_binned/"
MASK_REF_PATH = "/home/chorus/test/masque.jpg"
SAVE_PATH     = "/home/chorus/data_rms.npy"

dx = (0.06 / 2048) * 8   # m/pixel

# ROI fixe (pixels) — même valeur que ROIfixed_deriv.py
R_EVAL_PIX = 190
num_pixels = int(np.pi * R_EVAL_PIX**2)
s_roi = 0.06 / 512

# Paramètres compute_rms_radius
RMS_BLUR_CENTER_SIGMA  = 10.0
RMS_BLUR_MOMENTS_SIGMA = 2.0
RMS_RING_RMIN_FRAC     = 0.38
RMS_RING_RMAX_FRAC     = 0.48
RMS_THRESH_K_SIGMA     = 4.0
RMS_ROI_K              = 2.5
RMS_ROI_RMIN_PIX       = 25.0
RMS_ROI_RMAX_FRAC      = 0.48

# Fraction de masse définissant le rayon effectif : 68 % = 1σ pour une gaussienne
MASS_FRACTION_1SIGMA = 0.68

# Offset pour la détection du min de d(mean)/dt après i0
OFFSET_AFTER_I0 = 30

# ==========================================================
# EXPÉRIENCES À TRAITER
# ==========================================================
REQUEST = [
    (10, "fine"),
    (0,  "fine"),
    (0,  "coarse"),
    (10, "coarse"),
    (3,  "coarse"),
    (6,  "coarse"),
    (6,  "fine"),
    (3,  "fine"),
    (10, "stokes"),
    (10, "contactless"),
]

# ==========================================================
# DISCOVERY
# ==========================================================
def select_exact_combinations(base_path, combinations):
    selected = []
    for L_mm, sand in combinations:
        path = os.path.join(base_path, f"{L_mm}mm", sand.lower())
        if not os.path.exists(path):
            print(f"[WARN] Folder not found: {path}")
            continue
        for d in os.listdir(path):
            exp_path = os.path.join(path, d)
            if not os.path.isdir(exp_path):
                continue
            tif_files = glob(os.path.join(exp_path, "**", "*.tif"), recursive=True)
            csv_path  = os.path.join(exp_path, "weight_data.csv")
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
    return open(p).read().strip() if os.path.exists(p) else os.path.basename(root)

# ==========================================================
# TEMPS : grille d'images à partir du CSV (identique ROIfixed_deriv)
# ==========================================================
def build_t_img_from_csv(root_folder, n_images):
    df = pd.read_csv(os.path.join(root_folder, "weight_data.csv"))
    t  = df["Timestamp"].values.astype(float)
    dt = np.median(np.diff(t))
    return t[0] + dt * np.arange(n_images)

# ==========================================================
# STRIPE CORRECTION
# ==========================================================
def compute_col_profile_from_frame0(frame0, smooth=2, eps=1e-6):
    col_profile = np.nanmean(frame0, axis=0)
    if smooth and smooth > 3:
        col_profile = ndimage.median_filter(col_profile, size=smooth)
    col_profile /= np.nanmean(col_profile)
    return np.maximum(col_profile, eps)


def apply_col_profile(frame, col_profile):
    return frame / col_profile[None, :]

# ==========================================================
# COMPUTE_RMS_RADIUS
#
# Méthode :
#   1. Localise le centroïde du blob (image fortement lissée, fond estimé
#      sur un anneau externe).
#   2. Calcule la distribution radiale de masse autour du centroïde.
#   3. Définit un rayon effectif r_eff à la fraction de masse MASS_FRACTION_1SIGMA
#      (= 0.68, convention 1σ pour une gaussienne isotrope).
#   4. ROI adaptative de rayon roi_k * r_eff.
#   5. Moments d'ordre 2 pondérés par l'intensité dans cette ROI :
#         σ_x² = Σ w (x - xc)² / Σ w
#         σ_y² = Σ w (y - yc)² / Σ w
#      σ_m = sqrt((σ_x² + σ_y²) / 2)   →   écart-type moyen en mètres
#
# C'est bien un estimateur d'écart-type (1σ) de la distribution de
# concentration projetée — pas un rayon à 95 % ou un rayon à mi-hauteur.
# ==========================================================
def compute_rms_radius(img,
                       blur_center_sigma=RMS_BLUR_CENTER_SIGMA,
                       blur_moments_sigma=RMS_BLUR_MOMENTS_SIGMA,
                       ring_rmin_frac=RMS_RING_RMIN_FRAC,
                       ring_rmax_frac=RMS_RING_RMAX_FRAC,
                       thresh_k_sigma=RMS_THRESH_K_SIGMA,
                       roi_k=RMS_ROI_K,
                       roi_rmin_pix=RMS_ROI_RMIN_PIX,
                       roi_rmax_frac=RMS_ROI_RMAX_FRAC,
                       mass_fraction=MASS_FRACTION_1SIGMA,
                       eps=1e-12):
    img  = img.astype(np.float32)
    H, W = img.shape
    yy, xx = np.indices((H, W))

    # --- centre grossier (image lissée fortement) ---
    env_c = cv2.GaussianBlur(img, (0, 0), blur_center_sigma)
    cx0, cy0 = W / 2.0, H / 2.0
    r0 = np.sqrt((xx - cx0)**2 + (yy - cy0)**2)

    rmax_global = roi_rmax_frac * min(H, W)
    rmin_ring   = ring_rmin_frac * min(H, W)
    rmax_ring   = min(ring_rmax_frac * min(H, W), rmax_global)

    ring = (r0 >= rmin_ring) & (r0 <= rmax_ring)
    if not np.any(ring):
        bg, sigma_bg = float(np.percentile(env_c, 5.0)), float(np.std(env_c))
    else:
        vals = env_c[ring]
        bg   = float(np.median(vals))
        sigma_bg = float(1.4826 * np.median(np.abs(vals - bg)))

    w_c = np.clip(env_c - bg, 0, None)
    Mc  = w_c.sum()
    if Mc <= eps:
        return np.nan, (np.nan, np.nan)

    xc = float((w_c * xx).sum() / Mc)
    yc = float((w_c * yy).sum() / Mc)

    # --- moments d'ordre 2 (image lissée finement) ---
    env_m = cv2.GaussianBlur(img, (0, 0), blur_moments_sigma)
    r     = np.sqrt((xx - xc)**2 + (yy - yc)**2)

    ring2 = (r >= rmin_ring) & (r <= rmax_ring)
    if np.any(ring2):
        vals2     = env_m[ring2]
        bg2       = float(np.median(vals2))
        sigma_bg2 = float(1.4826 * np.median(np.abs(vals2 - bg2)))
    else:
        bg2, sigma_bg2 = bg, sigma_bg

    w = np.clip(env_m - bg2, 0, None)
    if thresh_k_sigma and thresh_k_sigma > 0:
        thr = bg2 + thresh_k_sigma * sigma_bg2
        w   = np.where(env_m >= thr, w, 0.0)

    wpos = w.copy()
    wpos[r > rmax_global] = 0.0
    M = wpos.sum()
    if M <= eps:
        return np.nan, (xc, yc)

    # --- rayon effectif à 1σ (68 % de la masse) → ROI adaptative ---
    rbin  = np.floor(r).astype(np.int32)
    nbins = max(int(np.floor(rmax_global)) + 1, 10)
    E     = np.bincount(rbin.ravel(), weights=wpos.ravel(), minlength=nbins).astype(np.float64)
    Ecum  = np.cumsum(E)
    Etot  = Ecum[-1]
    if Etot <= eps:
        return np.nan, (xc, yc)

    # Fraction de masse = 1σ (68 %) pour une gaussienne isotrope
    idx_1sigma = int(np.searchsorted(Ecum, mass_fraction * Etot))
    r_eff = float(np.clip(idx_1sigma, 1, rmax_global))
    roi_r = float(np.clip(roi_k * r_eff, roi_rmin_pix, rmax_global))
    roi   = r <= roi_r

    w_roi = np.where(roi, w, 0.0)
    M2    = w_roi.sum()
    if M2 <= eps:
        return np.nan, (xc, yc)

    # --- moments d'ordre 2 dans la ROI ---
    xc2 = float((w_roi * xx).sum() / M2)
    yc2 = float((w_roi * yy).sum() / M2)

    var_x = float((w_roi * (xx - xc2)**2).sum() / M2)
    var_y = float((w_roi * (yy - yc2)**2).sum() / M2)

    # AFTER (correct):
    var_x_m = max(var_x, 0.0) * dx ** 2
    var_y_m = max(var_y, 0.0) * dx ** 2
    R_rms = np.sqrt(var_x_m + var_y_m)  # ← true radial RMS: R² = σ_x² + σ_y²
    return R_rms, (xc2, yc2)

# ==========================================================
# ROI STATS
# ==========================================================
def roi_stats(frame, Xpix, Ypix, xc_pix, yc_pix, r_eval_pix):
    r2  = (Xpix - xc_pix)**2 + (Ypix - yc_pix)**2
    roi = r2 <= r_eval_pix**2
    vals = frame[roi]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.var(vals))

# ==========================================================
# PROCESS EXPERIMENT
# ==========================================================
def process_experiment(root_folder):
    img_folder = find_image_folder(root_folder)
    files      = list_tifs(img_folder)
    n          = len(files)

    # --- frame 0 + stripe correction ---
    img0 = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise RuntimeError(f"Cannot read {files[0]}")
    frame0      = img0.astype(np.float32) / 65535.0
    col_profile = compute_col_profile_from_frame0(frame0)

    H, W = frame0.shape
    Ypix, Xpix = np.indices((H, W))

    # --- grille temporelle ---
    t_img = build_t_img_from_csv(root_folder, n)

    # --- passe 1 : rayon RMS sur toutes les frames ---
    sigma_m = np.full(n, np.nan, dtype=float)
    xc_arr  = np.full(n, np.nan, dtype=float)
    yc_arr  = np.full(n, np.nan, dtype=float)

    for i, f in enumerate(files):
        img_u16 = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img_u16 is None:
            continue
        frame = img_u16.astype(np.float32) / 65535.0
        frame = apply_col_profile(frame, col_profile)
        tmp   = np.nan_to_num(frame, nan=0.0)

        s, (xc, yc) = compute_rms_radius(tmp)
        sigma_m[i]  = s
        xc_arr[i]   = xc
        yc_arr[i]   = yc

    # --- passe 2 : mean/var dans la ROI fixe centrée sur l'image ---
    # ROI FIXE de rayon R_EVAL_PIX pixels, centrée sur (W/2, H/2)
    xc_fixed = W / 2.0
    yc_fixed = H / 2.0

    mean = np.full(n, np.nan, dtype=float)
    var  = np.full(n, np.nan, dtype=float)

    for i, f in enumerate(files):
        img_u16 = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img_u16 is None:
            continue
        frame  = img_u16.astype(np.float32) / 65535.0
        frame  = apply_col_profile(frame, col_profile)
        m, v   = roi_stats(frame, Xpix, Ypix, xc_fixed, yc_fixed, R_EVAL_PIX)
        mean[i] = m
        var[i]  = v

    # ==========================================================
    # i0 = argmax(var / mean²)  (pic de variance normalisée)
    # ==========================================================
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_var = var / (mean**2)
    i0 = int(np.nanargmax(norm_var))

    # ==========================================================
    # dérivée de la masse + détection du min après i0 + OFFSET
    # ==========================================================
    dmean_dt = np.gradient(mean, t_img)
    i_start = min(i0 + OFFSET_AFTER_I0, len(mean) - 1)

    if i_start < len(mean) - 1:
        i_min = i_start + int(np.nanargmin(dmean_dt[i_start:]))
    else:
        i_min = i0

    # ==========================================================
    # Ta via la dérivée de la masse
    # Ta_30 = temps entre i0 et i_min  (parcours de 30 cm typiquement)
    # Ta    = Ta_30 / 30  → temps d'advection sur 1 cm
    # ==========================================================
    Ta_30 = t_img[i_min] - t_img[i0]
    Ta    = Ta_30 / 30.0
    u     = 0.01 / Ta

    # Temps réduit
    time = (t_img - t_img[i0]) / Ta

    # Aire ROI en m² (cohérence avec les autres scripts)
    A = num_pixels * s_roi**2

    return {
        "label":   read_label(root_folder),
        "time":    time,
        "Ta":      Ta,
        "i0":      i0,
        "mean":    mean,
        "var":     var,
        "sigma_m": sigma_m,   # rayon RMS (1σ) [m], toutes frames
        "A":       A,
    }

# ==========================================================
# MULTIPROCESS
# ==========================================================
def run_one_experiment(root):
    try:
        res = process_experiment(root)
        print(f"  [OK] {os.path.basename(root)}  Ta={res['Ta']:.2f} s  "
              f"i0={res['i0']}")
        return res
    except Exception as e:
        print(f"  [ERROR] {root}: {e}")
        return None

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    root_folders = select_exact_combinations(BASE_PATH, REQUEST)
    print(f"Expériences trouvées : {len(root_folders)}")
    if not root_folders:
        raise RuntimeError("Aucune expérience sélectionnée.")

    nproc = max(1, min(cpu_count() - 2, len(root_folders)))
    with Pool(nproc) as pool:
        results = pool.map(run_one_experiment, root_folders)

    results = [r for r in results if r is not None]
    print(f"\nExpériences OK : {len(results)}")

    np.save(SAVE_PATH, results, allow_pickle=True)
    print(f"Sauvegardé : {SAVE_PATH}")

    # ==========================================================
    # PLOT  σ_m(t/Ta)²/σ_m(i0)² - 1  [adim]
    # ==========================================================
    from matplotlib.lines import Line2D

    style_map = {
        (10, "fine"):        dict(marker="o", color="tab:blue",   mfc="tab:blue",   ms=5, alpha=0.8),
        (10, "coarse"):      dict(marker="D", color="tab:blue",   mfc="none",       mew=1, ms=5, alpha=0.8),
        (6,  "fine"):        dict(marker="o", color="tab:orange", mfc="tab:orange", ms=5, alpha=0.8),
        (6,  "coarse"):      dict(marker="D", color="tab:orange", mfc="none",       mew=1, ms=5, alpha=0.8),
        (3,  "fine"):        dict(marker="o", color="tab:red",    mfc="tab:red",    ms=5, alpha=0.8),
        (3,  "coarse"):      dict(marker="D", color="tab:red",    mfc="none",       mew=1, ms=5, alpha=0.8),
        (0,  "fine"):        dict(marker="o", color="tab:green",  mfc="tab:green",  ms=5, alpha=0.8),
        (0,  "coarse"):      dict(marker="D", color="tab:green",  mfc="none",       mew=1, ms=5, alpha=0.8),
        (10, "stokes"):      dict(marker="x", color="tab:purple", mfc="none",       mew=1, ms=5, alpha=0.8),
        (10, "contactless"): dict(marker="o", color="tab:pink",   mfc="tab:pink",   ms=4, alpha=0.8),
    }

    def parse_label(label_text):
        lines = [ln.strip().lower() for ln in label_text.split("\n") if ln.strip()]
        if len(lines) < 3:
            return None
        try:
            return int(lines[1].replace("mm", "")), lines[2]
        except Exception:
            return None

    fig, ax = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

    for res in results:
        info = parse_label(res.get("label", ""))
        if info is None:
            continue
        L_mm, sand = info
        key   = (L_mm, sand)
        style = style_map.get(key, dict(marker="s", color="gray", mfc="gray", ms=4, alpha=0.8))

        i0      = res["i0"]
        time    = res["time"]
        sigma_m = res["sigma_m"]
        sigma_m = sigma_m #/ sigma_m[i0]
        valid   = np.isfinite(time) & np.isfinite(sigma_m) & (sigma_m > 0)
        ax.plot(2*time[valid][::3], (sigma_m[valid][::3]) ,
                linestyle="None", **style)

    legend_elements = []
    for (L_mm, sand), style in style_map.items():
        legend_elements.append(
            Line2D([0], [0],
                   marker=style["marker"], linestyle="None",
                   color=style["color"],
                   markerfacecolor=style.get("mfc", style["color"]),
                   markeredgewidth=style.get("mew", 1.0),
                   markersize=style.get("ms", 5),
                   alpha=style.get("alpha", 1.0),
                   label=f"{L_mm} mm, {sand}")
        )

    # t_ref  = np.logspace(np.log10(1), np.log10(60), 100)
    # factor = 0.5
    # ax.plot(t_ref, factor * t_ref, 'k--', label=r"$\propto t^1$")

    ax.legend(handles=legend_elements, ncol=2)
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel(r"$2t \;/\; T_a$")
    ax.set_ylabel(r"$R$")
    ax.set_xlim(left=-4, right=60)
    ax.set_ylim( top=0.015)
    # ax.set_ylim(bottom=0.5, top=100)

    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig("/home/chorus/radiuses.pdf")
    plt.show()