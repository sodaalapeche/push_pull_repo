import os
from glob import glob
import re
import gc
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from multiprocessing import Pool, cpu_count



# ==========================================================
# GLOBAL PARAMETERS
# ==========================================================
BASE_PATH = "/home/chorus/HETEROGENE_binned/"
dx = (0.065 / 2048) * 4  # m/pixel

# --- Méthode n°1 (moments pondérés / RMS) ---
BG_PERCENTILE = 10.0     # estimation du fond (percentile bas)
BLUR_SIGMA = 2.0       # stabilise le centre / moments (low-pass)
MAX_R_FRAC = 0.48       # on limite le calcul à un disque central (évite bruit loin)
MIN_SIGNAL = 1e-6      # seuil de sécurité


# ==========================================================
# IMAGE UTILITIES (repris)
# ==========================================================
def find_image_folder(root):
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and glob(os.path.join(p, "*.tif")):
            return p
    raise RuntimeError(f"No TIFF folder found under: {root}")


def load_full_sequence_16bit(folder):
    files = sorted(
        glob(os.path.join(folder, "*.tif")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    if not files:
        raise RuntimeError(f"No .tif found in {folder}")

    img0 = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise RuntimeError(f"Cannot read {files[0]}")

    h, w = img0.shape
    stack = np.empty((len(files), h, w), dtype=np.float32)

    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Cannot read {f}")
        stack[i] = img.astype(np.float32) / 65535.0

    return stack, files


def correct_vertical_stripes(I, smooth=2, eps=1e-6):
    ref = I[0]
    col_profile = np.nanmean(ref, axis=0)

    if smooth and smooth > 3:
        col_profile = ndimage.median_filter(col_profile, size=smooth)

    col_profile /= np.nanmean(col_profile)
    col_profile = np.maximum(col_profile, eps)

    return I / col_profile[None, None, :]


# ==========================================================
# TIME UTILITIES (repris)
# ==========================================================
def build_t_img_from_csv(root_folder, n_images):
    """
    Reconstruit le temps image à partir de weight_data.csv
    Hypothèse : 1 image ↔ 1 ligne du CSV, pas de temps constant
    """
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)

    t = df["Timestamp"].values.astype(float)

    if len(t) < 2:
        raise ValueError("Pas assez de timestamps dans weight_data.csv")

    # pas de temps physique (robuste)
    dt = np.median(np.diff(t))

    # temps image centré sur t[0]
    t_img = t[0] + dt * np.arange(n_images)

    return t_img, dt


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
        A_pore = A_tot * eps_sable * 1
    elif Tip == "homo fine":
        A_pore = A_tot * eps_sable * 0.8
    elif Tip == "0.1":
        A_pore = A_tot * eps_sable * (1 - f_billes) * 1.7
    elif Tip == "0.6":
        A_pore = A_tot * eps_sable * (1 - f_billes) * 0.85
    elif Tip == "0.3 coarse":
        A_pore = A_tot * eps_sable * (1 - f_billes)
    elif Tip == "0.6 fine":
        A_pore = A_tot * eps_sable * (1 - f_billes) * 1.23
    else:
        A_pore = A_tot * eps_sable * (1 - f_billes)

    vp = Q_m3_s / A_pore
    Ta = L / vp
    return Ta, vp


def extract_Ta_from_csv(root_folder, colonne="grande", Tip="homo"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)

    D = 0.055 if colonne == "grande" else 0.027
    _ = read_label(root_folder)

    # ton choix: forcer L=0.01
    L = 0.01

    Ta, vp = Ta_for_bead(df["Timestamp"], df["Weight"], D=D, L=L, Tip=Tip)
    return Ta, vp


# ==========================================================
# MÉTHODE N°1 : moments pondérés / RMS
# ==========================================================
def compute_rms_radius(
    img,
    dx,
    blur_center_sigma=10.0,     # gros lissage pour centre
    blur_moments_sigma=2.0,     # petit lissage pour moments (évite dilatation)
    ring_rmin_frac=0.38,        # couronne fond: r in [rmin, rmax]
    ring_rmax_frac=0.48,
    thresh_k_sigma=4.0,         # seuil: bg + k*sigma (met 0 pour désactiver)
    roi_k=2.5,                  # ROI radius = roi_k * r_eff
    roi_rmin_pix=25.0,          # ROI minimale en pixels
    roi_rmax_frac=0.48,         # ROI max (fraction de min(H,W))
    eps=1e-12
):
    """
    Moments pondérés (RMS) robustes:
      - centre sur enveloppe lissée (blur_center_sigma)
      - bg et sigma_bg estimés sur couronne (loin du blob)
      - moments sur ROI adaptative autour du centre
      - option: seuil bg + k*sigma pour éviter les résidus étalés

    Retour:
      sigma_m, (xc_pix, yc_pix), (sigma_x_m, sigma_y_m), debug
    """

    img = img.astype(np.float32)
    H, W = img.shape
    yy, xx = np.indices((H, W))

    # 1) centre stable (sur image très lissée)
    env_c = cv2.GaussianBlur(img, (0, 0), blur_center_sigma)

    # centre provisoire = centre image (sert à définir la couronne)
    cx0, cy0 = W / 2.0, H / 2.0
    r0 = np.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)

    rmax_global = roi_rmax_frac * min(H, W)
    rmin_ring = ring_rmin_frac * min(H, W)
    rmax_ring = min(ring_rmax_frac * min(H, W), rmax_global)

    ring = (r0 >= rmin_ring) & (r0 <= rmax_ring)
    if not np.any(ring):
        # fallback: fond global percentile si la couronne est invalide
        bg = float(np.percentile(env_c, 5.0))
        sigma_bg = float(np.std(env_c))
    else:
        vals = env_c[ring]
        bg = float(np.median(vals))
        sigma_bg = float(1.4826 * np.median(np.abs(vals - bg)))  # MAD->sigma

    w_c = env_c - bg
    w_c[w_c < 0] = 0.0
    Mc = w_c.sum()
    if Mc <= eps:
        return np.nan, (np.nan, np.nan), (np.nan, np.nan), {"reason": "no_signal_center"}

    xc = float((w_c * xx).sum() / Mc)
    yc = float((w_c * yy).sum() / Mc)

    # 2) image pour moments (peu lissée)
    env_m = cv2.GaussianBlur(img, (0, 0), blur_moments_sigma)

    # fond/sigma re-estimés sur couronne centrée sur (xc,yc)
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
    ring2 = (r >= rmin_ring) & (r <= rmax_ring)
    if np.any(ring2):
        vals2 = env_m[ring2]
        bg2 = float(np.median(vals2))
        sigma_bg2 = float(1.4826 * np.median(np.abs(vals2 - bg2)))
    else:
        bg2 = bg
        sigma_bg2 = sigma_bg

    w = env_m - bg2
    w[w < 0] = 0.0

    # 3) seuil bruit (évite que le voile lointain domine quand blob est petit)
    if thresh_k_sigma and thresh_k_sigma > 0:
        thr = bg2 + thresh_k_sigma * sigma_bg2
        keep = env_m >= thr
        w = np.where(keep, w, 0.0)

    # 4) ROI adaptative: on estime une échelle r_eff via énergie cumulative sur w
    #    r_eff = rayon contenant 80% de l'énergie (sur w), puis ROI = roi_k * r_eff
    #    (ça suit l'apparition très tôt)
    wpos = w.copy()
    wpos[r > rmax_global] = 0.0
    M = wpos.sum()
    if M <= eps:
        return np.nan, (xc, yc), (np.nan, np.nan), {"reason": "no_signal_moments"}

    # profil radial grossier (bins de 1 px)
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

    # 5) moments d'ordre 2 dans la ROI
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
        "r_eff_pix": r_eff
    }
    return sigma_m, (xc2, yc2), (sigma_x_m, sigma_y_m), debug
# ==========================================================
# PROCESS EXPERIMENT: calcule sigma(t) + live display
# ==========================================================
def process_experiment_for_sigma(root_folder, out_dir, colonne="grande"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {root_folder} (missing weight_data.csv)")
        return None

    img_folder = find_image_folder(root_folder)
    I, files = load_full_sequence_16bit(img_folder)

    # correction stries verticales
    I = correct_vertical_stripes(I)

    # temps normalisé
    t_img, dt = build_t_img_from_csv(root_folder, I.shape[0])
    Ta, vp = extract_Ta_from_csv(root_folder, colonne=colonne)
    time = t_img / Ta
    var = np.nanvar(I, axis=(1, 2))
    mean= np.nanmean(I,axis=(1,2))
    i0 = np.argmax(var)
    time = time - time[i0]
    nt = I.shape[0]
    sigma = np.full(nt, np.nan, dtype=float)
    xc = np.full(nt, np.nan, dtype=float)
    yc = np.full(nt, np.nan, dtype=float)

    os.makedirs(out_dir, exist_ok=True)

    # # LIVE DISPLAY
    # plt.ion()
    # fig, (ax_r, ax_im) = plt.subplots(
    #     2, 1, figsize=(7, 9),
    #     gridspec_kw={"height_ratios": [1, 3]}
    # )
    #
    # line_r, = ax_r.plot([], [], "-o", ms=3)
    # ax_r.set_xlabel("t / Ta")
    # ax_r.set_ylabel(r"$\sigma_{RMS}$ (mm)")
    # ax_r.grid(True, ls="--", alpha=0.3)
    #
    # im_handle = ax_im.imshow(I[0], cmap="viridis")
    # ax_im.axis("off")
    #
    # # contraste stable
    # sample_idx = np.linspace(0, nt - 1, min(nt, 10), dtype=int)
    # sample_vals = np.concatenate([I[k].ravel() for k in sample_idx])
    # vmin, vmax = np.percentile(sample_vals, (2, 99.5))
    # im_handle.set_clim(vmin, vmax)
    #
    for t in range(nt):
        sigma_m, (xc_pix, yc_pix), (sx_m, sy_m), dbg = compute_rms_radius(I[t], dx)

        sigma[t] = sigma_m
        xc[t] = xc_pix
        yc[t] = yc_pix

    #     # update courbe
    #     line_r.set_data(time[:t + 1], sigma[:t + 1] * 1e3)
    #     ax_r.relim()
    #     ax_r.autoscale_view()
    #
    #     # update image
    #     im_handle.set_data(I[t])
    #     if np.isfinite(sigma[t]):
    #         title = (
    #             f"frame {t + 1}/{nt} | t/Ta={time[t]:.4g} | "
    #             fr"$\sigma$={sigma[t] * 1e3:.3f} mm | ROI={dbg.get('roi_r_pix', np.nan):.0f}px"
    #         )
    #     else:
    #         title = f"frame {t + 1}/{nt} | t/Ta={time[t]:.4g} | sigma=NaN"
    #     ax_im.set_title(title)
    #
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
    #     plt.pause(0.001)
    #
    # plt.ioff()
    # plt.show()


    return {
        "root": os.path.basename(root_folder),
        "img_folder": img_folder,
        "time": time,
        "sigma_m": sigma,
        "xc_pix": xc,
        "yc_pix": yc,
        "files": files,
        "var":var,
        'mean':mean
    }

def plot_minus_one_slope(ax,slope, start_x, start_y, length_decades=1, offset_factor=1.2, **kwargs):
    """
    Ajoute une droite de pente -1 en log-log, partant de (start_x, start_y*offset_factor).
    length_decades : nombre de décades sur x à tracer.
    """
    # Génère une plage de x logarithmique
    x_vals = np.logspace(
        np.log10(start_x),
        np.log10(start_x) + length_decades,
        100
    )
    # Courbe de pente -1
    y_vals = start_y * offset_factor * (x_vals / start_x) ** (slope)

    # Trace sur les axes existants (uniquement bottom log-log)
    ax.plot(x_vals, y_vals, label=r"$t^{-1}$", **kwargs)

    # Place un petit texte à la fin de la courbe
    ax.text(
        x_vals[-1]*0.7, y_vals[-1]*2,
        f"slope = {slope}",
        fontsize="x-large",
        ha="left", va="top"
    )
# ==========================================================
# DISCOVERY HELPERS
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

            tif_files = glob(os.path.join(exp_path, "*", "*.tif"), recursive=True)
            csv_path = os.path.join(exp_path, "weight_data.csv")

            if tif_files and os.path.isfile(csv_path):
                selected.append(exp_path)
            else:
                if not tif_files:
                    print(f"[SKIP] No TIFF in experiment folder: {exp_path}")
                if not os.path.isfile(csv_path):
                    print(f"[SKIP] Missing weight_data.csv: {exp_path}")
    return selected

from multiprocessing import Pool, cpu_count
import gc
import os

def run_one_experiment(root_folder):
    out_dir = os.path.join(root_folder, "rms_moments")
    try:
        res = process_experiment_for_sigma(root_folder, out_dir, colonne="grande")
        return res
    except Exception as e:
        print(f"[FAIL] {root_folder} -> {e}")
        return None
# ==========================================================
# MAIN
REQUEST = [
    (0, "coarse"),
    (10, "fine"),
    (10, "coarse"),
    (10,"stokes"),
    (6, "fine"),
    (6, "coarse"),
    (3, "coarse"),
    (0, "fine"),
]

root_folders = select_exact_combinations(BASE_PATH, REQUEST)
print("Experiments sélectionnées :", len(root_folders))

if len(root_folders) == 0:
    raise RuntimeError("Aucune expérience sélectionnée (vérifie la recherche de .tif et weight_data.csv).")

nproc = max(1, min(cpu_count() - 1, len(root_folders)))
with Pool(nproc) as pool:
    results = pool.map(run_one_experiment, root_folders)

# filtre None
results = [r for r in results if r is not None]
print("Experiments OK :", len(results))

# --- PLOT UNIQUE (toutes les courbes sur le même graph) ---
fig, ax = plt.subplots()

for res in results:
    time = res["time"]
    sigma = res["sigma_m"]
    mean= res['mean']# mm
    var= res["var"]
    var = (var/mean**2
           )
    var= var/np.max(var)

    max2 = sigma[40]
    sigma = (sigma)
             #**2 * var)
    mask = (time > 0) & np.isfinite(sigma) & (sigma > 0)
    if np.any(mask):
        label = res.get("root", "exp")
        ax.plot(time[mask], sigma[mask], "-o", ms=2, label=label)

    gc.collect()

ax.set_xlabel("t / Ta")
ax.set_ylabel(r"$\sigma_{RMS}$ (mm)")
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid(True, ls="--", alpha=0.3)
ax.set_xlim(0, 33)
# ax.set_ylim(-0.1, 15)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
plt.show()