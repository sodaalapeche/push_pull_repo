import os
from glob import glob
import re
import math as m

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.signal import fftconvolve
import skimage.filters as sk


# ==========================================================
# GLOBAL PARAMETERS (repris de ton script)
# ==========================================================
BASE_PATH = "/home/chorus/HETEROGENE_binned/"
MASK_REF_PATH = "/home/chorus/test/masque.jpg"

dx = (0.06 / 2048) * 4  # m/pixel (comme chez toi)

# --- Paramètres encircled energy ---
P_ENERGY = 0.4      # R80 par défaut (tu peux mettre 0.5, 0.9, etc.)
BG_PERCENTILE = 5.0      # estimation du fond (percentile bas)
CLOSE_KSIZE = 5        # comblement des lignes sombres (closing)
BLUR_SIGMA = 2.0 # enveloppe (low-pass) pour stabiliser centre + intégrale
MAX_R_FRAC =0.48        # limite de rayon max (fraction de min(H,W))
DR_PIX = 1               # pas radial en pixels

# --- Recalage masque hex (repris) ---
ANGLE_MAX = 2.0
ANGLE_STEP = 0.1
TRANS_MAX = 30


# ==========================================================
# IMAGE UTILITIES (repris/adaptés)
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

def extract_structure_sato(img):
    I_sato = sk.sato(img, sigmas=range(4, 10, 1))
    I_sato = ndimage.median_filter(I_sato, size=13)

    mask_sato = I_sato > 1e-5

    H, W = img.shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask_circle = (X - W/2)**2 + (Y - H/2)**2 > 213**2

    structure = mask_sato & (~mask_circle)
    return structure.astype(np.uint8)

def build_t_img_from_csv(root_folder, n_images):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)
    t = df["Timestamp"].values.astype(float)
    if len(t) < 2:
        raise ValueError("Pas assez de timestamps dans weight_data.csv")

    dt = np.median(np.diff(t))
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
    A_tot = np.pi * (D / 2)**2

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
    label = read_label(root_folder)

    D = 0.055 if colonne == "grande" else 0.027
    L = extract_L_from_label(root_folder)
    # Ici ton script force L=0.01 à la fin — je reproduis ce choix :
    L = 0.01

    Ta, vp = Ta_for_bead(df["Timestamp"], df["Weight"], D=D, L=L, Tip=Tip)
    return Ta, vp


# ==========================================================
# MÉTHODE N°2 : rayon par énergie encerclée avec correction couverture
# ==========================================================
def _make_aperture_mask_from_hex(hex_mask_bool, ksize=31):
    """
    Construit un masque d'ouverture "aperture" (contour externe / ROI globale),
    en "bouchant" les lignes du honeycomb.
    """
    u8 = (hex_mask_bool.astype(np.uint8) * 255)
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    aperture = closed > 0
    aperture = ndimage.binary_fill_holes(aperture)
    return aperture

def compute_encircled_energy_radius(
    img,
    dx,
    p=P_ENERGY,
    close_ksize=CLOSE_KSIZE,
    blur_sigma=BLUR_SIGMA,
    bg_percentile=BG_PERCENTILE,
    max_r_frac=MAX_R_FRAC,
    dr_pix=DR_PIX,
    eps=1e-12
):
    """
    Version simplifiée sans masque hexagonal.
    Rayon basé sur énergie encerclée directe.
    """

    img = img.astype(np.float32)
    H, W = img.shape

    # --- lissage léger pour stabilité ---
    k = int(close_ksize)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    env = cv2.GaussianBlur(img_close, (0, 0), blur_sigma)

    # --- soustraction fond ---
    bg = np.percentile(env, bg_percentile)
    env2 = env - bg
    env2[env2 < 0] = 0.0

    M = env2.sum()
    if M <= eps:
        return np.nan, (np.nan, np.nan), {}

    yy, xx = np.indices((H, W))
    xc = (env2 * xx).sum() / M
    yc = (env2 * yy).sum() / M

    r = np.sqrt((xx - xc)**2 + (yy - yc)**2)

    rmax = max_r_frac * min(H, W)
    nbins = int(np.floor(rmax / dr_pix))
    if nbins < 5:
        return np.nan, (xc, yc), {}

    rbin = np.floor(r / dr_pix).astype(np.int32)
    valid = (rbin >= 0) & (rbin < nbins)

    E = np.bincount(
        rbin[valid].ravel(),
        weights=env2[valid].ravel(),
        minlength=nbins
    ).astype(np.float64)

    Ecum = np.cumsum(E)
    Etot = Ecum[-1]
    if Etot <= eps:
        return np.nan, (xc, yc), {}

    target = p * Etot
    idx = int(np.searchsorted(Ecum, target))
    idx = np.clip(idx, 0, nbins - 1)

    Rp_pix = (idx + 0.5) * dr_pix
    Rp_m = Rp_pix * dx

    return Rp_m, (xc, yc), {}

# ==========================================================
# PROCESS EXPERIMENT: calcule Rp(t) + overlays
# ==========================================================
def process_experiment_for_Rp(root_folder, out_dir, colonne="grande"):

    csv_path = os.path.join(root_folder, "weight_data.csv")
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {root_folder} (missing weight_data.csv)")
        return None

    img_folder = find_image_folder(root_folder)
    I, files = load_full_sequence_16bit(img_folder)

    # correction stries verticales
    I = correct_vertical_stripes(I)

    # --- temps normalisé ---
    t_img, dt = build_t_img_from_csv(root_folder, I.shape[0])
    Ta, vp = extract_Ta_from_csv(root_folder, colonne=colonne)
    time = t_img / Ta

    nt = I.shape[0]
    Rp = np.full(nt, np.nan, dtype=float)
    xc = np.full(nt, np.nan, dtype=float)
    yc = np.full(nt, np.nan, dtype=float)

    os.makedirs(out_dir, exist_ok=True)
    var = np.nanvar(I)
    # -------- LIVE DISPLAY --------
    plt.ion()
    fig, (ax_r, ax_im) = plt.subplots(
        2, 1, figsize=(7, 9),
        gridspec_kw={"height_ratios": [1, 3]}
    )

    line_r, = ax_r.plot([], [], "-o", ms=3)
    ax_r.set_xlabel("t / Ta")
    ax_r.set_ylabel(f"R{int(P_ENERGY * 100)} (mm)")
    ax_r.grid(True, ls="--", alpha=0.3)

    im_handle = ax_im.imshow(I[0], cmap="viridis")
    ax_im.axis("off")

    # contraste stable
    sample_idx = np.linspace(0, nt - 1, min(nt, 10), dtype=int)
    sample_vals = np.concatenate([I[k].ravel() for k in sample_idx])
    vmin, vmax = np.percentile(sample_vals, (2, 99.5))
    im_handle.set_clim(vmin, vmax)

    # -------- BOUCLE TEMPORELLE --------
    for t in range(nt):

        Rp_m, (xc_pix, yc_pix), _ = compute_encircled_energy_radius(I[t], dx)

        Rp[t] = Rp_m
        xc[t] = xc_pix
        yc[t] = yc_pix

        # update courbe R(t)
        line_r.set_data(time[:t + 1], Rp[:t + 1] * 1e3)
        ax_r.relim()
        ax_r.autoscale_view()

        # update image
    #     im_handle.set_data(I[t])
    #     ax_im.set_title(
    #         f"frame {t + 1}/{nt} | t/Ta={time[t]:.4g} | "
    #         f"R{int(P_ENERGY * 100)}={Rp[t] * 1e3:.3f} mm"
    #     )
    #
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
    #
    plt.ioff()
    plt.show()
    t0=np.argmax(var)
    time=time-time[t0]
    # -------- FIGURE FINALE R(t) --------
    fig, ax = plt.subplots()
    ax.plot(time, Rp , "-o", ms=3)
    ax.set_xlabel("t / Ta")
    ax.set_ylabel(f"R{int(P_ENERGY * 100)} (mm)")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Rp_vs_time.png"), dpi=150)
    plt.show()

    return {
        "root": os.path.basename(root_folder),
        "img_folder": img_folder,
        "time": time,
        "Rp_m": Rp,
        "xc_pix": xc,
        "yc_pix": yc,
        "files": files
    }


# ==========================================================
# DISCOVERY HELPERS (repris de ta logique)
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


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    # Comme dans ton script: liste de combinaisons (L_mm, "fine/coarse")
    REQUEST = [
        (10, "fine"),
        # (10, "coarse"),
        # (6, "fine"),
        # ...
    ]

    root_folders = select_exact_combinations(BASE_PATH, REQUEST)
    print("Experiments sélectionnées :", len(root_folders))

    for root in root_folders:
        out_dir = os.path.join(root, f"encircled_R{int(P_ENERGY*100)}")
        print(f"\n[RUN] {root}")
        res = process_experiment_for_Rp(root, out_dir, colonne="grande")
        if res is None:
            continue
        time=res["time"]
        Rp=res["Rp_m"]
        fig, ax = plt.subplots()
        ax.plot(time, Rp * 1e3, "-o", ms=3)
        ax.set_xlabel("t / Ta")
        ax.set_ylabel(f"R{int(P_ENERGY * 100)} (mm)")
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        plt.show()
        print(f"[DONE] overlays -> {out_dir}")