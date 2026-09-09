import os
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

width     = 18/2.1   # cm
height    = 10/2.1
# font_size = 15
inches    = 2.54

# ==========================================================
# PARAMÈTRES GLOBAUX
# ==========================================================
BASE_PATH     = "/home/chorus/HETEROGENE_binned/"
MASK_REF_PATH = "/home/chorus/test/masque.jpg"
SAVE_PATH     = "/home/chorus/data_roi.npy"

s          = 0.06 / 512
R_EVAL_PIX = 190
num_pixels = int(np.pi * R_EVAL_PIX**2)

# ==========================================================
# CORRECTION DE POROSITÉ POUR LES EXPÉRIENCES CONTACTLESS
#
# Dans les expériences "contactless", la fraction volumique de billes φ_billes
# a été réduite pour éviter les contacts. La porosité effective est donc plus
# grande que dans les expériences de référence (contact) :
#
#   φ_contact     = φ_sable × (1 - φ_billes_contact)
#   φ_contactless = φ_sable × (1 - φ_billes_contactless)
#
# Le temps réduit t/Ta est basé sur la vitesse de Darcy u = 0.01/Ta, mais
# la vitesse interstitielle réelle est u_i = u / φ. Pour comparer les
# expériences à vitesse interstitielle équivalente, on rescale l'axe temporel
# des expériences contactless par le rapport des porosités :
#
#   t_corr = t × (φ_contactless / φ_contact) = t × ALPHA_CONTACTLESS
#
# ALPHA_CONTACTLESS = φ_contactless / φ_contact
#   - Si φ_contactless ≈ 2 × φ_contact  →  ALPHA ≈ 2  (ordre de grandeur)
#   - À affiner quand les porosités seront mesurées par pesée.
#
# Mettre ALPHA_CONTACTLESS = 1.0 pour désactiver la correction.
# ==========================================================
ALPHA_CONTACTLESS = 0.38  # ← à affiner avec la mesure de porosité

# Ensemble des labels "sand" qui reçoivent la correction
CONTACTLESS_LABELS = {"contactless"}

# ==========================================================
# DISCOVERY
# ==========================================================
def select_exact_combinations(base_path, combinations):
    selected = []
    for L_mm, sand in combinations:
        path = os.path.join(base_path, f"{L_mm}mm", sand.lower())
        if not os.path.exists(path):
            continue
        for d in os.listdir(path):
            exp_path = os.path.join(path, d)
            if not os.path.isdir(exp_path):
                continue
            tif_files = glob(os.path.join(exp_path, "**", "*.tif"), recursive=True)
            csv_path  = os.path.join(exp_path, "weight_data.csv")
            if tif_files and os.path.isfile(csv_path):
                selected.append(exp_path)
    return selected


def find_image_folder(root):
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and glob(os.path.join(p, "*.tif")):
            return p
    raise RuntimeError(f"No TIFF folder found under: {root}")


def list_tifs(folder):
    return sorted(
        glob(os.path.join(folder, "*.tif")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )


def read_label(root):
    p = os.path.join(root, "label.txt")
    return open(p).read().strip() if os.path.exists(p) else os.path.basename(root)

# ==========================================================
# TEMPS
# ==========================================================
def build_t_img_from_csv(root_folder, n_images):
    df = pd.read_csv(os.path.join(root_folder, "weight_data.csv"))
    t  = df["Timestamp"].values.astype(float)
    dt = np.median(np.diff(t))
    return t[0] + dt * np.arange(n_images)

# ==========================================================
# STRIPE CORRECTION
# ==========================================================
def compute_col_profile(frame0):
    col_profile = np.mean(frame0, axis=0)
    col_profile /= np.mean(col_profile)
    return np.maximum(col_profile, 1e-6)


def apply_col_profile(frame, col_profile):
    return frame / col_profile[None, :]

# ==========================================================
# ROI STATS
# ==========================================================
def roi_stats(frame, Xpix, Ypix, xc, yc, r_eval_pix):
    r2   = (Xpix - xc)**2 + (Ypix - yc)**2
    roi  = r2 <= r_eval_pix**2
    vals = frame[roi]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.var(vals))

# ==========================================================
# CORE PROCESS
# ==========================================================
def process_experiment(root_folder):
    img_folder = find_image_folder(root_folder)
    files      = list_tifs(img_folder)
    n          = len(files)

    frame0      = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    col_profile = compute_col_profile(frame0)

    H, W = frame0.shape
    Ypix, Xpix = np.indices((H, W))
    xc, yc = W / 2, H / 2

    t_img = build_t_img_from_csv(root_folder, n)

    mean = np.full(n, np.nan)
    var  = np.full(n, np.nan)

    for i, f in enumerate(files):
        frame   = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        frame   = apply_col_profile(frame, col_profile)
        m, v    = roi_stats(frame, Xpix, Ypix, xc, yc, R_EVAL_PIX)
        mean[i] = m
        var[i]  = v

    # i0 = pic de variance
    i0 = int(np.nanargmax(var))

    # dérivée + détection min après i0 + 30 frames
    dmean_dt = np.gradient(mean, t_img)
    i_start  = min(i0 + 30, len(mean) - 1)
    if i_start < len(mean) - 1:
        i_min = i_start + int(np.nanargmin(dmean_dt[i_start:]))
    else:
        i_min = i0

    Ta   = (t_img[i_min] - t_img[i0]) / 30.0
    u    = 0.01 / Ta   # noqa: F841
    time = (t_img - t_img[i0]) / Ta

    return {
        "label": read_label(root_folder),
        "time":  time,
        "mean":  mean,
        "var":   var,
        "Ta":    Ta,
        "i0":    i0,
        "A":     num_pixels * s**2,
    }

# ==========================================================
# MULTIPROCESS
# ==========================================================
def run_one_experiment(root):
    try:
        return process_experiment(root)
    except Exception as e:
        print(f"[ERROR] {root}: {e}")
        return None

# ==========================================================
# HELPER : parse label → (L_mm, sand)
# ==========================================================
def parse_label(label_text):
    lines = [ln.strip().lower() for ln in label_text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return None
    try:
        return int(lines[1].replace("mm", "")), lines[2]
    except Exception:
        return None

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    from matplotlib.lines import Line2D

    REQUEST = [
        (0,  "fine"),
        (10, "fine"),
        (0,  "coarse"),
        (6,  "fine"),
        (3,  "fine"),
        (10, "coarse"),
        (6, "coarse"),
        (3, "coarse")

        # (10, "contactless"),
    ]

    root_folders = select_exact_combinations(BASE_PATH, REQUEST)
    print(f"Experiments: {len(root_folders)}")

    with Pool(max(1, cpu_count() - 2)) as pool:
        results = pool.map(run_one_experiment, root_folders)

    results = [r for r in results if r is not None]

    np.save(SAVE_PATH, results, allow_pickle=True)
    print(f"Saved {len(results)} experiments to {SAVE_PATH}")

    # ==========================================================
    # STYLE
    # ==========================================================
    style_map = {

        # (10, "coarse"):      dict(marker="D", color="tab:blue",   mfc="none",       mew=1.2, ms=5, alpha=0.8),
        # (6,  "coarse"):      dict(marker="D", color="tab:orange", mfc="none",       mew=1.2, ms=5, alpha=0.8),
        # (3,  "coarse"):      dict(marker="D", color="tab:red",    mfc="none",       mew=1.2, ms=5, alpha=0.8),
        # (0,  "coarse"):      dict(marker="D", color="tab:green",  mfc="none",       mew=1.2, ms=5, alpha=0.8),
        (10, "fine"): dict(marker="^", color="tab:blue", mfc="tab:blue", ms=4, alpha=0.8),

        # (6,  "fine"):        dict(marker="o", color="tab:orange", mfc="tab:orange", ms=4, alpha=0.8),
        # (3, "fine"): dict(marker="o", color="tab:red", mfc="tab:red", ms=4, alpha=0.8),

        (0,  "fine"):        dict(marker="o", color="tab:green",  mfc="tab:green",  ms=4, alpha=0.8),

        # (10, "contactless"): dict(marker="s", color="tab:pink",   mfc="tab:pink",   ms=5, alpha=0.8),
        }

    # fig, ax = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

    fig, ax = plt.subplots(figsize=(width / inches, height / inches), layout="constrained")

    # ==========================================================
    # Pour L_mm == 10, plusieurs résultats existent pour une même
    # combinaison : on choisit lequel tracer (0 = premier, 1 = deuxième, etc.)
    # Pour les autres L_mm, toutes les courbes sont tracées normalement.
    # ==========================================================
    DUPLICATE_INDEX_FOR_L10 = 2  # <-- change ici le numéro de la courbe à garder pour L_mm==10

    from collections import defaultdict

    grouped = defaultdict(list)
    for res in results:
        info = parse_label(res.get("label", ""))
        if info is None:
            continue
        key = info
        if key not in style_map:
            continue
        grouped[key].append(res)

    for key, res_list in grouped.items():
        L_mm, sand = key
        style = style_map[key]

        if L_mm == 10:
            if DUPLICATE_INDEX_FOR_L10 >= len(res_list):
                print(f"[warning] {key}: seulement {len(res_list)} résultat(s), "
                      f"index {DUPLICATE_INDEX_FOR_L10} indisponible, on saute.")
                continue
            res_list_to_plot = [res_list[DUPLICATE_INDEX_FOR_L10]]
        else:
            res_list_to_plot = res_list  # toutes les courbes comme avant

        for res in res_list_to_plot:
            time = res["time"].copy()
            mean = res["mean"]
            var = res["var"]
            i0 = res["i0"]
            A0 = res["A"]

            if sand in CONTACTLESS_LABELS:
                time = time * ALPHA_CONTACTLESS

            Sigma = var / (A0 * mean ** 2)
            Sigma /= Sigma[i0]

            valid = np.isfinite(time) & np.isfinite(Sigma) & (Sigma > 0)
            ax.plot(2 * time[valid][::2], Sigma[valid][::2],
                    linestyle="None", **style)
    # légende
    legend_elements = []
    for (L_mm, sand), style in style_map.items():
        # label_str = (f"{L_mm} mm, {sand}"
        #              if L_mm > 0 else f"{sand} sand only")
        label_str = (f"bidisperse - $Pe_M = 245$"
                     if L_mm > 0 else f"monodisperse")
        if sand in CONTACTLESS_LABELS:
            label_str += rf" (×{ALPHA_CONTACTLESS:.1f})"
        legend_elements.append(
            Line2D([0], [0],
                   marker=style["marker"], linestyle="None",
                   color=style["color"],
                   markerfacecolor=style.get("mfc", style["color"]),
                   markeredgewidth=style.get("mew", 1.0),
                   markersize=style.get("ms", 6),
                   alpha=style.get("alpha", 1.0),
                   label=label_str)
        )
    # ==========================================================
    # DEUX DROITES GUIDES tau_1^-1 et tau_2^-1 (courbe bidisperse, L_mm=10)
    # Pas un fit : simple sécante entre les deux bornes de chaque fenêtre,
    # tracée légèrement en dessous des données pour rester lisible.
    # Fenêtres : [0, 5] cm et [25, 50] cm
    # ==========================================================
    GUIDE_WINDOWS = {
        r"$\ell_1^{-1}$": (0, 5),
        r"$\ell_2^{-1}$": (26, 46),
    }
    OFFSET = 0.5  # facteur multiplicatif pour décaler la droite sous la courbe

    # Courbe bidisperse (L_mm=10, fine) réellement tracée
    key_bidisperse = (10, "fine")
    res_bidisperse = grouped[key_bidisperse][DUPLICATE_INDEX_FOR_L10]

    time_b = res_bidisperse["time"].copy()
    mean_b = res_bidisperse["mean"]
    var_b = res_bidisperse["var"]
    i0_b = res_bidisperse["i0"]
    A0_b = res_bidisperse["A"]

    Sigma_b = var_b / (A0_b * mean_b ** 2)
    Sigma_b /= Sigma_b[i0_b]
    x_b = 2 * time_b

    valid_b = np.isfinite(x_b) & np.isfinite(Sigma_b) & (Sigma_b > 0)
    x_b = x_b[valid_b]
    Sigma_b = Sigma_b[valid_b]
    i = 0

    for label, (x_lo, x_hi) in GUIDE_WINDOWS.items():
        # point de données le plus proche de chaque borne
        i_lo = np.argmin(np.abs(x_b - x_lo))
        i_hi = np.argmin(np.abs(x_b - x_hi))

        x0, x1 = x_b[i_lo], x_b[i_hi]
        y0, y1 = Sigma_b[i_lo] * OFFSET, Sigma_b[i_hi] * OFFSET

        # pente en ln(Sigma) vs x -> tau^-1 = -pente
        slope = (np.log(y1) - np.log(y0)) / (x1 - x0)
        tau_inv = -slope

        if i==0:
            ax.annotate(
            label,
            xy=(x1-3, y1*0.85),
            xytext=(0, 0), textcoords="offset points",
            va="top", ha="left",
        )
            i+=1
        else:
            tau_inv = -slope*0.82

            ax.annotate(
                label,
                xy=(x1-24, y1*1.2),
                xytext=(-0, 0), textcoords="offset points",
            )
        ax.plot([x0, x1], [y0, y1], "k-", lw=1.2, zorder=5)
        print(f"{label} : {x_lo}-{x_hi} cm -> tau^-1 ~ {tau_inv:.4f} cm^-1")
    # ax.legend(handles=legend_elements,fontsize=8)
    ax.set_xlabel(r"$2t\cdot u$ (cm)")
    ax.set_ylabel(r"$\sigma^2/(\mu^2 \cdot A)$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.003, top=4)
    ax.set_xlim(left=0, right=70)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    plt.savefig("/home/chorus/TexProject/clement draft avance (document de travail)/figures/fig1.pdf")
    plt.show()