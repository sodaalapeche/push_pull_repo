import os
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

width     = 18   # cm
height    = width * 0.5
font_size = 15
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
        (10, "fine"): dict(marker="o", color="tab:blue", mfc="tab:blue", ms=4, alpha=0.8),

        (6,  "fine"):        dict(marker="o", color="tab:orange", mfc="tab:orange", ms=4, alpha=0.8),
        (3, "fine"): dict(marker="o", color="tab:red", mfc="tab:red", ms=4, alpha=0.8),

        (0,  "fine"):        dict(marker="o", color="tab:green",  mfc="tab:green",  ms=4, alpha=0.8),

        # (10, "contactless"): dict(marker="s", color="tab:pink",   mfc="tab:pink",   ms=5, alpha=0.8),
        }

    fig, ax = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

    for res in results:
        info = parse_label(res.get("label", ""))
        if info is None:
            continue
        L_mm, sand = info
        key = (L_mm, sand)
        if key not in style_map:
            continue
        style = style_map[key]

        time = res["time"].copy()
        mean = res["mean"]
        var  = res["var"]
        i0   = res["i0"]
        A0   = res["A"]

        # ----------------------------------------------------------
        # CORRECTION DE POROSITÉ pour les expériences contactless
        #
        # On rescale l'axe temporel adimensionnel par ALPHA_CONTACTLESS
        # pour ramener ces expériences à une vitesse interstitielle
        # équivalente aux expériences de référence.
        #
        #   t_corr = t × ALPHA   avec ALPHA = φ_contactless / φ_contact
        #
        # Physiquement : à même débit imposé, la vitesse interstitielle
        # est plus faible dans le milieu contactless (plus poreux), donc
        # les particules de fluide avancent plus lentement par unité de
        # temps de Darcy. Le rescaling corrige cet écart.
        # ----------------------------------------------------------
        if sand in CONTACTLESS_LABELS:
            time = time * ALPHA_CONTACTLESS

        Sigma  = var / (A0 * mean**2)
        Sigma /= Sigma[i0]

        valid = np.isfinite(time) & np.isfinite(Sigma) & (Sigma > 0)
        ax.plot(2 * time[valid][::2], Sigma[valid][::2],
                linestyle="None", **style)

    # légende
    legend_elements = []
    for (L_mm, sand), style in style_map.items():
        # label_str = (f"{L_mm} mm, {sand}"
        #              if L_mm > 0 else f"{sand} sand only")
        label_str = (f"sand + {L_mm}mm inclusions"
                     if L_mm > 0 else f"sand")
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

    ax.legend(handles=legend_elements,fontsize=8)
    ax.set_xlabel(r"$2t/T_a$")
    ax.set_ylabel(r"$\Sigma = \frac{\sigma_c^2}{\mu_c^2 \cdot A}$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.007, top=1.1)
    ax.set_xlim(left=1, right=64)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    plt.savefig("/home/chorus/scalardecay.pdf")
    plt.show()