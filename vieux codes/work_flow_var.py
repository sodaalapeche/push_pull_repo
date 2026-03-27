import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
import skimage.filters as sk
from scipy import ndimage
from scipy.optimize import minimize
import re

import math as m
# ==========================================================
# GLOBAL PARAMETERS
# ==========================================================

BASE_PATH = "/home/chorus/HETEROGENE_binned/"
MASK_REF_PATH = "/home/chorus/test/masque.jpg"

dx = (0.06 / 2048)*4
idx0 = 50
frac = 0.2
fit = False

ANGLE_MAX = 2.0      # degrés max autorisés (±)
ANGLE_STEP = 0.1    # résolution angulaire

TRANS_MAX = 30
ROI_FRAC = 0.35
#"exp_09_02_4","15_10_3","16_10_2","exp_05_02_1","exp_05_02_2","exp_28_01_5","exp_28_01_3","exp_3_11_3"
# ==========================================================
# FOLDER DISCOVERY
# ==========================================================
#,"exp_28_01_5","exp_15_01_2",'exp_12_01_2'
# root_folders = [
#     os.path.join(BASE_PATH, d)
#     for d in sorted(os.listdir(BASE_PATH))
#     if d.startswith(("16_10_4","16_10_2","exp_02_02_2","exp_03_02_4","16_10_3","exp_28_01_2","exp_28_01_3"))
#     and os.path.isdir(os.path.join(BASE_PATH, d))_
# ]
root_folders = [
    os.path.join(BASE_PATH, d)
    for d in sorted(os.listdir(BASE_PATH))
    if d.startswith(("exp_12_02","exp_13_02","exp_18_02_2","exp_18_02_3"
                     ))
    and os.path.isdir(os.path.join(BASE_PATH, d))
]#"15_10_3","16_10_4","exp_18_02_4"

root_folders = [
    os.path.join(BASE_PATH, d)
    for d in sorted(os.listdir(BASE_PATH))
    if d.startswith(("exp_18_02_4","exp_19_02"))
    and os.path.isdir(os.path.join(BASE_PATH, d))
]

# =========_=================================================
# IMAGE UTILITIES
# ==========================================================
def extract_structure_sato(img):
    """
    Extrait une carte binaire robuste du réseau hexagonal
    à partir du filtre de Sato
    """
    I_sato = sk.sato(img, sigmas=range(4, 10, 1))
    I_sato = ndimage.median_filter(I_sato, size=13)

    mask_sato = I_sato > 1e-5

    H, W = img.shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask_circle = (X - W/2)**2 + (Y - H/2)**2 > 213**2

    structure = mask_sato & (~mask_circle)

    return structure.astype(np.uint8)
def save_results_to_h5(results, h5_path="experiments.h5", compression="gzip", compression_level=4):
    """
    Sauvegarde TOUTES les expériences dans un seul fichier HDF5.
    - results: liste de dicts (comme tes retours de run_one_experiment), avec éventuellement des None
    - 1 groupe par expérience: /<root> (ou /exp_XXXX si root absent)
    - stocke strings, scalaires, np.ndarray (compressés), et sérialise le reste en JSON si besoin
    """
    import numpy as np
    import h5py
    import json
    from datetime import datetime

    def _save_value(group, key, val):
        # strings
        if isinstance(val, str):
            group.create_dataset(key, data=np.string_(val))
            return

        # scalaires numériques/bool
        if isinstance(val, (bool, int, float, np.integer, np.floating)):
            group.create_dataset(key, data=val)
            return

        # numpy arrays / listes -> ndarray
        if isinstance(val, (list, tuple)):
            val = np.asarray(val)

        if isinstance(val, np.ndarray):
            # cas objet -> JSON
            if val.dtype == object:
                group.create_dataset(key + "_json", data=np.string_(json.dumps(val.tolist(), ensure_ascii=False)))
                return

            # dataset compressé
            group.create_dataset(
                key,
                data=val,
                compression=compression,
                compression_opts=compression_level,
                shuffle=True,
            )
            return

        # fallback -> JSON (dict, etc.)
        group.create_dataset(key + "_json", data=np.string_(json.dumps(val, ensure_ascii=False)))

    # nettoyage
    clean = [r for r in results if r is not None]

    with h5py.File(h5_path, "w") as f:
        f.attrs["format"] = "experiments_h5_v1"
        f.attrs["created_utc"] = datetime.utcnow().isoformat() + "Z"
        f.attrs["n_experiments"] = len(clean)

        for i, res in enumerate(clean, start=1):
            root = res.get("root", f"exp_{i:04d}")

            # nom de groupe safe
            grp_name = str(root).replace("/", "_")
            if grp_name in f:
                del f[grp_name]
            g = f.create_group(grp_name)

            # meta minimal
            g.create_dataset("meta_json", data=np.string_(json.dumps({"root": root, "index": i}, ensure_ascii=False)))

            for k, v in res.items():
                if k == "root":
                    continue
                _save_value(g, k, v)

    print(f"[SAVE] {len(clean)} expériences sauvegardées dans: {h5_path}")
    return h5_path

def select_exact_combinations(base_path, combinations):
    selected = []

    for L_mm, sand in combinations:
        size_folder = f"{L_mm}mm"
        sand_folder = sand.lower()
        path = os.path.join(base_path, size_folder, sand_folder)

        if not os.path.exists(path):
            print(f"[WARN] Folder not found: {path}")
            continue

        # Parcours chaque dossier d'expérience
        for d in os.listdir(path):
            exp_path = os.path.join(path, d)
            if not os.path.isdir(exp_path):
                continue

            # Cherche TIFF dans exp_path ou ses sous-dossiers (comme "serie1")
            tif_files = glob(os.path.join(exp_path, "**", "*.tif"), recursive=True)
            if tif_files:
                selected.append(exp_path)
            else:
                print(f"[WARN] No TIFF in experiment folder: {exp_path}")

    return selected
def calculate_blob_radius(image, mask, dx,
                          close_ksize=11,
                          blur_sigma=12.0,
                          thr_percentile=95.0,
                          weight_mode="intensity"):
    """
    Calcule Rg, x_c, y_c de manière robuste sur images avec motif hexagonal sombre.
    - close_ksize : taille du kernel pour combler les lignes sombres (impair, ~9-15)
    - blur_sigma  : lissage de l'enveloppe (typ. 8-20)
    - thr_percentile : 90-98 selon si blob très diffus
    - weight_mode : "intensity" (recommandé) ou "binary"
    """
    import numpy as np
    import cv2

    img = image.astype(np.float32)
    m0 = mask.astype(bool)

    # Appliquer masque global + nettoyer NaN
    img = np.where(m0, img, 0.0)
    if np.all(img == 0):
        raise ValueError("Image vide après masque global")

    # 1) Combler les lignes sombres de l'hexagone (closing)
    k = int(close_ksize)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # 2) Lissage -> enveloppe du blob
    env = cv2.GaussianBlur(img_close, (0, 0), blur_sigma)

    # 3) Seuil robuste (percentile dans la zone masquée)
    vals = env[m0]
    if vals.size == 0:
        raise ValueError("Masque global vide")
    thr = np.percentile(vals, thr_percentile)
    blob = (env >= thr) & m0

    # Nettoyage léger
    blob_u8 = blob.astype(np.uint8)
    blob_u8 = cv2.morphologyEx(blob_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    blob_u8 = cv2.morphologyEx(blob_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # Garder la plus grande composante connexe
    nlab, labels = cv2.connectedComponents(blob_u8)
    if nlab <= 1:
        raise ValueError("Masque blob vide (seuil trop haut ?)")
    areas = np.bincount(labels.ravel())[1:]  # sans fond
    lab = 1 + int(np.argmax(areas))
    blob = (labels == lab)

    # 4) Calcul moments -> centre et Rg
    if weight_mode == "binary":
        W = blob.astype(np.float64)
    else:
        # pondération par l'intensité (après comblement), plus stable que l'image brute
        W = (img_close.astype(np.float64)) * blob

    M = W.sum()
    if M <= 0:
        raise ValueError("Somme des poids nulle")

    ny, nx = img.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y, indexing="xy")

    x_c = (W * X).sum() / M
    y_c = (W * Y).sum() / M
    r2 = (X - x_c) ** 2 + (Y - y_c) ** 2
    Rg = np.sqrt((W * r2).sum() / M)

    return Rg, x_c, y_c

def compute_hex_mask_from_sato(img0, mask_ref):
    """
    Recale le masque hexagonal sur la structure extraite par Sato
    """
    H, W = img0.shape

    # --- structure expérimentale (vérité terrain)
    structure = extract_structure_sato(img0)

    # --- bords du masque hexagonal
    edges_mask = cv2.Canny(mask_ref * 255, 50, 150)

    # --- bords de la structure Sato
    edges_struct = cv2.Canny(structure * 255, 50, 150)

    # ---------- ANGLE ----------
    best_angle, best_score = 0.0, -np.inf
    center = (W / 2, H / 2)

    angles = np.arange(-ANGLE_MAX, ANGLE_MAX + ANGLE_STEP, ANGLE_STEP)

    best_angle, best_score = 0.0, -np.inf

    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(
            edges_mask, M, (W, H),
            flags=cv2.INTER_NEAREST, borderValue=0
        )
        score = np.sum(rot * edges_struct)

        if score > best_score:
            best_score = score
            best_angle = angle

    # ---------- TRANSLATION (corrélation robuste) ----------
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    edges_mask_rot = cv2.warpAffine(
        edges_mask, M, (W, H),
        flags=cv2.INTER_NEAREST, borderValue=0
    )
    corr = fftconvolve(
        edges_struct.astype(float),
        edges_mask_rot[::-1, ::-1].astype(float),
        mode="same"
    )

    cy, cx = H // 2, W // 2

    corr_window = corr[
                  cy - TRANS_MAX: cy + TRANS_MAX + 1,
                  cx - TRANS_MAX: cx + TRANS_MAX + 1
                  ]

    dy, dx = np.unravel_index(np.argmax(corr_window), corr_window.shape)

    ty = dy - TRANS_MAX
    tx = dx - TRANS_MAX

    # ---------- MASQUE FINAL ----------
    M[0, 2] += tx
    M[1, 2] += ty

    mask_aligned = cv2.warpAffine(
        mask_ref, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )
    mask_aligned = ndimage.binary_erosion(mask_aligned, iterations=4)
    return mask_aligned.astype(bool)

def find_image_folder(root):
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and glob(os.path.join(p, "*.tif")):
            return p
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

def autoscale_for_edges(img, low=2, high=98):
    vmin, vmax = np.percentile(img, (low, high))
    img = np.clip(img, vmin, vmax)
    return ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)

# ==========================================================
# HEX MASK COMPUTATION (ONCE)
# ==========================================================

from scipy.signal import fftconvolve

# ==========================================================
# METADATA UTILITIES
# ==========================================================

def plot_semi_log_slope(ax, start_x, start_y, slope=-0.06, length=10, offset_factor=1.2, **kwargs):
    """
    Ajoute une droite de pente donnée en semi-log (y log, x linéaire).

    Args:
        ax : matplotlib axes
        start_x (float) : point de départ en x
        start_y (float) : point de départ en y (sera multiplié par offset_factor)
        slope (float) : pente dans l'espace semi-log (ex: -0.06)
        length (float) : intervalle en x sur lequel tracer la droite
        offset_factor (float) : décalage vertical multiplicatif
        **kwargs : arguments pour ax.plot()
    """
    # Plage en x (linéaire)
    x_vals = np.linspace(start_x, start_x + length, 200)

    # Courbe exponentielle correspondant à la pente demandée
    y_vals = start_y * offset_factor * np.exp(slope * (x_vals - start_x))
    ax.plot(x_vals, y_vals,linestyle='dashed', label=fr"$slope = {slope}$", **kwargs)

    # Étiquette à la fin de la courbe
    ax.text(
        x_vals[-1]*0.8, y_vals[-1]*1.2,
        fr"slope = {slope}",
        fontsize="x-large", ha="left", va="top"
    )

    return ax
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


def Ta_for_bead(timestamps, weights_g, D=0.055, L=0.01, f_billes=0.3,Tip="homo"):
    weights_g = np.array(weights_g, dtype=float)
    eps_sable = 0.5
    valid = weights_g > 1000
    ts = timestamps[valid]
    ws = weights_g[valid]

    coeffs = np.polyfit(ts, ws, 1)
    print(coeffs)
    dMdt_g_s = coeffs[0]

    Q_m3_s = dMdt_g_s * 1e-6
    A_tot = np.pi * (D / 2)**2
    if Tip=="homo coarse":
        A_pore = A_tot * eps_sable*0.93
    elif Tip=="homo fine":
        A_pore = A_tot * eps_sable*0.8
    elif Tip=="0.1":
        A_pore = A_tot*eps_sable*(1-f_billes)*1.7
    elif Tip == "0.6":
        A_pore = A_tot*eps_sable*(1-f_billes)*0.85
        print("tick0.6)")
    elif Tip == "0.3 coarse":
        A_pore = A_tot*eps_sable*(1-f_billes)
    elif Tip=="0.6 fine":
        A_pore = A_tot *eps_sable*(1-f_billes)*1.23
    elif Tip=="Hetero":
        A_pore = A_tot*eps_sable*(1-f_billes)
    else :
        A_pore = A_tot *eps_sable*(1-f_billes)
    print()
    print(1/A_pore)
    vp = Q_m3_s / A_pore
    Ta = L / vp
    print("Q (g/s)=",dMdt_g_s)
    print("vp(m/s) : ",vp)

    return Ta, vp



def extract_Ta_from_csv(root_folder, colonne="grande",Tip="homo"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)
    label = read_label(root_folder)

    D = 0.055 if colonne == "grande" else 0.027
    L = extract_L_from_label(root_folder)
    print("L =",L)
    if L==0.001:
        Tip='0.1'
    if L==0.01 and "fine" in label:
        Tip="Hetero"
    if L==0.006 and "coarse" in label:
        Tip="0.6"
    if L==0.006 and "fine" in label:
        Tip="0.6 fine"
    if L==0.003:
        Tip="0.3 coarse"
    if L==0.0 and "fine" in label:
        Tip="homo fine"
        L=0.01
    if L==0.0 and "coarse" in label:
        Tip="homo coarse"
        L=0.01
    L=0.01


    Ta, vp = Ta_for_bead(
        df["Timestamp"],
        df["Weight"],
        D=D,
        L=L,
        Tip=Tip
    )
    print(Ta,"s")
    return Ta, vp


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
# DISPERSION RADIUS
# ==========================================================
import matplotlib.pyplot as plt


def compute_dispersion_radius(I, dx, frac=0.3):
    nt, ny, nx = I.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)

    R = np.full(nt, np.nan)

    for t in range(nt):
        C = np.nan_to_num(I[t], nan=0.0)
        if C.sum() == 0:
            continue
        M = C.sum()
        xc = np.sum(X * C) / M
        yc = np.sum(Y * C) / M
        r = np.sqrt((X - xc)**2 + (Y - yc)**2)
        idx = np.argsort(r.ravel())
        R[t] = r.ravel()[idx][np.searchsorted(np.cumsum(C.ravel()[idx]), frac * M)]

    return R

def correct_vertical_stripes(I, smooth=2, eps=1e-6):
    """
    Corrige les stries verticales en normalisant par le profil colonne
    extrait de la première image du stack.
    """
    ref = I[0]

    # Profil moyen par colonne
    col_profile = np.nanmean(ref, axis=0)

    # Lissage pour éviter d'amplifier le bruit
    if smooth and smooth > 3:
        col_profile = ndimage.median_filter(col_profile, size=smooth)

    # Normalisation globale pour ne pas changer l’échelle
    col_profile /= np.nanmean(col_profile)

    # Sécurité divisions
    col_profile = np.maximum(col_profile, eps)

    # Broadcast division sur tout le stack
    I_corr = I / col_profile[None, None, :]

    return I_corr

# ==========================================================
# MAIN EXPERIMENT PROCESSING
# ==========================================================
def process_experiment(root_folder, colonne="grande"):
    img_folder = find_image_folder(root_folder)
    I = load_full_sequence_16bit(img_folder)

    # Correction des stries verticales
    I = correct_vertical_stripes(I)

    # --- masque hexagonal de référence
    mask_ref = cv2.imread(MASK_REF_PATH, cv2.IMREAD_GRAYSCALE)
    mask_ref = (mask_ref > 128).astype(np.uint8)
    mask_ref = cv2.resize(
        mask_ref, I[0].shape[::-1],
        interpolation=cv2.INTER_NEAREST
    )

    # --- recalage à partir de Sato (ROBUSTE)
    hex_mask = compute_hex_mask_from_sato(I[0], mask_ref)

    # --- application à toute la séquence
    I[:, ~hex_mask] = np.nan

    # --- le reste de TON pipeline
    mean = np.nanmean(I, axis=(1, 2))
    var = np.nanvar(I, axis=(1, 2))
    R =0
    A=m.pi*(0.055/2)**2
    t_img, dt = build_t_img_from_csv(root_folder, I.shape[0])
    Ta, vp = extract_Ta_from_csv(root_folder)
    print(img_folder)
    print("ta (s) :", Ta)
    time = t_img / Ta

    label = read_label(root_folder)
    print("1/A :")
    print(1/A)

    return {
        "root": os.path.basename(root_folder),
        "label": label,
        "time": time,
        "mean": mean,
        "var": var,
        "A": A,
        "R": R,
        "mask": hex_mask,
        "I0": I[0],
        "I": I
    }
def run_one_experiment(root):
    try:
        res = process_experiment(root)
        time = res["time"]
        mean = res["mean"]
        var = res["var"]
        A = res["A"]
        I = res["I"]  # Stack complet des images

        Sigma = (var)
                 #/ (mean**2 * A))

        valid = (~np.isnan(Sigma) & (Sigma > 0))
        time_m = time[valid].ravel()
        Sigma_m = Sigma[valid]
        i0 = np.argmax(Sigma_m)

        # Calcul du rayon de giration pour l'image i0
        # --- Rayon de giration pour TOUTES les images ---

        # Centre et Rg au temps i0 (si tu veux garder l’info)
        Rg0, x_c, y_c = calculate_blob_radius(I[i0], res["mask"], dx)
        time_m = time_m-time_m[i0]

        return {
            "root": res["root"],
            "label": res["label"],
            "time": time_m,
            "value": Sigma_m,
            "mean": mean,
            "x_c": x_c,  # Coordonnée x du centre de masse (m)
            "y_c": y_c,  # Coordonnée y du centre de masse (m)
            "I0": I[0],  # Ajoute I0 ici
            "mask": res["mask"],
            "Rg0": Rg0,
        }
    except Exception as e:
        print(f"[ERROR] {root}: {e}")
        return None
def parse_label(label_text):
    lines = [l.strip().lower() for l in label_text.split("\n") if l.strip()]
    system = lines[0]
    size = lines[1]     # ex: "10mm"
    sand = lines[2]     # "fine" ou "coarse"
    return system, size, sand

def plot_blob_with_center(image, mask, dx, Rg, x_c, y_c):
    masked_image = image * mask
    plt.imshow(masked_image, cmap="viridis")
    plt.scatter(x_c / dx, y_c / dx, color='red', marker='x', s=100, label='Centre de masse')
    circle = plt.Circle((x_c / dx, y_c / dx), Rg / dx, color='red', fill=False, linestyle='--', label=f'Rg = {Rg:.2e} m')
    plt.gca().add_patch(circle)
    plt.legend()
    plt.title("Blob avec centre de masse et rayon de giration")
    plt.colorbar()
    plt.show()

# Exemple d'utilisation :

# ==========================================================
# PARALLEL EXECUTION
# ==========================================================



if __name__ == "__main__":

    REQUEST = [
    (0,"fine"),
    (0,"coarse"),
    (6, "fine"),
    (10, "coarse"),
    (10, "stokes"),
    (10, "fine"),
    (3, "coarse"),
    (6, "coarse"),
    (6, "fine"),

    ]
    root_folders = select_exact_combinations(BASE_PATH, REQUEST)

    print("Experiments sélectionnées :", len(root_folders))

    # --- Run pipeline ---
    nproc = min(cpu_count()-1, len(root_folders))
    with Pool(nproc) as pool:
        results = pool.map(run_one_experiment, root_folders)
        #save_results_to_h5(results, "experiments.h5")

    # ==========================================================
    # CLEAN PLOTTING (mm + fine/coarse only)
    # ==========================================================

    fig, ax = plt.subplots()

    style_map = {
        (10, "fine"): dict(marker="o", color="tab:blue",
                           mfc="tab:blue", ms=4, alpha=0.8),
        (10, "coarse"): dict(marker="D", color="tab:blue",
                             mfc="none", mew=1.2, ms=5, alpha=0.8),

        (6, "fine"): dict(marker="o", color="tab:orange",
                          mfc="tab:orange", ms=4, alpha=0.8),
        (6, "coarse"): dict(marker="D", color="tab:orange",
                            mfc="none", mew=1.2, ms=5, alpha=0.8),

        (3, "fine"): dict(marker="o", color="tab:red",
                          mfc="tab:red", ms=4, alpha=0.8),
        (3, "coarse"): dict(marker="D", color="tab:red",
                            mfc="none", mew=1.2, ms=5, alpha=0.8),

        (0, "fine"): dict(marker="o", color="tab:green",
                          mfc="tab:green", ms=4, alpha=0.8),
        (0, "coarse"): dict(marker="D", color="tab:green",
                            mfc="none", mew=1.2, ms=5, alpha=0.8)
    }

    for res in results:
        if res is None:
            continue
        folder_name = res["root"]  # ex: "exp_18_02_4"
        clean_name = folder_name
        if folder_name.lower().startswith("exp"):
            clean_name = folder_name[3:]
            if clean_name.startswith("_") or clean_name.startswith("-"):
                clean_name = clean_name[1:]

        # Récupération des données à tracer
        time = res["time"]
        Rg=res["Rg0"]
        value = res["value"]
        # value = Rg**2*(res["value"])
        value = (res["value"]/np.max(res["value"]))
        # value = (value/np.nanmean(value))
        # value = res["mean"]/np.max(res["mean"])
        # value=res["value"]*Rg**2

        label_text = res["label"]
        lines = [l.strip().lower() for l in label_text.split("\n") if l.strip()]

        if len(lines) < 3:
            continue

        size_str = lines[1]      # "10mm"
        sand = lines[2]          # "fine" ou "coarse"

        L_mm = int(size_str.replace("mm", ""))

        key = (L_mm, sand)

        if key not in style_map:
            continue

        style = style_map[key]
        #
        ax.plot(
            res["time"][::4],  # ← un point sur 5
            value[::4],
            linestyle="None",
            **style
        )
        #ax.plot(res["time"], res['mean'], **style)
        # target_time = 3
        # idx_annot = np.argmin(np.abs(time - target_time))
        #
        # # On vérifie que le point existe bien dans la fenêtre visible
        # if np.isfinite(value[idx_annot]):
        #     ax.text(
        #         time[idx_annot],
        #         value[idx_annot],
        #         clean_name,
        #         fontsize=6,
        #         ha="left",
        #         va="center",
        #         color=style["color"],
        #         alpha=1
        #     )

    # -------- Legend automatique --------

    from matplotlib.lines import Line2D

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
                label=f"{L_mm} mm, {sand}"
            )
        )

    plt.legend(handles=legend_elements)

    plt.legend(handles=legend_elements)
    plt.axvline(0, color="k", ls="--", alpha=0.5)
    plt.xlabel(r"$(t - t_0)/t_a$")
    # plt.ylabel(r"$\frac{\sigma_c^2 / A\mu_c^2}{\sigma_{0}^2 / A\mu_0^2}$")
    plt.ylabel(r"$\frac{\sigma_c^2 \cdot R_c**2}{mean(\sigma_c^2 \cdot R_c**2)}$")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.xlim(left=1, right=35)

    # plt.ylim(bottom=0.5,top=10**1)
    plt.yscale("log")
    plt.xscale('log')
    # plot_semi_log_slope(ax,10,0.03,-0.09)
    # plot_semi_log_slope(ax,15,0.03,-0.03)
    # --- modèle 2D asymptotique : Sigma(τ) = 1 / (8π (D Ta) τ)
    # L0 = 0.01  # ta longueur arbitraire (m) dans Ta = L0/U
    # alphaT = 0.1  # D = alphaT * L * U
    # s0=0.002
    # L = 0.0001 # "10mm" -> 0.01 m
    # DTa = 2*alphaT * L * L0
    # C = 1.0 / (s0**2+ 8.0 * np.pi * DTa)
    # # courbe
    # mfit = np.isfinite(time) & (time > 0)
    # C0=value[0]
    # C = C0 / (s0 ** 2 + 8.0 * np.pi * DTa)
    # tt = np.linspace(max(1e-6, np.nanmin(time[mfit])), np.nanmax(time[mfit]), 300)
    # yy = C/ tt
    #
    # ax.plot(tt, yy, color=style["color"], lw=1.6, alpha=0.9)
    plt.show()