
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
import skimage.filters as sk
from scipy import ndimage

base_path = "/home/chorus/EXP_convert_binned/"
# "exp_13","exp_12","exp_22",'
root_folders = [
    os.path.join(base_path, d)
    for d in sorted(os.listdir(base_path))
    if d.startswith(("exp_05")) and os.path.isdir(os.path.join(base_path, d))
]
#root_folders.append("/home/chorus/EXP_convert_binned/exp_12_01_2/")
#root_folders.append("/home/chorus/EXP_convert_binned/exp_15_01_2/")
mesureta=False
dx = 0.06 / 2048
idx0 = 50
frac = 0.2
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
    if Tip=="homo":
        A_pore = A_tot * eps_sable
    else :
        A_pore = A_tot *eps_sable*(1-f_billes)
    vp = Q_m3_s / A_pore
    Ta = L / vp

    return Ta, vp



def extract_Ta_from_csv(root_folder, colonne="grande",Tip="homo"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)

    D = 0.055 if colonne == "grande" else 0.027
    L = extract_L_from_label(root_folder)

    Ta, vp = Ta_for_bead(
        df["Timestamp"],
        df["Weight"],
        D=D,
        L=L,
        Tip=Tip
    )
    print(Ta,"s")
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


def read_label(root_folder):
    label_path = os.path.join(root_folder, "label.txt")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            return f.read().strip()
    return os.path.basename(os.path.normpath(root_folder))


import re

def extract_L_from_label(root_folder, default_L=0.01):
    """
    Extract inclusion size L (in meters) from label.txt.
    Expected patterns: '1cm', '2 cm', '10cm', etc.
    """
    label_path = os.path.join(root_folder, "label.txt")

    if not os.path.exists(label_path):
        return default_L

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()

    match = re.search(r'(\d+(?:\.\d+)?)\s*cm', text)
    if match:
        L_cm = float(match.group(1))
        return L_cm * 1e-2  # cm → m

    return default_L

def compute_dispersion_radius(I, dx, frac=0.3):
    nt, ny, nx = I.shape

    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)

    R = np.full(nt, np.nan)
    xc = np.full(nt, np.nan)
    yc = np.full(nt, np.nan)

    for t in range(nt):
        C = I[t]
        if np.all(np.isnan(C)):
            continue

        C = np.nan_to_num(C, nan=0.0)
        if C.sum() == 0:
            continue

        M = C.sum()
        xc[t] = np.sum(X * C) / M
        yc[t] = np.sum(Y * C) / M

        r = np.sqrt((X - xc[t])**2 + (Y - yc[t])**2)
        mask = C > 0
        r_flat = r[mask]
        m_flat = C[mask]

        idx = np.argsort(r_flat)
        r_sorted = r_flat[idx]
        m_sorted = m_flat[idx]

        m_cum = np.cumsum(m_sorted)
        R[t] = r_sorted[np.searchsorted(m_cum, frac * M)]

    return R, xc, yc
def process_experiment(root_folder, colonne="grande"):
    img_folder = find_image_folder(root_folder)
    I_full = load_full_sequence_16bit(img_folder)
    I = I_full.copy()
    lab = read_label(root_folder)
    L = extract_L_from_label(root_folder)

    I_canny = sk.sato(I[0], sigmas=range(4, 10, 1))
    I_canny = ndimage.median_filter(I_canny, 13)

    masksato = I_canny > 0.00003
    x, y = np.meshgrid(np.arange(I[0].shape[0]), np.arange(I[0].shape[1]))
    maskcercle = (x - np.shape(I[0])[0]/2)**2 + (y - np.shape(I[0])[1]/2)**2 > 213**2
    maskcercle = maskcercle + masksato
    I[0][maskcercle] = np.nan

    Inorm = I[0] / np.nanmean(I[0], axis=0)

    if "homo" in lab:
        mask2 = np.swapaxes(
            np.swapaxes(
                np.tile((I[idx0 + 10] < np.nanquantile(I[30], 0.6)).reshape(I.shape[1], I.shape[2], 1), I.shape[0]),
                0, 2),
            1, 2
        )
    else:
        mask2 = np.swapaxes(
            np.swapaxes(
                np.tile((I[idx0 + 10] < np.nanquantile(I[40], 0.01)).reshape(I.shape[1], I.shape[2], 1), I.shape[0]),
                0, 2),
            1, 2
        )

    I[mask2] = np.nan
    mean = np.nanmean(I, axis=(1, 2))
    var = np.nanvar(I, axis=(1, 2))
    A = (np.nansum(maskcercle[0]) + np.nansum(mask2[0])) * dx**2

    if "homo" in lab:
        Ta, vp = extract_Ta_from_csv(root_folder, colonne, Tip="homo")
    else:
        Ta, vp = extract_Ta_from_csv(root_folder, colonne, Tip="hetero")

    t_img, dt = build_t_img_from_csv(root_folder, I.shape[0])
    time = t_img / Ta

    R, xc, yc = compute_dispersion_radius(I, dx, frac)

    return {
        "root": os.path.basename(root_folder),
        "I_canny": I_canny,
        "maskcercle": maskcercle,
        "Inorm": Inorm,
        "mask2": mask2[0],
        "I0": I[0],
        "mean": mean,
        "time": time,
        "var": var,
        "A": A,
        "R": R,
        "label": lab
    }

def run_one_experiment(root):
    try:
        res = process_experiment(root)

        time = res["time"]
        mean = res["mean"]
        var = res["var"]
        A = res["A"]
        R = res["R"]

        Sigma = (var * A) / mean**2

        mask = (
            ~np.isnan(R) &
            ~np.isnan(Sigma) &
            (R > 0) &
            (Sigma > 0)
        )

        time_m = time[mask].ravel()
        Sigma_m = Sigma[mask]

        noise = np.mean(Sigma_m[0:2])
        mask2 = Sigma_m > (100 * noise)
        i0 = np.argmax(mask2) if mask.any() else None
        t0 = time_m[i0]

        time_m = time_m - t0
        Sigma_m = Sigma_m / np.max(Sigma_m)

        return {
            "label": res["label"],
            "time": time_m,
            "value": Sigma_m,
            "mass": mean,
            "debug": res
        }

    except Exception as e:
        print(f"[ERROR] {root}: {e}")
        return None


# ==========================================================
# EXECUTION & PLOT
# ==========================================================

fig, ax = plt.subplots()
nproc = min(cpu_count(), len(root_folders))
print(f"Using {nproc} processes")

with Pool(processes=nproc) as pool:
    results = pool.map(run_one_experiment, root_folders)

plt.ion()

for res in results:
    if res is None:
        continue

    dbg = res["debug"]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.ravel()

    axs[0].imshow(dbg["I_canny"], cmap="viridis")
    axs[1].imshow(dbg["maskcercle"], cmap="gray")
    axs[2].imshow(dbg["Inorm"], cmap="viridis")
    axs[3].imshow(dbg["mask2"], cmap="gray")
    axs[4].imshow(dbg["I0"], cmap="viridis")
    axs[5].plot(dbg["mean"])

    fig.suptitle(dbg["root"], fontsize=16)
    plt.tight_layout()
    plt.show(block=False)

    label = res['label'].lower()
#     # plt.plot(res['time'],res['mass'],label=label+'mass')

fig, ax = plt.subplots()

for res in results:
    if res is None:
        continue

    label = res["label"].lower()

    if "2cm" in label:
        ax.plot(res["time"], res["value"], 'yo', ms=3)

    elif "0.6cm" in label:
        ax.plot(res["time"], res["value"], 'co', ms=3)

    elif "1cm" in label and "grossier" in label:
        ax.plot(res["time"], res["value"], 'bD', mfc='none', ms=3)

    elif "1cm" in label and "fin" in label:
        ax.plot(res["time"], res["value"], 'mo', ms=3)

    elif "homo" in label and "fin" in label:
        ax.plot(res["time"], res["value"], 'go', ms=3)

    elif "homo" in label and "grossier" in label:
        ax.plot(res["time"], res["value"], 'gD', mfc="none", ms=3)

from matplotlib.lines import Line2D
#
#
legend_elements = [
    # Marker meaning (sand type)
    Line2D([0], [0], marker='o', color='black', linestyle='None',
           markersize=8, label='Fine sand'),
    Line2D([0], [0], marker='D', mfc='none',color='black', linestyle='None',
           markersize=8, label='Coarse sand'),
    Line2D([0], [0], marker='o', color='m', linestyle='None',
           markersize=8, label='1 cm beads'),
    Line2D([0], [0], marker='o', color='y', linestyle='None',
           markersize=8, label='2 cm beads'),
    Line2D([0], [0], marker='o', color='c', linestyle='None',
           markersize=8, label='0.6 cm beads'),
    Line2D([0], [0], marker='o', color='g', linestyle='None',
           markersize=8, label='no beads, fine sand'),
    Line2D([0], [0], marker='D',mfc='none', color='g', linestyle='None',
           markersize=8, label='no beads, coarse sand'),
]
# legend_elements = [
#     # Marker meaning (sand type)
#     Line2D([0], [0], marker='D', color='m', linestyle='None',
#            markersize=8, label='débit asymétrique'),
#     Line2D([0], [0], marker='D', mfc='none',color='b', linestyle='None',
#            markersize=8, label=' débit symétrique')]

plt.legend(handles=legend_elements)
plot_semi_log_slope(ax,3.5,1.1,-0.24,length=10)
plot_semi_log_slope(ax,0,0.8,-0.6,length=5)
plot_semi_log_slope(ax,15,0.01,-0.05,length=12)

plt.axvline(0, color="k", ls="--", alpha=0.5)
plt.xlabel(r"$(t - t_0)/t_a$")
plt.ylabel(r"$\frac{\sigma_c^2 / \cdot A\mu_c^2)}{\sigma_0^2 / \cdot A\mu_0^2}$")
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.xlim(left=0.1,right=40)
plt.yscale('log')
plt.tight_layout()
plt.show()





# ==========================================================
# RAYON DE DISPERSION vs TEMPS
# ==========================================================
#
# plt.figure(figsize=(6,4))
# plt.plot(time, R, 'o', ms=4)
# plt.xlabel(r"$t/t_a$")
# plt.ylabel("Dispersion radius $R$ (m)")
# plt.grid(True, alpha=0.3)
# plt.xscale("log")
# plt.yscale("log")
# plt.tight_layout()
# plt.show()
# # ==========================================================
# # VARIANCE SCALAIRE vs TEMPS
# # ==========================================================
#
# plt.figure(figsize=(6,4))
# plt.plot(time, var, 'o', ms=4)
# plt.xlabel(r"$t/t_a$")
# plt.ylabel(r"Scalar variance $\sigma_c^2$")
# plt.xscale("log")
# plt.yscale("log")
# plt.grid(True, alpha=alpha)
# plt.tight_layout()
# plt.show()

