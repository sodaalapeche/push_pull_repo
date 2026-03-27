import gc

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import math
import pandas as pd

# ==============================
# PARAMÈTRES GLOBAUX
# ==============================

root_folders = [
    "/home/chorus/EXP/"
]
base_path = "/home/chorus/EXP_convert_binned/"

root_folders = [
    os.path.join(base_path, d)
    for d in sorted(os.listdir(base_path))
    if d.startswith(("exp","15","16")) and os.path.isdir(os.path.join(base_path, d))
]

dt = 2
dx = 0.018/ 573
m = -1.5
idx0 = 40

# ==============================
# OUTILS DÉBIT / TEMPS D’ADVECTION
# ==============================

def compute_fps(timestamps):
    ts = np.array(timestamps, dtype=float)
    dt = np.diff(ts)
    dt = dt[dt > 0]
    dt_mean = float(np.mean(dt))
    fps_auto = 1.0 / dt_mean
    return fps_auto, dt_mean


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
    A_pore = A_tot
    vp = Q_m3_s / A_tot
    Ta = L / vp

    return Ta, vp

def save_I_video(I, root_folder, fps=10, vmin=None, vmax=None):
    """
    Sauvegarde la séquence I en vidéo AVI avec colormap viridis
    """
    import matplotlib.cm as cm

    video_path = os.path.join(root_folder, "I_sequence_viridis.avi")

    n_frames, h, w = I.shape

    if vmin is None:
        vmin = np.nanpercentile(I, 5)
    if vmax is None:
        vmax = np.nanpercentile(I, 95)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for k in range(n_frames):
        frame = I[k].copy()
        frame[np.isnan(frame)] = vmin

        rgba = cmap(norm(frame))          # RGBA float [0,1]
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        writer.write(bgr)

    writer.release()
    print(f"Vidéo sauvegardée : {video_path}")

def extract_Ta_from_csv(root_folder, colonne="grande"):
    csv_path = os.path.join(root_folder, "weight_data.csv")
    df = pd.read_csv(csv_path)

    timestamps = df["Timestamp"].values
    weights = df["Weight"].values

    if colonne == "grande":
        D = 0.055
    else:
        D = 0.027

    Ta, vp = Ta_for_bead(timestamps, weights, D=D)
    return Ta, vp


# ==============================
# OUTILS IMAGES
# ==============================

def find_image_folder(root):
    for d in os.listdir(root):
        path = os.path.join(root, d)
        if os.path.isdir(path) and glob(os.path.join(path, "*.tif")):
            return path
    raise RuntimeError("Aucun dossier .tif trouvé")


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


# ==============================
# PIPELINE PRINCIPAL
# ==============================
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


def process_experiment(root_folder, colonne="grande"):
    img_folder = find_image_folder(root_folder)
    I_full = load_full_sequence_16bit(img_folder)

    means_full = I_full.mean(axis=(1, 2))
    best_frame = I_full[np.argmax(means_full)]

    I = I_full
    mask1 = np.swapaxes(
        np.swapaxes(
            np.tile((I[0] < 0.9* np.nanmedian(I[0])).reshape(I.shape[1], I.shape[2], 1), I.shape[0]),
            0, 2),
        1, 2
    )
    I[mask1] = np.nan

    mask2 = np.swapaxes(
        np.swapaxes(
            np.tile((I[idx0 + 10] < np.nanquantile(I[idx0], [0.7])).reshape(I.shape[1], I.shape[2], 1), I.shape[0]),
            0, 2),
        1, 2
    )
    I[mask2] = np.nan

    A = (np.nansum(mask1[0]) + np.nansum(mask2[0])) * dx ** 2

    mean = np.nanmean(I, axis=(1, 2))
    var = np.nanvar(I, axis=(1, 2))
    var = var * math.pi * 16
    ta, vp = extract_Ta_from_csv(root_folder, colonne=colonne)
    t_img, dt = build_t_img_from_csv(root_folder, I.shape[0])
    time = t_img /ta

    # idmax= np.argmax(var)
    # x0 = time[idmax]
    ##loglog
    y0 = var[idx0] / (A * mean[idx0] ** 2)
    # time = time - x0
    # b = np.log10(y0) - m * np.log10(x0)
    # D_eff = 10 ** (-b)
    # print(f"{os.path.basename(root_folder)} : D = {D_eff:.3e} , D/u = {D_eff / vp:.3e}")
    #
    #   fit_y = (10 ** b) * time ** m
    #save_I_video(I, root_folder, fps=int(1/dt))



    return time, mean, var, A, I


# ==============================
# EXÉCUTION
# ==============================

plt.figure()
k = 0

for folder in root_folders:
    time, mean, var, A, I = process_experiment(folder, colonne="grande")
    color = f"C{k}"
    label = os.path.basename(os.path.normpath(folder))
    sigma = var * A /mean**2
    sigma=sigma/sigma[0]
    xmax=np.argmax(sigma)
    time = time - time[xmax]
    plt.plot(time, var / (A * mean ** 2), 'o', color=color, ms=4, label=label)
    # plt.plot(time, 1e6*mean,'gx',label="mass")
    # if k == 0:
    #     plt.plot(time, fit_y, "r--", label=f"fit $\\propto t^{m}$")


    plt.yscale("log")
    plt.ylabel("$\\sigma_c^2 / (A\\mu_c^2)$")
    plt.xlabel("$t/t_a$")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    gc.collect()
    k += 1

plt.show()
