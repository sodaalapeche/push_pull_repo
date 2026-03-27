import os
import re
from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path

# --- OpenFOAM comparison
SIM_VTK_DIR = "/home/chorus/OpenFOAM/chorus-13/run/cylinderDiffusion2D/VTK"
USE_SIMULATION = True
SIM_TIME_SHIFT = 0.0   # optional shift in physical seconds
SIM_LABEL = "OpenFOAM"
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from scipy.signal import fftconvolve
import skimage.filters as sk

# ==========================================================
# GLOBAL PARAMETERS
# ==========================================================
BASE_PATH = "/home/chorus/HETEROGENE_binned/"
MASK_REF_PATH = "/home/chorus/test/masque.jpg"

# ATTENTION: dans tes scripts tu as deux dx différents (*4 et *2)
# Choisis UN dx cohérent avec les images que tu analyses.
dx = (0.06 / 2048) * 4  # m/pixel (mets *2 si c'est la bonne binning)

# Recalage masque hex
ANGLE_MAX = 2.0
ANGLE_STEP = 0.1
TRANS_MAX = 30

# ROI variance
R_EVAL_FACTOR = 1.4 # <-- demandé : 1.2 * rayon max RMS

# RMS radius (repris de ton script 2, sans affichage)
RMS_BLUR_CENTER_SIGMA = 10.0
RMS_BLUR_MOMENTS_SIGMA = 2.0
RMS_RING_RMIN_FRAC = 0.38
RMS_RING_RMAX_FRAC = 0.48
RMS_THRESH_K_SIGMA = 4.0
RMS_ROI_K = 2.5
RMS_ROI_RMIN_PIX = 25.0
RMS_ROI_RMAX_FRAC = 0.48


def _discover_case_prefix(vtk_dir):
    vtk_dir = Path(vtk_dir)
    vtk_files = sorted(vtk_dir.glob('*.vtk'))
    candidates = []
    for f in vtk_files:
        m = re.match(r'^(.*)_(\d+(?:\.\d+)?)\.vtk$', f.name)
        if m and not m.group(1).startswith('outerWall'):
            candidates.append(m.group(1))
    if not candidates:
        raise FileNotFoundError(f"No case VTK files found in {vtk_dir}")
    # most frequent prefix wins
    return max(set(candidates), key=candidates.count)


def load_simulation_sigma_curve(vtk_dir=SIM_VTK_DIR, field_name='T', case_prefix=None):
    """
    Load OpenFOAM simulation curve directly from VTK files.
    Only the simulation-curve logic is handled here; the rest of the script is untouched.
    """
    import pyvista as pv

    vtk_dir = Path(vtk_dir)
    if case_prefix is None:
        case_prefix = _discover_case_prefix(vtk_dir)

    pairs = []
    pat = re.compile(rf'^{re.escape(case_prefix)}_(\d+(?:\.\d+)?)\.vtk$')
    for f in vtk_dir.glob(f'{case_prefix}_*.vtk'):
        m = pat.match(f.name)
        if m:
            pairs.append((float(m.group(1)), f))

    pairs.sort(key=lambda x: x[0])
    if not pairs:
        raise FileNotFoundError(f"No VTK files found in {vtk_dir} for prefix {case_prefix}")

    rows = []
    for t, f in pairs:
        mesh = pv.read(str(f))

        if field_name in mesh.cell_data:
            T = np.asarray(mesh.cell_data[field_name]).reshape(-1)
        elif field_name in mesh.point_data:
            mesh = mesh.point_data_to_cell_data()
            T = np.asarray(mesh.cell_data[field_name]).reshape(-1)
        else:
            continue

        sized = mesh.compute_cell_sizes(length=False, area=True, volume=True)
        if 'Volume' in sized.cell_data:
            w = np.asarray(sized.cell_data['Volume']).reshape(-1)
            if not np.any(w > 0) and 'Area' in sized.cell_data:
                w = np.asarray(sized.cell_data['Area']).reshape(-1)
        elif 'Area' in sized.cell_data:
            w = np.asarray(sized.cell_data['Area']).reshape(-1)
        else:
            raise RuntimeError(f"No Volume/Area weights found in {f}")

        wsum = np.sum(w)
        mean_T = np.sum(w * T) / wsum
        var_T = np.sum(w * (T - mean_T)**2) / wsum
        sigma = np.nan if np.isclose(mean_T, 0.0) else var_T / (mean_T**2)
        rows.append((t, mean_T, var_T, sigma))

    if not rows:
        raise RuntimeError(f"No readable VTK fields found in {vtk_dir}")

    df = pd.DataFrame(rows, columns=['time', 'mean_T', 'variance_T', 'sigma'])
    valid = np.isfinite(df['sigma'].values)
    sigma0 = df.loc[valid, 'sigma'].iloc[0]
    df['sigma_over_sigma0'] = df['sigma'] / sigma0

    print('[SIM] case_prefix =', case_prefix)
    print('[SIM] loaded times =', df['time'].to_list())
    print('[SIM] max time =', float(df['time'].max()))

    time_s = np.asarray(df['time'], dtype=float)
    sigma_norm = np.asarray(df['sigma_over_sigma0'], dtype=float)
    valid = np.isfinite(time_s) & np.isfinite(sigma_norm)
    return {'time_s': time_s[valid], 'sigma_norm': sigma_norm[valid]}


def build_master_sim_curve(sim_data, Ta_ref):
    time_s = np.asarray(sim_data['time_s'], dtype=float)
    sigma_norm = np.asarray(sim_data['sigma_norm'], dtype=float)

    # keep simulation time in its own nondimensional form: tau = t / Ta_sim
    tau_sim = time_s / Ta_ref

    valid = np.isfinite(tau_sim) & np.isfinite(sigma_norm)
    return tau_sim[valid], sigma_norm[valid]
# DISCOVERY
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
# TIME / Ta utilities (repris, simplifiés)
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
        A_pore = A_tot * eps_sable*0.75
    elif Tip=="homo fine":
        A_pore = A_tot * eps_sable*0.67
    elif Tip=="0.1":
        A_pore = A_tot*eps_sable*(1-f_billes)*1.7
    elif Tip == "0.6":
        A_pore = A_tot*eps_sable*(1-f_billes)*0.8
        print("tick0.6)")
    elif Tip == "0.3 coarse":
        A_pore = A_tot*eps_sable*(1-f_billes)*0.9
    elif Tip=="0.6 fine":
        A_pore = A_tot *eps_sable*(1-f_billes)*1.23
    elif Tip=="Hetero":
        A_pore = A_tot*eps_sable*(1-f_billes)
    elif Tip == "0.1 coarse":
        A_pore = A_tot*eps_sable*(1-f_billes)*0.95
    elif Tip == "stokes 0.1":
        A_pore = A_tot*0.4
    else :
        A_pore = A_tot *eps_sable*(1-f_billes)

    print()
    print("A_pore",A_pore)
    vp = Q_m3_s / A_pore
    print("vp ==  ",vp,"  m/s")
    Ta = L / vp
    # print("Q (g/s)=",dMdt_g_s)
    # print("vp(m/s) : ",vp)

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
    if L==0.01 and "coarse" in label:
        Tip="0.1 coarse"
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
    if L==0.01 and "stokes" in label:
        Tip = "stokes 0.1"

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

# ==========================================================
# Stripe correction (idem)
# ==========================================================
def correct_vertical_stripes_stack_firstframe_ref(I_stack, smooth=2, eps=1e-6):
    ref = I_stack[0]
    col_profile = np.nanmean(ref, axis=0)
    if smooth and smooth > 3:
        col_profile = ndimage.median_filter(col_profile, size=smooth)
    col_profile /= np.nanmean(col_profile)
    col_profile = np.maximum(col_profile, eps)
    return I_stack / col_profile[None, None, :]

def compute_col_profile_from_frame0(frame0, smooth=2, eps=1e-6):
    col_profile = np.nanmean(frame0, axis=0)
    if smooth and smooth > 3:
        col_profile = ndimage.median_filter(col_profile, size=smooth)
    col_profile /= np.nanmean(col_profile)
    col_profile = np.maximum(col_profile, eps)
    return col_profile

def apply_col_profile(frame, col_profile):
    return frame / col_profile[None, :]

# ==========================================================
# HEX mask via Sato (repris + nettoyé)
# ==========================================================
def extract_structure_sato(img):
    I_sato = sk.sato(img, sigmas=range(4, 10, 1))
    I_sato = ndimage.median_filter(I_sato, size=13)
    mask_sato = I_sato > 1e-5

    H, W = img.shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask_circle = (X - W/2)**2 + (Y - H/2)**2 > 213**2
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
# RMS radius (batch) : repris de ton script rayon
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
        return np.nan, (np.nan, np.nan)

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
        return np.nan, (xc, yc)

    rbin = np.floor(r).astype(np.int32)
    nbins = int(np.floor(rmax_global)) + 1
    nbins = max(nbins, 10)
    E = np.bincount(rbin.ravel(), weights=wpos.ravel(), minlength=nbins).astype(np.float64)
    Ecum = np.cumsum(E)
    Etot = Ecum[-1]
    if Etot <= eps:
        return np.nan, (xc, yc)

    idx80 = int(np.searchsorted(Ecum, 0.80 * Etot))
    r_eff = float(np.clip(idx80, 1, rmax_global))

    roi_r = float(np.clip(roi_k * r_eff, roi_rmin_pix, rmax_global))
    roi = r <= roi_r

    w_roi = np.where(roi, w, 0.0)
    M2 = w_roi.sum()
    if M2 <= eps:
        return np.nan, (xc, yc)

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

    return sigma_m, (xc2, yc2)
def plot_blob_radius(image, mask, xc_pix, yc_pix, R_eval_m, dx, title=""):
    """
    Plot image traitée avec cercle du rayon d'évaluation
    """

    fig, ax = plt.subplots()

    img_plot = image.copy()
    # img_plot[~mask] = np.nan

    im = ax.imshow(img_plot, cmap="viridis")

    # centre
    ax.scatter(xc_pix, yc_pix, color="red", marker="x", s=100)

    # cercle
    R_pix = R_eval_m / dx
    circle = plt.Circle(
        (xc_pix, yc_pix),
        R_pix,
        color="red",
        fill=False,
        linewidth=2
    )

    ax.add_patch(circle)

    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im)

    plt.show()
# ==========================================================
# Variance in circular ROI (fast)
# ==========================================================
def roi_stats(frame, hex_mask, Xpix, Ypix, xc_pix, yc_pix, r_eval_pix):
    # disque centré sur (xc_pix, yc_pix)
    r2 = (Xpix - xc_pix) ** 2 + (Ypix - yc_pix) ** 2
    roi = (r2 <= (r_eval_pix ** 2))

    vals = frame[roi]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.var(vals))

# ==========================================================
# Main per experiment
# ==========================================================
def process_experiment(root_folder, colonne="grande"):
    img_folder = find_image_folder(root_folder)
    files = list_tifs(img_folder)
    n = len(files)

    # --- read first frame (for mask + stripe profile)
    img0_u16 = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if img0_u16 is None:
        raise RuntimeError(f"Cannot read {files[0]}")
    frame0 = img0_u16.astype(np.float32) / 65535.0

    # stripe correction profile
    col_profile = compute_col_profile_from_frame0(frame0)

    # mask ref
    mask_ref = cv2.imread(MASK_REF_PATH, cv2.IMREAD_GRAYSCALE)
    if mask_ref is None:
        raise RuntimeError(f"Cannot read mask ref: {MASK_REF_PATH}")
    mask_ref_u8 = (mask_ref > 128).astype(np.uint8)
    mask_ref_u8 = cv2.resize(mask_ref_u8, frame0.shape[::-1], interpolation=cv2.INTER_NEAREST)

    # compute hex mask once (use corrected frame0 for better edges)
    frame0c = apply_col_profile(frame0, col_profile)
    hex_mask = compute_hex_mask_from_sato(frame0c, mask_ref_u8)

    # precompute pixel grids for ROI stats
    H, W = frame0.shape
    Ypix, Xpix = np.indices((H, W))

    # --- time axis
    t_img, _ = build_t_img_from_csv(root_folder, n)
    Ta, _ = extract_Ta_from_csv(root_folder, colonne=colonne)
    time = t_img / Ta

    # --- 1) compute sigma_m(t) (RMS radius) to get Rmax
    sigma_m = np.full(n, np.nan, dtype=float)
    xc_pix = np.full(n, np.nan, dtype=float)
    yc_pix = np.full(n, np.nan, dtype=float)

    for i, f in enumerate(files):
        img_u16 = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img_u16 is None:
            continue
        frame = img_u16.astype(np.float32) / 65535.0
        frame = apply_col_profile(frame, col_profile)

        # apply hex mask (as NaN outside)
        frame_masked = frame.copy()
        # frame_masked[~hex_mask] = np.nan

        # RMS radius needs finite values -> replace NaN by 0 (outside mask)
        tmp = np.nan_to_num(frame_masked, nan=0.0)

        s, (xc, yc) = compute_rms_radius(tmp, dx)
        sigma_m[i] = s
        xc_pix[i] = xc
        yc_pix[i] = yc

    Rmax_m = np.nanmax(sigma_m)
    if not np.isfinite(Rmax_m) or Rmax_m <= 0:
        raise RuntimeError("Rmax RMS invalide (sigma_m).")

    R_eval_m = R_EVAL_FACTOR * Rmax_m
    r_eval_pix = R_eval_m / dx

    # --- 2) compute mean/var in disk of radius R_eval_m
    mean = np.full(n, np.nan, dtype=float)
    var = np.full(n, np.nan, dtype=float)

    for i, f in enumerate(files):
        img_u16 = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img_u16 is None:
            continue
        frame = img_u16.astype(np.float32) / 65535.0
        frame = apply_col_profile(frame, col_profile)

        # stats: centre par frame si dispo, sinon fallback centre image
        xc = xc_pix[i] if np.isfinite(xc_pix[i]) else (W / 2.0)
        yc = yc_pix[i] if np.isfinite(yc_pix[i]) else (H / 2.0)

        m, v = roi_stats(frame, hex_mask, Xpix, Ypix, xc, yc, r_eval_pix)
        mean[i] = m
        var[i] = v

    # --- choix de t0 comme max(var) (dans la ROI)
    i0 = int(np.nanargmax(var))

    time0 = time - time[i0]+0.1
    sigma_m0 = sigma_m.copy()  # même longueur que time
    return {
        "root": os.path.basename(root_folder),
        "label": read_label(root_folder),
        "time": time0,
        "Ta" : Ta,
        "mean": mean,
        "var": var,
        "sigma_m": sigma_m,
        "Rmax_m": Rmax_m,
        "R_eval_m": R_eval_m,
        "mask": hex_mask,
        "xc_pix": xc_pix[i0],
        "yc_pix": yc_pix[i0],
        "I0": frame,  # image utilisée pour le plot
    }

def run_one_experiment(root):
    try:
        return process_experiment(root)
    except Exception as e:
        print(f"[ERROR] {root}: {e}")
        return None
def sigma_fick_norm_tau(tau, alpha, L, R0):
    tau = np.asarray(tau, dtype=float)
    k = 8.0 * np.pi * alpha * L / (R0**2)
    return 1.0 / (1.0 + k * tau)
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    REQUEST = [
         # (10, "fine"),
        # (0, "fine"),
        # (0, "coarse"),
        #  (10, "coarse"),
        # (3, "coarse"),
        # (6, "coarse"),
        # (6, "fine"),
        # (10,"stokes"),
        (3, 'fine')
    ]

    root_folders = select_exact_combinations(BASE_PATH, REQUEST)
    print("Experiments sélectionnées :", len(root_folders))
    if not root_folders:
        raise RuntimeError("Aucune expérience sélectionnée.")

    nproc = max(1, min(cpu_count() - 1, len(root_folders)))
    with Pool(nproc) as pool:
        results = pool.map(run_one_experiment, root_folders)

    # garde uniquement OK
    results = [r for r in results if r is not None]

    print("Experiments OK :", len(results))
    sim_data = None
    SIM_TA = 3.6 # actual Ta [s] used for the OpenFOAM run plotted here

    Ta_ref = None

    if USE_SIMULATION:
        try:
            sim_data = load_simulation_sigma_curve(SIM_VTK_DIR)
            Ta_ref = SIM_TA
            print(f"[OK] Simulation loaded, using Ta_sim = {Ta_ref:.6g} s")
        except Exception as e:
            print(f"[WARN] Could not load simulation: {e}")
            sim_data = None
            Ta_ref = None

    # ==========================================================
    # CLEAN PLOTTING (mm + fine/coarse only) - comme ton script 1
    # ==========================================================
    fig, ax = plt.subplots()

    style_map = {
        (10, "fine"): dict(marker="o", color="tab:blue",
                           mfc="tab:blue", ms=4, alpha=0.8),
        (10, "coarse"): dict(marker="D", color="tab:blue",
                             mfc="none", mew=1.2, ms=5, alpha=0.8),
        (10, "stokes"): dict(marker="^", color="purple",
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

    # ==========================================================
    # CLEAN PLOTTING : Sigma + Rayon (mêmes conventions)
    # ==========================================================
    figS, axS = plt.subplots()  # Sigma


    for res in results:
        label_text = res.get("label", "")
        lines = [l.strip().lower() for l in label_text.split("\n") if l.strip()]
        if len(lines) < 3:
            continue

        size_str = lines[1]
        sand = lines[2]

        try:
            L_mm = int(size_str.replace("mm", ""))
        except Exception:
            continue

        key = (L_mm, sand)
        if key not in style_map:
            continue
        style = style_map[key]
        Ta=res['Ta']
        time = res["time"]  # déjà (t - t0)/Ta
        # time=time/Ta
        mean = res["mean"]
        var = res["var"]
        rad = res["sigma_m"]  # en mètres

        # --------- courbe Sigma (comme avant)
        Sigma = var / (mean ** 2)
        Sigma = Sigma / np.nanmax(Sigma)
        validS = np.isfinite(time) & np.isfinite(Sigma) & (Sigma > 0)
        if np.any(validS):
            axS.plot(time[validS][::4], Sigma[validS][::4], linestyle="-", **style)
        # --------- courbe Rayon mesuré
        # options d'affichage :
        #   - en mm : rad*1e3
        #   - normalisé par Rmax : rad / np.nanmax(rad)
        Rmax = np.nanmax(rad)
        # --------- courbe simulation OpenFOAM sur la même base t/Ta
    if sim_data is not None and Ta_ref is not None:
        tau_sim, sigma_sim = build_master_sim_curve(sim_data, Ta_ref)
        axS.plot(
            tau_sim, sigma_sim,
            "--", color="k", lw=1.8, alpha=0.95,
            label=SIM_LABEL
        )
    # -------- Legend identique sur les 2 figures
    legend_elements = []
    for (L_mm, sand), style in style_map.items():
        if L_mm==0:
            True
            # legend_elements.append(
            #     Line2D(
            #         [0], [0],
            #         marker=style.get("marker", "o"),
            #         linestyle="None",
            #         color=style.get("color", "k"),
            #         markerfacecolor=style.get("mfc", style.get("color", "k")),
            #         markeredgewidth=style.get("mew", 1.0),
            #         markersize=style.get("ms", 6),
            #         alpha=style.get("alpha", 1.0),
            #         label=f"{sand} sand"
            #     )
            # )
        else:
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
                    label=f"{L_mm}mm ,{sand}"
                )
            )

    if sim_data is not None:
        legend_elements.append(
            Line2D(
                [0], [0],
                linestyle="--",
                color="k",
                linewidth=1.8,
                label=SIM_LABEL
            )
        )

    # --- SIGMA figure style
    axS.legend(handles=legend_elements)
    axS.axvline(0, color="k", ls="--", alpha=0.5)
    axS.set_xlabel(r"$t/t_a$", fontsize=18)
    axS.set_ylabel(r"$\frac{\sigma_c^2(t)}{\mu_c^2}$", fontsize=18)
    axS.grid(True, which="both", ls="--", alpha=0.3)
    # axS.set_xscale("log")
    axS.set_yscale("log")
    axS.set_xlim(left=0.1, right=32)
    # axS.set_ylim(bottom=0.002, top=1.1)
    figS.tight_layout()
    # --- Modèle Fickien (en dispersivité)
    # ALPHA_FICK = 1e-5  # m  (ex: 0.2 mm -> 2e-4 m)
    # R0_FICK = 3.7e-3  # m  (ex: 2 mm)
    # L_CHAR = 1.0e-2  # m  (ton L dans ta = L/U ; chez toi souvent 0.01)
    # # --- Courbe modèle Fickien (une seule fois, indépendante des expériences)
    # # on prend la plage de tau visible sur le graphe
    # tmin, tmax = axS.get_xlim()
    # tmin = max(1e-6, tmin)
    #
    # tt = np.logspace(np.log10(tmin), np.log10(tmax), 300)
    # yy = sigma_fick_norm_tau(tt, ALPHA_FICK, L_CHAR, R0_FICK)
    #
    # axS.plot(
    #     tt, yy,
    #     color="k", ls="--", lw=1.6, alpha=0.9,
    #     label=fr"Fick: $\alpha$={ALPHA_FICK * 1e3:.2f} mm, $R_0$={R0_FICK * 1e3:.1f} mm"
    # )
    plt.show()
    # figR, axR = plt.subplots()  # Rayon
    # for res in results:
    #     label_text = res.get("label", "")
    #     lines = [l.strip().lower() for l in label_text.split("\n") if l.strip()]
    #     if len(lines) < 3:
    #         continue
    #
    #     size_str = lines[1]
    #     sand = lines[2]
    #
    #     try:
    #         L_mm = int(size_str.replace("mm", ""))
    #     except Exception:
    #         continue
    #
    #     key = (L_mm, sand)
    #     if key not in style_map:
    #         continue
    #     style = style_map[key]
    #
    #     time = res["time"]  # déjà (t - t0)/Ta
    #     mean = res["mean"]
    #     var = res["var"]
    #     rad = res["sigma_m"]  # en mètres
    #
    #     # --------- courbe Sigma (comme avant)
    #
    #     # --------- courbe Rayon mesuré
    #     # options d'affichage :
    #     #   - en mm : rad*1e3
    #     #   - normalisé par Rmax : rad / np.nanmax(rad)
    #     Rmax = np.nanmax(rad)
    #     validR = np.isfinite(time) & np.isfinite(rad) & (rad > 0)
    #
    #     if np.any(validR):
    #         # Choix 1: rayon en mm
    #         axR.plot(time[validR][::4], (rad[validR] * 1e3)[::4], linestyle="None", **style)
    #
    #         # Choix 2 (si tu préfères normalisé): décommente
    #         # axR.plot(time[validR][::4], (rad[validR] / Rmax)[::4], linestyle="None", **style)
    #
    # # -------- Legend identique sur les 2 figures
    # legend_elements = []
    # for (L_mm, sand), style in style_map.items():
    #     legend_elements.append(
    #         Line2D(
    #             [0], [0],
    #             marker=style.get("marker", "o"),
    #             linestyle="None",
    #             color=style.get("color", "k"),
    #             markerfacecolor=style.get("mfc", style.get("color", "k")),
    #             markeredgewidth=style.get("mew", 1.0),
    #             markersize=style.get("ms", 6),
    #             alpha=style.get("alpha", 1.0),
    #             label=f"{L_mm} mm, {sand}"
    #         )
    #     )
    # # # --- RADIUS figure style (mêmes conventions)
    # # axR.legend(handles=legend_elements)
    # # axR.axvline(0, color="k", ls="--", alpha=0.5)
    # # axR.set_xlabel(r"$(t - t_0)/t_a$")
    # # axR.set_ylabel(r"$R_{RMS}$ (mm)")  # ou "R normalisé" si tu utilises l'option 2
    # # axR.grid(True, which="both", ls="--", alpha=0.3)
    # # axR.set_xscale("log")
    # # axR.set_yscale("log")  # mets en log si tu veux, sinon laisse linéaire
    # # axR.set_xlim(left=0.1, right=35)
    #
    # plt.show()
    resultsA=np.array(results)
    np.save("/home/chorus/data2.npy", resultsA)
