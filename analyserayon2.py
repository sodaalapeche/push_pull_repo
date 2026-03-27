import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import pandas as pd
import re
from matplotlib.patches import Circle

# --- AJOUTE CES LIGNES JUSTE APRÈS LES IMPORTS ---
plt.rcParams['font.family'] = 'Ubuntu'          # Police Ubuntu
plt.rcParams['axes.titlesize'] = 'x-large'      # Taille titre
plt.rcParams['axes.labelsize'] = 'large'        # Taille labels
plt.rcParams['legend.fontsize'] = 'large'       # Taille légende
plt.rcParams['xtick.labelsize'] = 'large'       # Taille ticks X
plt.rcParams['ytick.labelsize'] = 'large'       # Taille ticks Y


# ---------------------------------------------------------------------
# 🔹 UTILITAIRES
# ---------------------------------------------------------------------

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else -1

def read_fps_from_param_file(folder_path):
    """
    Lit le fichier 'parametres.txt' du dossier et extrait le FPS.
    Retourne un float (ex: 2.0) ou None si non trouvé.
    Gère automatiquement les encodages (UTF-8, Latin-1, etc.)
    """
    param_path = os.path.join(folder_path, "parametres.txt")
    if not os.path.exists(param_path):
        print(f"⚠️ Fichier 'parametres.txt' introuvable dans {folder_path}")
        return None

    # On tente UTF-8 d'abord, puis Latin-1 en cas d'erreur
    for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            with open(param_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"⚠️ Impossible de lire {param_path} (encodage inconnu)")
        return None

    for line in lines:
        if "FPS" in line.upper():
            match = re.search(r"[\d\.]+", line)
            if match:
                return float(match.group())

    print(f"⚠️ FPS non trouvé dans {param_path}")
    return None



def Ta_for_bead(timestamps, weights_g, D=0.055, L=0.01,
                f_billes=0, phi_sable=0.4, tau=1.0):
    """
    Calcule le temps d'advection T_a à partir des données de poids.
    """
    timestamps = np.array(timestamps, dtype=float)
    weights_g = np.array(weights_g, dtype=float)
    valid_mask = weights_g > 1000
    timestamps = timestamps[valid_mask]
    weights_g = weights_g[valid_mask]

    if len(weights_g) < 5:
        raise ValueError("Pas assez de données physiques (>1000 g).")

    diff_w = np.diff(weights_g)
    jump_indices = np.where(diff_w < -0.5 * np.max(weights_g))[0]

    if len(jump_indices) > 0:
        start = 0
        segments = []
        for j in jump_indices:
            segments.append((timestamps[start:j+1], weights_g[start:j+1]))
            start = j + 1
        if start < len(timestamps):
            segments.append((timestamps[start:], weights_g[start:]))
    else:
        segments = [(timestamps, weights_g)]

    best_seg, best_score = None, -np.inf
    for ts, ws in segments:
        if len(ws) < 3:
            continue
        coeffs = np.polyfit(ts, ws, 1)
        slope = coeffs[0]
        fit = np.polyval(coeffs, ts)
        r2 = 1 - np.sum((ws - fit)**2) / np.sum((ws - np.mean(ws))**2)
        if slope > 0 and r2 > best_score:
            best_seg, best_score = (ts, ws, slope), r2

    if best_seg is None:
        raise ValueError("Aucun segment d'écoulement valide trouvé.")

    ts, ws, dMdt_g_s = best_seg
    Q = dMdt_g_s * 1e-6
    A_tot = np.pi * (D / 2.0)**2
    T_a = (L * A_tot * (1 - f_billes)) / (Q / tau)
    print(dMdt_g_s)
    print("r2:")
    print(best_score)
    if T_a > 30 or T_a < 1:
        T_a = 7.0
    return T_a


def measure_dispersion_radius_quantile(image, mass_fraction=0.3, eps=1e-12):
    """
    Calcule le barycentre et le rayon contenant une fraction de masse donnée (quantile).
    """
    img = image.astype(float)
    img -= img.min()
    total_mass = np.sum(img) + eps
    if total_mass < eps:
        return np.nan, np.nan, np.nan

    y, x = np.indices(img.shape)
    x_mean = np.sum(x * img) / total_mass
    y_mean = np.sum(y * img) / total_mass

    distances = np.sqrt((x - x_mean)**2 + (y - y_mean)**2)
    distances_flat = distances.ravel()
    mass_flat = img.ravel()

    sorted_indices = np.argsort(distances_flat)
    sorted_mass = mass_flat[sorted_indices]
    sorted_distances = distances_flat[sorted_indices]

    cum_mass = np.cumsum(sorted_mass) / total_mass
    idx = np.searchsorted(cum_mass, mass_fraction)
    r_quantile = sorted_distances[min(idx, len(sorted_distances) - 1)]

    return x_mean, y_mean, r_quantile


def show_circle_on_image(image, x_mean, y_mean, r_std, title=None):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    circle = Circle((x_mean, y_mean), r_std, color='r', fill=False, lw=2)
    ax.add_patch(circle)
    ax.set_title(title or f"Rayon = {r_std:.1f} px")
    ax.axis('off')
    plt.show()


def compute_background_mean(folder_path, n_first=5):
    tif_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tif")])
    if len(tif_files) == 0:
        raise ValueError("Aucune image .tif trouvée dans le dossier parent.")
    n_first = min(n_first, len(tif_files))
    images = [imread(os.path.join(folder_path, tif_files[i])).astype(float) for i in range(n_first)]
    return np.mean(images, axis=0)


def process_dispersion_series(folder_path, show_every=10, plot=False):
    """
    Analyse les images contenues dans le sous-dossier 'serie1'
    du dossier d'expérience donné, en utilisant le temps réel
    déduit des timestamps du fichier weight_data.csv.
    Adimensionne le temps par T_a et le rayon par R0 = 3e valeur.
    """

    serie_path = os.path.join(folder_path, "serie1")
    if not os.path.exists(serie_path):
        raise FileNotFoundError(f"Sous-dossier 'serie1' introuvable dans {folder_path}")

    csv_path = os.path.join(folder_path, "weight_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    # --- Lecture robuste du CSV ---
    for enc in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Impossible de lire weight_data.csv avec encodages standard")

    if not {'Image_Index', 'Timestamp', 'Weight'}.issubset(df.columns):
        raise ValueError("Le CSV doit contenir les colonnes : Image_Index, Timestamp, Weight")

    image_indices = df["Image_Index"].values
    timestamps = df["Timestamp"].values.astype(float)
    weights = df["Weight"].values.astype(float)

    # --- Calcul automatique du FPS à partir des timestamps ---
    timestamps = timestamps - timestamps[0]  # temps relatif
    dt = np.diff(timestamps)
    dt = dt[dt > 0]  # on ignore les valeurs nulles ou négatives
    dt_mean = np.mean(dt)
    fps_auto = 1.0 / dt_mean
    print(f"📸 FPS détecté : {fps_auto:.3f} (Δt moyen = {dt_mean:.3f} s)")

    # --- Calcul du T_a à partir des poids ---
    T_a = Ta_for_bead(timestamps, weights)
    print(f"🧮 T_a (physique) = {T_a:.3f} s")

    # --- Préparation des images ---
    tif_files = sorted([f for f in os.listdir(serie_path) if f.endswith(".tif")],
                       key=extract_number)
    if len(tif_files) <= 1:
        raise ValueError("Pas assez d'images (≥ 2)")

    bg_mean = compute_background_mean(folder_path)

    radii, time_seq = [], []
    for i, tif_file in enumerate(tif_files[1:]):  # on saute la première
        num = extract_number(tif_file)
        mask = image_indices == num
        if not np.any(mask):
            continue

        t = timestamps[mask][0]
        img = imread(os.path.join(serie_path, tif_file)).astype(float)
        img_corr = img - bg_mean
        img_corr[img_corr < 0] = 0

        x_mean, y_mean, r = measure_dispersion_radius_quantile(img_corr, eps=0.001)
        radii.append(r)
        time_seq.append(t)

        if plot and i % show_every == 0:
            show_circle_on_image(img, x_mean, y_mean, r, title=f"{tif_file} | R = {r:.1f}")

    if len(radii) == 0:
        raise ValueError("Aucune image valide trouvée.")

    time_seq = np.array(time_seq)
    radii = np.array(radii)

    # --- Conversion temps réel -> adimensionné ---
    time_dimless = time_seq / T_a
    t_shift = time_dimless[np.nanargmin(radii)]
    time_dimless -= t_shift

    # --- Conversion rayon en mm ---
    radii_mm = radii * 55 / 2048  # facteur d'échelle pixel → mm

    # --- Adimensionnement par R0 (3e valeur) ---
    if len(radii_mm) >= 3:
        R0 = radii_mm[2]
    else:
        R0 = radii_mm[0]
    if R0 <= 0 or np.isnan(R0):
        R0 = np.nanmean(radii_mm[:5])  # fallback
    radii_dimless = radii_mm / R0

    print(f"📏 R0 (3e rayon) = {R0:.3f} mm")

    return {
        "time": time_dimless,
        "radius": radii_mm,
        "Ta": T_a,
        "fps": fps_auto,
        "dt_mean": dt_mean,
        "R0": R0
    }

# ---------------------------------------------------------------------
# 🔹 TRACE DU RÉSULTAT
# ---------------------------------------------------------------------

def plot_dispersion(result, mode="semi-log", label=None):
    time = result["time"]
    radius = result["radius"]

    plt.scatter(time, radius, s=20, alpha=0.7, label=label)
    plt.xlabel(r"$t/T_a$", fontsize="x-large")
    plt.ylabel(r"Dispersion radius (mm)", fontsize="x-large")

    if mode == "semi-log":
        plt.yscale("log")
    elif mode == "loglog":
        plt.xscale("log")
        plt.yscale("log")

    plt.grid(True, ls="--", lw=0.6)
    plt.legend(fontsize="medium")
    plt.tight_layout()


def plot_minus_one_slope(ax, slope, start_x, start_y, length_decades=1, offset_factor=1.2, **kwargs):
    x_vals = np.logspace(np.log10(start_x), np.log10(start_x) + length_decades, 100)
    y_vals = start_y * offset_factor * (x_vals / start_x) ** (0.95*slope)
    ax.plot(x_vals, y_vals, label=r"$t^{-1}$", **kwargs)
    ax.text(x_vals[-1]*0.7, y_vals[-1]*0.7, f"slope = {slope}", fontsize="x-large", ha="left", va="top")


# ---------------------------------------------------------------------
# 🔹 FONCTION PRINCIPALE
# ---------------------------------------------------------------------

def run_dispersion(folder=False, nb=1, mode="semi-log"):
    if folder is False:
        folder = input("Chemin du dossier d'expérience : ")

    result = process_dispersion_series(folder, show_every=10)
    print(f"Analyse : {len(result['radius'])} images traitées")
    print(f"T_a = {result['Ta']:.2f} s | FPS = {result['fps']:.2f}")

    plot_dispersion(result, mode=mode, label=f"{nb}")
    return result


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 5))
    run_dispersion("/home/chorus/exp_septoct/exp_3_11_6/", nb="homogeneous", mode="loglog")
    #run_dispersion("/home/chorus/exp_septoct/16_10_4/", nb="heterogeneous", mode="loglog")
    plot_minus_one_slope(ax, 0.5, 1,  1.1, 0.5)
    #ax.get_legend().remove()
    plt.show()
