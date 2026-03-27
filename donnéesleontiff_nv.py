import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import re
import itertools
import pandas as pd
from matplotlib.ticker import ScalarFormatter, LogLocator
import matplotlib.pyplot as plt
import math



plt.rcParams['font.family'] = 'Ubuntu'          # Police Ubuntu
plt.rcParams['axes.titlesize'] = 'x-large'      # Taille titre
plt.rcParams['axes.labelsize'] = 'large'        # Taille labels
plt.rcParams['legend.fontsize'] = 'large'       # Taille légende
plt.rcParams['xtick.labelsize'] = 'large'       # Taille ticks X
plt.rcParams['ytick.labelsize'] = 'large'       # Taille ticks Y
def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else -1

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import linregress

from scipy.optimize import curve_fit

def exponential(x, y0, k):
    """Simple exponential function y = y0 * exp(k*x)"""
    return y0 * np.exp(k * x)


def fit_from_max(time_dimless, ratios, Ta, t_end_dimless):
    """
    Fit an exponential from the maximum of ratios to t_end (dimensionless time)

    Args:
        time_dimless (array): dimensionless times t/Ta
        ratios (array): ratio values
        Ta (float): advection time in seconds
        t_end_dimless (float): end time for the fit in units of Ta

    Returns:
        popt (tuple): fitted parameters (y0, k)
        tau_real (float): real-time exponential constant in seconds
    """
    # Center time at maximum
    max_idx = np.argmax(ratios)
    t0 = time_dimless[max_idx]
    mask = (time_dimless >= t0) & (time_dimless <= t_end_dimless)

    #x_fit = time_dimless[mask] - t0
    y_fit = ratios[mask]

    if len(time_dimless) < 2:
        raise ValueError("Not enough points for fitting in the selected range")

    popt, _ = curve_fit(exponential, time_dimless, y_fit, p0=(y_fit[0], -0.05))
    y0, k = popt
    tau_real = Ta / abs(k)

    print(f"Fit from max: y0={y0:.4f}, k={k:.4f} -> τ ≈ {tau_real:.2f} s")
    return popt, tau_real


def fit_between_bounds(time_dimless, ratios, Ta, A, B):
    """
    Fit an exponential between two dimensionless times A and B (units of Ta)

    Args:
        time_dimless (array): dimensionless times t/Ta
        ratios (array): ratio values
        Ta (float): advection time in seconds
        A (float): start of fit (dimensionless)
        B (float): end of fit (dimensionless)

    Returns:
        popt (tuple): fitted parameters (y0, k)
        tau_real (float): real-time exponential constant in seconds
    """
    mask = (time_dimless >= A) & (time_dimless <= B)
    x_fit = time_dimless[mask]
    y_fit = ratios[mask]

    if len(x_fit) < 2:
        raise ValueError("Not enough points for fitting in the selected range")

    # Shift time to start at zero for fitting
    #x_fit_shifted = x_fit - x_fit[0]

    popt, _ = curve_fit(exponential, x_fit, y_fit, p0=(y_fit[0], -0.05))
    y0, k = popt
    tau_real = Ta / abs(k)

    print(f"Fit A→B: y0={y0:.4f}, k={k:.4f} -> τ ≈ {tau_real:.2f} s")
    return popt, tau_real

markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h'])
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff7f0e', '#8c564b'])
def compute_fps(timestamps):
    """Return (fps_auto, dt_mean) computed from timestamps array (in seconds)."""
    ts = np.array(timestamps, dtype=float)
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Impossible de calculer dt à partir des timestamps fournis.")
    dt_mean = float(np.mean(dt))
    fps_auto = 1.0 / dt_mean
    return fps_auto, dt_mean

def read_experiment_parameters(param_file):
    """Lit parametres.txt et extrait les paramètres utiles (Qinj, Qmean, billes, sable, diamètre)."""

    # --- Lecture robuste ---
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            with open(param_file, "r", encoding=enc) as f:
                raw = f.read()
            break
        except UnicodeDecodeError:
            continue

    params = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            params[k.strip()] = v.strip()

    # --- Extraction D’INTÉRÊT ---

    # Qinj = flow injection (ml/min)
    Qinj_ml_min = float(params.get("flow injection (ml/min) pdt continu", 0))

    # Qmean = mean flowrate (g/min)
    Qmean_g_min = float(params.get("mean flowrate (g/min)", 0))

    # Medium type : "sable 100microns/billes6mm/colonnediam27mm"
    medium = params.get("medium type", "").lower()

    # sable
    if "100micron" in medium or "100microns" in medium:
        d_sable = 100e-6
    else:
        d_sable = 0  # fallback

    # billes
    import re
    m = re.search(r"cm", medium)
    if m:
        d_billes = 0.01
    else:
        d_billes = 0.006  # fallback

    # colonne diamètre
    m2 = re.search(r"colonnediam(\d+)\s*mm", medium)
    if m2:
        diam_col = float(m2.group(1)) / 1000.0
    else:
        diam_col = 0.027

    return {
        "Qinj_ml_min": Qinj_ml_min,
        "Qmean_g_min": Qmean_g_min,
        "d_billes_m": d_billes,
        "d_sable_m": d_sable,
        "diametre_colonne_m": diam_col
    }

def Ta_for_bead(timestamps, weights_g, D=0.055, L=0.01, f_billes=0.6, phi_sable=0.4, tau=1.0):
    """
    Retourne aussi : vp, Q (m3/s), Q_mean_g_s (g/s)
    """

    timestamps = np.array(timestamps, dtype=float)
    weights_g = np.array(weights_g, dtype=float)

    fps_auto, dt_mean = compute_fps(timestamps)

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
        r2 = 1 - np.sum((seg_ws - fit)**2) / np.sum((seg_ws - np.mean(seg_ws))**2)
        if slope > 0 and r2 > best_score:
            best_score = r2
            best_seg = (seg_ts, seg_ws, slope)

    seg_ts, seg_ws, dMdt_g_s = best_seg

    Q_mean_g_s = dMdt_g_s  # g/s
    Q_m3_s = dMdt_g_s * 1e-6
    print("débit : "+str(Q_mean_g_s*60)+"g/min")
    print("r^2 : "+str(r2))
    A_tot = np.pi * (D / 2.0)**2
    A_pore = A_tot * (1 - f_billes)**2

    vp = Q_m3_s / A_pore
    print("vp = "+str(vp))

    Ta = (L / vp)

    return Ta, fps_auto, dt_mean, vp, Q_m3_s, Q_mean_g_s


def process_tif_sequence(folder_path,colonne,max_files=20):
    """
    Traite un dossier contenant des sous-dossiers d'images .tif
    et un fichier weight_data.csv avec timestamp et weight.

    Retourne :
        dict: {nom_du_dossier: {"time", "ratios", "mass"}}
        ainsi que (timestamps, weights, Ta)
    """
    results = {}
    csv_path = os.path.join(folder_path, "weight_data.csv")

    if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    # Lecture du CSV
    df = pd.read_csv(csv_path)
    if not {'Image_Index', 'Timestamp', 'Weight'}.issubset(df.columns):
        raise ValueError("Le CSV doit contenir les colonnes : Image_Index, timestamp, weight")

    image_indices = df["Image_Index"].values
    timestamps = df["Timestamp"].values
    weights = df["Weight"].values

    # Calcul du temps d'advection à partir du débit moyen
    if colonne=='petite':
        D=0.027
        roi_radius=None
    elif colonne=='grande':
        D=0.055
        roi_radius=None
    T_a, fps_auto, dt_mean, vp, Q_m3_s, Q_mean_g_s = Ta_for_bead(timestamps, weights, D=D)
    print(f"[INFO] Ta={T_a:.3f}s, fps={fps_auto:.3f}Hz")
    folders = sorted([f for f in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, f))])
    folder_count = 0

    for folder in folders:
        full_path = os.path.join(folder_path, folder)
        tif_files = sorted([f for f in os.listdir(full_path) if f.endswith('.tif')],
                           key=extract_number)
        if len(tif_files) < 3:
            print(f"Pas assez d’images dans {folder}, ignoré.")
            continue

    variances, means, mass, time_seq,mean2 = [], [], [], [],[]

    for tif_file in tif_files:
        num = extract_number(tif_file)
        img_path = os.path.join(full_path, tif_file)
        img = imread(img_path)

        # --- Appliquer ROI circulaire ---
        if roi_radius is not None:
            cy, cx = np.array(img.shape) // 2  # centre de l'image
            Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= roi_radius ** 2
            img = img * mask  # zéro hors du cercle

        # on récupère le timestamp correspondant à l'Image_Index
        mask = image_indices == num
        if not np.any(mask):
            continue
        img = img
        t = timestamps[mask][0]
        time_seq.append(float(t))
        mass.append(np.nansum(img))
        variances.append(np.nanvar(img))
        means.append(np.nanmean(img))
        mean2.append(np.nanmean(img[img>10]**2))
    if len(time_seq) < 3:
        print(f"Pas assez de correspondances dans {folder}, ignoré.")


    variances = np.array(variances)
    means = np.array(means)
    ratios = variances / means
    # --- construction du temps réel pour cette série d'images ---
    time_seq = np.array(time_seq, dtype=float)
    # mettre t=0 au début de la série d'images
    time_seconds = time_seq - time_seq[0]

    # --- temps adimensionné en tenant compte du Ta calculé avec les timestamps réels ---
    time_dimless = time_seconds

    results[folder] = {
        "time": time_dimless,
        "ratios": ratios,
        "mass": means,
        "Ta": T_a,
        "fps": fps_auto,
        "mean2": np.array(mean2)
    }
    return results, (timestamps, weights, T_a)


def plot_weight(timestamps, weights, title="Débit au cours du temps"):
    """Affiche le débit (weight) en fonction du temps."""
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, weights, color='darkblue', linewidth=1.8)
    plt.scatter(timestamps, weights, color='royalblue', s=15)
    plt.xlabel("Temps (s)", fontsize="large")
    plt.ylabel("Débit (a.u.)", fontsize="large")
    plt.title(title, fontsize="x-large")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

def plot_results_semi_log(results, window=(2, 15), nb=1, auto_fit=False, fit_end_max=2, fit_bounds=None, variable="variance/mean"):
    """
    Trace en semi-log les résultats en fonction du mode choisi.
    variable : "means", "variance" ou "variance/mean"
    """
    for label, data in results.items():
        time = np.array(data["time"])
        means = np.array(data["mass"])
        Ta = data["Ta"]

        # Choix de la variable à tracer
        variances = np.array([np.var(m) for m in means]) if means.ndim > 1 else np.zeros_like(means)
        if variable == "variance/mean":
            ratios = np.array(data["ratios"])
            ratios = ratios / np.max(ratios)
            #ratios = ratios - np.mean(ratios[-10:])
            ylabel = r"$\frac{\sigma_c^2}{\mu_c}$"
        elif variable == "variance/mean2":
            ratios = np.array(data["ratios"])/np.array(means)
            ratios = ratios / np.max(ratios)

            ylabel = r"$\frac{\sigma_c^2}{\mu_c^2}$"

        elif variable == "variance":
            ratios = np.array(data["ratios"]) * np.array(data["mass"])  # approximation si besoin
            ylabel = r"$\sigma_c^2$"
        elif variable == "means":
            ratios = means
            ylabel = r"$\mu_c$"
        elif variable == "mean2":
            ratios = np.array(data["mean2"])
            ylabel = (r"$<c^2>$")

        else:
            raise ValueError(f"variable inconnue : {variable}")

        # Filtrage des valeurs positives
        mask = (time > 0) & (ratios > 0)
        time = time[mask]
        t0 = time[np.argmax(ratios)-2]
        time = time - t0
        ratios = ratios[mask]
        ax.scatter(time, ratios, s=15, label=f"{nb} ({label})", alpha=0.7)

    plt.xlabel(r"$t (s)$", fontsize="x-large")
    plt.ylabel(ylabel, fontsize="x-large")
    plt.yscale("log")
    plt.grid(True, ls="--", lw=0.5)
    plt.legend(fontsize="medium")

def plot_results_linear(results, window=(2, 15), nb=1, auto_fit=False, fit_end_max=2, fit_bounds=None, variable="variance/mean"):
    """
    Trace en semi-log les résultats en fonction du mode choisi.
    variable : "means", "variance" ou "variance/mean"
    """
    for label, data in results.items():
        time = np.array(data["time"])
        means = np.array(data["mass"])
        Ta = data["Ta"]

        # Choix de la variable à tracer
        variances = np.array([np.var(m) for m in means]) if means.ndim > 1 else np.zeros_like(means)
        if variable == "variance/mean":
            ratios = np.array(data["ratios"])/means
            ylabel = r"$\frac{\sigma_c^2}{\mu_c}$"
        elif variable == "variance":
            ratios = np.array(data["ratios"]) * np.array(data["mass"])  # approximation si besoin
            ylabel = r"$\sigma_c^2$"
        elif variable == "means":
            ratios = means
            ylabel = r"$\mu_c$"
        elif variable == "variance/mean2":
            s = 0.055/1900
            N = 2048
            c_mean= (means)
            var_c = np.array(data['ratios'])*means
            y = (8 *math.pi* var_c) / (s**2*c_mean)**2
            ratios=y

            ylabel = r"$\frac{\sigma_c^2}{\mu_c^2}$"
        else:
            raise ValueError(f"variable inconnue : {variable}")

        # Filtrage des valeurs positives
        mask = (time > 0) & (ratios > 0)
        time = time[mask]


        ratios = ratios[mask]

        ax.scatter(time, ratios, s=15, label=f"{nb} ({label})", alpha=0.7)


        # --- Fit linéaire en log-log directement ---
        log_x = (time)
        log_y = (ratios)


    plt.xlabel(r"$t/T_{a}$", fontsize="x-large")
    plt.ylabel(ylabel, fontsize="x-large")
    plt.grid(True, ls="--", lw=0.5)
    plt.legend(fontsize="medium")

def plot_results_loglog(results, window=(2, 15), nb=1, variable="variance/mean",start_index=30):
    """
    Trace les mêmes résultats mais en log-log.
    variable : "means", "variance" ou "variance/mean"
    """
    for label, data in results.items():
        time = np.array(data["time"])
        means = np.array(data["mass"])
        Ta = data["Ta"]

        # Choix de la variable à tracer
        if variable == "variance/mean":
            ratios = np.array(data["ratios"])
            ylabel = r"$\frac{\sigma_c^2 -\sigma_{0}^2 }{\mu_c}$"
        elif variable == "variance":
            ratios = np.array(data["ratios"]) * np.array(data["mass"])  # approximation
            ylabel = r"$\sigma_c^2$"
        elif variable == "means":
            ratios = means
            ylabel = r"$\mu_c$"
        elif variable == "variance/mean2":
            ratios = np.array(data["ratios"])/np.array(means)
            ylabel = r"$\frac{\sigma_c^2}{A \cdot \mu_c^2}$"
        else:
            raise ValueError(f"variable inconnue : {variable}")
        s = 0.055 / 1200
        N = 2048
        print(time.shape)
        c_mean = means[start_index]
        var_c = ratios * means
        A = 2048**2*s**2
        y = (8* math.pi * var_c*A) / (c_mean**2)
        ratios = y
        time = np.asarray(time).reshape(-1)
        ratios = np.asarray(ratios).reshape(-1)
        assert time.ndim == 1
        assert ratios.ndim == 1
        mask = (time > 0) & (ratios > 0)
        time = time[mask]
        ratios = ratios[mask]
        print(Ta)
        ax.scatter(time, ratios, s=15, label=f"{nb})", alpha=0.7)
        x0 = float(time[start_index])
        y0 = float(ratios[start_index])
        m = -1
        b = np.log10(y0) - m * np.log10(x0)
        b = float(b)
        fit_y = (10 ** b) * (time ** m)
        print("b = " + str(b))
        print("D = " + str(10 ** -b))
        plt.plot(time, fit_y, color='red', label=f"Fit pente -1")
    ax.legend(fontsize="medium")
    plt.xlabel(r"$t/T_{a}$", fontsize="x-large")
    plt.ylabel(ylabel, fontsize="x-large")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, ls="--", lw=0.5)


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

    return ax
def run(folder=False, nb=1, mode='semi-log', variable="variance/mean",colonne='grande',sindex=None):
    import sys
    if folder is False:
        root_directory = input("Entrez le chemin du dossier contenant les sous-dossiers d'images : ")
    else:
        root_directory = folder

    results, (timestamps, weights, T_a) = process_tif_sequence(root_directory,colonne)

    if not results:
        print("Aucun résultat obtenu.")
        sys.exit(0)

    if mode == "semi-log":
        plot_results_semi_log(results, nb=nb, variable=variable)
    elif mode == "loglog":
        plot_results_loglog(results, nb=nb, variable=variable,start_index=sindex)
    if mode == "linear":
        plot_results_linear(results, nb=nb, variable=variable)
    else:
        print("Mode invalide.")

    # ➤ Figure séparée du débit
    #plot_weight(timestamps, weights, title=f"Débit (T_a = {T_a:.2f} s)")
def process_dispersion_series(folder_path, show_every=10, plot=True):
    """
    Traite une série expérimentale à partir du fichier 'weight_data.csv'.
    - Détecte automatiquement le FPS à partir des timestamps.
    - Calcule le temps adimensionné t/Ta et le rayon adimensionné R/R0.
    - Compare plusieurs expériences sur la même échelle physique.
    """

    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Lecture du fichier ---
    data_path = os.path.join(folder_path, "weight_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier 'weight_data.csv' introuvable dans {folder_path}")

    # Essai automatique de lecture avec encodage flexible
    for enc in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Impossible de lire le fichier weight_data.csv avec les encodages standard")

    # Vérification des colonnes
    expected_cols = {"Image_Index", "Timestamp", "Weight"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Colonnes attendues : {expected_cols}, trouvé : {set(df.columns)}")

    # --- Extraction des données ---
    timestamps = df["Timestamp"].to_numpy().astype(float)
    radii = df["Weight"].to_numpy()  # ou conversion en rayon si nécessaire

    # --- Nettoyage du temps relatif ---
    timestamps_rel = timestamps - timestamps[0]
    fps_auto, dt_mean = compute_fps(timestamps)  # ou récupère via Ta_for_bead
    print(f"📊 FPS calculé : {fps_auto:.3f} (Δt moyen = {dt_mean:.3f} s)")

    # --- Ici on calcule Ta à partir du CSV (timestamps complets) et récupère fps aussi ---
    Ta, fps_auto, dt_mean = Ta_for_bead(timestamps, radii)  # si 'radii' représente poids (g)

    time_seconds = timestamps_rel  # temps réel relatif
    time_dimless = time_seconds / Ta
    # --- Calcul du rayon adimensionné ---
    R0 = radii[0]
    radii_dimless = radii / R0

    # --- Tracé optionnel ---
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(time_dimless, radii_dimless, "o-", label=os.path.basename(folder_path))
        plt.xlabel("Temps adimensionné t/Tₐ")
        plt.ylabel("Rayon adimensionné R/R₀")
        plt.title("Évolution du rayon adimensionné")
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Retour des résultats ---
    return {
        "time": time_dimless,
        "radius": radii_dimless,
        "Ta": Ta,
        "fps": fps_auto,
        "timestamps": time_seconds,
        "dt_mean": dt_mean
    }
def forced_minus_one_fit(time, ratios, start_index):
    """
    Fit forcé à pente = -1 en log-log,
    sur time[start_index:], ratios[start_index:].

    Retourne :
        A (ordonnée à l'origine dans log-log)
        C (préfacteur : y = C * t^{-1})
    """
    import numpy as np

    t = np.array(time[start_index:], float)
    r = np.array(ratios[start_index:], float)

    mask = (t > 0) & (r > 0)
    t = t[mask]
    r = r[mask]

    logt = np.log(t)
    logr = np.log(r)

    # Fit forcé slope = -1 :
    # logr = A - logt  =>  A = mean( logr + logt )
    A = np.mean(logr + logt)

    # Préfacteur C dans r = C * t^{-1}
    C = np.exp(A)

    return A, C

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
def run_on_folder(parent_folder, mode="semi-log", variable="variance/mean",
                  colonne="grande", sindex=None):
    """
    Exécute run() pour chaque sous-dossier contenant un weight_data.csv.
    Corrige le problème 'ax is not defined' en créant fig, ax avant run().
    Sauvegarde la figure générée.
    """

    import matplotlib.pyplot as plt
    import os

    subfolders = sorted([
        os.path.join(parent_folder, f)
        for f in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, f))
    ])

    print(f"📁 {len(subfolders)} dossiers trouvés.\n")

    for exp_folder in subfolders:

        csv_path = os.path.join(exp_folder, "weight_data.csv")
        if not os.path.exists(csv_path):
            continue

        print(f"\ntraitement de {exp_folder}")

        out_dir = os.path.join(exp_folder, "OUTPUT_PLOTS")
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots()

        globals()["ax"] = ax

        try:
            run(folder=exp_folder, nb=1, mode=mode,
                variable=variable, colonne=colonne, sindex=sindex)
        except Exception as e:
            plt.close(fig)
            continue

        # Sauvegarde
        filename = f"plot_{exp_folder[-6:]}.png"
        save_path = (parent_folder+"/"+filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"   figure sauvegardée dans {save_path}")

    print("\n terminé ! \n")

if __name__ == "__main__":
    fig, ax = plt.subplots()

    #run_on_folder("/home/chorus/EXP_TO_TREAT/,mode="linear",variable="variance/mean2",colonne="grande")
    #run('/home/chorus/EXP_TO_TREAT/exp_22_11_2/', "heterogene", mode="loglog",variable="variance/mean2",colonne='petite',sindex=24)
    #run('/home/chorus/EXP_TO_TREAT/16_10_2', "valve manuelle", mode="loglog",variable="variance/mean2",colonne='grande',sindex=20)

    # run('/home/chorus/EXP_TO_TREAT/15_10_4/', "heterogene", mode="loglog",variable="variance/mean2",colonne='grande',sindex=None)
    #run('/home/chorus/exp_17_11_2/','système de valve automatique',mode='loglog',variable="variance/mean2",colonne='grande',sindex=20)
    #run('/home/chorus/exp_17_12_2/','billes 2cm n° 2',mode='loglog',variable="variance/mean2",colonne='grande',sindex=50)
    #run('/home/chorus/exp_17_12_3/','billes 2cm n° 3',mode='loglog',variable="variance/mean2",colonne='grande',sindex=50)
    run('/home/chorus/exp_3_11_3/','homogene',mode='loglog',variable="variance/mean2",colonne='grande',sindex=50)
    #run('/home/chorus/exp_adrien_nv_5/','adrien 5',mode='semi-log',variable="variance/mean2",colonne='grande',sindex=25)
    # run('/home/chorus/exp_adrien_nv_4/','adrien 4',mode='semi-log',variable="variance/mean2",colonne='grande',sindex=25)
    #
    # #plt.ylim(top=2E7)
    #plt.ylim(bottom=0.1*1e-11,top=0.4*1e-11)
    #run('/home/chorus/exp_22_11_2/', "bille0.6 grosse 1", mode="loglog",variable="variance/mean",colonne='grande')
    # run('/home/chorus/exp_22_11_2/', "bille0.6 grosse 2", mode="semi-log",variable="variance/mean",colonne='grande')
    # run('/home/chorus/exp_22_11_3/', "bille0.6 grosse 3", mode="semi-log",variable="variance/mean",colonne='grande')
    # run('/home/chorus/exp_22_11_4/', "bille0.6 grosse 4", mode="semi-log",variable="variance/mean",colonne='grande')

    #run('/home/chorus/exp_22_11_5/', "bille0.6 grosse 5", mode="semi-log",variable="variance/mean",colonne='grande')


    #run('/home/chorus/exp_septoct/push_only/push_only_3_22_2/', "push only4", mode="linear",variable="means",colonne='grande')
    #run('/home/chorus/exp_septoct/15_10_5/', "15_10_5", mode="linear",variable="means")
    #run('/home/chorus/exp_septoct/15_10_6/', "15_10_6", mode="linear",variable="means")


    #run('/home/chorus/exp_septoct/15_10_4/', "15_10_4", mode="loglog",variable="variance/mean2")
    #run('/home/chorus/exp_septoct/15_10_2/', "15_10_2", mode="loglog",variable="variance/mean2")
    #run('/home/chorus/exp_septoct/15_10_3/', "15_10_3", mode="loglog",variable="variance/mean2")
     #run('/home/chorus/EXP_TO_TREAT/16_10_2/', "16_10_2", mode="semi-log",variable="variance/mean2",colonne='grande',sindex=25)
    # run('/home/chorus/EXP_TO_TREAT/16_10_3/', "16_10_3", mode="semi-log",variable="variance/mean2",colonne='grande',sindex=25)
    # run('/home/chorus/exp_17_11_1/', "17_11_1", mode="semi-log",variable="means",colonne='grande')
    # plt.show()
     #run('/home/chorus/EXP_TO_TREAT/exp_22_11_4/', "22_11_1", mode="semi-log",variable="variance/mean2",colonne='grande',sindex=None )
    #run('/home/chorus/exp_septoct/exp_1410/', "brut" , mode="semi-log")

    #run('/home/chorus/expjuillet/ALL_EXP/11_07_2/',2,mode="semi-log")

    #plot_minus_one_slope(ax,-1.6,20,2,length_decades=0.7)
    #
    # # run('/home/chorus/exp_septoct/16_10_2/', "16_10_2", mode="semi-log")
    #run('/home/chorus/exp_petite_colonne3/', "hetero bille 0.6", mode="loglog",variable="variance/mean",colonne='petite')
     #run('/home/chorus/EXP_TO_TREAT/exp_petite_colonne3/', "petite colonne 0.6 n2", mode="semi-log",variable="variance/mean2",colonne='petite')
    #run('/home/chorus/exp_petite_colonne5/', "hetero0.6 n3", mode="loglog",variable="variance/mean",colonne='petite')
    #run('/home/chorus/exp_petite_colonne7/', "hetero0.6 n4, s0 plus petit", mode="loglog",variable="variance/mean",colonne='petite')
    #run('/home/chorus/exp_petite_colonne8/', "hetero0.6 n5", mode="loglog",variable="variance/mean",colonne='petite')
    #analyze_and_log_dispersion("/home/chorus/exp_petite_colonne3/",exp_name="petit colonne3")
    #analyze_and_log_dispersion("/home/chorus/exp_petite_colonne2/",exp_name="petit colonne2")
    #analyze_and_log_dispersion("/home/chorus/exp_petite_colonne5/",exp_name="petit colonne5")
    #analyze_and_log_dispersion("/home/chorus/exp_petite_colonne7/",exp_name="petit colonne7")

    #run('/home/chorus/exp_septoct/exp_3_11_2/', "homo3_2", mode="loglog",variable="variance/mean")
    #run('/home/chorus/exp_septoct/exp_3_11_3/', "homo3_3", mode="loglog",variable="variance/mean")
    #run('/home/chorus/exp_septoct/exp_3_11_4/', "homo3_4", mode="loglog",variable="variance/mean")
    # run('/home/chorus/exp_septoct/16_10_3/', "16_10_3", mode="semi-log"
    # ax.get_legend().remove()
    #plot_semi_log_slope(ax,start_x=2,start_y=14,slope=-0.6,length=3,color="black")

    #plot_semi_log_slope(ax,start_x=5,start_y=0.3,slope=-0.3,length=5,color="black")
#plot_semi_log_slope(ax,start_x=40,start_y=1.3,slope=-0.017,length=30,color="black")
    #plot_semi_log_slope(ax,start_x=40,start_y=0.3,slope=-0.012,length=30,color="blue")

    # #


    plt.tight_layout()
    plt.show()
    # fig,ax = plt.subplots()
   # run('/home/chorus/exp_septoct/exp_homo_28_10/', "homo1", mode="loglog",variable="variance/mean")
    #run('/home/chorus/exp_septoct/exp_homo_28_10_2/', "homo2", mode="loglog",variable="variance/mean")
    #plot_minus_one_slope(ax,-1,6,4*10**-1,0.5)
    #plt.xlim(right=25)
    #plt.ylim(bottom=0.01)
