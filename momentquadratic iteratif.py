import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import re
import itertools

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else -1

def local_intensity_quadratic(img, n, epsilon=0):
    """
    Calcule le moment d'ordre n dans les zones
    où l'image est supérieure à epsilon :
    ⟨c^n | c > ε⟩ = (1/N) * Σ c^n
    """
    mask = img > epsilon
    if not np.any(mask):
        return 0.0
    values = img[mask].astype(np.float64)
    return np.mean(values ** n)

# -------- Conversion float64 -> uint16 [0,65500] --------
def scale_to_uint16_65500(arr, per_frame=False):
    MAX_VAL = 65500
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, MAX_VAL).astype(np.uint16)

    if per_frame:
        out = np.empty(arr.shape, dtype=np.uint16)
        for i in range(arr.shape[0]):
            frame = arr[i]
            mn = np.nanmin(frame)
            mx = np.nanmax(frame)
            if np.isnan(mn) or np.isnan(mx) or mn == mx:
                out[i] = np.zeros_like(frame, dtype=np.uint16)
            else:
                scaled = (frame - mn) / (mx - mn) * MAX_VAL
                out[i] = np.clip(np.rint(scaled), 0, MAX_VAL).astype(np.uint16)
        return out
    else:
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if np.isnan(mn) or np.isnan(mx) or mn == mx:
            return np.zeros_like(arr, dtype=np.uint16)
        scaled = (arr - mn) / (mx - mn) * MAX_VAL
        return np.clip(np.rint(scaled), 0, MAX_VAL).astype(np.uint16)

# -------- Traitement des .tif --------
def process_tif_sequence(folder_path, n, epsilon=100, max_files=20):
    results = {}
    folders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
    folder_count = 0

    for folder in folders:
        full_path = os.path.join(folder_path, folder)
        tif_files = sorted(
            [f for f in os.listdir(full_path) if f.endswith('.tif')],
            key=extract_number
        )
        if len(tif_files) < 3:
            print(f"Pas assez d’images dans {folder}, ignoré.")
            continue

        try:
            moments = []
            masses = []
            for tif_file in tif_files:
                image_path = os.path.join(full_path, tif_file)
                img = imread(image_path)
                moments.append(local_intensity_quadratic(img, n, epsilon=epsilon))
                masses.append(np.sum(img[img > epsilon]))

            moments = np.array(moments)
            masses = np.array(masses)

            results[folder] = {
                "time": np.arange(len(moments)) * 3 / 11,
                "moments": moments,
                "mass": masses,
                "n": n
            }

            folder_count += 1
            if folder_count >= max_files:
                break

        except Exception as e:
            print(f"Erreur dans {folder}: {e}")

    return results

# -------- Traitement d’un .npy --------
def process_npy_sequence(npy_path, n, epsilon=100, per_frame_scale=False):
    results = {}
    try:
        data = np.load(npy_path)
        if data.ndim != 3:
            raise ValueError(f"{npy_path} doit avoir shape (frames, X, Y)")

        epsilon = int(epsilon)
        epsilon = max(0, min(65500, epsilon))

        data_uint16 = scale_to_uint16_65500(data, per_frame=per_frame_scale)

        moments = []
        masses = []
        for frame in data_uint16:
            moments.append(local_intensity_quadratic(frame, n, epsilon=epsilon))
            masses.append(np.sum(frame[frame > epsilon]))

        moments = np.array(moments)
        masses = np.array(masses)

        results[os.path.basename(npy_path)] = {
            "time": np.arange(len(moments)) * 3 / 11,
            "moments": moments,
            "mass": masses,
            "n": n
        }

    except Exception as e:
        print(f"Erreur en lisant {npy_path}: {e}")

    return results
def plot_results_semi_log(results, window):
    slope, slope_err = np.nan, np.nan  # valeurs par défaut

    for label, data in results.items():
        time = np.array(data["time"])
        moments = np.array(data["moments"])
        mass = np.array(data["mass"])
        n = np.array(data["n"])

        mask = (time > 0) & (moments > 0)
        time = time[mask]
        moments = moments[mask]
        mass = mass[mask]
        window1, window2 = window
        max_index = int(window1 * 11 / 3)
        end_index = min(int(window2 * 11 / 3), len(moments))

        fit_time = time[max_index:end_index]
        fit_moments = moments[max_index:end_index]

        if len(fit_time) < 2:
            print(f"⚠️ Pas assez de points pour le fit dans {label}")
            continue

        log_fit_moments = np.log(fit_moments)
        (slope, intercept), cov = np.polyfit(fit_time, log_fit_moments, 1, cov=True)

        slope_err = np.sqrt(cov[0, 0])
        intercept_err = np.sqrt(cov[1, 1])

        fit_line = np.exp(intercept) * np.exp(slope * fit_time)

    return slope, slope_err


# -------- Fonctions Gamma --------
def gamman(n, window=(100,200)):
    root_directory = input("chemin du dossier contenant les sous-dossiers d'images .tif : ")
    epsilon = int(input("Valeur du seuil epsilon (16 bits) : "))
    slopes=[]
    Ns=[]
    slope_errs=[]
    for i in range(1,n+1):
        results = process_tif_sequence(root_directory, i, epsilon=epsilon)
        slope, slope_err = plot_results_semi_log(results, window)
        slopes.append(-1*slope)
        slope_errs.append(slope_err)
        Ns.append(i)
    plt.errorbar(Ns, slopes, yerr=slope_errs, fmt='o-', capsize=5)
    plt.title(root_directory)
    plt.xlabel("n")
    plt.ylabel("Gamma_n")
    plt.show()

    plt.plot([i for i in range(1,n+1)], [i/2 for i in range(1,n+1)], color='k')
    slopes = np.array(slopes)/slopes[1]
    plt.errorbar(Ns, slopes, yerr=slope_errs, fmt='o-', capsize=5)
    plt.title(root_directory)
    plt.xlabel("n")
    plt.ylabel("Gamma_n/gamma_2")
    plt.show()

def gamma_eps(eps=[1000,2500,3800], window=(23,57)):
    root_directory = input("chemin du dossier contenant les sous-dossiers d'images .tif : ")
    slopes = []
    slope_errs = []
    for i in eps:
        results = process_tif_sequence(root_directory, 2, epsilon=i)
        slope, slope_err = plot_results_semi_log(results, window)
        slopes.append(-1 * slope)
        slope_errs.append(slope_err)
    plt.errorbar(eps, slopes, yerr=slope_errs, fmt='o-', capsize=5)
    plt.title(root_directory)
    plt.xlabel("eps")
    plt.ylabel("Gamma_n")

def tafix_epsmax(n, root_directory, window=(3,30), absolu=False):
    eps = [3000]
    Ns=[]
    slopes = []
    slope_errs = []
    slopeseps = []

    for i in eps:
        results = process_tif_sequence(root_directory, 2, epsilon=i)
        slopeeps, slopeerrseps = plot_results_semi_log(results, window)
        slopeseps.append(-1 * slopeeps)
    eps_max = eps[np.argmax(slopeseps)]
    print("eps max pour exp " + str(root_directory) + ':   ' + str(eps_max))
    for j in range(1,n+1):
        results = process_tif_sequence(root_directory, j, epsilon=eps_max)
        slope, slope_err = plot_results_semi_log(results, window)
        slopes.append(-1 * slope)
        slope_errs.append(slope_err)
        Ns.append(j)
    if not absolu:
        slopes = np.array(slopes)/slopes[1]
    return Ns, slopes, slope_errs, root_directory

# -------- Main --------
if __name__ == '__main__':
    root_directory = input("all_exp type link")  # dossier principal
    n = 8
    absolu = False
    window = (3, 15)
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h'])
    colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff7f0e', '#8c564b'])

    plt.figure(figsize=(8, 6))
    for exp_folder in sorted(os.listdir(root_directory)):
        full_path = os.path.join(root_directory, exp_folder)
        print(f"\n=== Traitement de {exp_folder} ===")
        results = tafix_epsmax(n, full_path, window=window, absolu=absolu)
        Ns, slopes, slope_errs, rt = results
        marker = next(markers)
        color = next(colors)
        plt.grid(True)
        plt.errorbar(
            Ns, slopes, yerr=slope_errs,
            marker=marker, capsize=5,
            linestyle='None', linewidth=1.5,
            label=exp_folder, color=color)

        plt.xlabel("n", fontsize='x-large', fontweight='bold')
        if not absolu:
            plt.ylabel(r"$\frac{\alpha_{n}}{\alpha_{2}}$, $\epsilon_{max}$",
                       fontsize='x-large', fontweight='bold')
            plt.plot([i for i in range(1, n + 1)],
                     [i / 2 for i in range(1, n + 1)],
                     color='k', linestyle="--", linewidth=1,
                     label='auto-similar solution')
        else:
            plt.ylabel(r"$\alpha_{n}$, $\epsilon_{max}$",
                       fontsize='xx-large', fontweight='bold')
        plt.legend()
    plt.show()
