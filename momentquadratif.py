import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import re

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else -1


def local_intensity_quadratic(img, epsilon=0):
    """
    Calcule l'intensité quadratique moyenne dans les zones
    où l'image est supérieure à epsilon.

    ⟨c² | c > ε⟩ = (1/N) * Σ c²  (sur les pixels où c > ε)
    """
    mask = img > epsilon
    if not np.any(mask):
        return 0.0

    values = img[mask].astype(np.float64)
    return np.mean(values ** 2)


def process_tif_sequence(folder_path, epsilon=100, max_files=20):
    """
    Traite un dossier contenant une séquence d'images .tif (time-lapse).
    Remplace la variance scalaire par le moment quadratique spatial.
    """
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
                moments.append(local_intensity_quadratic(img, epsilon=epsilon))
                masses.append(np.sum(img[img > epsilon]))

            moments = np.array(moments)
            masses = np.array(masses)

            results[folder] = {
                "time": np.arange(len(moments))*3/11,
                "moments": moments,
                "mass": masses
            }

            folder_count += 1
            if folder_count >= max_files:
                break

        except Exception as e:
            print(f"Erreur dans {folder}: {e}")

    return results

def scale_to_uint16_65500(arr, per_frame=False):
    """
    Convertit un tableau numpy (float64 ou autre) en uint16 sur l'intervalle [0, 65500].
    - per_frame=False : scaling global (min->0, max->65500) sur tout l'array 3D.
    - per_frame=True  : scaling séparé pour chaque frame (préserve contraste intra-frame).
    """
    MAX_VAL = 65500
    # si déjà entier, on clip simplement
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, MAX_VAL).astype(np.uint16)

    if per_frame:
        out = np.empty(arr.shape, dtype=np.uint16)
        for i in range(arr.shape[0]):
            frame = arr[i]
            # gérer NaN éventuels
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


def process_npy_sequence(npy_path, epsilon=100, per_frame_scale=False):
    """
    Traite un fichier .npy contenant une séquence d'images (shape attendu : [Nframes, X, Y]).
    - Convertit en uint16 sur [0,65500] pour que epsilon (entier) fonctionne correctement.
    - Calcule le moment quadratique spatial (local_intensity_quadratic) et la masse (> epsilon).
    Retourne un dict compatible avec plot_results_semi_log.
    """
    results = {}
    try:
        data = np.load(npy_path)  # shape attendu : (Nframes, Nx, Ny)
        if data.ndim != 3:
            raise ValueError(f"Le fichier {npy_path} n’a pas la bonne forme (attendu: 3D frames,x,y).")

        # s'assurer que epsilon est un entier dans [0,65500]
        epsilon = int(epsilon)
        epsilon = max(0, min(65500, epsilon))

        # conversion en uint16 (0..65500)
        data_uint16 = scale_to_uint16_65500(data, per_frame=per_frame_scale)

        moments = []
        masses = []
        for frame in data_uint16:
            moments.append(local_intensity_quadratic(frame, epsilon=epsilon))
            masses.append(np.sum(frame[frame > epsilon]))

        moments = np.array(moments)
        masses = np.array(masses)

        results[os.path.basename(npy_path)] = {
            "time": np.arange(len(moments)) * 3 / 11,
            "moments": moments,
            "mass": masses
        }

    except Exception as e:
        print(f"Erreur en lisant {npy_path}: {e}")

    return results
def plot_results_semi_log(results, window):
    """
    Affiche les résultats en semi-log avec un fit exponentiel
    du moment quadratique spatial.
    """
    plt.figure(figsize=(10, 6))

    for label, data in results.items():
        time = np.array(data["time"])
        moments = np.array(data["moments"])
        mass = np.array(data["mass"])

        mask = (time > 0) & (moments > 0)
        time = time[mask]
        moments = moments[mask]
        mass = mass[mask]

        plt.plot(time, moments, label=f"{label} (moment²)")
        if True:
            max_index = int(window[0]*11/3)
        end_index = min(int(window[1]*11/3), len(moments))

        fit_time = time[max_index:end_index]
        fit_moments = moments[max_index:end_index]

        if len(fit_time) < 2:
            continue

        # fit exponentiel : ln(M2) = slope * t + intercept
        log_fit_moments = np.log(fit_moments)
        (slope, intercept), cov = np.polyfit(fit_time, log_fit_moments, 1, cov=True)

        slope_err = np.sqrt(cov[0, 0])
        intercept_err = np.sqrt(cov[1, 1])

        # droite de fit
        fit_line = np.exp(intercept) * np.exp(slope * fit_time)

        plt.plot(
            fit_time,
            fit_line,
            '--',
            label=f"{label} (slope={slope:.4f} ± {slope_err:.4f})"
        )
        # optionnel : tracer la masse aussi
        #plt.plot(time, mass, label=f"{label} (masse)")
    plt.xlabel("Temps (frames)")
    plt.ylabel("Moment quadratique spatial")
    plt.xscale("linear")
    plt.yscale("log")
    plt.title("Moment quadratique spatial en semi-log")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import sys
    path = input("Entrez le chemin d’un dossier (.tif) ou d’un fichier .npy : ")
    epsilon = int(input("Valeur du seuil epsilon (16 bits) : "))

    if os.path.isdir(path):
        results = process_tif_sequence(path, epsilon=epsilon)
    elif path.endswith(".npy"):
        results = process_npy_sequence(path, epsilon=epsilon)
    else:
        print("Chemin invalide : doit être un dossier .tif ou un fichier .npy")
        sys.exit(0)

    if not results:
        print("Aucun résultat obtenu.")
        sys.exit(0)

    plot_results_semi_log(results, (2, 15))
