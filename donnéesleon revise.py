import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import re

def calculta(file):
    pattern = r"debit\s*(\d+)"
    matches = re.findall(pattern, file, re.IGNORECASE)
    debit = [int(match) for match in matches]
    if debit==[] :
        flux=((0.5/60)*(10**(-3)))
    else :
        flux = float((debit[0])/600)*(10**-3)
        print('debit ===='+str(debit[0]))
        if len(str(debit[0]))==3:
            flux=flux/10
    if "fin" in file:
        ta=0.35*(0.057/2)**2*np.pi*0.5/flux
        color='b'
    elif "inclu" in file:
        ta = 0.35 * (0.057 / 2) ** 2 * np.pi * 0.5/ flux
        color='r'

    else :
        ta = 0.35 * (0.057 / 2) ** 2 * np.pi * 0.5/ flux
        color='g'
    print(ta)
    return ta,color

def process_ome_tif_files(root_dir, max_files=20):
    """
    Parcourt les dossiers pour traiter les fichiers .ome.tif et extrait les informations nécessaires.

    Args:
        root_dir (str): Le chemin racine contenant les dossiers et fichiers .ome.tif.
        max_files (int): Nombre maximum de fichiers à traiter.

    Returns:
        dict: Un dictionnaire contenant les résultats (temps, ratio variance/moyenne) pour chaque fichier.
    """
    results = {}
    file_count = 0

    # Parcourir tous les sous-dossiers et fichiers
    for subdir, _, files in os.walk(root_dir):

        for file in files:
            if file.endswith('.ome.tif'):
                file_path = os.path.join(subdir, file)

                # Lire le fichier .ome.tif
                try:
                    data = imread(file_path)  # Chargement du fichier

                    # Vérification des dimensions (supposons que l'axe du temps soit le premier)
                    if data.ndim < 3:
                        print(f"Le fichier {file_path} ne contient pas plusieurs trames.")
                        continue

                    # Calcul des statistiques (variance / moyenne) pour chaque trame
                    variances = []
                    means = []
                    for frame in data:
                        variances.append(np.var(frame))
                        means.append(np.mean(frame))

                    variances = np.array(variances)
                    means = np.array(means)
                    ratios = variances / means

                    # Sauvegarder les résultats
                    results[file] = {
                        "time": np.arange(len(ratios)),
                        "ratios": ratios
                    }

                    # Incrémenter le compteur de fichiers
                    file_count += 1
                    if file_count >= max_files:
                        break
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier {file_path}: {e}")

        if file_count >= max_files:
            break

    return results

def plot_results_loglog(results):
    """
    Trace les résultats pour tous les fichiers et ajoute un fit en log-log
    entre le maximum de la courbe et les 60 prochaines valeurs.

    Args:
        results (dict): Dictionnaire contenant les temps et ratios pour chaque fichier.
    """
    plt.figure(figsize=(10, 6))

    for file, data in results.items():
        ta=calculta(file)[0]
        c=calculta(file)[1]
        time = np.array(data["time"])/ta
        ratios = np.array(data["ratios"])

        # Filtrer les valeurs positives pour éviter les erreurs avec le log
        mask = (time > 0) & (ratios > 0)
        time = time[mask]
        ratios = ratios[mask]

        # Tracer les données originales
        #plt.plot(time, ratios, label=f"Data: {''.join([char for char in file if char.isdigit()])}",color=c)
        plt.plot(time, ratios, color=c)

        # Localiser l'indice du maximum
        max_index = np.argmax(ratios)

        # Extraire le sous-ensemble pour le fit
        end_index = min(max_index +60, len(ratios))

        fit_time = time[max_index:end_index]
        fit_ratios = ratios[max_index:end_index]

        # Transformation semi-log et ajustement
        log_fit_time = np.log(fit_time)
        log_fit_ratios = np.log(fit_ratios)
        coeffs = np.polyfit(log_fit_time, log_fit_ratios, 1)  # Ajustement linéaire
        slope, intercept = coeffs

        # Générer les valeurs de la ligne de fit
        fit_line = np.exp(intercept) * fit_time **slope
        plt.plot(fit_time, fit_line, linestyle="--", label=f"Fit: {''.join([char for char in file if char.isdigit()])} (slope={slope:.2f})")

    plt.xlabel("Temps adimensionné)")
    plt.ylabel("Variance / Moyenne")
    plt.yscale("log")
    plt.xscale("log")  # Mettre l'échelle x en log pour correspondre au fit
    plt.title("Ratio Variance / Moyenne en fonction du temps adimmensionné")
    plt.legend(fontsize=7)

    plt.grid()
    plt.show()

def plot_results(results):
    """
    Trace les résultats pour tous les fichiers et ajoute un fit en log-log
    entre le maximum de la courbe et les 60 prochaines valeurs.

    Args:
        results (dict): Dictionnaire contenant les temps et ratios pour chaque fichier.
    """
    plt.figure(figsize=(10, 6))

    for file, data in results.items():
        time = np.array(data["time"])
        ratios = np.array(data["ratios"])

        # Filtrer les valeurs positives pour éviter les erreurs avec le log
        mask = (time > 0) & (ratios > 0)
        time = time[mask]
        ratios = ratios[mask]

        # Tracer les données originales
        plt.plot(time, ratios, label=f"Data: {''.join([char for char in file if char.isdigit()])}")

        # Localiser l'indice du maximum
        max_index = np.argmax(ratios)

        # Extraire le sous-ensemble pour le fit
        end_index = min(max_index + 70, len(ratios))
        fit_time = time[max_index:end_index]
        fit_ratios = ratios[max_index:end_index]

        # Transformation semi-log et ajustement
        log_fit_time = fit_time
        log_fit_ratios = np.log(fit_ratios)
        coeffs = np.polyfit(fit_time, log_fit_ratios, 1)  # Ajustement linéaire
        slope, intercept = coeffs

        # Générer les valeurs de la ligne de fit
        fit_line = np.exp(intercept) * np.exp(fit_time *slope)
        plt.plot(fit_time, fit_line, linestyle="--", label=f"Fit: {''.join([char for char in file if char.isdigit()])} (slope={slope:.2f})")

    plt.xlabel("Temps (frames)")
    plt.ylabel("Variance / Moyenne")
    plt.xscale("log")  # Mettre l'échelle x en log pour correspondre au fit
    plt.title("Ratio Variance / Moyenne en fonction du Temps")
    plt.legend()

    plt.grid()
    plt.show()
def plot_results_loglog_rgb(results):
    """
    Trace les résultats pour tous les fichiers et ajoute un fit en log-log
    entre le maximum de la courbe et les 60 prochaines valeurs.

    Args:
        results (dict): Dictionnaire contenant les temps et ratios pour chaque fichier.
    """
    plt.figure(figsize=(10, 6))

    # Dictionnaire pour associer couleurs et labels
    color_labels = {
        'b': 'Sable fin + bille',
        'r': 'Inclusions de sable',
        'g': 'Homogène'
    }
    color_used = set()  # Pour éviter les doublons dans la légende

    for file, data in results.items():
        ta = calculta(file)[0]
        c = calculta(file)[1]
        time = np.array(data["time"])/ta
        ratios = np.array(data["ratios"])

        # Filtrer les valeurs positives pour éviter les erreurs avec le log
        mask = (time > 0) & (ratios > 0)
        time = time[mask]
        ratios = ratios[mask]

        # Tracer les données originales
        plt.plot(time, ratios, color=c)

        # Localiser l'indice du maximum
        max_index = np.argmax(ratios)

        # Extraire le sous-ensemble pour le fit
        end_index = min(max_index + 60, len(ratios))

        fit_time = time[max_index:end_index]
        fit_ratios = ratios[max_index:end_index]

        # Transformation semi-log et ajustement
        log_fit_time = np.log(fit_time)
        log_fit_ratios = np.log(fit_ratios)
        coeffs = np.polyfit(log_fit_time, log_fit_ratios, 1)  # Ajustement linéaire
        slope, intercept = coeffs

        # Générer les valeurs de la ligne de fit
        fit_line = np.exp(intercept) * fit_time ** slope
        plt.plot(fit_time, fit_line, linestyle="--")

        # Ajouter la couleur utilisée à l'ensemble
        color_used.add(c)

    # Ajouter une légende séparée avec les labels correspondant aux couleurs
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=label)
        for color, label in color_labels.items() if color in color_used
    ]
    plt.legend(handles=handles, title="Types de milieux")

    plt.xlabel("Temps adimensionné")
    plt.ylabel("Variance / Moyenne")
    plt.yscale("log")
    plt.xscale("log")  # Mettre l'échelle x en log pour correspondre au fit
    plt.title("Ratio Variance / Moyenne en fonction du temps adimensionné")
    plt.grid()
    plt.show()
from collections import defaultdict

def compute_mean_curve(curves):
    """Calcule la courbe moyenne à partir d'une liste de (time, ratio)."""
    min_len = min(len(r[0]) for r in curves)
    times = np.array([r[0][:min_len] for r in curves])
    ratios = np.array([r[1][:min_len] for r in curves])
    mean_time = np.mean(times, axis=0)
    mean_ratio = np.mean(ratios, axis=0)
    return mean_time, mean_ratio
def process_folders_and_plot_avg(root_dir):
    """
    Traite les .ome.tif dans chaque sous-dossier et trace une courbe moyenne par sous-dossier.
    """
    subfolder_curves = defaultdict(list)

    for subfolder in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(full_path):
            results = process_ome_tif_files(full_path, max_files=1000)  # ou plus si besoin
            for file, data in results.items():
                ta, color = calculta(file)
                time = np.array(data["time"]) / ta
                ratios = np.array(data["ratios"])
                mask = (time > 0) & (ratios > 0)
                subfolder_curves[subfolder].append((time[mask], ratios[mask]))

    # Tracer les courbes moyennes
    plt.figure(figsize=(10, 6))
    for folder, curves in subfolder_curves.items():
        if not curves:
            continue
        mean_time, mean_ratios = compute_mean_curve(curves)

        plt.plot(mean_time, mean_ratios, label=folder)

        # Fit log-log sur max -> max+60
        max_idx = np.argmax(mean_ratios)
        fit_end = min(max_idx + 110, len(mean_ratios))
        fit_time = mean_time[max_idx:fit_end]
        fit_ratios = mean_ratios[max_idx:fit_end]
        log_fit_time = np.log(fit_time)
        log_fit_ratios = np.log(fit_ratios)
        coeffs = np.polyfit(log_fit_time, log_fit_ratios, 1)
        slope, intercept = coeffs
        fit_line = np.exp(intercept) * fit_time**slope
        plt.plot(fit_time, fit_line, linestyle="--", label=f"{folder} fit (slope={slope:.2f})")

    plt.xlabel("Temps adimensionné")
    plt.ylabel("Variance / Moyenne")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Courbes moyennes par sous-dossier")
    plt.legend()
    plt.grid()
    plt.show()
    return subfolder_curves
if __name__ == "__main__":
    root_directory = input("Entrez le chemin du dossier racine contenant les sous-dossiers: ")
    A=process_folders_and_plot_avg(root_directory)
