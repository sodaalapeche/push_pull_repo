import os
import numpy as np
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width
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
        end_index = min(max_index + 50, len(ratios))
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
    i=0
    j=0
    for file, data in results.items():
        ta = calculta(file)[0]
        c = calculta(file)[1]
        time = np.array(data["time"]) / ta
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
        coeffs, cov = np.polyfit(log_fit_time, log_fit_ratios, 1, cov=True)
        slope, intercept = coeffs
        slope_error = np.sqrt(cov[0, 0])

        # Generate fit line
        fit_line = np.exp(intercept) * fit_time**slope
        if "10" in ''.join([char for char in file if char.isdigit()]) :
            label = f"{"Heterogeneous"} n°{str(i)} (slope={slope:.2f}±{slope_error:.2f})"
            i+=1
        else:
            label = f"{"Homogeneous"} n°{str(j)} (slope={slope:.2f}±{slope_error:.2f})"
            j+=1
        plt.xlabel("Dimensionless time", weight='bold')
        plt.ylabel("Variance / Mean", weight='bold')
        plt.yscale("log")
        plt.xlim([0.7,2])
        plt.ylim([150,10**5])
        plt.xscale("log")
        #plt.title("Variance-to-Mean Ratio vs. Dimensionless Time")
        plt.grid()
        plt.plot(fit_time, fit_line, linestyle="--",linewidth=2,label=label)

        # Ajouter la couleur utilisée à l'ensemble
        color_used.add(c)
    plt.legend()
    plt.show()
    # Ajouter une légende séparée avec les labels correspondant aux couleurs
    handles = [
        plt.Line2D([0], [0], color=color, lw=5, label=label)
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
if __name__ == "__main__":
    repertoire1="/home/chorus/expjuillet/"
    how=input("loglog ou semi-log")
    root_directory = input("Entrez le chemin du dossier contenant les fichiers .ome.tif: ")

    # Traiter les fichiers

    extracted_results = process_ome_tif_files(root_directory)

    # Tracer les résultats
    if extracted_results and how=="loglog":
        plot_results_loglog_rgb(extracted_results)
    elif extracted_results and how=="semi-log":
        plot_results(extracted_results)

    else:
        print("Aucun fichier traité avec seuccès.")
