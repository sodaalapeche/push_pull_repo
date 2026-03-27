import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import re
import itertools
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogLocator
def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else -1

markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h'])
lab=[1,2,3]


colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff7f0e', '#8c564b'])

def process_tif_sequence(folder_path, max_files=20):
    """
    Traite un dossier contenant une séquence d'images .tif (ex : time-lapse).

    Args:
        folder_path (str): Dossier contenant les images .tif (1 image = 1 frame).
        max_files (int): Nombre maximum de dossiers à traiter.

    Returns:
        dict: Dictionnaire {nom_du_dossier: {time, ratios}}
    """
    results = {}
    folders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
    folder_count = 0

    for folder in folders:
        full_path = os.path.join(folder_path, folder)
        tif_files = sorted(
            [f for f in os.listdir(full_path) if f.endswith('.tif')],
            key=extract_number)
        if len(tif_files) < 3:
            print(f"Pas assez d’images dans {folder}, ignoré.")
            continue

        try:
            variances = []
            means = []
            mass=[]
            for tif_file in tif_files:
                image_path = os.path.join(full_path, tif_file)
                img = imread(image_path)
                mass.append(np.sum(img))
                variances.append(np.var(img))
                means.append(np.mean(img))

            variances = np.array(variances)
            means = np.array(means)
            ratios =variances/means
            #ratios=means
            results[folder] = {
                "time": np.arange(len(ratios))*3.5/9,
                "ratios": ratios,
                "mass":means
            }

            folder_count += 1
            if folder_count >= max_files:
                break

        except Exception as e:
            print(f"Erreur dans {folder}: {e}")

    return results

def plot_results_loglog(results, window=(0, 40), nb=1, ax=None):
    """
    Affiche les résultats en semi-log avec un fit exponentiel.
    Si ax est None, utilise les axes courants.
    """

    if ax is None:
        ax = plt.gca()

    for label, data in results.items():
        time = np.array(data["time"])
        ratios = np.array(data["ratios"])
        mass = np.array(data["mass"])

        mask = (time > time[15]) & (ratios > 0)
        time = time[mask]
        #time= time * 100
        ratios = ratios[mask]
        max_val = np.max(ratios)
        ratios = ratios / max_val
        mass = mass[mask]

        marker = next(markers)
        color = next(colors)
        ax.scatter(time, ratios, color=color, marker=marker,
                   label='exp n°' + str(nb), s=15)

        max_index = int(window[0] * 9.5 / 3)
        end_index = min(int(window[1] * 9.5 / 3), len(ratios))

        fit_time = time[max_index:end_index]
        fit_ratios = ratios[max_index:end_index]

        if len(fit_time) < 2:
            continue  # trop peu de points

        log_fit_time = np.log(fit_time)
        log_fit_ratios = np.log(fit_ratios)

        (slope, intercept), cov = np.polyfit(log_fit_time, log_fit_ratios, 1, cov=True)
        fit_line = np.exp(intercept) * fit_time**slope
        slope_err = np.sqrt(cov[0, 0])
        color=next(colors)
        marker = next(markers)
        #ax.plot(fit_time, fit_line, '--', label=f"power-law fit : (slope={slope:.2f} ± {slope_err:.3f}))

    # Bottom axis only
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r"$\frac{t}{T_{a}}$", fontsize="x-large")
    ax.set_ylabel(r"$\sigma^{2}_{c}$", fontsize="x-large")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()
    #plt.xlim(right=28)
    # Custom ticks bottom
    ax.set_xticks([5,10, 15,20])
    plt.setp(ax.get_xticklabels(), rotation=45)

    return ax  # return ax s to allow further modifications


def process_npy_sequence(npy_path):
    """
    Traite un fichier .npy contenant une séquence d'images (array 3D : [frame, x, y]).

    Args:
        npy_path (str): chemin du fichier .npy

    Returns:
        dict: {nom_du_fichier: {time, ratios, mass}}
    """
    results = {}
    try:
        data = np.load(npy_path)  # shape attendu : (Nframes, Nx, Ny)

        if data.ndim != 3:
            raise ValueError(f"Le fichier {npy_path} n’a pas la bonne forme (attendu: 3D).")

        variances = []
        means = []
        mass = []

        for frame in data:
            mass.append(np.sum(frame))
            variances.append(np.var(frame))
            means.append(np.mean(frame))

        variances = np.array(variances)
        means = np.array(means)
        ratios = variances / means

        # on reprend ton time-step : Δt = 3/11
        time = np.arange(len(ratios))
        results[os.path.basename(npy_path)] = {
            "time": time,
            "ratios": ratios,
            "mass": means
        }

    except Exception as e:
        print(f"Erreur en lisant {npy_path}: {e}")

    return results

def plot_minus_one_slope(ax, start_x, start_y, length_decades=1, offset_factor=1.2, **kwargs):
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
    y_vals = start_y * offset_factor * (x_vals / start_x) ** (-1)

    # Trace sur les axes existants (uniquement bottom log-log)
    ax.plot(x_vals, y_vals, label=r"$t^{-1}$", **kwargs)

    # Place un petit texte à la fin de la courbe
    ax.text(
        x_vals[-1]*0.7, y_vals[-1]*2,
        r"slope = -1",
        fontsize="x-large",
        ha="left", va="top"
    )

    return ax


    return ax
def plot_results_semi_log(results,window=(2,15), nb=1):
    import itertools
    """
    Affiche les résultats en semi-log avec un fit exponentiel.

    Args:
        results (dict): Résultats extraits par process_tif_sequence.
    """

    for label, data in results.items():
        time = np.array(data["time"])
        ratios = np.array(data["ratios"])
        mass = np.array(data["mass"])
        max = np.max(ratios)
        ratios = ratios/max
        mask = (time > 0) & (ratios > 0)
        time = time[mask]

        #max_temps=time[-1]
        #time=time/max_temps
        ratios = ratios[mask]
        mass = mass[mask]
        marker=next(markers)
        color=next(colors)
        plt.scatter(time, ratios,color=color,marker=marker, label='exp n°'+str(nb),s=15)
        color=next(colors)
        marker=next(markers)
        #plt.scatter(time, mass,color=color,marker=marker, label='mass', s=15)
        max_index = int(window[0]*11/4.5)
        end_index = min(int(window[1]*11/4.5), len(ratios))

        #max_index = int(window[0])
        #end_index = min(int(window[1]), len(ratios))

        fit_time = time[max_index:end_index]
        fit_ratios = ratios[max_index:end_index]

        if len(fit_time) < 2:
            continue

        log_fit_ratios = np.log(fit_ratios)
        (slope, intercept), cov = np.polyfit(fit_time, log_fit_ratios, 1, cov=True)

        # erreur standard sur slope et intercept
        slope_err = np.sqrt(cov[0, 0])
        intercept_err = np.sqrt(cov[1, 1])

        # droite de fit (dans l’espace exponentiel)
        fit_line = np.exp(intercept) * np.exp(slope * fit_time)
        color = next(colors)
        # tracé avec pente ± erreur
        #plt.plot(fit_time,fit_line,'--',label=f"exp fit (slope={slope:.2f} ± {slope_err:.3f})",color=color,alpha =0.8)
        print(slope)
        lyapunov=-1.8*slope
        #villmeunier= ((lyapunov / (6 * time)) ** 0.25) * np.exp(-lyapunov * time / 3)
        #plt.plot(time,mass,label=label)
        #plt.plot(time,villmeunier,color='purple')

    plt.xlabel(r"$\frac{t}{T_{a}}$",fontsize="x-large")
    plt.ylabel(r"$\sigma^{2}_{c}$",fontsize="x-large")
    plt.xlim(left=0, right=window[1]+70)
    plt.yscale("log")


    plt.tight_layout()
    plt.grid(True)
    plt.legend(fontsize='x-large')


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
        x_vals[-1]*0.4, y_vals[-1],
        fr"slope = {slope}",
        fontsize="x-large", ha="left", va="top"
    )

    return ax


def run(folder=False,nb=1,mode='semi-log'):
    import sys
    if folder==False:
        root_directory = input("Entrez le chemin du dossier contenant les sous-dossiers d'images .tif : ")
        mode = input("Mode d'affichage ? (loglog / semi-log): ").strip().lower()
    else:
        root_directory = folder

    results = process_tif_sequence(root_directory)

    if not results:
        print("Aucun résultat obtenu.")
        sys.exit(0)

    if mode == "loglog":
        plot_results_loglog(results,(10,30),nb)
    elif mode == "semi-log":
        plot_results_semi_log(results,(2,20),nb)
    else:
        print("Mode invalide.")
if __name__ == "__main__":
    fig, ax = plt.subplots()
    #run('/home/chorus/expjuillet/ALL_EXP/11_07_1/',1,mode="semi-log")
    #run('/home/chorus/expjuillet/ALL_EXP/11_07_2/',2,mode="semi-log")
    #run('/home/chorus/expjuillet/ALL_EXP/11_07_3/',3,mode="semi-log")
    #run('/home/chorus/expjuillet/ALL_EXP/11_074/', 4, mode="semi-log")
    run('/home/chorus/exp_septoct/exp_1310_6/', 5, mode="semi-log")
    #plot_semi_log_slope(ax, start_x=2, start_y=0.85, slope=-0.05, length=50, color="black")
    plot_semi_log_slope(ax,start_x=10,start_y=0.36,slope=-0.06,length=70,color="black")
    #A=np.load("/home/chorus/simu/pushpointkostis_inclusions29e-09.npy")
    #results=process_npy_sequence("/home/chorus/simu/pullpointkostis_inclusions29e-09.npy")
    #plot_results_semi_log(results)
    # plot_semi_log_slope(ax,50,0.6,slope=-0.02)
    # plt.show()
    # fig,ax=plt.subplots()
    # plot_results_loglog(results,window=(0,300))
    # plot_minus_one_slope(ax,100,0.25,length_decades=0.45)
    #
    # plt.show()
    # plt.imshow(np.log(A)[:, :, 30])

#run('/home/chorus/expjuillet/ALL_EXP/11_07_3/',3)
    #run('/home/chorus/expjuillet/banned exp2/homogene/', 1, mode = 'loglog')
    #run('/home/chorus/expjuillet/banned exp2/homogene_2/', 2, mode = 'loglog')
    #plt.tight_layout()
    #run('/home/chorus/expjuillet/ALL_EXP/homogeneppinj2/', 3, mode='loglog')
#    run('/home/chorus/expjuillet/ALL_EXP/homogeneppinj3/', 4, mode='loglog')
    plt.show()