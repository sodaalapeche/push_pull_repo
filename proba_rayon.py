import os
import numpy as np
from glob import glob
import cv2
import tifffile as tiff
from conda.instructions import PRINT
from skimage.transform import resize
from pathlib import Path
import h5py
from path.to.venv.histo2densité_HDF5 import process_hdf5_images

global pcov
global coeffs
import math
import re
import traceback


def subtract_annular_background(image_stack):
    """
    Soustrait la moyenne des niveaux de gris d'un anneau extérieur dans chaque image.
    L'anneau est défini entre 1.5 et 1.8 fois le rayon centré de l'intensité.

    Parameters:
        image_stack (numpy.ndarray): Pile d'images (3D array où chaque couche correspond à une image).

    Returns:
        Generator: Génère une image corrigée à la fois pour économiser de la mémoire.
    """
    height, width = image_stack.shape[1:]  # Récupère la taille des images
    center_x, center_y = width // 2, height // 2  # Centre de l'image

    # Meshgrid pour les coordonnées des pixels
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    for i in range(len(image_stack)):
        image=image_stack[i]
        # Normalisation des intensités pour obtenir des probabilités
        total_intensity = np.sum(image, dtype=np.float32)
        probability_image = (image / total_intensity) if total_intensity > 0 else np.zeros_like(image, dtype=np.float32)

        # Calcul des moments pour estimer le rayon central
        center_x = np.sum(x_indices * probability_image, dtype=np.float32)
        center_y = np.sum(y_indices * probability_image, dtype=np.float32)
        x_central_moment = np.sum(((x_indices - center_x) ** 2) * probability_image, dtype=np.float32)
        y_central_moment = np.sum(((y_indices - center_y) ** 2) * probability_image, dtype=np.float32)
        radius = np.sqrt(x_central_moment + y_central_moment)

        # Définition de l'anneau
        radius_inner =  1* radius
        radius_outer = 1.2 * radius
        distance_to_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        annular_mask = (distance_to_center >= radius_inner) & (distance_to_center <= radius_outer)
        image[image < 200] = 0
        # Calcul de la moyenne du fond dans l'anneau
        annular_values = image[annular_mask]
        background_mean = np.mean(annular_values, dtype=np.float32) if annular_values.size > 0 else 0
        # Soustraction et mise à zéro des valeurs négatives
        corrected_image = np.clip(image - background_mean, 0, None).astype(
            np.uint16)  # Conversion pour réduire la mémoire

        #if i%10==0:
         #   plt.imshow(corrected_image)
           # plt.show()

        image_stack[i] = corrected_image
    return image_stack


def compute_probability(image):

    total_intensity = np.sum(image)
    if total_intensity > 0:
        P = image / total_intensity
    else:
        P = np.zeros_like(image)
    return P


def plot_image_with_circle(image, xmoy, ymoy, radius, image_index):
    """
    Trace une image avec un point rouge au niveau de (xmoy, ymoy) et un cercle de rayon.

    Parameters:
        image (numpy.ndarray): L'image à tracer.
        xmoy (float): Coordonnée x du centre du cercle.
        ymoy (float): Coordonnée y du centre du cercle.
        radius (float): Rayon du cercle.
        image_index (int): Index de l'image dans la pile.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='viridis')
    ax.plot(xmoy, ymoy, 'ro', label='Centre (xmoy, ymoy)')  # Point rouge
    circle = plt.Circle((xmoy, ymoy), radius, color='blue', fill=False, linestyle='--', label=f'Rayon={radius:.2f}')
    ax.add_patch(circle)
    ax.set_title(f"Image {image_index}: xmoy={xmoy}, ymoy={ymoy}, rayon={radius:.2f}")
    ax.legend()
    plt.savefig(f'{"/home/chorus/video"}/image_{image_index:03d}.png')
    plt.show()


def compute_moments(probability_stack):
    """
    Calcule les moments premiers et seconds centrés pour chaque image d'une pile de probabilités,
    et affiche l'image avec un cercle et un point rouge au centre.

    Parameters:
        probability_stack (numpy.ndarray): Pile de probabilités (3D array où chaque couche correspond à une image).

    Returns:
        list: Liste de dictionnaires contenant les moments pour chaque image.
    """
    moments = []
    for i, probability_image in enumerate(probability_stack):
        # Coordonnées des pixels
        y_indices, x_indices = np.meshgrid(np.arange(probability_image.shape[0]),
            np.arange(probability_image.shape[1]),
            indexing='ij'
        )


        # Moments premiers
        x_mean = np.sum(x_indices * probability_image)
        y_mean = np.sum(y_indices * probability_image)
        # Moments seconds centrés
        x_central_moment = np.sum(((x_indices - x_mean) ** 2) * probability_image)
        y_central_moment = np.sum(((y_indices - y_mean) ** 2) * probability_image)

        moments.append({
            'x_mean': x_mean,
            'y_mean': y_mean,
            'x_central_moment': x_central_moment,
            'y_central_moment': y_central_moment
        })

        #print(
         #   f"Image {i}: x_mean={x_mean}, y_mean={y_mean}, x_central_moment={x_central_moment}, y_central_moment={y_central_moment}")

        # Afficher l'image avec le cercle et le point rouge tous les 50 indices
        if i in range(0,len(probability_stack),20):
            radius = np.sqrt(x_central_moment + y_central_moment)
            #plot_image_with_circle(probability_image, x_mean, y_mean, radius, i)

    return moments
def verify_probabilities(probability_stack):
    """
    Vérifie que la somme des probabilités pour chaque image vaut 1.

    Parameters:
        probability_stack (numpy.ndarray): Pile de probabilités (3D array où chaque couche correspond à une image).

    Returns:
        list: Liste des booléens indiquant si la somme vaut 1 pour chaque image.
    """
    verification_results = []
    for i, probability_image in enumerate(probability_stack):
        total_probability = np.sum(probability_image)
        is_valid = np.isclose(total_probability, 1.0)
        verification_results.append(is_valid)
        #print(f"Image {i}: Total Probability = {total_probability}, Valid = {is_valid}")
    return verification_results
def subtract_background(image_stack):
    """
    Soustrait la première image de la pile à toutes les autres images.

    Parameters:
        image_stack (numpy.ndarray): Pile d'images (3D array où chaque couche correspond à une image).

    Returns:
        numpy.ndarray: Pile d'images avec le fond soustrait.
    """
    background = image_stack[0] # Première image de la pile
    corrected_stack = image_stack - background
    corrected_stack[corrected_stack < 0.012] = 0  # Éliminer les valeurs négatives
    return corrected_stack

import json
def process_ome_tif_files(base_dir, subtract_background_option=False):
    """
    Parcourt les sous-dossiers d'un répertoire pour charger et traiter les fichiers .ome.tif.

    Parameters:
        base_dir (str): Chemin du répertoire de base contenant les sous-dossiers avec fichiers .ome.tif.
        subtract_background_option (bool): Indique si l'on doit soustraire le fond (première image) des autres.

    Returns:
        dict: Un dictionnaire contenant les chemins des fichiers comme clés et les résultats calculés comme valeurs.
    """
    results = {}
    base_path = Path(base_dir)
    folder2 = "/home/chorus/video/"
    for filename in os.listdir(folder2):
        J = str(filename)
        file_path = os.path.join(folder2, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Recherche des fichiers .ome.tif dans tous les sous-dossiers
    ome_files = list(base_path.rglob("*.ome.tif"))

    for ome_file in ome_files:
        print(f"Processing {ome_file}...")
        try:
            # Chargement des images
            image_stack = tiff.imread(ome_file, is_mmstack=False)

            # Réduction de la résolution par un facteur de 4
            if image_stack.ndim == 3:
                resized_stack = np.array([
                    resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
                    for image in image_stack
                ])

                # Soustraire le fond si l'option est activée
                if subtract_background_option:
                    resized_stack = subtract_background(resized_stack)

                # Appliquer la correction d'arrière-plan sur l'anneau extérieur
                resized_stack = subtract_annular_background(resized_stack)

                probabilities = compute_probability(resized_stack)
                # Vérification des probabilités
                verify_probabilities(probabilities)

                # Calcul des moments
                moments = compute_moments(probabilities)

                # Calcul et traçage du rayon
                radii = compute_radius_evolution(moments)
                plot_radius_evolution_with_fit(radii)
            else:
                print(f"Skipping {ome_file} as it is not a stack of images.")
        except Exception as e:
            print(f"Error processing {ome_file}: {e}")
            traceback.print_exc()  # Affiche la stack trace complète pour comprendre l'erreur
            continue
    return results
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from glob import glob
import cv2
from skimage.transform import resize
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import tifffile
def find_extracted_images(directory):
    image_files = sorted(glob(os.path.join(directory, '*.png'), recursive=True))
    image_stack = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_files]

    return np.array(image_stack)
def find_metadata(folder_path):
    ome_tiff_file = None

    # Recherche du fichier .ome.tif dans le dossier donné
    for file in os.listdir(folder_path):
        if file.endswith(".ome.tif"):
            ome_tiff_file = os.path.join(folder_path, file)
            break

    if not ome_tiff_file:
        raise FileNotFoundError("Aucun fichier .ome.tif trouvé dans le dossier.")

    # Ouverture du fichier .ome.tif et extraction des métadonnées
    with tifffile.TiffFile(ome_tiff_file) as tif:
        metadata_xml = tif.ome_metadata  # Extraction du XML des métadonnées
    if metadata_xml is not None:

        root = ET.fromstring(metadata_xml)
        namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

        # 🔹 Extraction de l'intervalle de temps (TimeIncrement en ms dans <Pixels>)
        intervalle_ms = None
        pixels_element = root.find(".//ome:Pixels", namespace)
        if pixels_element is not None:
            interval_attr = pixels_element.get("TimeIncrement")
            if interval_attr:
                intervalle_ms = float(interval_attr)  # Convertir en float

        # 🔹 Extraction du débit (présent dans la description)
        debit = None
        description_element = root.find(".//ome:Image/ome:Description", namespace)
        if description_element is not None:
            description_text = description_element.text
            if description_text:
                # Recherche d'une ligne contenant "debit" et extraction de la valeur
                for line in description_text.split("\n"):
                    if "debit" in line.lower():
                        parts = line.split()
                        for i, word in enumerate(parts):
                            if word.lower() == "debit" and i + 1 < len(parts):
                                debit = parts[i + 1]  # Prend la valeur après "debit"
                                break
        if "fin" in folder_path:
            d=0.1
        else:
            d=0.6

        return {"intervalle_ms": float(intervalle_ms), "debit": float(debit),"d":float(d)}
    else:
        print("PAS DE METADONNEE XML ATTENTION DEBIT/FPS")
        return {"intervalle_ms": float(2500), "debit": float(90),"d":float(0.6)}
def find_hdf5_file(folder):
    for file in os.listdir(folder):
        if file.endswith('.h5'):
            return os.path.join(folder, file)
    return None

def load_hdf5_images(hdf5_path, nframe=None):
    with h5py.File(hdf5_path, 'r') as f:
        dataset_name = list(f.keys())[0]
        dataset = f[dataset_name]
        if nframe is not None:
            return dataset[nframe].squeeze()
        else:
            return np.squeeze(dataset[:])

import xml.etree.ElementTree as ET

def process_hdf5_images(base_dir, subtract_background_option=True, nframe=None):
    """
    Processes images stored in an HDF5 file found in the given directory.

    Parameters:
        base_dir (str): Path to the directory containing the HDF5 file.
        subtract_background_option (bool): Whether to subtract the background (first image) from the others.
        nframe (int, optional): Specific frame index to load. If None, loads the entire dataset.

    Returns:
        tuple: Computed slope, velocity (u), diffusion coefficient (D), error in D, dispersivity, and error in dispersivity.
    """
    results = {}

    # Locate the HDF5 file
    hdf5_path = find_hdf5_file(base_dir)
    if not hdf5_path:
        raise FileNotFoundError("No HDF5 file found in the specified directory.")

    print(f"Loading images from {hdf5_path}...")

    # Load image stack from the HDF5 file

    image_stack = load_hdf5_images(hdf5_path, nframe)

    print('trouve les metadata')
    # Load metadata from the .ome.tif file
    metadata = find_metadata(base_dir)
    print(metadata["intervalle_ms"], metadata["debit"])
    fps = 1 / (0.001 * metadata["intervalle_ms"])
    debit = metadata["debit"] * 0.001 / 60
    u = debit / (10**3 * math.pi * (55 * 0.001 * 0.5) ** 2)
    d= metadata["d"]*0.001
    image_stack = subtract_annular_background(image_stack)

    # Apply annular background correction
    image_stack = image_stack.astype(np.float32)

    for i in range(len(image_stack)):
        image_stack[i]=(compute_probability(image_stack[i]))  # Append instead of indexing

    moments = compute_moments(image_stack)
    radii = compute_radius_evolution(moments)
    (slope, u, D, errD, dispersive, errDisp) = plot_radius_evolution_with_fit(radii, fps, u)



    return slope, u, D, errD, dispersive, errDisp,d


def process_jpg_images(base_dir, subtract_background_option=False):
    """
    Traite les fichiers JPG dans un dossier donné.

    Parameters:
        base_dir (str): Chemin du dossier contenant les fichiers JPG.
        subtract_background_option (bool): Indique si l'on doit soustraire le fond (première image) des autres.

    Returns:
        dict: Résultats du traitement.
    """
    results = {}
    base_path = base_dir+'extracted_images/'

    print(f"Chargement des images depuis {base_dir}...")
    image_stack = find_extracted_images(base_path)
    text=find_metadata(base_dir)

    print(text["intervalle_ms"],text['debit'])
    #print(text['interval_ms'])
    fps=1/(0.001*text["intervalle_ms"])
    debit=text["debit"]*0.001/60
    u= debit/(10**3 * math.pi * (55*0.001*0.5)**2)
    resized_stack= image_stack

    # Soustraire le fond si l'option est activée
    if subtract_background_option:
        resized_stack = subtract_background(resized_stack)

        # Appliquer la correction d'arrière-plan sur l'anneau extérieur
    resized_stack = subtract_annular_background(resized_stack)

    probabilities = compute_probability(resized_stack)
    verify_probabilities(probabilities)
    moments = compute_moments(probabilities)
    radii = compute_radius_evolution(moments)
    (slope,u,D,errD,dispersivite,errDisp)=plot_radius_evolution_with_fit(radii,fps,u)

    results[base_dir] = probabilities


    return slope,u,D,errD,dispersivite,errDisp


def power_law(x, a, b):
    """
    Fonction de loi de puissance : y = a * x^b
    """
    return a * x**b


def compute_radius_evolution(moments):
    radii = []
    for m in moments:
        radius = np.sqrt(m['x_central_moment'] + m['y_central_moment'])
        radii.append(radius)
    return np.array(radii)


def plot_radius_evolution_with_fit(radii, fps, u):
    """
    Trace l'évolution du rayon caractéristique avec un fit power-law,
    en appliquant un filtre sur les indices de temps (entre 8 et 100)
    et les valeurs de `radii` (supérieures à 5).

    Les unités sont converties en mm (rayon) et secondes (temps).
    """
    px_to_mm = 0.11 * 0.001  # Conversion des pixels en mm
    time = np.arange(len(radii)) / fps  # Temps en secondes
    radii_mm = radii * px_to_mm  # Rayon en mm

    # Filtrer les données
    time_condition = time < 2 * 10 ** 2
    radii_condition = radii_mm > 10 ** -2
    combined_condition = radii_condition & time_condition

    filtered_radii = radii_mm[combined_condition]
    filtered_time = time[combined_condition]

    if len(filtered_radii) < 2:
        print("Pas assez de points après filtrage pour effectuer un fit.")
        return

    # Trouver l'indice du minimum après filtrage
    min_index = np.argmin(filtered_radii)
    t_min = filtered_time[min_index]

    # Décaler le temps pour que t_min soit 0
    adjusted_time = filtered_time - t_min

    adjusted_radii = filtered_radii  # Assurer que les dimensions correspondent
    fit_window = 110

    fit_time = adjusted_time[min_index+20:min_index + fit_window]
    fit_radii = adjusted_radii[min_index+20:min_index + fit_window]

    # Transformation log-log et ajustement
    valid_fit_mask = fit_time > 0  # Éviter log(0)
    log_fit_time = np.log(fit_time[valid_fit_mask])
    log_fit_radii = np.log(fit_radii[valid_fit_mask])

    coeffs, covariance = np.polyfit(log_fit_time, log_fit_radii, 1, cov=True)
    slope, intercept = coeffs
    std_err = np.sqrt(covariance[0, 0])
    std_err_inter = np.sqrt(covariance[1, 1])
    fit_line = np.exp(intercept) * fit_time ** slope

    plt.figure(figsize=(8, 6))
    plt.plot(adjusted_time, adjusted_radii, marker='o', label="Expérience")
    plt.plot(fit_time, fit_line, linestyle="--", color="red",
             label=f"Fit power-law (slope={slope:.2f} ± {std_err:.4f})")

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Temps ajusté (s)")
    plt.ylabel("Rayon (m)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    print(f"Slope (pente): {slope}, Intercept: {intercept}")
    D = math.exp(2 * intercept) / 2
    print(f"Diffusivité : {D} m²/s")
    print(f"Erreur std sur D : {math.exp(2 * intercept) * std_err_inter}")
    print('VITESSE : ', u, ' m.s-1')
    print('Dispersivité :', D / u)
    print(f"Erreur sur la dispersivité : {(math.exp(2 * intercept) * std_err_inter / D) * (D / u)}")

    return slope, u, D, math.exp(2 * intercept) * std_err_inter, D / u, (
                math.exp(2 * intercept) * std_err_inter / D) * (D / u)


#base_directory ="/home/chorus/exp/heterogenous/ppull_sablefin_bille_1cm_09-12_3/"
base_directory = "/home/chorus/exp/heterogenous/ppull_sablefin_bille_1cm_09-12_2/"
#results = process_ome_tif_files(base_directory,subtract_background_option=True)
#results=process_hdf5_images(base_directory)
import shutil

#%%
def process_all_folders(base_folder):
    L=[]
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):  # Vérifie que c'est un dossier
            print(f"Traitement du dossier : {folder_path}")
            a=process_hdf5_images(folder_path+'/')  # Appel de la fonction sur chaque dossier
            L.append(a)
        gc.collect()
    return np.array(L)
if '__main__' == __name__:
    import gc
    L= process_all_folders('/home/chorus/exp2/Homogenous/')
    np.save('/home/chorus/exp2/data.npy', L)
    L1= process_all_folders('/home/chorus/exp2/heterogenous/')
    np.save('/home/chorus/exp2/data1.npy', L1)
    disp=[]
    us=[]
    label=[]
    errD=[]
    h=[]
    d=[]
    #A=np.delete(L1,3,0)
    #L1=A
    for i in range(len(L)):
        d.append(L[i][6])
        disp.append(L[i][4])
        us.append(L[i][1])
        label.append(L[i][0])
        errD.append(L[i][5])
        h.append('red')
    for i in range(len(L1)):
        d.append(L1[i][6])
        disp.append(L1[i][4])
        us.append(L1[i][1])
        label.append(L1[i][0])
        errD.append(L1[i][5])
        h.append('blue')
    for i in range(len(disp)):
        plt.errorbar(us[i], disp[i]/d[i],yerr=errD[i]/d[i],marker=str('x' if d[i]<=0.0002 else 'o'), label=f"slope = {str(label[i])[0:4]}", color=h[i])
        plt.annotate(str(label[i])[0:4],(float(us[i]), float(disp[i]/d[i])),(float(us[i]*1.01), float(disp[i]/d[i])))
    plt.xlabel("vitesse u (m/s)")
    plt.ylabel(r"$\frac{\alpha}{d}$")
    plt.show()

