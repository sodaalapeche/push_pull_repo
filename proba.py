import os
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from pathlib import Path

def compute_probability(image_stack):
    """
    Calcule la probabilité P(x, y) pour chaque image d'une pile d'images,
    en sélectionnant uniquement les pixels dans une ROI circulaire.

    Parameters:
        image_stack (numpy.ndarray): Pile d'images (3D array où chaque couche correspond à une image).

    Returns:
        numpy.ndarray: Pile de probabilités P(x, y) de même dimension que image_stack, limitées à la ROI.
    """
    probabilities = []
    height, width = image_stack[0].shape
    radius = 0.45 * width  # Rayon de la ROI = 40% de la largeur de l'image
    center_x, center_y = width // 2, height // 2  # Centre de l'image

    # Création du masque circulaire
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    distance_to_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    circular_mask = distance_to_center <= radius

    for image in image_stack:
        # Appliquer le masque circulairen
        masked_image = np.where(circular_mask, image, 0)

        # Calcul de la somme des niveaux de gris pour normalisation
        total_intensity = np.sum(masked_image)
        if total_intensity > 0:
            P = masked_image / total_intensity
        else:
            P = np.zeros_like(image)
        probabilities.append(P)

    return np.array(probabilities)


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

        print(
            f"Image {i}: x_mean={x_mean}, y_mean={y_mean}, x_central_moment={x_central_moment}, y_central_moment={y_central_moment}")

        # Afficher l'image avec le cercle et le point rouge tous les 50 indices
        if i in [0,1,2,5,10,15,20,40,50,60,80,100,150,200]:
            radius = np.sqrt(x_central_moment + y_central_moment)
            plot_image_with_circle(probability_image, x_mean, y_mean, radius, i)

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
        print(f"Image {i}: Total Probability = {total_probability}, Valid = {is_valid}")
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
    corrected_stack[corrected_stack < 0.004] = 0  # Éliminer les valeurs négatives
    return corrected_stack


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
                    resize(image,
                           (image.shape[0] // 4, image.shape[1] // 4),
                           anti_aliasing=True) for image in image_stack
                ])

                # Soustraire le fond si l'option est activée
                if subtract_background_option:
                    resized_stack = subtract_background(resized_stack)

                probabilities = compute_probability(resized_stack)
                results[str(ome_file)] = probabilities

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
            continue

    return results
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def power_law(x, a, b):
    """
    Fonction de loi de puissance : y = a * x^b
    """
    return a * x**b

def fit_power_law(time, radii):
    """
    Ajuste une loi de puissance sur les données de rayon caractéristique.

    Parameters:
        time (numpy.ndarray): Tableau des temps (indices).
        radii (numpy.ndarray): Tableau des rayons caractéristiques.

    Returns:
        tuple: Paramètres ajustés (a, b) et covariance de l'ajustement.
    """
    # Prendre uniquement la partie décroissante après le minimum
    min_index = np.argmin(radii)
    decreasing_time = time[min_index:]
    decreasing_radii = radii[min_index:]

    # Conversion en échelle log-log pour ajustement
    log_time = np.log(decreasing_time)
    log_radii = np.log(decreasing_radii)

    # Ajustement de la loi de puissance
    popt, pcov = curve_fit(lambda t, a, b: a + b * t, log_time, log_radii)

    # Revenir aux paramètres de la loi de puissance (a, b)
    a_fit = np.exp(popt[0])  # Remettre l'échelle exponentielle pour a
    b_fit = popt[1]          # Exposant
    return a_fit, b_fit, pcov

def compute_radius_evolution(moments):
    """
    Calcule l'évolution du rayon en fonction du temps à partir des moments centrés.

    Parameters:
        moments (list): Liste de dictionnaires contenant les moments pour chaque image.

    Returns:
        numpy.ndarray: Rayon caractéristique pour chaque image.
    """
    radii = []
    for m in moments:
        radius = np.sqrt(m['x_central_moment'] + m['y_central_moment'])
        radii.append(radius)
    return np.array(radii)


def plot_radius_evolution_with_fit(radii):
    """
    Trace l'évolution du rayon caractéristique avec un fit power-law,
    en appliquant un filtre sur les indices de temps (entre 8 et 100)
    et les valeurs de `radii` (supérieures à 5).

    Parameters:
        radii (numpy.ndarray): Rayon caractéristique pour chaque image.
    """
    time = np.arange(len(radii))  # Temps discrétisé (indices des images)

    # Filtrer les données
    time_condition = (time >= 8) & (time <= 100)
    radii_condition = radii > 5
    combined_condition = time_condition & radii_condition

    filtered_radii = radii[combined_condition]
    filtered_time = time[combined_condition]

    # Vérifier qu'il reste suffisamment de points pour effectuer un fit
    if len(filtered_radii) < 2:
        print("Pas assez de points après filtrage pour effectuer un fit.")
        return

    # Trouver l'indice du minimum après filtrage
    min_index = np.argmin(filtered_radii)
    print(f"Index du minimum (filtré): {min_index}, Rayon minimum: {filtered_radii[min_index]}")

    # Extraire les données pour le fit (par exemple, les 60 points après le minimum)
    fit_window = 80
    end_index = min(min_index + fit_window, len(filtered_radii))
    fit_time = filtered_time[min_index:end_index]
    fit_radii = filtered_radii[min_index:end_index]

    # Transformation log-log et ajustement
    log_fit_time = np.log(fit_time)
    log_fit_radii = np.log(fit_radii)

    # Ajustement linéaire
    coeffs = np.polyfit(log_fit_time, log_fit_radii, 1)
    slope, intercept = coeffs

    # Générer les valeurs pour la ligne de fit
    fit_line = np.exp(intercept) * fit_time ** slope

    # Tracé des données et du fit
    plt.figure(figsize=(8, 6))
    plt.plot(time, radii, marker='o', label="experience")
    plt.plot(fit_time, fit_line, linestyle="--", color="red", label=f"Fit power-law (slope={slope:.2f})")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Time (frame)")
    plt.ylabel("Rayon (px)")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    print(f"Slope (pente): {slope}, Intercept: {intercept}")
base_directory ="/home/chorus/heterogenous/ppull_sablefin_bille_1cm_09-12_3/"
base_directory = "/home/chorus/Homogenous/"
results = process_ome_tif_files(base_directory,subtract_background_option=True)

# Sauvegarde des résultats si nécessaire
for file_path, probability_stack in results.items():
    output_path = Path(file_path).with_name(Path(file_path).stem + "_probabilities.tif")
    tiff.imwrite(output_path, probability_stack.astype(np.float32))
    print(f"Probabilities saved to {output_path}")
# Chemin vers le répertoire contenant les fichiers .ome.tif
