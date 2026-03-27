import os
import numpy as np
import cv2
import tifffile as tiff
import shutil
import traceback
import matplotlib.pyplot as plt
from skimage.transform import resize
from pathlib import Path
from glob import glob
from scipy.optimize import curve_fit


def subtract_annular_background(image_stack):
    """
    Soustrait la moyenne des niveaux de gris d'un anneau extérieur dans l'image.
    """
    corrected_stack = []
    height, width = image_stack[0].shape
    radius_inner = 0.65 * width
    radius_outer = 0.75 * width
    center_x, center_y = width // 2, height // 2

    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    distance_to_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    annular_mask = (distance_to_center >= radius_inner) & (distance_to_center <= radius_outer)

    for image in image_stack:
        annular_values = image[annular_mask]
        background_mean = np.mean(annular_values) if annular_values.size > 0 else 0
        corrected_image = np.clip(image - background_mean, 0, None)
        corrected_stack.append(corrected_image)

    return np.array(corrected_stack)


def compute_probability(image_stack):
    """
    Calcule la probabilité P(x, y) pour chaque image d'une pile d'images.
    """
    probabilities = []
    height, width = image_stack[0].shape
    radius = 0.45 * width
    center_x, center_y = width // 2, height // 2

    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    distance_to_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    circular_mask = distance_to_center <= radius

    for image in image_stack:
        masked_image = image * circular_mask
        total_intensity = np.sum(masked_image)
        P = masked_image / total_intensity if total_intensity > 0 else np.zeros_like(image)
        probabilities.append(P)

    return np.array(probabilities)


def verify_probabilities(probability_stack):
    """
    Vérifie que la somme des probabilités pour chaque image vaut 1.
    """
    for i, probability_image in enumerate(probability_stack):
        total_probability = np.sum(probability_image)
        is_valid = np.isclose(total_probability, 1.0)
        print(f"Image {i}: Total Probability = {total_probability}, Valid = {is_valid}")


def subtract_background(image_stack):
    """
    Soustrait la première image de la pile à toutes les autres images.
    """
    background = image_stack[0]
    corrected_stack = image_stack - background
    corrected_stack[corrected_stack < 0.009] = 0
    return corrected_stack


def process_ome_tif_files(base_dir, subtract_background_option=False):
    """
    Traite les fichiers .ome.tif dans un répertoire donné.
    """
    results = {}
    base_path = Path(base_dir)
    output_dir = Path("/home/chorus/video/")

    # Nettoyage du dossier de sortie
    for file in output_dir.glob("*"):
        try:
            if file.is_file() or file.is_symlink():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        except Exception as e:
            print(f'Erreur lors de la suppression de {file}: {e}')

    ome_files = list(base_path.rglob("*.ome.tif"))

    for ome_file in ome_files:
        print(f"Processing {ome_file}...")
        try:
            image_stack = tiff.imread(ome_file)

            if image_stack.ndim == 3:
                resized_stack = np.array([
                    resize(img, (img.shape[0] // 4, img.shape[1] // 4), anti_aliasing=True)
                    for img in image_stack
                ])

                if subtract_background_option:
                    resized_stack = subtract_background(resized_stack)

                resized_stack = subtract_annular_background(resized_stack)
                probabilities = compute_probability(resized_stack)
                verify_probabilities(probabilities)
                results[str(ome_file)] = probabilities
            else:
                print(f"Skipping {ome_file} (not a stack of images).")
        except Exception as e:
            print(f"Error processing {ome_file}: {e}")
            traceback.print_exc()

    return results


def save_video_from_images(image_folder, output_path, fps=10):
    """
    Génère une vidéo à partir des images d'un dossier donné.
    """
    images = sorted(glob(f"{image_folder}/image_*.png"))
    if not images:
        print("Aucune image trouvée pour générer la vidéo.")
        return

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    video.release()
    print(f"Vidéo sauvegardée sous {output_path}")


# Exécution principale
base_directory = "/home/chorus/Homogenous/ppull_SABLE_homog_22_11_2/"
results = process_ome_tif_files(base_directory, subtract_background_option=True)

for file_path, probability_stack in results.items():
    output_path = Path(file_path).with_name(Path(file_path).stem + "_probabilities.tif")
    tiff.imwrite(output_path, probability_stack.astype(np.float32))
    print(f"Probabilities saved to {output_path}")

save_video_from_images("/home/chorus/video", "/home/chorus/video_rayon.mp4")
