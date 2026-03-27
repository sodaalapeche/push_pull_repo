import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from path.to.venv.proba import base_directory
from proba_rayon import subtract_annular_background,process_jpg_images,find_extracted_images,find_metadata
from skimage.transform import resize

# Échelle spatiale (0.11 mm/px = 0.11 * 10⁻³ m/px)
scale = 0.11 * 1e-3


def find_ome_tif_files(folder):
    """Trouve tous les fichiers .ome.tif dans le dossier et ses sous-dossiers."""
    ome_tif_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.ome.tif'):
                ome_tif_files.append(os.path.join(root, file))
    return sorted(ome_tif_files)  # Trier les fichiers pour garder l'ordre


def load_ome_tif_images(folder):
    """Charge toutes les images .ome.tif trouvées."""

    file_paths = find_ome_tif_files(folder)
    for path in file_paths:
        print("treating image of videos "+str(path))
        img = tifffile.imread(path,is_mmstack=False)

    return img


def reduce_resolution(images, scale_factor=4):

    if images.ndim == 3:
        resized_stack = np.array([
            resize(image, (image.shape[0] //scale_factor, image.shape[1] // scale_factor), anti_aliasing=True)
            for image in images
        ])
    return  resized_stack


def compute_center_of_mass(image):
    """Calcule le centre de masse en utilisant la probabilité P(x,y) = C(x,y) / intégrale(C dxdy)."""
    height, width = image.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    total_intensity = np.sum(image)
    if total_intensity == 0:
        return width // 2, height // 2  # Retourne le centre géométrique si l'image est vide

    x_mean = np.sum(x_indices * image) / total_intensity
    y_mean = np.sum(y_indices * image) / total_intensity
    return int(x_mean), int(y_mean)


def compute_radial_profile(image, center, dr=15):
    """Calcule la moyenne du niveau de gris en fonction du rayon par anneaux concentriques."""
    height, width = image.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Calcul des distances au centre
    distances = np.sqrt((x_indices - center[0]) ** 2 + (y_indices - center[1]) ** 2)

    max_radius = int(np.max(distances))
    radial_profile = np.zeros(max_radius)
    counts = np.zeros(max_radius)

    for r in range(max_radius):
        mask = (distances >= r) & (distances < r + dr)
        radial_profile[r] = np.sum(image[mask])/np.max(image)
        counts[r] = np.sum(mask)

    radial_profile[counts > 0] /= counts[counts > 0]
    integral = np.trapz(radial_profile, dx=dr)  # Approximation de l'intégrale
    if integral > 0:
        radial_profile /= integral
    # Conversion en mètres
    radii_m = np.arange(max_radius) * scale
    return radii_m, radial_profile


def process_sequence(folder):
    """Charge, traite et affiche les profils radiaux pour une image sur 5."""
    images = find_extracted_images(folder+'/extracted_images/')
    images = subtract_annular_background(images)

    for i, img in enumerate(images):
        center = compute_center_of_mass(img)
        if i == 50:  # Une image sur 5
            radii, profile = compute_radial_profile(img, center)
            # 🔹 Affichage de l'image expérimentale avec le centre de masse
            plt.plot(radii, profile, label='radial profile')
            plt.title(f'frame n°{i}')
            plt.yscale('log')
            plt.show()

def compare_image(folder1, folder2):

    img1 =cv2.imread(folder1,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(folder2,cv2.IMREAD_GRAYSCALE)

    #print('IMAGE1 :  ',str([img1]), type(img1))

    img1=subtract_annular_background(np.array([img1]))
    img2=subtract_annular_background(np.array([img2]))
    plt.imshow(img1, cmap='viridis')
    plt.show()
    plt.imshow(img2, cmap='viridis')
    plt.show()
    center1 = compute_center_of_mass(img1)
    center2 = compute_center_of_mass(img2)
    radii, profile1 = compute_radial_profile(img1, center1)
    radii2, profile2 = compute_radial_profile(img2, center2)
    plt.plot(radii, profile1,linestyle='--', label='cas hétérogène')
    plt.plot(radii2, profile2, label='cas homogène')
    plt.legend()
    #plt.xlim((0,0.024))
    plt.xlabel('r (m)')
    plt.ylabel('log(c/cmax)')
    plt.yscale('log')
    plt.show()

#base_directory ="/home/chorus/heterogenous/ppull_sablefin_bille_1cm_09-12_3/"
#base_directory = "/home/chorus/exp/Homogenous/ppull_SABLE_homog_22_11_2/"
base_directory = '/home/chorus/exp/heterogenous/ppull_sablefin_bille_1cm_09-12_3/extracted_images/ppull_sablefin_bille_1cm_09-12_3_MMStack_Default.ome_frame_0110.png'
base_directory1 = "/home/chorus/exp/Homogenous/ppull_SABLE_homog_22_11_2/extracted_images/ppull_SABLE_homog_22_11_2_MMStack_Default.ome_frame_0110.png"
#process_sequence(base_directory)
A=compare_image(base_directory, base_directory1)

