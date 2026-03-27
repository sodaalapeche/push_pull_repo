import os
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
from skimage import measure
import matplotlib.pyplot as plt
def process_tiff_files(directory):
    radii = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ome.tif'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                radii.extend(process_tiff_file(file_path))
    return radii

def process_tiff_file(file_path):
    radii = []  # Liste pour stocker les rayons calculés
    with tiff.TiffFile(file_path) as tif:
        for page in tif.pages:
            # Convertir la page en un tableau NumPy
            image = np.array(page.asarray())
            return image
            # Vérifier si l'image est binaire ou nécessite un prétraitement
            # Exemple : si l'image contient des valeurs de pixels dans une plage spécifique,
            # tu peux la binariser (si nécessaire) pour mesurer les objets
            if image.max() > 1:  # Si l'image n'est pas binaire
                image = image > 128  # Exemple de binarisation (à ajuster selon l'image)

            # Calculer le rayon à l'aide de la fonction mesure (adapte cela selon ta logique)
            radius = measure_radius(image)
            radii.append(radius)

    return radii

def measure_radius(image):
    # Apply Gaussian filter to smooth the image
    filtered_image = gaussian_filter(image, sigma=1)
    # Threshold the image to create a binary mask
    threshold = np.mean(filtered_image)
    binary_mask = filtered_image > threshold
    # Label connected regions
    labeled_image, num_features = measure.label(binary_mask, return_num=True)
    # Measure properties of labeled regions
    properties = measure.regionprops(labeled_image)
    # Find the largest region (assuming it's the blob of interest)
    largest_region = max(properties, key=lambda prop: prop.area)
    # Calculate the equivalent radius of the largest region
    radius = np.sqrt(largest_region.area / np.pi)
    return radius
def plot_radii(radii):
    plt.figure()
    plt.plot(radii, marker='o')
    plt.xlabel('Frame')
    plt.ylabel('Radius')
    plt.title('Radius of Luminous Blob vs Frame')
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    directory = '/home/chorus/best/homo/'
    radii = process_tiff_files(directory)
    #plot_radii(radii)
