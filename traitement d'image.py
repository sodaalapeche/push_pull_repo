import math
import os
import numpy as np
from PIL import Image,ImageSequence
import matplotlib.pyplot as plt

def process_tif_images(directory):
    """
    Ouvre les fichiers TIF dans un répertoire, calcule la somme des niveaux de gris
    pour chaque image contenue dans chaque fichier TIF et trace le résultat en fonction du temps.

    Parameters:
        directory (str): Le chemin vers le répertoire contenant les fichiers TIF.
    """
    # Récupère la liste des fichiers TIF dans le répertoire
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif') or f.endswith('.tiff')]

    if not tif_files:
        print("Aucun fichier TIF trouvé dans le répertoire spécifié.")
        return

    # Trie les fichiers pour s'assurer qu'ils sont dans l'ordre temporel
    tif_files.sort()

    # Stocke les sommes des niveaux de gris
    grayscale_sums = []

    # Parcourt les fichiers et calcule les sommes des niveaux de gris pour chaque image
    for tif_file in tif_files:
        filepath = os.path.join(directory, tif_file)
        with Image.open(filepath) as img:
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                # Convertit chaque frame en niveaux de gris si ce n'est pas déjà le cas
                grayscale_image = frame.convert('L')
                # Convertit en tableau numpy
                image_array = np.array(grayscale_image)
                # Calcule la somme des niveaux de gris
                grayscale_sums.append(np.mean(image_array))

    # Génère l'axe des temps en fonction du nombre d'images
    time_points = np.arange(len(grayscale_sums))

    # Trace le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, grayscale_sums, marker='o', linestyle='-', color='b')
    plt.title('Somme des niveaux de gris en fonction du temps')
    plt.xlabel('Temps (images indexées)')
    plt.ylabel('moyenne des niveaux de gris')
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    directory = "/home/chorus/exp16-17/btc_3d_inclu_bleue2_17_12_5"
    process_tif_images(directory)