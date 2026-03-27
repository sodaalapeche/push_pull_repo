import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

# Variables globales
root_dir = "/home/chorus/Homogenous/"  # Chemin vers le dossier contenant les fichiers
plage_temps = 0.005 # Taille de la plage de temps adimensionné (ex. 0.1 pour regrouper par 0.0-0.1, 0.1-0.2, etc.)

# Fonction pour adimensionner les temps
def adimensionner_temps(temps, ta):
    return temps / ta
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
    return ta
# Lecture des séquences et regroupement par temps adimensionné
# Lecture des séquences et regroupement par temps adimensionné
def regrouper_images_par_temps(root_dir, calcul_temps_advection, plage_temps):
    groupes_images = defaultdict(list)

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.ome.tif'):
                filepath = os.path.join(dirpath, file)

                # Chargement de la séquence d'images
                sequence = tifffile.imread(filepath)

                # Calcul du temps d'advection (déjà fourni par l'utilisateur)
                ta = calcul_temps_advection(filepath)

                # Supposons que le temps initial est incrémenté régulièrement pour chaque image
                num_images = sequence.shape[0]
                temps_initiaux = np.linspace(0, num_images - 1, num_images)

                # Adimensionnement des temps
                temps_adimensionnes = adimensionner_temps(temps_initiaux, ta)

                # Regroupement par plages
                for i, img in enumerate(sequence):
                    temps_adim = temps_adimensionnes[i]
                    plage = int(temps_adim // plage_temps) * plage_temps
                    groupes_images[plage].append((temps_adim, img))

    return groupes_images

# Fonction pour afficher les images d'une plage de temps

def afficher_groupes_images(groupes_images, temps_cible, max_images=10, max_resolution=512):
    # Limiter le nombre d'images à afficher
    nb_images = min(len(groupes_images), max_images)

    # Créer une figure pour afficher toutes les images sur le même plot
    plt.figure(figsize=(15, 15))

    for idx, (plage, images) in enumerate(sorted(groupes_images.items())):
        if idx >= nb_images:
            break

        # Chercher l'image correspondant au temps_cible (le plus proche)
        image_choisie = None

        for temps, img in images:
            # Si le temps est proche du temps cible (ou exactement égal)
            if abs(temps - temps_cible) < abs(image_choisie[0] - temps_cible) if image_choisie else True:
                image_choisie = (temps, img)

        # Extraire l'image et son temps
        temps, img = image_choisie

        # Si img est une liste (extraction d'une seule image de la séquence)
        if isinstance(img, list):
            img = img[0]  # Prendre la première image (s'il y en a plusieurs dans la liste)

        # Vérifier que l'image est bien un tableau 2D
        if isinstance(img, np.ndarray) and img.ndim == 2:
            # Normalisation des images pour l'affichage
            img_normalisee = img / np.max(img)

            # Réduire la taille de l'image si nécessaire
            if img.shape[0] > max_resolution or img.shape[1] > max_resolution:
                img_normalisee = img_normalisee[::int(img.shape[0] / max_resolution), ::int(img.shape[1] / max_resolution)]

            # Affichage de l'image dans la figure avec un subplot
            plt.subplot(1, nb_images, idx + 1)  # Afficher les images dans une ligne
            plt.imshow(img_normalisee, cmap='viridis', vmin=0, vmax=1)

            plt.axis('off')
        else:
            print(f"Warning: L'image à t={temps:.2f} n'est pas valide ou est mal formatée.")

    # Ajuster l'espacement pour que les titres et images ne se chevauchent pas
    plt.tight_layout()
    plt.show()
# Exécution
groupes_images = regrouper_images_par_temps(root_dir, calculta, plage_temps)
temps_cible = 0.8  # Temps adimensionné spécifique
afficher_groupes_images(groupes_images, temps_cible, max_images=5)
