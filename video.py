import os
import re
import numpy as np
import cv2
import tifffile as tiff
from glob import glob
from tqdm import tqdm

def calculta(file):
    pattern = r"debit\s*(\d+)"
    matches = re.findall(pattern, file, re.IGNORECASE)
    debit = [int(match) for match in matches]
    if debit == []:
        flux = ((0.5 / 60) * (10 ** -3))
    else:
        flux = float((debit[0]) / 600) * (10 ** -3)
        print('debit ====' + str(debit[0]))
        if len(str(debit[0])) == 3:
            flux = flux / 10
    if "fin" in file:
        ta = 0.35 * (0.057 / 2) ** 2 * np.pi * 0.5 / flux
        color = 'b'
    elif "inclu" in file:
        ta = 0.35 * (0.057 / 2) ** 2 * np.pi * 0.5 / flux
        color = 'r'
    else:
        ta = 0.35 * (0.057 / 2) ** 2 * np.pi * 0.5 / flux
        color = 'g'
    print(ta)
    return ta

def resize_image(image, scale=0.25):
    """Reduit la taille de l'image par un facteur donné."""
    height, width = image.shape
    new_dim = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

def load_and_process_tiff(file_path, scale=0.25):
    """Charge un fichier .ome.tif et réduit la taille des images."""
    images = tiff.imread(file_path)  # Charge toutes les images
    resized_images = [resize_image(img, scale) for img in images]
    return resized_images
def get_fps_from_metadata(file_path):
    """Récupère le fps à partir des métadonnées du fichier TIFF."""
    try:
        with tiff.TiffFile(file_path) as tif:
            # Chercher dans les métadonnées, ici l'exemple est basé sur une clé commune
            # "time_interval" qui peut être en secondes, ou bien une autre clé spécifique
            if hasattr(tif.pages[0], 'tags'):
                tags = tif.pages[0].tags
                for tag in tags.values():
                    if tag.name.lower() == 'timeinterval':
                        time_interval = tag.value
                        fps = 1 / time_interval
                        return fps
            # Si pas de time_interval trouvé, retourne une valeur par défaut
            return 5  # Valeur par défaut si aucune info n'est trouvée
    except Exception as e:
        print(f"Erreur lors de la récupération des métadonnées pour {file_path}: {e}")
        return 5  # Valeur par défaut si une erreur survient
def adimensionalize_times(images, ta):
    """Associe les temps adimensionnels à chaque image."""
    num_frames = len(images)
    return {ta * i / num_frames: images[i] for i in range(num_frames)}

def synchronize_videos(video_data, ta_values, fps_values):
    """Synchronise les vidéos en fonction des temps adimensionnels sans interpolation."""
    all_times = sorted(set().union(*[data.keys() for data in video_data]))
    synchronized_frames = []

    for i, t in enumerate(all_times):
        frames_at_t = []
        for j, data in enumerate(video_data):
            # Trouver la frame la plus proche dans les temps disponibles
            closest_time = min(data.keys(), key=lambda k: abs(k - t))
            frames_at_t.append(data[closest_time])

        concatenated_frame = np.concatenate(frames_at_t, axis=1)

        # Calcul du temps réel de l'image
        fps = fps_values[0]  # Utilisation du fps de la première vidéo
        time_real = i / fps  # Le temps réel de l'image en secondes

        # Calcul du temps adimensionnel
        adimensional_time = time_real / ta_values[0]  # Utilisation du ta de la première vidéo

        # Texte à afficher (temps adimensionnel)
        text = f"t* = {adimensional_time:.2f}"

        # Calculer la taille du texte et sa position pour le mettre en haut à gauche
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2  # Taille du texte
        thickness = 10  # Épaisseur du texte

        # Positionner le texte en haut à gauche
        text_x, text_y = 20, 50  # Décalage pour éviter qu'il touche le bord

        # Ajouter le texte sur l'image avec un contour noir épais
        cv2.putText(
            concatenated_frame, text, (text_x, text_y),
            font, font_scale, (8000,8000,8000), thickness, lineType=cv2.LINE_AA  # Texte blanc épais
        )

        synchronized_frames.append(concatenated_frame)

    return synchronized_frames

def save_tiff(frames, output_path):
    """Sauvegarde une séquence d'images sous forme de fichier TIFF."""
    tiff.imwrite(output_path, np.array(frames, dtype=np.uint16))

def main(input_folder, output_tiff, scale=0.5):
    """Assemble les vidéos côte à côte avec temps adimensionnés."""
    # Recherche récursive dans les sous-dossiers
    tif_files = sorted(glob(os.path.join(input_folder, "**", "*.ome.tif"), recursive=True))

    if not tif_files:
        raise ValueError("Aucun fichier .ome.tif trouvé dans le dossier spécifié.")

    video_data = []
    ta_values = []  # Liste pour stocker les valeurs de ta pour chaque vidéo
    fps_values = []  # Liste pour stocker les valeurs de fps pour chaque vidéo

    # Charger et traiter chaque vidéo
    for i, tif_file in enumerate(tqdm(tif_files, desc="Traitement des fichiers")):
        images = load_and_process_tiff(tif_file, scale=scale)
        if not images:
            print(f"Aucune image chargée pour {tif_file}. Ignoré.")
            continue
        ta = calculta(tif_file)  # Calculer ta à partir du nom du fichier
        ta_values.append(ta)  # Ajouter la valeur de ta à la liste
        fps = get_fps_from_metadata(tif_file)  # Récupérer le fps depuis les métadonnées
        fps_values.append(fps)  # Ajouter le fps à la liste
        adim_data = adimensionalize_times(images, ta)
        video_data.append(adim_data)

    if not video_data:
        raise ValueError("Aucune donnée de vidéo valide n'a été générée.")

    # Synchroniser les vidéos
    print("Synchronisation des vidéos...")
    synchronized_frames = synchronize_videos(video_data, ta_values, fps_values)

    if not synchronized_frames:
        raise ValueError("Aucune frame synchronisée disponible après la synchronisation.")

    # Sauvegarder le fichier TIFF final
    print(f"Sauvegarde de la séquence d'images dans {output_tiff}...")
    save_tiff(synchronized_frames, output_tiff)


if __name__ == "__main__":
    # Exemple d'utilisation
    input_folder = "/home/chorus/best/"  # Dossier contenant les fichiers .ome.tif
    output_tiff = "/home/chorus/output2.tiff"  # Nom du fichier TIFF final
    scale = 0.25 # Réduction de taille

    main(input_folder, output_tiff, scale=scale)


