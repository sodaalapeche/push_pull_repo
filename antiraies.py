import numpy as np
import skimage.io
import skimage.transform
import glob
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm  # <-- barre de progression

# --- Dossier contenant les fichiers .tif ---
folder = '/home/chorus/exp_septoct/exp_1310_6/serie1/'

# --- Récupération et tri des fichiers ---
files = sorted(glob.glob(os.path.join(folder, '*.tif')))
print(f"{len(files)} fichiers trouvés dans {folder}")

# --- Création du sous-dossier de sortie ---
output_folder = os.path.join(folder, 'corrige')
os.makedirs(output_folder, exist_ok=True)

# --- Paramètres globaux ---
angle0 = 2.2
Angles = np.linspace(angle0 - 2.0, angle0 + 2.7, 10)

# --- Boucle sur les fichiers avec barre de progression ---
for f in tqdm(files, desc="Traitement des images", unit="image"):
    # Lecture et normalisation
    I = skimage.io.imread(f).astype(np.float32) / (2**16 - 1)
    n0 = I.shape[0]
    Iall = np.copy(I)

    # Boucle sur les angles
    for i, angle in enumerate(Angles):
        Ir = skimage.transform.rotate(I, angle, preserve_range=True)
        norm = np.percentile(Ir, 98, axis=0)
        norm_mean = gaussian_filter(norm, 100)
        fact = norm / (norm_mean + 1e-8)
        Iplot = skimage.transform.rotate(Ir / fact, angle0 - angle, preserve_range=True)

        # Mise à jour partielle
        x1 = -(i + 1) * n0 // len(Angles)
        x2 = -(i) * n0 // len(Angles)
        Iall[:, x1:x2] = Iplot[:, x1:x2]

        # Libération mémoire
        del Ir, Iplot, norm, norm_mean, fact

    # Sauvegarde directe
    base = os.path.basename(f)
    name, ext = os.path.splitext(base)
    output_path = os.path.join(output_folder, f"{name}_corrige{ext}")
    skimage.io.imsave(output_path, np.uint16(Iall * (2**16 - 1)))

    # Libération mémoire
    del I, Iall

print("\n✅ Traitement terminé pour tous les fichiers.")
print(f"Images corrigées sauvegardées dans : {output_folder}")
