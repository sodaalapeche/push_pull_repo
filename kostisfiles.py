import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Dossier principal
data_folder = "/media/chorus/USB DISK/push_only_new"

# Chercher les sous-dossiers (expériences)
subfolders = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder)
                     if os.path.isdir(os.path.join(data_folder, f))])

if len(subfolders) < 2:
    raise ValueError("Moins de deux expériences trouvées.")

import re

def numerical_sort(value):
    # Extraire tous les nombres dans le nom de fichier et retourner une liste d'entiers
    parts = re.findall(r'\d+', value)
    return list(map(int, parts))

def load_experiment(folder):
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif")]
    images.sort(key=numerical_sort)  # tri numérique
    intensities = []
    for img_path in images:
        img = Image.open(img_path)
        data = np.array(img, dtype=np.float32)
        intensities.append(np.mean(data))  # moyenne des pixels
    return np.array(intensities), len(intensities)

# Charger les deux expériences
intensity1, n1 = load_experiment(subfolders[0])
intensity2, n2 = load_experiment(subfolders[1])

# Normaliser par le maximum de chaque expérience
C1 = intensity1-intensity1[0]
C2 = intensity2-intensity2[0]

# Générer temps (supposons images espacées uniformément)
t1 = np.arange(n1)
t2 = np.arange(n2)

# Tracé semi-log
plt.figure(figsize=(8,6))
#plt.semilogy(t1, C1, marker="o", label=f"Expérience 1 ({os.path.basename(subfolders[0])})",linestyle='None')
#plt.semilogy(t2, C2, marker="s", label=f"Expérience 2 ({os.path.basename(subfolders[1])})",linestyle='None')
plt.loglog(t1, C1, marker="o", label=f"Expérience 1 ({os.path.basename(subfolders[0])})",linestyle='None')
plt.loglog(t2, C2, marker="s", label=f"Expérience 2 ({os.path.basename(subfolders[1])})",linestyle='None')

plt.xlabel("Temps (frames)")
plt.ylabel("C-C₀ (Intensité normalisée)")
plt.title("Breakthrough curves semi-log")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
