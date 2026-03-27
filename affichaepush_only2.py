import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# === Configuration ===
root_dir = "/home/chorus/exp_septoct/push_only/"
colormap = "magma"  # alternatives : "inferno", "plasma", "viridis"

# === Recherche de tous les sous-dossiers ===
exp_dirs = sorted(
    glob.glob(os.path.join(root_dir, "push_only_*"))
)

print("Sous-dossiers trouvés :")
for d in exp_dirs:
    print(" -", os.path.basename(d))

images = []
labels = []

for exp_dir in exp_dirs:
    serie_dir = os.path.join(exp_dir, "serie1")
    if not os.path.isdir(serie_dir):
        print(f"⚠️ Pas de dossier 'serie1' dans {exp_dir}")
        continue

    tif_files = sorted(glob.glob(os.path.join(serie_dir, "*.tif")))
    if not tif_files:
        print(f"⚠️ Aucune image .tif trouvée dans {serie_dir}")
        continue

    last_tif = tif_files[-1]
    print(f"→ {os.path.basename(exp_dir)} : {os.path.basename(last_tif)}")

    img = np.array(Image.open(last_tif))
    images.append(img)
    labels.append(os.path.basename(exp_dir))

# === Affichage côte à côte ===
n = len(images)
if n == 0:
    print("🚫 Aucune image à afficher.")
else:
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(np.log(img), cmap=colormap)
        ax.set_title(label, fontsize=9)
        ax.axis('off')

    plt.show()
