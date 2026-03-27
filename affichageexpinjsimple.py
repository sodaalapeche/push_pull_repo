import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import re
# Chemin racine
root_dir = "/home/chorus/exp_septoct/push_only/"


def sort_key_serie(name):
    """Trie serie1 < serie2 < serie3 numériquement."""
    match = re.search(r"serie(\d+)", name.lower())
    if match:
        return int(match.group(1))
    return 999

# Récupérer les chemins des images choisies
data = {}  # exp -> [img_serie1, img_serie2, img_serie3]
for exp in sorted(os.listdir(root_dir)):
    exp_path = os.path.join(root_dir, exp)
    if os.path.isdir(exp_path):
        series = sorted([s for s in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, s))],
                        key=sort_key_serie)
        imgs = []
        for serie in series:
            serie_path = os.path.join(exp_path, serie)
            images = sorted([f for f in os.listdir(serie_path) if f.lower().endswith(".tif")],
                            key=lambda x: int(os.path.splitext(x)[0]))
            if len(images) >= 20:
                img_name = images[-20]
                img_path = os.path.join(serie_path, img_name)
                imgs.append(tifffile.imread(img_path))
        if imgs:
            data[exp] = np.array(imgs)

all_pixels = np.concatenate([img.ravel() for imgs in data.values() for img in imgs])
vmin, vmax = np.percentile(all_pixels, [20, 99.8])

col_titles = ["Pé = 210, Inj size = 2mm", "Pé = 90, injec size = 4mm", "Pé = 90, injec size = 2mm"]
row_titles = ["inj point 1 \n ~16cm", "inj point 2 \n ~30cm ", "inj point  3 \n ~43cm", "inj point  4 \n ~55cm"]

# Affichage
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

for row, (exp, imgs) in enumerate(zip(sorted(data.keys()), data.values())):
    for col, img in enumerate(imgs):
        ax = axes[row, col]
        ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.axis("off")
        if row == 0:
            ax.set_title(col_titles[col], fontsize=12, pad=10)

# Ajouter titres de lignes dans la marge gauche
for row, title in enumerate(row_titles):
    fig.text(0.04, 0.89 - row*0.23, title, va="center", ha="right", fontsize=16, rotation=90)

plt.tight_layout(rect=[0.08, 0, 1, 1])  # laisser de la place à gauche
plt.show()