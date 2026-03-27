import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- seuils manuels à ajuster ---
THRESHOLDS = [2000, 10000, 30000]  # 3 seuils → 4 classes

def histo_and_segment(folder_path, n, thresholds=THRESHOLDS):
    # --- Chargement image ---
    tif_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])
    image_path = os.path.join(folder_path, tif_files[n])
    image = np.array(Image.open(image_path), dtype=np.uint16)

    # --- PDF via np.unique ---
    pixels = image.flatten()
    unique_vals, counts = np.unique(pixels, return_counts=True)
    pdf = counts / counts.sum()

    mask = pdf > 0
    x = unique_vals[mask]
    y = pdf[mask]

    # Sous-échantillonnage pour log-log
    x = x[::2]
    y = y[::2]
    indices = np.unique(np.logspace(0, np.log10(len(x) - 1), num=500, dtype=int))
    x_sub = x[indices]
    y_sub = y[indices]

    # --- Segmentation par seuils manuels ---
    segmented = np.digitize(image, bins=thresholds)

    # --- Affichage ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # PDF avec seuils
    axs[0].loglog(x_sub, y_sub, marker='o', linestyle='dashed')  # <-- décommenté
    for t in thresholds:
        axs[0].axvline(t, color='r', linestyle='--', alpha=0.7)
    axs[0].set_title(f'Semi-Log PDF\nRange: {x.min()}-{x.max()}')
    axs[0].set_ylim([10**-8, 0.05])
    axs[0].set_xlim([60, 66000])
    axs[0].set_ylabel('pdf(c)')
    axs[0].set_xlabel('c')

    # Image segmentée
    axs[1].imshow(segmented, cmap='nipy_spectral')
    axs[1].set_title("Segmentation par seuils manuels")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    for i in range(1,100, 3):  # test sur 4 images
        histo_and_segment('/home/chorus/expjuillet/11_07_1/serie1/', i)
