import os
import numpy as np
from tifffile import imread
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


def circular_roi_mask(shape, ratio=0.75):
    ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny / 2, nx / 2
    r = 0.5 * ratio * min(nx, ny)
    return (x - cx)**2 + (y - cy)**2 <= r**2

def estimate_porosity_two_phases(folder, n_samples=50, roi_ratio=0.75,
                                 show_example=True, filter_size=3):
    """
    Segmentation 2 phases : billes (sombres) vs matrice sableuse.
    Calcule la porosité comme la fraction non-billes.

    Paramètres :
        folder : str - dossier contenant les .tif
        n_samples : int - nombre de slices utilisées
        roi_ratio : float - diamètre du cercle / taille de l’image
        show_example : bool - affiche segmentation
        filter_size : int - taille du filtre médian (anti-bruit)
    """
    tif_files = sorted([os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.lower().endswith('.tif')])
    if len(tif_files) == 0:
        raise FileNotFoundError(f"Aucune image .tif trouvée dans {folder}")

    # --- distribute samples evenly across all slices ---
    total_files = len(tif_files)
    if n_samples >= total_files:
        sampled_files = tif_files
    else:
        indices = np.linspace(0, total_files - 1, n_samples, dtype=int)
        sampled_files = [tif_files[i] for i in indices]

    imgs = []
    mask = None
    for f in sampled_files:
        img = imread(f).astype(np.float32)
        if img.ndim > 2:
            img = img[..., 0]
        img = median_filter(img, size=filter_size)
        if mask is None:
            mask = circular_roi_mask(img.shape, roi_ratio)
        imgs.append(img)
    imgs = np.array(imgs)

    # Application du masque sur toutes les slices
    flat = np.stack([img[mask] for img in imgs]).ravel()

    # --- Segmentation en 2 classes ---
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
    kmeans.fit(flat.reshape(-1, 1))
    centers = np.sort(kmeans.cluster_centers_.flatten())
    thresh = (centers[0] + centers[1]) / 2

    # La phase la plus sombre = billes
    billes_mask = flat < thresh
    f_billes = np.sum(billes_mask) / len(flat)
    f_sable = 1 - f_billes
    phi = f_sable  # porosité macroscopique = fraction hors billes

    print(f"[RESULT] Fraction de billes = {f_billes:.3f}")
    print(f"[RESULT] Porosité (hors billes) φ = {phi:.3f}")

    if show_example:
        example = imgs[len(imgs)//2]
        seg = np.zeros_like(example, dtype=np.uint8)
        seg[mask & (example < thresh)] = 0  # billes
        seg[mask & (example >= thresh)] = 1  # sable

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(example, cmap='gray')
        plt.title("Slice originale")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(seg, cmap='gray')
        plt.title("Segmentation : 0=billes, 1=sable")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return {"billes": f_billes, "sable": f_sable}, phi


if __name__ == '__main__':
    fractions, phi = estimate_porosity_two_phases(
        folder="/media/chorus/T7/tomo/20250707_COLONNE_PETITJEAN_01_SlicesY/",
        n_samples=100,
        roi_ratio=0.8,
        show_example=True
    )