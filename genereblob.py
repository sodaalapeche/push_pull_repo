import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_gaussian_blob(size=2048, sigma=300):
    """
    Génère une image 16-bit en niveaux de gris avec un blob gaussien,
    en utilisant une solution analytique basée sur un meshgrid.
    """
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)

    # Calcul de la fonction gaussienne
    gaussian_blob = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    # Normalisation à la plage des entiers 16-bit
    gaussian_blob = (gaussian_blob / gaussian_blob.max() * np.iinfo(np.uint16).max).astype(np.uint16)

    return gaussian_blob
def display_image(image):
    """Affiche une image en niveaux de gris."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title("Blob Gaussien 16-bit")
    plt.show()
def histo(image):
    image = np.squeeze(image)
    ##fit log lin exponentiel:
    filtered_pixels = image.flatten().astype(np.float64) / 2 ** 16
    log_bins = np.logspace(-4, 1, num=400)
    density, bins = np.histogram(filtered_pixels, bins=log_bins, density=True)
    bins = bins[1:]
    plt.plot(bins, density)
    plt.yscale('log')
    plt.xlim([0,1.1])
    plt.show()

if __name__ == "__main__":
    img = generate_gaussian_blob()
    display_image(img)
    histo(img)