import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from matplotlib import cm
import gc
def find_hdf5_file(directory):
    hdf5_files = glob(os.path.join(directory, '*.h5'))
    return hdf5_files[0] if hdf5_files else None

def read_hdf5_images(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        dataset_name = list(f.keys())[0]
        images = f[dataset_name][:]
    return images

from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * np.power(x, b)

def fit_initial_slope(image, log_bins, fit_range=(0.0018, 0.01)):
    filtered_pixels = image.flatten().astype(np.float64) / 2 ** 16
    density, bins = np.histogram(filtered_pixels, bins=log_bins, density=True)
    bins = bins[1:]

    mask = (bins >= fit_range[0]) & (bins <= fit_range[1])
    x_fit = bins[mask]
    y_fit = density[mask]

    popt, _ = curve_fit(power_law, x_fit, y_fit, p0=(1, -1))
    a_fit, b_fit = popt

    start_bin = x_fit[0]
    start_density = y_fit[0]

    plt.figure(figsize=(8, 6))
    plt.plot(bins, density, label='Densité', color='blue')
    plt.plot(x_fit, power_law(x_fit, *popt), 'r--', label=f'pente={b_fit:.2f}')
    ref_line1 = start_density * np.power(bins / start_bin, -1)
    ref_line2 = start_density * np.power(bins / start_bin, -2)
    plt.plot(bins, ref_line1, 'g-.', label='Pente -1')
    plt.plot(bins, ref_line2, 'orange', linestyle='-.', label='Pente -2')
    plt.ylim(bottom=0.000001)
    plt.xlim(left=0.001, right=1.2)
    plt.yscale('log')
    plt.xlabel('Valeur du pixel')
    plt.ylabel('Densité de probabilité')
    plt.legend()
    plt.show()

    print(f"Fit terminé: a = {a_fit:.2e}, b = {b_fit:.2f}")
    return a_fit, b_fit

# Nouvelle fonction pour détecter le cutoff

def find_cutoff(density, bins, threshold=1e-4, range_cutoff=(0.01, 1)):
    mask = (bins >= range_cutoff[0]) & (bins <= range_cutoff[1])
    valid_bins = bins[mask]
    valid_density = density[mask]
    cutoff_idx = np.argmax(valid_density < threshold)
    return valid_bins[cutoff_idx] if cutoff_idx > 0 else None

# Ajout du traçage de l'évolution du cutoff

def plot_cutoff_evolution(images, log_bins, discr=2):
    cutoffs = []
    for i in range(0, len(images), discr):
        images=np.squeeze([i-10,i+10])

        filtered_pixels = images.flatten().astype(np.float64) / 2 ** 16

        density, bins = np.histogram(filtered_pixels, bins=log_bins, density=True)

        bins = bins[1:]

        cutoff = find_cutoff(density, bins)

        cutoffs.append(cutoff)

    plt.figure(figsize=(8, 6))
    plt.plot(range(0, len(images), discr), cutoffs, 'o', color='red')
    plt.ylabel('Cutoff (pixel value)')
    plt.title(f'{dir[-30::]}')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
def f(x, c=1):
    return c * np.exp(-x ** 2)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors




def cutoffshow(images, discr=20, a=0.05, b=0.223):
    cmap = plt.colormaps.get_cmap('plasma')
    frame_indices = np.arange(0, len(images), discr)  # Frames utilisées
    norm = mcolors.BoundaryNorm(boundaries=frame_indices, ncolors=cmap.N)

    for i in frame_indices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        image = np.squeeze(images[i])
        filtered_pixels = image.astype(np.float64) / image.max()
        mask_red = (filtered_pixels >= a) & (filtered_pixels <= b)
        mask_green = (filtered_pixels > b)
        gray_image = np.stack([filtered_pixels] * 3, axis=-1)
        highlight_red = np.zeros_like(gray_image)
        highlight_red[..., 0] = 1
        highlight_green = np.zeros_like(gray_image)
        highlight_green[..., 1] = 1
        alpha = 0.3
        highlighted_image = np.where(mask_red[..., None], alpha * gray_image + (1 - alpha) * highlight_red, gray_image)
        highlighted_image = np.where(mask_green[..., None], alpha * gray_image + (1 - alpha) * highlight_green, highlighted_image)

        axes[0].imshow(highlighted_image)
        axes[0].set_title(f"Image Frame {i} (Rouge: {a} ≤ x ≤ {b}, Vert: x > {b})")
        axes[0].axis("off")
        log_bins = np.logspace(-4, 1, num=400)
        density, bins = np.histogram(filtered_pixels.flatten(), bins=log_bins, density=True)
        bins = bins[1:]
        color = cmap(norm(i))
        axes[1].plot(bins, density, color=color)

        axes[1].set_xlim([0, 1.3])
        axes[1].set_xscale('linear')
        axes[1].set_yscale('log')
        axes[1].set_title(f"{j} - Frame {i}")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[1], label="Numéro de frame")
        cbar.set_ticks(frame_indices)
        cbar.set_ticklabels([str(i) for i in frame_indices])
        plt.show()

def exp(x,a,b):
    return b*np.exp(a*x**2)
def power_law(x,a,b):
    return a*x**b
def fit_cutoff(path,num,fit_range=(0.275,0.32)):
    hdf5_files = find_hdf5_file(path)
    images=read_hdf5_images(hdf5_files)
    image=np.squeeze(images[num:num+10])
    filtered_pixels = image.flatten().astype(np.float64) / 2 ** 16
    log_bins = np.logspace(-4, 1, num=1000)
    density, bins = np.histogram(filtered_pixels, bins=log_bins, density=True)
    bins = bins[1:]
    mask = (bins >= fit_range[0]) & (bins <= fit_range[1])
    plt.plot(bins, density)
    plt.yscale('log')
    plt.xlim(left=0.000001,right=0.7)
    x_fit = bins[mask]
    y_fit = density[mask]

    popt, _ = curve_fit(power_law, x_fit, y_fit, p0=(1, -1))
    a_fit, b_fit = popt
    plt.plot(x_fit,power_law(x_fit,a_fit,b_fit),color='r',linestyle='--',label=f'{b_fit:.1e}exp({a_fit:.2e}*x)')
    plt.legend()
    plt.show()


# Chargement et analyse des images
directory = '/home/chorus/exp/Homogenous/ppull_SABLE_homog_22_11_3/'
directory ='/home/chorus/exp/heterogenous/ppull_sablefin_bille_1cm_09-12_3/'
directory="/home/chorus/exp/Homogenous/test/"
directory="/home/chorus/exp/fin/"
for dirs in os.walk(directory):
    path=dirs[0]
    for j in dirs[1]:
        dir=os.path.join(path,j)
        print(f'treating exp : {j}')
        discr = 10
        hdf5_path = find_hdf5_file(dir)
        images = read_hdf5_images(hdf5_path)
        images=images[15:160]
        cutoffshow(images,discr)






#fit_cutoff(directory,50,fit_range=(0.1,0.3))