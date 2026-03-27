import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def f(x, c=1):
    return c * np.exp(-x ** 2)
def find_hdf5_file(directory):
    hdf5_files = glob(os.path.join(directory, '*.h5'))
    return hdf5_files[0] if hdf5_files else None

def read_hdf5_images(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        dataset_name = list(f.keys())[0]  # Assuming first dataset is the image sequence
        images = f[dataset_name][:]  # Load images as numpy array
    return images
directory = ('/home/chorus/exp/heterogenous/ppull_sablefin_bille_1cm_10-12_2')
hdf5_path = find_hdf5_file(directory)
images = read_hdf5_images(hdf5_path)
for i in range(len(images)):
    if i%10==0:
        image = np.squeeze(images[i])
        # cmax = np.max(image)
        # threshold = 0.02 * cmax
        # filtered_image = np.where(image > threshold, image, 1)
        filtered_pixels = image.flatten().astype(np.float64) / 2 ** 16
        log_bins = np.logspace(-4, 1, num=400)
        density, bins = np.histogram(filtered_pixels, bins=log_bins, density=True)
        bins = bins[1:]
        y_vals = f(bins, c=np.max(density))
        plt.plot(bins, density)
        plt.xlim(left=0.005)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
