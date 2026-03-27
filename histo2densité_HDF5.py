import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from glob import glob

def find_hdf5_file(directory):
    hdf5_files = glob(os.path.join(directory, '*.h5'))
    return hdf5_files[0] if hdf5_files else None

def read_hdf5_images(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        dataset_name = list(f.keys())[0]  # Assuming first dataset is the image sequence
        images = f[dataset_name][:]  # Load images as numpy array
    return images  # Returns a 3D array (frames, height, width)
import math


def calculate_probability_density(image):
    cmax = np.max(image)
    threshold = 0.02 * cmax
    filtered_image = np.where(image > threshold, image, 1)
    #filtered_image = np.where(filtered_image > 0.95*cmax, 0, filtered_image)
    filtered_pixels = filtered_image.flatten().astype(np.float64)

    if filtered_pixels.size == 0:
        return None, None
    mean = np.mean(filtered_pixels)
    std_dev = np.std(filtered_pixels)
    normalized_values = filtered_pixels
    log_bins = np.logspace(-4, 1, num=400)

    density, bins = np.histogram(normalized_values, bins=log_bins, density=True)
    return density, bins


def f(x, c=1):
    return c * np.exp(-x ** 2)

def linear_fit(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    n = len(log_x)
    sum_x = np.sum(log_x)
    sum_y = np.sum(log_y)
    sum_xx = np.sum(log_x * log_x)
    sum_xy = np.sum(log_x * log_y)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

def plot_averaged_histogram(images, frame_number, n):
    cumulative_density = None
    cumulative_bins = None
    num_valid_images = 0

    for image in images:
        density, bins = calculate_probability_density(image)
        if cumulative_density is None:
            cumulative_density = density
            cumulative_bins = bins
        else:
            cumulative_density += density
        num_valid_images += 1

    if num_valid_images == 0:
        print("No valid images for averaging.")
        return

    averaged_density = cumulative_density / num_valid_images
    bin_centers = (cumulative_bins[:-1] + cumulative_bins[1:]) / 2
    #LES BINS SONT CONSERVES
    y_vals = f(bin_centers, c=np.max(averaged_density))
    valid_indices = (y_vals > 0) & (np.abs(bin_centers) > 0)
    y_vals = y_vals[valid_indices]
    averaged_density = averaged_density[valid_indices]
    dx_dy = 1 / (2 * np.abs(bin_centers[valid_indices]) * y_vals)
    p_y_exp = averaged_density * dx_dy
    valid_prob_indices = np.isfinite(p_y_exp) & (p_y_exp > 0)
    y_vals = y_vals[valid_prob_indices]
    p_y_exp = p_y_exp[valid_prob_indices]


    #FIT LINEAIRE
  #  low_indices = y_vals < np.percentile(y_vals, 10)  # Adjust as needed
   # slope, intercept = linear_fit(y_vals[low_indices], p_y_exp[low_indices])
    #fit_line = np.exp(intercept) * y_vals**slope

    plt.figure(figsize=(8, 6))
    plt.loglog(y_vals, p_y_exp, color='blue', label=f'Averaged over {n} frames')
    #plt.loglog(y_vals[low_indices], fit_line[low_indices], color='red', linestyle='--', label=f'Linear Fit (slope={slope:.2f})')
    plt.xlabel('y')
    plt.grid(True)
    plt.ylabel('p(y)')
    plt.title(f'Frame number {frame_number} to {n + frame_number}')
    plt.legend()
    #plt.xlim(left=0.000001,right=10**2)
    plt.tight_layout()
    os.makedirs('processed_images', exist_ok=True)
    plt.savefig(f'processed_images/averaged_histogram_{frame_number:03d}.png')
    plt.show()

def process_hdf5_images(directory, n=5):
    hdf5_path = find_hdf5_file(directory)
    if not hdf5_path:
        print("No HDF5 file found.")
        return

    images = read_hdf5_images(hdf5_path)
    num_frames = images.shape[0]

    for idx in range(0, num_frames, n):
        frame_group = images[idx:idx + n]
        print(f"Processing frames {idx} to {idx + len(frame_group) - 1}")
        plot_averaged_histogram(frame_group, idx, n)

def create_video(image_folder, output_video):
    images = sorted(glob(f'{image_folder}/averaged_histogram_*.png'))
    if not images:
        print('No images found for video. Aborting.')
        return
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 5, (width, height))
    for image in images:
        video.write(cv2.imread(image))
    video.release()
    print(f'Video saved as {output_video}')

import copy
def image1(folder,numero,a=1,b=10):
    hdf5_path = find_hdf5_file(folder)
    if not hdf5_path:
        print("No HDF5 file found.")
        return

    images = read_hdf5_images(hdf5_path)
    image=images[numero][0,0]

    cmax = np.max(image)

    filtered_pixels = image.flatten().astype(np.float64)
    #mean = np.mean(filtered_pixels)
    #y_values = filtered_pixels
    #log_y_values = np.log(np.abs(y_values) + 1e-8)
    #image_filtered = np.zeros_like(image, dtype=np.uint16)
    #mask = (log_y_values >= a) & (log_y_values <= b)
    #image_filtered[image > threshold] = image[image > threshold] * mask
    density, bins = calculate_probability_density(filtered_pixels)
    y_vals = f(bins, c=np.max(density))
    p_y_exp = density

    plt.loglog(y_vals, p_y_exp, color='blue')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="viridis")
    plt.title(f"Pixels où {a} ≤ log(y) ≤ {b} de l'image n° {numero}")
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    folder = '/home/chorus/exp/Homogenous/test/ppull_SABLE_homog_22_11_3/'

    #process_hdf5_images(folder, n=1)
    A=image1(folder,90)
