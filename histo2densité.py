import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob


def find_extracted_images(directory):
    image_files = sorted(glob(os.path.join(directory, '*.tif'), recursive=True))
    return image_files if image_files else None


def calculate_probability_density(image):
    pixel_values = image.flatten()
    mean = np.mean(pixel_values)
    std_dev = np.std(pixel_values)
    normalized_values = np.log((pixel_values - mean) / std_dev)
    log_bins = np.logspace(np.log10(0.9), np.log10(10), num=2000)
    density, bins = np.histogram(normalized_values, bins=log_bins, density=True)
    return density, bins


def f(x, c=1):
    return c * np.exp(-x ** 2)


def plot_averaged_histogram(images, frame_number, n):
    cumulative_density = None
    cumulative_bins = None
    num_valid_images = 0

    for image_path in images:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image
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
    y_vals = f(bin_centers, c=np.max(averaged_density))
    valid_indices = (y_vals > 0) & (np.abs(bin_centers) > 0)
    y_vals = y_vals[valid_indices]
    averaged_density = averaged_density[valid_indices]
    dx_dy = 1 / (2 * np.abs(bin_centers[valid_indices]) * y_vals)
    p_y_exp = averaged_density * dx_dy
    valid_prob_indices = np.isfinite(p_y_exp) & (p_y_exp > 0)
    y_vals = y_vals[valid_prob_indices]
    p_y_exp = p_y_exp[valid_prob_indices]

    plt.figure(figsize=(8, 6))
    plt.loglog(y_vals, p_y_exp, color='blue', label=f'Averaged over {n} frames')
    plt.xlabel('log(y)')
    plt.ylabel('log(p(y))')
    plt.title(f'frame number {frame_number} to {n+int(frame_number)}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'processed_images/averaged_histogram_{frame_number:03d}.png')
    plt.show()


def process_images(directory, n=5):
    images = find_extracted_images(directory + "extracted_images/")


    if not images:
        print("Aucune image trouvée dans extracted_images.")
        return
    os.makedirs('video/processed_images', exist_ok=True)

    for idx in range(0, len(images), n):
        frame_group = images[idx:idx + n]
        print(f"Traitement des images {idx} à {idx + len(frame_group) - 1}")
        plot_averaged_histogram(frame_group, idx, n)
    create_video('processed_images', 'video_output.mp4')


def create_video(image_folder, output_video):
    images = sorted(glob(f'{image_folder}/averaged_histogram_*.png'))
    if not images:
        print('Aucune image pour la vidéo. Annulation. ')
        return
    frame = cv2.imread(images[0])
    return frame
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 5, (width, height))
    for image in images:
        video.write(cv2.imread(image))
    video.release()
    print(f'Vidéo sauvegardée sous {output_video}')


if __name__ == '__main__':

    folder='/home/chorus/exp/heterogenous/ppull_sablefin_bille_1cm_09-12_3/'
    #folder="/home/chorus/expjuillet/11_07_1"
    A=process_images(folder, n=1)

