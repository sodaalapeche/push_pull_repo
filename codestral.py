import os
from mistralai import Mistral

api_key = 'cyLssFUBjNRV1FgGMJzJz2An2P61QgGe'

client = Mistral(api_key=api_key)

model = "codestral-latest"

message = [
    {
        "role": "user",
        "content":""" Here is my code. can you modify it so that it can open .ome.h5 files, with the h5py library ? Also, I need it to average values of the histogram over 5 frames. Can you do it ?
        
        No need to create a video, just plot the images
        
import os
        
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from glob import glob
import cv2
import shutil
import math

def find_ome_H5_file(directory):
    tif_files = sorted(glob(os.path.join(directory, '**', '*.ome.tif'), recursive=True))
    if not tif_files:
        return None
    return tif_files

def calculate_probability_density(image):
    pixel_values = image.flatten()
    mean = np.mean(pixel_values)
    std_dev = np.std(pixel_values)
    normalized_values = np.log((pixel_values - mean) / std_dev)
    log_bins = np.logspace(np.log10(0.9), np.log10(10), num=2000)
    density, bins = np.histogram(normalized_values, bins=log_bins, density=True)
    return density, bins
def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))
def linear_regression(log_y, log_p_y):
    # Convert inputs to numpy arrays for ease of calculation
    x = np.array(log_y)
    y = np.array(log_p_y)

    # Number of data points
    N = len(x)

    # Calculate the necessary sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)

    # Calculate the slope (m) and intercept (b)
    m = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / N

    # Predicted values based on the linear model
    y_pred = m * x + b

    # Calculate residuals (errors)
    residuals = y - y_pred

    # Calculate R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    # Calculate standard error (SE)
    # Standard error of the regression (std_err) formula
    std_err = np.sqrt(ss_residual / (N - 2))

    # Calculate the t-statistic (slope / SE of slope)
    se_slope = std_err / np.sqrt(np.sum((x - np.mean(x)) ** 2))
    t_statistic = m / se_slope

    # Approximate p-value based on t-distribution for large N
    # For a rough estimate, we use the t-distribution formula:
    # p_value = 2 * (1 - CDF(t_statistic))
    # For large sample sizes, we can approximate p-value using a normal distribution:
    # p_value ≈ 2 * (1 - Normal CDF(t_statistic))
    # This approximation is generally acceptable for large N.
    p_value = 2 * (1 - normal_cdf(abs(t_statistic)))

    return m, b, r_squared, p_value, std_err

def f(x, c=1):
    return c * np.exp(-x ** 2)


# Fonction pour afficher l'image et l'histogramme transformé
def plot_image_and_histogram(image, frame_number):
    pixelvalues = image
    mean = pixelvalues.mean()
    std_dev = pixelvalues.std()
    normalized_values = (pixelvalues - mean) / std_dev

    plt.figure(figsize=(12, 6))

    # Image plot
    plt.subplot(1, 2, 1)
    plt.imshow(normalized_values, cmap='viridis', clim=[-2, 3])
    plt.colorbar()
    plt.title(f'Frame {frame_number}')
    plt.axis('off')

    # Histogram plot with log-log scale
    density, bins = calculate_probability_density(image)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    y_vals = f(bin_centers,c=np.max(density))

    # Éviter les divisions par zéro ou infinis
    valid_indices = (y_vals > 0) & (np.abs(bin_centers) > 0)
    y_vals = y_vals[valid_indices]
    density = density[valid_indices]

    dx_dy = 1 / (2 * np.abs(bin_centers[valid_indices]) * y_vals)
    p_y_exp = density * dx_dy

    valid_prob_indices = np.isfinite(p_y_exp) & (p_y_exp > 0)
    y_vals = y_vals[valid_prob_indices]
    p_y_exp = p_y_exp[valid_prob_indices]
    slope, intercept, r_value, p_value, std_err = linear_regression(y_vals, p_y_exp)
    print(slope)

    plt.subplot(1, 2, 2)
    plt.loglog(y_vals, p_y_exp, color='green', label='Densité transformée')
    plt.xlabel('log(y)')
    plt.ylabel('log(p(y))')
    plt.title('Densité de probabilité transformée')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def generate_gaussian_circle(image_size, center=None, radius=None):
    if center is None:
        center = (image_size[0] // 2, image_size[1] // 2)  # Center of the image
    if radius is None:
        radius = min(image_size) // 14 # Use a quarter of the smallest dimension as the radius

    x = np.linspace(0, image_size[1] - 1, image_size[1])  # X-axis
    y = np.linspace(0, image_size[0] - 1, image_size[0])  # Y-axis
    X, Y = np.meshgrid(x, y)  # Create grid of coordinates

    # Calculate the Gaussian function
    gaussian_circle = np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * radius ** 2))

    return gaussian_circle


def test_with_gaussian_circle():
    np.random.seed()  # Optional: for reproducibility
    image_size = (2000, 2000)  # Example image size (200x200)
    gaussian_circle = generate_gaussian_circle(image_size)
    plot_image_and_histogram(gaussian_circle, frame_number=0)
# Fonction de test avec une distribution gaussienne
def test_with_normal_distribution():
    np.random.seed()
    gaussian_data = np.random.normal(loc=0, scale=1, size=(2000, 2000))  # Génère une image 100x100
    plot_image_and_histogram(gaussian_data, frame_number=0)
def main():
    global J
    directory = '/home/chorus/exp/'
    #ppull_sablefin_bille_1cm_09-12_2/
    ome_tif_files = find_ome_H5_file(directory)
    if ome_tif_files is None:
        print('No .ome.tif file found in the specified directory.')
        return

    output_folder = '/home/chorus/video'
    folder = output_folder + '/'

    for i in ome_tif_files:
        print('treating video  '+str(i))
        J = str(i)
        print('Deleting images of previous videos')
        for filename in os.listdir(folder):

            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        p = []
        images = imread(i, is_mmstack=False)
        scaled_images = [cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA) for img
                         in images]

        images = np.array(scaled_images)
        height, width = images[0].shape
        radius = 0.45 * width
        center_x, center_y = width // 2, height // 2
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        distance_to_center = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        circular_mask = distance_to_center <= radius
        inner_radius = 0.34 * width
        outer_radius = 0.4* width
        annular_mask = (distance_to_center >= inner_radius) & (distance_to_center <= outer_radius)

        for image in images:
            # Calcul de l'intensité moyenne dans l'anneau
            annular_intensity = np.mean(image[annular_mask])
            masked_image = image - annular_intensity
            masked_image[masked_image < 0] = 0
            total_intensity = np.sum(masked_image)
            if total_intensity > 0:
                P = masked_image
            else:
                P = np.zeros_like(masked_image)

            p.append(P)
            background = images[0]  # Garde le fond de la première image

        filtered_images = np.array(p)

        for j in range(0, len(filtered_images), 5):
            print(f'Image n°{j} is being treated')
            plot_image_and_histogram(filtered_images[j], j)
            plt.savefig(f'{output_folder}/image_{j:03d}.png')
            plt.close()

        images = sorted(glob(f'{output_folder}/image_*.png'))

        if not images:
            print('No images found in the output folder.')
            return

        frame = cv2.imread(images[0])
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f'/home/chorus/video_histo_'+J[-30::]+'.mp4', fourcc, 5, (width, height))

        for image in images:
            video.write(cv2.imread(image))

        video.release()
        cv2.destroyAllWindows()
        print(f'Video saved to {output_folder}/video_histo.mp4')

if __name__ == '__main__':
    test_with_gaussian_circle()
    #main()"""
    }]
chat_response = client.chat.complete(
    model=model,
    messages=message
)
print(chat_response.choices[0].message.content)