import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit
global mass

# Échelle spatiale (0.11 mm/px = 0.11 * 10⁻³ m/px)

from proba_rayon import subtract_annular_background, process_jpg_images, find_extracted_images, find_metadata


def find_hdf5_file(folder):
    for file in os.listdir(folder):
        if file.endswith('.h5'):
            return os.path.join(folder, file)
    return None


def load_hdf5_images(hdf5_path, nframe=None):
    with h5py.File(hdf5_path, 'r') as f:
        dataset_name = list(f.keys())[0]
        dataset = f[dataset_name]
        return dataset[nframe].squeeze() if nframe is not None else dataset[:].squeeze()


def compute_center_of_mass(image):
    height, width = image.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    total_intensity = np.sum(image)
    if total_intensity == 0:
        return width // 2, height // 2
    x_mean = np.sum(x_indices * image) / total_intensity
    y_mean = np.sum(y_indices * image) / total_intensity
    return int(x_mean), int(y_mean)


def compute_radial_profile(image, center, dr=10):
    height, width = image.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    distances = np.sqrt((x_indices - center[0]) ** 2 + (y_indices - center[1]) ** 2)
    scale = 0.060 / height
    max_radius = int(np.max(distances))
    radial_profile = np.zeros(max_radius, dtype=np.float32)
    counts = np.zeros(max_radius, dtype=np.int32)
    mass=0
    for r in range(max_radius):
        mask = (distances >= r) & (distances < r + dr)
        counts[r] = np.sum(mask)
        if counts[r] !=0:

            radial_profile[r] = np.sum(image[mask])/counts[r]

            mass+=radial_profile[r]
        else : radial_profile[r] = 0.0


    radial_profile /= mass


    radii_m = np.arange(max_radius) * scale/(0.055/2)
    return radii_m, radial_profile


def quadratic_fit(x, a, b):
    return a * x ** 2 + b


def compare_hdf5_images(folder1, folder2, n,threshold=8*10**-7):
    hdf5_path1 = find_hdf5_file(folder1)
    hdf5_path2 = find_hdf5_file(folder2)
    if not hdf5_path1 or not hdf5_path2:
        print("Fichiers HDF5 non trouvés.")
        return

    images1 = load_hdf5_images(hdf5_path1, nframe=n)
    images2 = load_hdf5_images(hdf5_path2, nframe=n)

    img1 =np.squeeze( subtract_annular_background(np.array([images1])))
    img2 = np.squeeze(subtract_annular_background(np.array([images2])))

    plt.imshow(img1, cmap='viridis')
    plt.title('Image 1')
    plt.show()
    plt.imshow(img2, cmap='viridis')
    plt.title('Image 2')
    plt.show()

    center1 = compute_center_of_mass(img1)
    center2 = compute_center_of_mass(img2)

    radii1, profile1 = compute_radial_profile(img1, center1)
    radii2, profile2 = compute_radial_profile(img2, center2)
    plt.plot(radii1, profile1, linestyle='--', label='Cas hétérogène')
    plt.plot(radii2, profile2, label='Cas homogène')
    invalid_indices = np.where(profile1<threshold)[0]
   #print(np.where(profile1 >threshold)[0])
    if len(invalid_indices) > 0:
        last_valid_index = invalid_indices[0]
        radii_fit = radii1[:last_valid_index]
        profile_fit = profile1[:last_valid_index]
    else:
        radii_fit = radii1
        profile_fit = radii1
    valid_indices = np.isfinite(np.log(profile_fit))  # Find indices where log(profile_fit) is valid
    popt, _ = curve_fit(quadratic_fit, radii_fit[valid_indices], np.log(profile_fit[valid_indices]))
    integrate = np.trapz(profile1)
    print('valeur intégrale  :  ', integrate)
    fit_x = np.linspace(0, max(radii_fit), 100)
    fit_y = np.exp(quadratic_fit(fit_x, *popt))
    residuals = np.log(profile_fit) - quadratic_fit(radii_fit, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.log(profile_fit) - np.mean(np.log(profile_fit))) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.plot(fit_x, fit_y, color='r',linestyle=':', label=f'Fit -x², f = {popt[0]:.1f}*x² + {popt[1]:.1f}, R² = {r_squared:.3f}')
    plt.xlim(0,radii1[last_valid_index])
    a=popt[0]

    D=0.25*(1/(n*(1/2.5)*abs(a)*(0.055/2)**2))
    print(f'D : {D} ')
    u =  0.0003507
    alpha = D/u
    print(f'alpha : {alpha}')
    plt.legend()
    plt.xlabel('r/d')
    plt.ylabel('c/integrale(c)')
    plt.yscale('log')
    plt.show()

    return (popt, _)
import gc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quadratic(x, a, c):
    return a * x**2 + c


def computeD(folder, meta, threshold=3.7*10 ** -4):
    hdf5_path1 = find_hdf5_file(folder)
    tscale = 1 / (0.001 * meta["intervalle_ms"])

    if not hdf5_path1:
        print("Fichiers HDF5 non trouvés.")
        return

    images1 = load_hdf5_images(hdf5_path1)
    images1 = images1[31:101]
    D = []
    D_err = []  # Stockage des erreurs sur D
    t = []

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.colormaps.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=len(images1) * tscale)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    rmin = 1
    for i in range(len(images1)):
        if i % 15 == 1:
            print(i)
            img1 = images1[i]
            #img1 = np.squeeze(subtract_annular_background(np.array([img1])))

            temps = i * tscale
            t.append(temps)
            center1 = compute_center_of_mass(img1)

            radii1, profile1 = compute_radial_profile(img1, center1)
            invalid_indices = np.where(profile1 < threshold)[0]

            if len(invalid_indices) > 0:
                last_valid_index = invalid_indices[0]
                radii_fit = radii1[:last_valid_index]
                profile_fit = profile1[:last_valid_index]
            else:
                radii_fit = radii1
                profile_fit = profile1

            valid_indices = np.isfinite(np.log(profile_fit))
            popt, pcov = curve_fit(quadratic, radii_fit[valid_indices], np.log(profile_fit[valid_indices]))
            fit_x = np.linspace(0, max(radii_fit), len(valid_indices))
            color = cmap(norm(temps))
            ax.plot(radii_fit, profile_fit, color=color, label=f't={temps:.1f}')
            ax.set_yscale('log')

            # Calcul du R²
            y_pred = quadratic(fit_x, *popt)
            ylog = np.log(profile_fit)
            ss_res = np.sum((ylog - y_pred) ** 2)
            ss_tot = np.sum((ylog - np.mean(ylog)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmin = min(r_squared, rmin)

            a = popt[0]
            a_err = np.sqrt(pcov[0, 0])  # Incertitude sur a

            Diff = 1/a
            D.append(Diff)

            Diff_err = a_err  # Propagation de l'erreur
            D_err.append(Diff_err)

    print("rmin = ", rmin)
    fig.colorbar(sm, ax=ax, label='Temps')
    ax.set_ylabel('c/integrale(c)')
    ax.set_xlabel('r/rmax')
    plt.show()
    return t, D, D_err, rmin


def fit(B):

    t, D, D_err,rmin = B
    s0=np.abs(D[0])
    log_t = np.log(t)
    log_D = np.log(np.abs(D)-s0)
    log_t=log_t[2:]
    log_D=log_D[2:]
    print("log T" , log_t)
    print("LOG D",log_D)
    coeffs, cov = np.polyfit(log_t, log_D, 1, cov=True)
    a, b = coeffs
    print(coeffs)
    D= b/4
    b_err = np.sqrt(cov[1, 1])  # Erreur sur b
    plt.plot(log_t, log_D)
    plt.show()
    D_fit =np.exp(2*b)/4
    D_fit_err = D_fit * b_err  # Propagation de l'erreur
      # Propagation
    return D_fit, D_fit_err

def process_all_folders(base_folder):
    L = []
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            print('TRAITEMENT DU FOLDER : ',folder)
            meta = find_metadata(folder_path)
            B = computeD(folder_path, meta)
            D_fit , D_fit_erre= fit(B)
            L.append([D_fit, D_fit_erre , meta['debit'], meta['d']])
        gc.collect()
    return np.array(L)

if __name__ == '__main__':
    C = process_all_folders('/home/chorus/exp/heterogenous')
    D = process_all_folders('/home/chorus/exp/Homogenous')


    CT = np.transpose(C)

    DT = np.transpose(D)

    plt.errorbar(DT[2], DT[0] / (DT[2] * DT[3] *0.001* 0.00035 / 50), yerr=DT[1], fmt='ro', label='homogène')
    plt.errorbar(CT[2], CT[0] / (CT[2] * CT[3] *0.001* 0.00035 / 50), yerr=CT[1], fmt='bx', label='hétérogène')
    plt.xlabel("débit (g/min)")
    plt.ylabel(r"$\frac{\alpha}{d}$")
    plt.legend()
    plt.ylim()
    plt.show()
