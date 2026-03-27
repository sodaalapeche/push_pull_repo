import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# 1. Load 16-bit image
img = imageio.imread("/home/chorus/exp_18_12_2/serie1/40.tif")

# If RGB, convert to grayscale
if img.ndim == 3:
    img = img.mean(axis=2)

# 2. Convert to float and normalize
img = img.astype(np.float64)
img /= img.max()

# 3. Remove mean
import numpy as np

threshold = 0.5 * np.nanmedian(img)
mask = img >= threshold

img_filt = img.copy()
img_filt[~mask] = np.nan

# 4. Optional: apply a window (Hann)
ny, nx = img_filt.shape
window = np.outer(np.hanning(ny), np.hanning(nx))
img_filt *= window

mean_val = np.nanmean(img_filt)
img_filt = img_filt - mean_val

# Replace NaNs by zero
img_filt = np.nan_to_num(img_filt, nan=0.0)

# 5. 2D FFT
F = np.fft.fft2(img_filt)
F_shift = np.fft.fftshift(F)

# 6. Power spectrum
P = np.abs(F_shift)**2

# 7. Display log-spectrum
plt.figure(figsize=(6,6))
plt.imshow(np.log10(P + 1e-12), cmap="inferno")
plt.colorbar(label="log10 Power")
plt.title("2D Fourier Power Spectrum")
plt.axis("off")
plt.show()

# Create wavenumber grid
ky = np.fft.fftshift(np.fft.fftfreq(ny))
kx = np.fft.fftshift(np.fft.fftfreq(nx))
KX, KY = np.meshgrid(kx, ky)
K = np.sqrt(KX**2 + KY**2)
# Radial bins
nbins = 300
k_bins = np.linspace(0, K.max(), nbins)
P_radial = np.zeros(nbins-1)

for i in range(nbins-1):
    mask = (K >= k_bins[i]) & (K < k_bins[i+1])
    if np.any(mask):
        P_radial[i] = P[mask].mean()

k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
# Ignore very low-k (bulk scale)
k_min_cut = 3
k = k_centers[k_min_cut:]
Pk = P_radial[k_min_cut:]

# Normalize
Pk /= Pk.max()
from scipy.signal import find_peaks

# Find peaks
peaks, props = find_peaks(
    Pk,
    prominence=0.05,   # sensitivity
    distance=5         # avoid clustering
)

# Sort by peak strength
peak_strength = props["prominences"]
order = np.argsort(peak_strength)[::-1]

top_peaks = peaks[order][:5]  # top 5 frequencies
s = 0.6 / 2048  # physical size of one pixel

print("Top mixing scales:")
for i, p in enumerate(top_peaks):
    k_peak = k[p]               # cycles / pixel
    length_pixels = 1 / k_peak
    length_phys = length_pixels * s

    print(
        f"{i+1}: k = {k_peak:.4e} cyc/pix  |  "
        f"λ ≈ {length_pixels:.1f} px  |  "
        f"λ ≈ {length_phys:.4e} physical units"
    )

plt.figure()
plt.loglog(k, Pk, label="Spectrum")
plt.scatter(k[top_peaks], Pk[top_peaks], color="red", zorder=5, label="Peaks")
plt.xlabel("Spatial frequency k")
plt.ylabel("Normalized power")
plt.title("Dominant Mixing Frequencies")
plt.legend()
plt.grid(True, which="both")
plt.show()
