import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os
import cv2

# root_folder = "/home/chorus/exp_adrien_nv_5/"
root_folder = "/home/chorus/exp_18_12_2/"


# root_folder = '/home/chorus/exp_17_11_2/'

dt = 2
ta = 35
dx = 0.06 / 2048

def load_full_sequence(folder_path):
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tif")]
    tif_files = sorted(tif_files, key=lambda x: int(os.path.splitext(x)[0]))  # <── OBLIGATOIRE

    stack = []
    for f in tif_files:
        img = skimage.io.imread(os.path.join(folder_path, f))
        stack.append(np.float32(img) / (2**16 - 1))  # ou img / 65535 si tu veux
    return np.array(stack)

subfolders = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
subfolder = subfolders[0]

I_full = load_full_sequence(subfolder)
means = I_full.mean(axis=(1,2))
best_frame = I_full[np.argmax(means)]
best_frame_uint8 = np.uint8(255 * (best_frame - best_frame.min()) / (best_frame.max() - best_frame.min()))

cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select ROI", 1200, 800)
roi = cv2.selectROI("Select ROI", best_frame_uint8, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()
x, y, w, h = roi

I = I_full[:, y:y+h, x:x+w]

# mask1 = np.swapaxes(np.swapaxes(np.tile((I[0]>0.97*np.nanmedian(I[0,:,:])).reshape(I.shape[1],I.shape[2],1),I.shape[0]),0,2),1,2)
# mask2 = np.swapaxes(np.swapaxes(np.tile((I[123]>0.002).reshape(I.shape[1],I.shape[2],1),I.shape[0]),0,2),1,2)
# I[mask1*mask2]=np.nan
# A = np.sum(mask1[0]*mask2[0])*dx**2

A=w*h*dx**2

# plt.imshow(I[0,:,:] > eps)
# plt.show()

#I[I < eps] = np.nan
# c0 = np.nanmax(I)
# I = I / c0
#
# plt.figure()
# plt.imshow(I[40,:,:])
# plt.colorbar()
# plt.show()


var = np.nanvar(I.reshape(I.shape[0], -1), axis=1)
mean = np.nanmean(I.reshape(I.shape[0], -1), axis=1)
time = np.arange(I.shape[0]) * dt / ta

m = -1
idx0 = 60
x0 = time[idx0]
##loglog
y0 = var[idx0] / (A * mean[idx0]**2)
b = np.log10(y0) - m * np.log10(x0)
D = 10 ** (-b)
fit_y = (10**b) * (time**m)
##semi-log
b = np.log(y0) - m * x0    # intercept in ln-space
D = np.exp(b)              # convert intercept back to linear scale

# Fitted curve
fit_y_semi = D * np.exp(m * time)
plt.figure()
#plt.plot(time, fit_y_semi, 'p--', label=f'exp fit : slope={m}')
plt.plot(time, fit_y, 'r--', label=f"fit: y = 10^{b:.2f} · t^{m}")
plt.plot(time, var / (A*mean**2), "bo",ms=4, label="$\\sigma_c^2 / \\mu_c^2 / A$")
plt.plot(time, 10**6*mean, 'gx',ms=3, label="$\\mu_c$")
plt.yscale("log")
plt.xscale("log")

plt.xlabel("$t/t_a$")
#plt.plot(time, 4e5 / time**2, "k-", label="$t^{-2}$")
plt.legend()
plt.show()