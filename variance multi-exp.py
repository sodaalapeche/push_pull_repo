import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # for reading images
def plot_semi_log_slope(ax, start_x, start_y, slope=-0.06, length=10, offset_factor=1.2, **kwargs):
    """
    Ajoute une droite de pente donnée en semi-log (y log, x linéaire).

    Args:
        ax : matplotlib axes
        start_x (float) : point de départ en x
        start_y (float) : point de départ en y (sera multiplié par offset_factor)
        slope (float) : pente dans l'espace semi-log (ex: -0.06)
        length (float) : intervalle en x sur lequel tracer la droite
        offset_factor (float) : décalage vertical multiplicatif
        **kwargs : arguments pour ax.plot()
    """
    # Plage en x (linéaire)
    x_vals = np.linspace(start_x, start_x + length, 200)

    # Courbe exponentielle correspondant à la pente demandée
    y_vals = start_y * offset_factor * np.exp(slope * (x_vals - start_x))

    ax.plot(x_vals, y_vals,linestyle='dashed', label=fr"$slope = {slope}$", **kwargs)

    # Étiquette à la fin de la courbe
    ax.text(
        x_vals[-1]*0.4, y_vals[-1],
        fr"slope = {slope}",
        fontsize="x-large", ha="left", va="top"
    )

    return ax

# Base folder
base_path = "/media/chorus/T7/inj simple/"
base_path = "/home/chorus/exp_septoct/push_only/"
# Detect experiments (all subfolders in base_path)
experiments = sorted([
    f for f in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, f))
])
# Define the series to process
series = ["serie1", "serie2", "serie3"]
series = ["serie1"]

# Dictionary to store variances
variances = {serie: [] for serie in series}

# Short labels (last 10 chars of folder name)
exp_labels = [7,16,30,43,55]

for exp in experiments:
    exp_path = os.path.join(base_path, exp)
    for serie in series:
        serie_path = os.path.join(exp_path, serie)
        if not os.path.isdir(serie_path):
            print(f"Missing {serie_path}, skipping...")
            continue

        # List images in folder
        images = sorted(
            [f for f in os.listdir(serie_path) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))],
            key=lambda x: int(os.path.splitext(x)[0])  # sort by numeric part
        )

        if not images:
            print(f"No images found in {serie_path}")
            continue

        # Take the last image
        last_image_path = os.path.join(serie_path, images[-4])
        img = imageio.imread(last_image_path)

        pixels = img[img > 0].astype(np.float64)

        # Compute second moment
        second_moment = np.mean(pixels)
        #second_moment = np.var(pixels)/np.mean(pixels)
        # second_moment = np.mean(pixels)

        print(str(second_moment)+" : " +str(exp[-9:])+"......"+str(serie))
        variances[serie].append(second_moment)
# Plot results: 3 different figures
fig,ax=plt.subplots()

for i in range(len(series)):
    #variances[series[i]]=variances[series[i]]/(variances[series[i]][0])
    series2 = ["Pé=210, 2mm", "Pé=90, 4mm", "Pé=90, 2mm"]
    series2= ["inj 0.5 débit 80"]

    plt.plot(exp_labels, variances[series[i]], marker="x",linestyle='None',label=series2[i],markersize=8)
    # plt.xlabel("distance d'injection (cm)")
    # if serie=="serie1":
    #     plt.title(f"Pe : 210; taille injection : 2mm")
    # if serie=="serie2":
    #     plt.title(f"Pe : 90; taille injection : 4mm")
    # if serie=="serie3":
    #     plt.title(f"Pe : 90; taille injection : 2mm")
#plt.ylim(bottom=0)
#plt.ylabel(r"$\frac{\sigma_{c}^{2}}{\sigma_{1}^{2}}$", fontsize=14,fontweight="bold")
plt.ylabel(r"$\frac{\sigma_{c}^{2}}{\mu_c}$", fontsize=14,fontweight="bold")
plt.ylabel(r"$\mu_c$", fontsize=14,fontweight="bold")
# plt.yscale("log")
#plt.xscale("log")
#plot_semi_log_slope(ax,5,10000,-0.06,40)
plt.xlabel(r"$d_{inj}$")
#plt.xlim([10,100])
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()