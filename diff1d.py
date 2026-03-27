import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Ubuntu'          # Police Ubuntu
plt.rcParams['axes.titlesize'] = 'x-large'      # Taille titre
plt.rcParams['axes.labelsize'] = 'large'        # Taille labels
plt.rcParams['legend.fontsize'] = 'large'       # Taille légende
plt.rcParams['xtick.labelsize'] = 'large'       # Taille ticks X
plt.rcParams['ytick.labelsize'] = 'large'       # Taille ticks Y
# ---------------------------------------------------
# PARAMÈTRES PHYSIQUES
# ---------------------------------------------------
D = 0.5          # diffusivité
A = 1.0          # amplitude initiale
sigma0 = 0.5     # sigma initial de la gaussienne
x0 = 0.0         # centre initial du pic

# ---------------------------------------------------
# ESPACE
# ---------------------------------------------------
x_min, x_max = -100, 100
nx = 4001
x = np.linspace(x_min, x_max, nx)

# ---------------------------------------------------
# TEMPS
# ---------------------------------------------------
t_max = 200
nt = 600
times = np.linspace(0, t_max, nt)

# ---------------------------------------------------
# ROI FIXE
# ---------------------------------------------------
a = -13.0    # borne gauche du ROI
b =  13.0    # borne droite du ROI
L = b - a   # largeur du ROI

# ---------------------------------------------------
# SOLUTION ANALYTIQUE DE LA DIFFUSION
# ---------------------------------------------------
def C_analytic(x, t):
    """Solution analytique : diffusion d'une gaussienne."""
    if t == 0:
        return A * np.exp(-(x-x0)**2 / (2*sigma0**2))
    s2 = sigma0**2 + 2*D*t
    pref = A * np.sqrt(sigma0**2 / s2)
    return pref * np.exp(-(x-x0)**2 / (2*s2))

# ---------------------------------------------------
# CALCUL : variance spatiale de C(x,t) dans le ROI fixe
# ---------------------------------------------------
variance_roi = np.zeros(nt)
mean_roi = np.zeros(nt)

mask = (x >= a) & (x <= b)

for i, t in enumerate(times):
    Cx = C_analytic(x, t)

    # moyenne spatiale dans le ROI
    mean = np.trapz(Cx[mask], x[mask]) / L
    mean_roi[i] = mean

    # variance spatiale E[C²] - (E[C])²
    Ex2 = np.trapz(Cx[mask]**2, x[mask]) / L
    variance_roi[i] = Ex2 - mean**2

# ---------------------------------------------------
# FIGURE 1 : Profils C(x,t) avec ROI fixe
# ---------------------------------------------------
plt.figure(figsize=(8,4))
for T in [t_max//10, t_max//5,t_max//2, t_max//1]:
    Cx = C_analytic(x, T)
    plt.plot(x, Cx, label=f"t={T}")

plt.axvspan(a, b, color="orange", alpha=0.08, label="ROI")
plt.xlabel("x")
plt.ylabel("c(x,t)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
idfit= (nt//4)
# ---------------------------------------------------
# FIGURE 2 : Variance dans ROI fixe
# ---------------------------------------------------
plt.figure(figsize=(8,4))
plt.plot(times, variance_roi,'b-', label="Variance in ROI")
plt.plot(times, mean_roi,'r--', label="total mass in ROI")
plt.xlabel(f"$t$")
plt.legend()
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.plot(times,variance_roi[0]/times**0.5,label="Analytical solution for 1D diffusion scalar variance")
plt.plot(times[idfit:], variance_roi[idfit]/times[idfit:]**2)
plt.tight_layout()
plt.show()