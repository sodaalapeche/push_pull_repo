import math

import numpy as np
from scipy.special import i0
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Solution exacte de Carslaw & Jaeger pour un disque initial
# ---------------------------------------------------------

def C_exact(r, t, a, C0, D=1.0, Nr_prime=300):
    """
    Concentration radiale C(r,t) pour un disque uniforme de rayon a,
    solution exacte en 2D.
    """
    if t <= 0:
        raise ValueError("t must be > 0")

    # Quadrature sur r'
    rprime = np.linspace(0, a, Nr_prime)
    exp_rp2 = np.exp(-rprime**2/(4*D*t))

    denom = 2*D*t
    prefactor = C0 / denom
    exp_r2 = np.exp(-r**2/(4*D*t))

    # Produit de Bessel
    arg = np.outer(r, rprime) / (2*D*t)
    I = i0(arg)

    integrand = I * exp_rp2[np.newaxis, :] * rprime[np.newaxis, :]
    integral = np.trapz(integrand, rprime, axis=1)

    return prefactor * exp_r2 * integral


# ---------------------------------------------------------
# Calcul des moyennes spatiales et de la variance
# ---------------------------------------------------------

def compute_variance(t_vals, a, M=1.0, D=1.0,
                     Nr=600, Rfactor=6.0):
    """
    Calcule <c>, <c^2>, sigma^2 pour un rayon initial a
    avec masse M constante.

    Renvoie:
        t_vals, sigma2_vals, mean_vals, mean2_vals
    """
    # Ajuste C0 pour masse totale M
    C0 = M / (np.pi * a**2)

    tmax = np.max(t_vals)
    Rcam = 0.2 # exemple : 2 mm
    r = np.linspace(0, Rcam, Nr)


    sigma2 = []
    mean_c = []
    mean_c2 = []

    for t in t_vals:
        C = C_exact(r, t, a, C0, D=D)

        # moyennes sur le disque
        Cr = C * r
        mean = 2 / Rcam ** 2 * np.trapz(C * r, r)
        mean2 = 2 / Rcam ** 2 * np.trapz(C ** 2 * r, r)

        sigma2.append(mean2 - mean**2)
        mean_c.append(mean)
        mean_c2.append(mean2)

    return np.array(sigma2), np.array(mean_c), np.array(mean_c2)

# ---------------------------------------------------------

if __name__ == "__main__":
    D = 1e-6     # Diffusivité
    M = 1.0      # Masse totale constante
    t_vals = np.logspace(-1,3,300 )  # de 0.1 à 30 s

    a_list = [0.01,0.001]

    plt.figure(figsize=(7,5))

    for a in a_list:
        sigma2, mean_c, mean2_c = compute_variance(t_vals, a, M=M, D=D)
        plt.plot(t_vals, sigma2, label=f"a={a}")

    plt.xlabel("t")
    plt.ylabel(r"$\sigma_c^2(t)$")
    plt.title("Variance de concentration — solution exacte 2D, masse constante")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.show()
