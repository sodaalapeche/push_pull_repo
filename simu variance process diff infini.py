import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.integrate import quad

# Paramètres
s0 = 0.1
D = 0.01
theta0 = 1.0

# Temps
t = np.logspace(-3, 1, 300)

# Solution exacte du créneau via convolution radiale
def theta_creneau(r, t, s0, D, theta0):
    # intégrale sur r' : theta(r,t) = (theta0 / (2 D t)) * ∫_0^s0 r' dr' e^{-(r^2+r'^2)/(4Dt)} I0(r r'/(2Dt))
    r_prime = np.linspace(0, s0, 200)
    dr = r_prime[1] - r_prime[0]
    integrand = r_prime * np.exp(-(r**2 + r_prime**2)/(4*D*t)) * i0(r*r_prime/(2*D*t))
    return (theta0 / (2*D*t)) * np.sum(integrand * dr)

# Calcul de la variance pour le créneau
Var_creneau = []
R_max = 10.0
r_vals = np.linspace(0, R_max, 500)
dr = r_vals[1] - r_vals[0]

for ti in t:
    theta_r = np.array([theta_creneau(r, ti, s0, D, theta0) for r in r_vals])
    integrand = theta_r**2 * r_vals
    Var = 2 * np.pi * np.sum(integrand * dr)
    Var_creneau.append(Var)
Var_creneau = np.array(Var_creneau)

# Variance pour gaussienne
Var_gauss = np.pi * (theta0 * s0**2)**2 / (2 * s0**2 + 4*D*t)

# Plot
plt.figure(figsize=(8,6))
plt.loglog(t, Var_creneau, label="Créneau (disque)")
plt.loglog(t, Var_gauss, label="Gaussienne")
plt.loglog([t[0], t[-1]], [Var_creneau[0], Var_creneau[0]*(t[0]/t[-1])], 'k--', label="pente -1")
plt.xlabel("Temps t")
plt.ylabel("Variance du scalaire")
plt.title("Comparaison de la variance : créneau vs gaussien")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
