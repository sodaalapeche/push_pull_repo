import numpy as np
import matplotlib.pyplot as plt
from scipy import special

params = {'legend.fontsize':10, 'legend.handlelength': 1, 'font.weight': 'bold'}

def calcul_variance(R, s0, D, t, c0=1.0, N_modes=1000):

    # Modes
    beta = special.jn_zeros(1, N_modes)
    J0_beta = special.j0(beta)


    Nn = (R**2 / 2) * (J0_beta**2)
    b_n = c0 * (R / beta) * s0 * special.j1(beta * s0 / R)

    coeff_n = (4 * c0**2 * (s0**2) / (R**2)) * (
        special.j1(beta * s0 / R)**2 / (beta**2 * J0_beta**2)
    )

    expo = np.exp(-2 * D * (beta**2)[:,None] * t[None,:] / (R**2))
    sigma2 = np.sum(coeff_n[:,None] * expo, axis=0)

    return sigma2, beta[0]

N=600
R_values  = [24e-3]   # m
s0_values = [1.5e-3]            # m
D_values  = [10**-8]            # m²/s
plot=False
t = np.logspace(-3,4 , N)       # s



plt.figure(figsize=(10,7))

stored_curves = []

for R in R_values:
    for s0 in s0_values:
        for D in D_values:

            sigma2, beta1 = calcul_variance(R, s0, D, t)

            stored_curves.append((R, s0, D, sigma2, beta1))
            # sigma2=sigma2/(s0**2)
            sigma2 = sigma2
            label = f"R={R*1e3:.0f} mm, s0={s0*1e3:.1f} mm, D={D:.0e}"
            plt.semilogy(t, sigma2, label=label)



R0, s00, D0, sigma_ref, beta1 = stored_curves[0]

tau = R0**2 / D0
alpha = 2 * beta1**2


n_tail = N//5
t_tail = t[-n_tail:]
t0 = t_tail[0]
sigma0 = sigma_ref[-n_tail]
pente = sigma0 * np.exp(-alpha * (t_tail - t0) / tau)
if plot==True:
    plt.semilogy(t_tail, pente, 'k--', linewidth=3,
                 label=r"asymptote : $\exp[-2 \beta_{1}^{2} \cdot (t/\tau)],  τ=\frac{R^{2}}{L}$")

#pente = sigma0*(t_tail/tau)**-1 ###pente -1loglog
#plt.loglog(t_tail,pente)
plt.xlabel(" t (s)")
# plt.ylabel(r"Variance $\frac{\sigma_c^2(t)}{s_{0}^{2}}$")
plt.ylabel(r"Variance $\sigma_c^2(t)$")

plt.title("diffusion sur un disque fini")
plt.grid(True, which="both")
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
