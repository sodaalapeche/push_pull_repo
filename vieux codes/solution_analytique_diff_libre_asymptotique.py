# (This is the full working script used above — it numerically evaluates the integral,
# computes sigma^2(t) on a disk and plots results for several a.)

import numpy as np
from scipy.special import i0
import matplotlib.pyplot as plt

def concentration_profile(r_vals, t, a, C0=1.0, D=1.0, Nrp=300):
    if t <= 0: raise ValueError("t must be > 0")
    rprime = np.linspace(0, a, Nrp)
    if a == 0: return np.zeros_like(r_vals)
    denom = 2*D*t
    prefactor = C0/denom
    exp_r2 = np.exp(-r_vals**2/(4*D*t))
    exp_rp2 = np.exp(-rprime**2/(4*D*t))
    arg = np.outer(r_vals, rprime) / (2*D*t)
    I = i0(arg)
    integrand_full = I * exp_rp2[np.newaxis, :] * rprime[np.newaxis, :]
    integral = np.trapz(integrand_full, rprime, axis=1)
    C = prefactor * exp_r2 * integral
    return C

def spatial_variance_vs_time(t_vals, a, C0=1.0, D=1.0, Rfactor=0.05, Nr=600, Nrp=300):
    t_max = np.max(t_vals)
    Rmax = max(Rfactor*np.sqrt(4*D*t_max), 4*a + 1e-8)
    r = np.linspace(0, Rmax, Nr)
    sigma2 = np.zeros_like(t_vals)
    mean_vals = np.zeros_like(t_vals)
    mean_sq_vals = np.zeros_like(t_vals)
    for i,t in enumerate(t_vals):
        C = concentration_profile(r, t, a, C0=C0, D=D, Nrp=Nrp)
        integral_c = np.trapz(C * r, r)
        integral_c2 = np.trapz((C**2) * r, r)
        mean = (2.0 * integral_c) / (Rmax**2)
        mean_sq = (2.0 * integral_c2) / (Rmax**2)
        sigma2[i] = mean_sq - mean**2
        mean_vals[i] = mean
        mean_sq_vals[i] = mean_sq
    return sigma2, mean_vals, mean_sq_vals

# Example usage
D = 1e-5
C0 = 1.0
a_list = [0.01,0.001,0.003]
t_vals = np.linspace(0.01, 1000,400)
results = {}
for a in a_list:
    s2, m, msq = spatial_variance_vs_time(t_vals, a, C0=C0, D=D, Rfactor=6.0, Nr=500, Nrp=300)
    results[a] = dict(sigma2=s2, mean=m, mean_sq=msq)

# Plotting
plt.figure()

for a in a_list:
    s2=results[a]['sigma2']
    # s2 = s2/(a**4)
    x=np.linspace(1,10,len(s2))
    plt.plot(t_vals, s2, label=f"$s_0={a}$")
plt.xlabel("t")

plt.ylabel(r"$\sigma_c^2(t)=\langle c^2\rangle-\langle c\rangle^2$")
plt.xscale('log')
plt.tight_layout()
plt.yscale('log')
plt.legend(); plt.grid(True); plt.show()
