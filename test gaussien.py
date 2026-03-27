import numpy as np
import matplotlib.pyplot as plt

# Définition de la densité de probabilité normale
def normal_pdf(x, cmax):
    return cmax / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

# Discrétisation de x
x_vals = np.linspace(-5, 5, 1000)

# Valeurs de cmax à tester
cmax_vals = np.logspace(-2, 1, 50)

# Calcul des probabilités p(c) (intégration numérique de la densité)
p_c_normal_vals = [np.trapz(normal_pdf(x_vals, c), x_vals) for c in cmax_vals]

# Tracé de log(p(c)) en fonction de log(c) pour la loi normale
plt.figure(figsize=(7,5))
plt.plot(np.log(cmax_vals), np.log(p_c_normal_vals), marker='o', linestyle='-', color='r')

plt.xlabel(r'$\log(c)$')
plt.ylabel(r'$\log(p(c))$')
plt.title(r'$\log(p(c))$ en fonction de $\log(c)$ (loi normale)')
plt.grid()
plt.show()
