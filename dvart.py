"""
Dérivée logarithmique de la variance : β(t) = d(ln Σ)/dt.

Objectif : mettre en évidence DEUX régimes exponentiels distincts de Σ(t).
Si Σ ~ exp(β t) sur un intervalle, alors d(ln Σ)/dt = β y est constant
-> chaque régime exponentiel apparaît comme un PLATEAU de β(t),
   dont la hauteur est directement le β du fit semi-log correspondant.

Σ est calculé exactement comme dans le script de scaling :
    Σ = var / (A * mean**2)
La constante A et la normalisation Σ[i0] disparaissent de la dérivée ;
en revanche mean(t) varie et compte, donc on garde le Σ complet.

Pente locale : régression linéaire glissante de ln Σ sur [t-h, t+h]
(= β local), robuste au bruit et directement comparable au fit global.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
import scienceplots

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")
width  = 15.5      # cm
height = width * 0.5
inches = 2.54

# ==========================================================
# PARAMÈTRES
# ==========================================================
DATA_PATH = "/home/chorus/data_roi.npy"

TIME_SCALE  = 2.0   # facteur sur l'axe temporel (2.0 pour coller à scaling fig1)
HALF_WIDTH  = 1   # demi-fenetre de regression locale, en unites de l'axe trace
MIN_PTS     = 2  # nb min de points dans la fenetre pour calculer une pente
SMOOTH_HALF = 0     # lissage final de la courbe beta(t) (0 = aucun)
SUBSAMPLE   = 1     # n'affiche qu'un point sur N (1 = tous)

# ==========================================================
# STYLE MAP  (couleur = confinement, plein = fine / tirets = coarse)
# ==========================================================
style_map = {
    (10, "fine"):   dict(color="tab:blue",   ls="-"),
    (10, "coarse"): dict(color="tab:blue",   ls="--"),
    (6,  "fine"):   dict(color="tab:orange", ls="-"),
    (6,  "coarse"): dict(color="tab:orange", ls="--"),
    (3,  "fine"):   dict(color="tab:red",    ls="-"),
    (3,  "coarse"): dict(color="tab:red",    ls="--"),
}

# ==========================================================
# PARSING
# ==========================================================
def parse_label(label_text):
    lines = [ln.strip().lower() for ln in label_text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return None
    try:
        return int(lines[1].replace("mm", "")), lines[2]
    except Exception:
        return None

# ==========================================================
# PENTE LOG LOCALE : beta(t) = pente de ln(Σ) sur [t-h, t+h]
# ==========================================================
def rolling_logslope(t, lny, half_width, min_pts=4):
    beta = np.full(t.size, np.nan)
    for i in range(t.size):
        m = (t >= t[i] - half_width) & (t <= t[i] + half_width)
        if np.count_nonzero(m) >= min_pts:
            beta[i] = np.polyfit(t[m], lny[m], 1)[0]
    return beta

# ==========================================================
# CHARGEMENT
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
print(f"Chargé : {len(results)} expériences")

# ==========================================================
# COLLECTE + DÉRIVÉE LOG + TRACÉ
# ==========================================================
fig, ax = plt.subplots(figsize=(width / inches, height / inches),
                       layout="constrained")

n_kept = 0
for res in results:
    info = parse_label(res.get("label", ""))
    if info is None:
        continue
    L_mm, sand = info
    if L_mm == 0:
        continue
    if L_mm ==3 and sand=="fine":
        continue
    style = style_map.get((L_mm, sand))
    if style is None:
        continue

    time = np.asarray(res["time"], dtype=float)
    mean = np.asarray(res["mean"], dtype=float)
    var  = np.asarray(res["var"],  dtype=float)
    A0   = res["A"]

    # --- Σ complet (mean(t) varie -> indispensable) ------
    Sigma = var / (A0 * mean**2)

    # --- axe temporel ------------------------------------
    t = TIME_SCALE * time

    # --- nettoyage (ln exige Σ > 0) + tri ----------------
    ok = np.isfinite(t) & np.isfinite(Sigma) & (Sigma > 0)
    t, Sigma = t[ok], Sigma[ok]
    order = np.argsort(t)
    t, Sigma = t[order], Sigma[order]
    if t.size < MIN_PTS:
        continue

    # --- pente log locale = β(t) -------------------------
    lnS  = np.log(Sigma)
    beta = rolling_logslope(t, lnS, HALF_WIDTH, MIN_PTS)

    # --- lissage final optionnel (ignore les NaN) --------
    if SMOOTH_HALF > 0:
        good = np.isfinite(beta)
        if np.count_nonzero(good) > 2 * SMOOTH_HALF + 1:
            beta[good] = uniform_filter1d(beta[good],
                                          size=2 * SMOOTH_HALF + 1,
                                          mode="nearest")

    ax.plot(t[::SUBSAMPLE], beta[::SUBSAMPLE], lw=1.3, **style)
    n_kept += 1

print(f"Expériences tracées : {n_kept}")

# ==========================================================
# LÉGENDE
# ==========================================================
legend_elements = [
    Line2D([0], [0], color=st["color"], ls=st["ls"], lw=1.3,
           label=f"{L_mm} mm, {sand}")
    for (L_mm, sand), st in style_map.items()
]
ax.legend(handles=legend_elements, ncol=2)

ax.axhline(0, color="k", lw=0.6, alpha=0.5)
ax.set_xlabel(fr"${str(TIME_SCALE)[0]}t \;/\; T_a$")
ax.set_ylabel(r"$\beta(t) = \mathrm{d}(\ln \Sigma)\;/\;\mathrm{d}(t/T_a)$")
ax.grid(True, ls="--", alpha=0.3)
ax.set_ylim(bottom=-0.6,top=0.2)
ax.set_xlim(left=-1,right=50)
fig.tight_layout()
plt.savefig("/home/chorus/dlogvar_dt.pdf")
plt.show()