"""
Figure 3bis : Sigma/Sigma_0 vs R^2/R_c^2 (plot paramétrique, temps éliminé)
R_c = taille de la colonne (grandeur indépendante, fixe), et non plus R_0
(premier point de la série) : ceci évite la circularité de la normalisation.
Si comportement fickien : points alignés sur la droite Sigma/Sigma_0 = x0/x,
de pente -1 en log-log (x0 = R_0^2/R_c^2 du premier point de référence).
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.style.use("science")

# ==========================================================
# PARAMÈTRES
# ==========================================================
width  = 18/2.1        # cm — un seul subplot maintenant
height = 10/2.1
inches = 2.54
h=10
COLOR_MAP = {
    10: "tab:blue",
    6:  "tab:orange",
    3:  "tab:red",
    0:  "tab:green",
}

# Taille de colonne (grandeur physique indépendante utilisée pour normaliser R^2).
# ATTENTION UNITÉS : sigma_m est supposé être en mm (comme d2 dans les labels)
# -> R_C exprimé en mm également. Adapter R_C si sigma_m est dans une autre unité.
R_C = 0.025  # mm  (0.025 m)

# ==========================================================
# CHEMINS
# ==========================================================
ROI_PATH = "/home/chorus/data_roi.npy"
RMS_PATH = "/home/chorus/data_rms.npy"

# ==========================================================
# CHARGEMENT
# ==========================================================
roi_results = np.load(ROI_PATH, allow_pickle=True)
rms_results = np.load(RMS_PATH, allow_pickle=True)
print(f"data_roi : {len(roi_results)} expériences")
print(f"data_rms : {len(rms_results)} expériences")

# ==========================================================
# PARSING / INDEXATION
# ==========================================================
def parse_label(label_text):
    lines = [ln.strip().lower() for ln in label_text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return None
    try:
        return int(lines[1].replace("mm", "")), lines[2]
    except Exception:
        return None

def index_results(results):
    idx = {}
    for r in results:
        info = parse_label(r.get("label", ""))
        if info is None:
            continue
        idx.setdefault(info, []).append(r)
    return idx

roi_idx = index_results(roi_results)
rms_idx = index_results(rms_results)

# ==========================================================
# HELPERS
# ==========================================================
# Fenêtre temporelle à conserver, en unités de 2t/Ta (comme sur le plot original)
T_WINDOW = (0.0, 51.0)

def sigma_raw(res):
    """Retourne (time [t/Ta], Sigma BRUT, non normalisé)."""
    time  = res["time"]
    var   = res["var"]
    mean  = res["mean"]
    A0    = res["A"]
    Sigma = var / (A0 * mean**2)
    return time, Sigma

def r2_raw(res):
    """Retourne (time [t/Ta], R^2 BRUT [mm^2], non normalisé)."""
    time    = res["time"]
    sigma_m = res["sigma_m"]
    R2 = sigma_m**2
    return time, R2

def paired_sigma_vs_r2c(res_roi, res_rms):
    """
    Aligne Sigma et R^2/R_c^2 sur le même axe temporel.
    - Sigma est normalisé par sa propre valeur au premier point conservé
      (Sigma/Sigma_0), comme dans le script d'origine.
    - R^2 est normalisé par la grandeur physique indépendante R_c^2
      (taille de colonne), PAS par le premier point -> plus de circularité.
    Retourne également x0 = (R^2/R_c^2) au premier point conservé, utile
    pour placer correctement la droite fickienne de référence.
    """

    t_sigma, Sigma_r = sigma_raw(res_roi)
    t_r, R2_r = r2_raw(res_rms)

    # Domaine commun
    tmin = max(t_sigma.min(), t_r.min())
    tmax = min(t_sigma.max(), t_r.max())

    mask = (t_sigma >= tmin) & (t_sigma <= tmax)

    t_common = t_sigma[mask]
    Sigma_c = Sigma_r[mask]

    # interpolation de R^2 sur la grille temporelle de Sigma
    f_interp = interp1d(
        t_r,
        R2_r,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    R2_c = f_interp(t_common)

    # temps réduit
    t2 = 2 * t_common

    # filtre AVANT normalisation
    valid = (
        (t2 >= T_WINDOW[0])
        & (t2 <= T_WINDOW[1])
        & np.isfinite(Sigma_c)
        & np.isfinite(R2_c)
        & (Sigma_c > 0)
        & (R2_c > 0)
    )

    Sigma_c = Sigma_c[valid]
    R2_c = R2_c[valid]
    t2 = t2[valid]

    if len(Sigma_c) == 0:
        return np.array([]), np.array([]), np.array([]), None

    i = 0
    if L_mm == 0:
        i = 0   # même point de référence que dans le script d'origine

    Sigma_c = Sigma_c / Sigma_c[i]      # Sigma / Sigma_0
    x_c = R2_c / R_C**2                 # R^2 / R_c^2  (grandeur indépendante)

    mask = (Sigma_c <= 1)
    Sigma_c = Sigma_c[mask]
    x_c = x_c[mask]
    t2 = t2[mask]

    x0 = x_c[i] if len(x_c) > i else (x_c[0] if len(x_c) else None)

    return Sigma_c[::1], x_c[::1], t2[::1], x0

# ==========================================================
# FIGURE : plot paramétrique fine sand, L = 0 et 10 mm
# ==========================================================

fig, ax = plt.subplots(
    figsize=(width/inches, height/inches),
    # layout="tight"
)


SAND = "coarse"
L_VALUES = [0, h]

MARKER_MAP = {
    0:  "o",   # cercles pleins pour d2 = 0 mm
    h: "^",   # triangles pour d2 = 10 mm
}

CMAP = "plasma"

# on aura besoin des bornes de temps globales pour une colormap commune
# aux deux séries -> on fait un premier passage pour tout calculer,
# puis un second pour tracer.²
all_data = []   # liste de (L_mm, Sigma_c, x_c, t_c)
x0_ref = None   # x0 = R_0^2/R_c^2 de la série de référence (d2 = 0 mm)

for L_mm in L_VALUES:

    roi_list = roi_idx.get((L_mm, SAND), [])
    rms_list = rms_idx.get((L_mm, SAND), [])

    # Pour L=10 mm on ne garde que 2 expériences (mêmes que script précédent)
    if L_mm == h:
        roi_list = roi_list[4:5]
        rms_list = rms_list[4:5]

    # on suppose un appariement 1-à-1 entre roi_list et rms_list
    # (même expérience -> même position dans les deux listes)
    for res_roi, res_rms in zip(roi_list, rms_list):
        Sigma_c, x_c, t_c, x0 = paired_sigma_vs_r2c(res_roi, res_rms)
        if len(Sigma_c) == 0:
            continue
        all_data.append((L_mm, Sigma_c, x_c, t_c))
        if L_mm == 0 and x0_ref is None:
            x0_ref = x0

t_all = np.concatenate([d[3] for d in all_data])
tnorm_min, tnorm_max = t_all.min(), t_all.max()

sc = None
for L_mm, Sigma_c, x_c, t_c in all_data:

    sc = ax.scatter(
        x_c,
        Sigma_c,
        c=t_c,
        cmap=CMAP,
        vmin=tnorm_min,
        vmax=tnorm_max,
        marker=MARKER_MAP[L_mm],
        s=22,
        linewidths=0.3,
        edgecolors="k",
        alpha=0.85,
    )


cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"$2t \cdot u$ (cm)")

# ==========================================================
# Limites d'axes calculées à partir des données (arrondies aux décades),
# plutôt que fixées à la main.
# ==========================================================
x_all = np.concatenate([d[2] for d in all_data])
y_all = np.concatenate([d[1] for d in all_data])

x_min = 10 ** np.floor(np.log10(x_all.min()))
x_max = 10 ** np.ceil(np.log10(x_all.max()))
y_min = 10 ** np.floor(np.log10(y_all.min()))
y_max = 10 ** np.ceil(np.log10(y_all.max()))

# ==========================================================
# Droite de référence fickienne : Sigma/Sigma_0 = x0 / x  (pente -1, log-log)
# x0 pris au premier point de la série de référence d2 = 0 mm.
# ==========================================================
xs_line = np.array([x_min, x_max])
ys_line = x0_ref / xs_line
ax.plot(xs_line, ys_line, color="k", ls="--", lw=0.8, alpha=0.6, zorder=0)
x0_ref = x0
xs_line = np.array([x_min, x_max])
ys_line = x0_ref / xs_line
ax.plot(xs_line, ys_line, color="k", ls="--", lw=0.8, alpha=0.6, zorder=0)

# ==========================================================
# Mise en forme
# ==========================================================
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(0.04,1.5)
ax.set_ylim(0.01, 1.9)

ax.set_xlabel(r"$R^2/R_c^2$", labelpad=4)
ax.set_ylabel(r"$(\sigma/\mu) / (\sigma_0/\mu_0)$", labelpad=4)

# ax.set_aspect("equal")
ax.grid(True, ls="--", alpha=0.3)
ax.tick_params(direction="in", top=True, right=True, which="both")

# ==========================================================
# Légende
# ==========================================================
legend_elements = [
    Line2D(
        [0], [0], marker=MARKER_MAP[0], linestyle="None",
        color="gray", mfc="gray", mec="k", mew=0.3, ms=6,
        label=r"$d_M=0$ mm",
    ),
    Line2D(
        [0], [0], marker=MARKER_MAP[h], linestyle="None",
        color="gray", mfc="gray", mec="k", mew=0.3, ms=6,
        label=r"$d_M=10$ mm",
    ),
    Line2D(
        [0], [0], color="k", ls="--", lw=0.8,
        label=r"$y = x^{-1}$",
    ),
]
#
# ax.legend(
#     handles=legend_elements,
#     loc="lower left",
#     frameon=False,
#     # fontsize=7,
#     handletextpad=0.5,
#     labelspacing=0.4,
#     borderaxespad=0.6,
# )

fig.tight_layout()
plt.savefig("/home/chorus/TexProject/clement draft avance (document de travail)/figures/fickian_parametric.pdf")
plt.show()