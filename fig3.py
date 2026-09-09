"""
Figure 3 : Sigma/Sigma_0 et R_0^2/R^2 vs 2t/Ta
Deux subplots côte à côte : fine (gauche) / coarse (droite)
Double axe Y : Sigma (gauche) / R0²/R² (droite)
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

plt.style.use("science")

# ==========================================================
# PARAMÈTRES (mêmes que le script précédent)
# ==========================================================
width  = 18          # cm — un peu plus large car 2 subplots
height = width * 0.35
inches = 2.54

COLOR_MAP = {
    10: "tab:blue",
    6:  "tab:orange",
    3:  "tab:red",
    0:  "tab:green",
}

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
def sigma_curve(res):
    """Retourne (time [t/Ta], Sigma normalisé à i0)."""
    time  = res["time"]
    var   = res["var"]
    mean  = res["mean"]
    A0    = res["A"]
    i0    = res["i0"]
    Sigma = var / (A0 * mean**2)
    Sigma = Sigma / Sigma[i0]
    return time, Sigma, i0

def inv_r2_curve(res):
    """Retourne (time [t/Ta], R0²/R² = sigma_m[i0]²/sigma_m²)."""
    time    = res["time"]
    sigma_m = res["sigma_m"]
    i0      = res["i0"]

    inv_r2  = 1.0 / sigma_m**2
    inv_r2  = inv_r2 / inv_r2[i0]   # R0²/R²(t), vaut 1 à t=0
    return time, inv_r2

# ==========================================================
# FIGURE : deux subplots fine | coarse, double axe Y
# ==========================================================
fig, axes = plt.subplots(1, 2,
                         figsize=(width/inches, height/inches),
                         layout="constrained", sharey=True)

SANDS = ["fine", "coarse"]
SUBSAMPLE = 3   # pas d'échantillonnage pour alléger l'affichage

for ax, sand in zip(axes, SANDS):
    ax_r = ax.twinx()   # axe droit pour R0²/R²

    # Pour chaque taille d'inclusion, tracer Sigma (gauche) + R0²/R² (droite)
    for L_mm in sorted(COLOR_MAP.keys()):
        color = COLOR_MAP[L_mm]

        # ---- Sigma sur l'axe de gauche ----
        for res in roi_idx.get((L_mm, sand), []):
            time, Sigma, i0 = sigma_curve(res)
            valid = np.isfinite(time) & np.isfinite(Sigma) & (Sigma > 0)
            ax.plot(2*time[valid][::SUBSAMPLE], Sigma[valid][::SUBSAMPLE],
                    marker="o", linestyle="None",
                    color=color, mfc=color, ms=4, mew=0.3, alpha=0.85)

        # ---- R0²/R² sur l'axe de droite ----
        for res in rms_idx.get((L_mm, sand), []):
            time, inv_r2 = inv_r2_curve(res)
            valid = np.isfinite(time) & np.isfinite(inv_r2) & (inv_r2 > 0)
            ax_r.plot(2*time[valid][::SUBSAMPLE], inv_r2[valid][::SUBSAMPLE],
                      marker="s", linestyle="None",
                      color=color, mfc="none", ms=4, mew=0.8, alpha=0.6)

    # ---- mise en forme ----
    ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.5)
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax_r.set_yscale("log")

    ax.set_xlim(1, 70)
    ax.set_ylim(8e-3, 1.3)
    ax_r.set_ylim(8e-3, 1.3)   # mêmes bornes pour comparer visuellement

    ax.set_xlabel(r"$2t\,/\,T_a$")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_title(f"{sand} sand", fontsize=11)

    # labels Y uniquement sur les axes extrêmes
    if sand == "fine":
        ax.set_ylabel(r"$\Sigma\,/\,\Sigma_0$")
        ax_r.set_yticklabels([])   # cache les ticks du twin interne
    else:
        ax.set_yticklabels([])
        ax_r.set_ylabel(r"$R_0^2\,/\,R^2$")

# ---- Légende globale ----
legend_elements = []
for L_mm in sorted(COLOR_MAP.keys()):
    legend_elements.append(
        Line2D([0], [0], marker="o", linestyle="None",
               color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm],
               ms=5, mew=0.3, label=fr"$d_2$ = {L_mm} mm")
    )
# séparateurs marqueurs
legend_elements.append(
    Line2D([0], [0], marker="o", linestyle="None",
           color="gray", mfc="gray", ms=5, label=r"$\Sigma/\Sigma_0$")
)
legend_elements.append(
    Line2D([0], [0], marker="s", linestyle="None",
           color="gray", mfc="none", ms=5, mew=0.8, label=r"$R_0^2/R^2$")
)

fig.legend(handles=legend_elements, loc="outside lower center",
           ncol=6, fontsize=8, frameon=False)

fig.savefig("/home/chorus/figure3_sigma_invR2.pdf")
plt.show()