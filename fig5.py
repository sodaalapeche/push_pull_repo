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
width  = 18/1.3          # cm — un peu plus large car 2 subplots
height = 10/1.3
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
    return time[::2], Sigma[::2], i0

def inv_r2_curve(res):
    """Retourne (time [t/Ta], R0²/R² = sigma_m[i0]²/sigma_m²)."""
    time    = res["time"]
    sigma_m = res["sigma_m"]
    i0      = res["i0"]

    inv_r2  = 1.0 / sigma_m**2
    inv_r2  = inv_r2 / inv_r2[i0]   # R0²/R²(t), vaut 1 à t=0
    return time[::2], inv_r2[::2]

# ==========================================================
# FIGURE : deux subplots fine | coarse, double axe Y
# ==========================================================
# ==========================================================
# FIGURE : fine sand uniquement, L = 0 et 10 mm
# ==========================================================

# ==========================================================
# FIGURE : fine sand uniquement, L = 0 et 10 mm
# ==========================================================

fig, ax = plt.subplots(
    figsize=(width/inches, height/inches),
    layout="constrained"
)
ax_r = ax.twinx()

SAND = "fine"
L_VALUES = [0, 10]

for L_mm in L_VALUES:

    color = COLOR_MAP[L_mm]

    # =====================================================
    # Sélection des expériences à afficher
    # =====================================================

    roi_list = roi_idx.get((L_mm, SAND), [])
    rms_list = rms_idx.get((L_mm, SAND), [])

    # Pour L=10 mm on ne garde que 2 expériences
    if L_mm == 10:
        roi_list = roi_list[1:2]
        rms_list = rms_list[1:2]

    # =====================================================
    # Sigma / Sigma0
    # =====================================================

    for res in roi_list:

        time, Sigma, i0 = sigma_curve(res)

        valid = (
            np.isfinite(time)
            & np.isfinite(Sigma)
            & (Sigma > 0)
        )

        ax.plot(
            2 * time[valid],
            Sigma[valid],
            marker="o",
            linestyle="None",
            color=color,
            mfc=color,
            ms=5,
            mew=0.3,
            alpha=0.85,
        )

    # =====================================================
    # R0² / R²
    # =====================================================

    for res in rms_list:

        time, inv_r2 = inv_r2_curve(res)

        valid = (
            np.isfinite(time)
            & np.isfinite(inv_r2)
            & (inv_r2 > 0)
        )

        ax_r.plot(
            2 * time[valid],
            inv_r2[valid],
            marker="s",
            linestyle="None",
            color=color,
            mfc="none",
            ms=5,
            mew=0.8,
            alpha=0.6,
        )

# ==========================================================
# Mise en forme
# ==========================================================

ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.5)

ax.set_yscale("log")
ax_r.set_yscale("log")

ax.set_xlim(-0.5, 57)
ax.set_ylim(8e-3, 1.2)
ax_r.set_ylim(8e-3, 1.2)

ax.set_xlabel(r"$2t/T_a$")
ax.set_ylabel(r"$\Sigma/\Sigma_0$")
ax_r.set_ylabel(r"$R_0^2/R^2$")

ax.grid(True, ls="--", alpha=0.3)

# ==========================================================
# Légende
# ==========================================================

legend_elements = [

    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        color="tab:green",
        mfc="tab:green",
        ms=5,
        label=r"$d_2=0$ mm --- $\Sigma$"
    ),

    Line2D(
        [0], [0],
        marker="s",
        linestyle="None",
        color="tab:green",
        mfc="none",
        ms=5,
        mew=0.8,
        label=r"$d_2=0$ mm --- $R_0^2/R^2$"
    ),

    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        color="tab:blue",
        mfc="tab:blue",
        ms=5,
        label=r"$d_2=10$ mm --- $\Sigma$"
    ),

    Line2D(
        [0], [0],
        marker="s",
        linestyle="None",
        color="tab:blue",
        mfc="none",
        ms=5,
        mew=0.8,
        label=r"$d_2=10$ mm --- $R_0^2/R^2$"
    ),
]

ax.legend(
    handles=legend_elements,
    loc="lower left",
    frameon=False,
    fontsize=9,
)

plt.savefig("fickianornotfickian.pdf")
plt.show()