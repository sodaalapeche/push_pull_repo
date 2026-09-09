"""
Figure : Var and 1/R² vs time for selected experiments
- Homogeneous fine sand (L_mm = 0)
- Heterogeneous fine sand with L_mm = 10 mm inclusions

Left Y-axis: Var (variance of concentration)
Right Y-axis: 1/R² (inverse of squared RMS radius)
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

plt.style.use("science")
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
SUBSAMPLE = 1

for L_mm in L_VALUES:

    color = COLOR_MAP[L_mm]

    # ---------------- Sigma ----------------
    for res in roi_idx.get((L_mm, SAND), []):

        time, Sigma, i0 = sigma_curve(res)

        valid = (
            np.isfinite(time)
            & np.isfinite(Sigma)
            & (Sigma > 0)
        )

        ax.plot(
            2*time[valid][::SUBSAMPLE],
            Sigma[valid][::SUBSAMPLE],
            "o",
            color=color,
            mfc=color,
            ms=5,
            mew=0.3,
            alpha=0.9,
            label=None
        )

    # ---------------- R0²/R² ----------------
    for res in rms_idx.get((L_mm, SAND), []):

        time, inv_r2 = inv_r2_curve(res)

        valid = (
            np.isfinite(time)
            & np.isfinite(inv_r2)
            & (inv_r2 > 0)
        )

        ax_r.plot(
            2*time[valid][::SUBSAMPLE],
            inv_r2[valid][::SUBSAMPLE],
            "s",
            color=color,
            mfc="none",
            ms=5,
            mew=0.9,
            alpha=0.7,
            label=None
        )

# ---------------- Mise en forme ----------------

ax.set_yscale("log")
ax_r.set_yscale("log")

ax.set_xlim(-0.5, 57)
ax.set_ylim(3e-3, 1.2)
ax_r.set_ylim(3e-3, 1.2)

ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)

ax.set_xlabel(r"$2t/T_a$")
ax.set_ylabel(r"$\Sigma/\Sigma_0$")
ax_r.set_ylabel(r"$R_0^2/R^2$")

ax.grid(True, ls="--", alpha=0.25)

legend_elements = [

    Line2D(
        [0],[0],
        marker="o",
        linestyle="None",
        color="tab:green",
        mfc="tab:green",
        ms=6,
        label=r"$d_2=0$ mm --- $\Sigma$"
    ),

    Line2D(
        [0],[0],
        marker="s",
        linestyle="None",
        color="tab:green",
        mfc="none",
        ms=6,
        mew=0.9,
        label=r"$d_2=0$ mm --- $R_0^2/R^2$"
    ),

    Line2D(
        [0],[0],
        marker="o",
        linestyle="None",
        color="tab:blue",
        mfc="tab:blue",
        ms=6,
        label=r"$d_2=10$ mm --- $\Sigma$"
    ),

    Line2D(
        [0],[0],
        marker="s",
        linestyle="None",
        color="tab:blue",
        mfc="none",
        ms=6,
        mew=0.9,
        label=r"$d_2=10$ mm --- $R_0^2/R^2$"
    ),
]

ax.legend(
    handles=legend_elements,
    loc="lower left",
    frameon=False,
    fontsize=9
)

plt.savefig("figure_sigma_invR2_fine.pdf")
plt.show()