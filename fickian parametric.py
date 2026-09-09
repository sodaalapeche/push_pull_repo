"""
Figure 3bis : trois figures séparées :
- fig1 : d2 = 0 mm (axes inversés)
- fig2 : d2 = 6 mm (axes inversés)
- fig3 : écart à la droite y=x en fonction du temps (distance to x-y)
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import os

plt.style.use("science")

# ==========================================================
# PARAMÈTRES (conservés)
# ==========================================================
width  = 18/1.8        # cm
height = 18/1.8
inches = 2.54
h = 6                  # distance d2 en mm
SAND = "coarse"
T_WINDOW = (0.0, 50.0)
NORM_INDEX_0MM = 0

# Limites d'axes
lims0 = [8e-2, 1.2]    # pour d2 = 0 mm
lims10 = [8e-3, 1.2]   # pour d2 = 6 mm

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
def sigma_raw(res):
    time  = res["time"]
    var   = res["var"]
    mean  = res["mean"]
    A0    = res["A"]
    Sigma = var / (A0 * mean**2)
    return time, Sigma

def inv_r2_raw(res):
    time    = res["time"]
    sigma_m = res["sigma_m"]
    inv_r2  = 1.0 / sigma_m**2
    return time, inv_r2

def paired_sigma_vs_invr2(res_roi, res_rms, L_mm):
    t_sigma, Sigma_r = sigma_raw(res_roi)
    t_r, inv_r2_r = inv_r2_raw(res_rms)

    tmin = max(t_sigma.min(), t_r.min())
    tmax = min(t_sigma.max(), t_r.max())

    mask = (t_sigma >= tmin) & (t_sigma <= tmax)
    t_common = t_sigma[mask]
    Sigma_c = Sigma_r[mask]

    f_interp = interp1d(t_r, inv_r2_r, kind="linear",
                        bounds_error=False, fill_value=np.nan)
    invr2_c = f_interp(t_common)

    t2 = 2 * t_common

    valid = ((t2 >= T_WINDOW[0]) & (t2 <= T_WINDOW[1]) &
             np.isfinite(Sigma_c) & np.isfinite(invr2_c) &
             (Sigma_c > 0) & (invr2_c > 0))

    Sigma_c = Sigma_c[valid]
    invr2_c = invr2_c[valid]
    t2 = t2[valid]

    if len(Sigma_c) == 0:
        return np.array([]), np.array([]), np.array([])

    idx_norm = NORM_INDEX_0MM if L_mm == 0 else 0
    if len(Sigma_c) <= idx_norm:
        idx_norm = 0
    Sigma_c /= Sigma_c[idx_norm]
    invr2_c /= invr2_c[idx_norm]

    return Sigma_c[::2], invr2_c[::2], t2[::2]

# ==========================================================
# PRÉPARATION DES DONNÉES
# ==========================================================
L_VALUES = [0, h]
MARKER_MAP = {0: "o", h: "^"}

data_dict = {L: [] for L in L_VALUES}
data_times = {L: [] for L in L_VALUES}

for L_mm in L_VALUES:
    roi_list = roi_idx.get((L_mm, SAND), [])
    rms_list = rms_idx.get((L_mm, SAND), [])

    if L_mm == h:
        roi_list = roi_list[0:1]
        rms_list = rms_list[0:1]

    for res_roi, res_rms in zip(roi_list, rms_list):
        Sigma_c, invr2_c, t_c = paired_sigma_vs_invr2(res_roi, res_rms, L_mm)
        if len(Sigma_c) > 0:
            data_dict[L_mm].append((Sigma_c, invr2_c, t_c))
            data_times[L_mm].append((t_c, Sigma_c, invr2_c))

# Échelles de temps pour la colormap (commune)
t_all = np.concatenate([t for L in L_VALUES for _, _, t in data_dict[L]])
tnorm_min, tnorm_max = t_all.min(), t_all.max()

# Création du dossier de sortie
out_dir = "/home/chorus/figures_sep/"
os.makedirs(out_dir, exist_ok=True)

# ==========================================================
# FIGURE 1 : d2 = 0 mm
# ==========================================================
fig1, ax = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")
for Sigma_c, invr2_c, t_c in data_dict[0]:
    sc0 = ax.scatter(Sigma_c, invr2_c, c=t_c, cmap="plasma",
                     vmin=tnorm_min, vmax=tnorm_max,
                     marker=MARKER_MAP[0], s=22,
                     linewidths=0.3, edgecolors="k", alpha=0.85)
ax.plot(lims0, lims0, color="k", ls="--", lw=0.8, alpha=0.6, zorder=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims0)
ax.set_ylim(*lims0)
ax.set_xlabel(r"$\Sigma/\Sigma_0$")
ax.set_ylabel(r"$R_0^2/R^2$")
ax.set_title(r"$d_2 = 0$ mm")
ax.grid(True, ls="--", alpha=0.3)
cbar = fig1.colorbar(sc0, ax=ax)
cbar.set_label(r"$2t/T_a$")
fig1.savefig(os.path.join(out_dir, "figure_d2_0mm.pdf"), bbox_inches='tight')
plt.close(fig1)

# ==========================================================
# FIGURE 2 : d2 = 6 mm
# ==========================================================
fig2, ax = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")
for Sigma_c, invr2_c, t_c in data_dict[h]:
    sc1 = ax.scatter(Sigma_c, invr2_c, c=t_c, cmap="viridis",
                     vmin=tnorm_min, vmax=tnorm_max,
                     marker=MARKER_MAP[h], s=22,
                     linewidths=0.3, edgecolors="k", alpha=0.85)
ax.plot(lims10, lims10, color="k", ls="--", lw=0.8, alpha=0.6, zorder=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims10)
ax.set_ylim(*lims10)
ax.set_xlabel(r"$\Sigma/\Sigma_0$")
ax.set_ylabel(r"$R_0^2/R^2$")
ax.set_title(f"$d_2 = {h}$ mm")
ax.grid(True, ls="--", alpha=0.3)
cbar = fig2.colorbar(sc1, ax=ax)
cbar.set_label(r"$2t/T_a$")
fig2.savefig(os.path.join(out_dir, f"figure_d2_{h}mm.pdf"), bbox_inches='tight')
plt.close(fig2)

# ==========================================================
# FIGURE 3 : Écart (distance to x-y) en fonction du temps
# ==========================================================
width  = 18/1.8        # cm
height = 10/1.8
inches = 2.54
fig3, ax = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")
color_map = {0: 'tab:blue', h: 'tab:orange'}
for L_mm in L_VALUES:
    for t_c, Sigma_c, invr2_c in data_times[L_mm]:
        delta = np.log10(invr2_c) - np.log10(Sigma_c)
        ax.plot(t_c, delta, marker=MARKER_MAP[L_mm],
                linestyle='-', markersize=4, linewidth=1.5,
                label=f"$d_2={L_mm}$ mm",
                color=color_map[L_mm])

ax.axhline(y=0, color='k', linestyle='--', linewidth=1.0, alpha=0.7, zorder=0)
ax.set_xlabel(r"$2tu$")
ax.set_ylabel("distance to x-y")   # étiquette demandée
ax.grid(True, ls="--", alpha=0.3)
ax.legend(loc='best', frameon=False)
# ax.set_xscale("log")
# ax.set_yscale("log")
# Ajustement des marges verticales
ymin, ymax = ax.get_ylim()
ymargin = 0.1 * (ymax - ymin)
ax.set_ylim(ymin - ymargin, ymax + ymargin)
fig3.savefig(os.path.join(out_dir, "figure_distance_to_xy.pdf"), bbox_inches='tight')
plt.close(fig3)

print(f"Toutes les figures ont été sauvegardées dans le dossier '{out_dir}'.")