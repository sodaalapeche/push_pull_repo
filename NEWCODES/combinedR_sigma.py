"""
Post-traitement combiné : data_roi.npy  +  data_rms.npy
Sable fin uniquement.

Figure 1 (deux axes Y) :
  axe gauche  — Σ = σ²_c / (A · μ²_c)  normalisé à t=0  [log]
  axe droit   — 1 / R²  [m⁻²]  (R = sigma_m du script rms.py)
  couleur = L_mm,  marqueur = rond plein (fine)

Figure 2 :
  Σ(L_mm) / Σ_homo  vs t/Ta
  (division point-à-point après interpolation sur la grille de temps de chaque exp)
  couleur = L_mm,  marqueur = rond plein (fine)
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")
A= 7
B=15
width   = 18     # cm
height  = width * 0.5
inches  = 2.54

# Couleur par L_mm  (fine = rond plein dans tous les cas)
COLOR_MAP = {
    10: "tab:blue",
    6:  "tab:orange",
    3:  "tab:red",
    0:  "tab:green",
}

def style_fine(L_mm, ms=6):
    return dict(marker="o", color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm],
                ms=ms, mew=0.4, alpha=0.85, linestyle="None")

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
# PARSING LABEL
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
# INDEXATION : construit des dicts  (L_mm, sand) -> liste de résultats
# ==========================================================
def index_results(results):
    idx = {}
    for r in results:
        info = parse_label(r.get("label", ""))
        if info is None:
            continue
        key = info   # (L_mm, sand)
        idx.setdefault(key, []).append(r)
    return idx

roi_idx = index_results(roi_results)
rms_idx = index_results(rms_results)
import numpy as np
import matplotlib.pyplot as plt
def fit_diffusion_D(time, R2):
    valid = np.isfinite(time) & np.isfinite(R2) & (time > 0)

    t = time[valid]
    y = R2[valid]

    if len(t) < 5:
        return np.nan

    slope, intercept = np.polyfit(t, y, 1)

    D = slope / 4
    return D
# ==========================================================
# HELPERS
# ==========================================================
def sigma_curve(res):
    """Retourne (time, Sigma) normalisé à i0, depuis data_roi."""
    time = res["time"]
    var  = res["var"]
    mean = res["mean"]
    A0   = res["A"]
    i0   = res["i0"]
    Sigma = var / (A0 * mean**2)
    Sigma = Sigma / Sigma[i0]
    return time, Sigma,i0

def inv_r2_curve(res):
    """Retourne (time, 1/R²) depuis data_rms."""
    time    = res["time"]
    sigma_m = res["sigma_m"]
    i0 = res["i0"]
    inv_r2  = 1.0 / sigma_m**2
    inv_r2 = inv_r2 / inv_r2[i0]
    return time, inv_r2

# ==========================================================
# FIGURE 1 — Σ  et  1/R²  vs  t/Ta   (sable fin, double axe Y)
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")
# ax1r = ax1.twinx()

fine_keys_sorted = sorted([k for k in roi_idx if k[1] == "fine"],
                          key=lambda k: k[0])
keys_sorted = sorted([k for k in roi_idx],
                          key=lambda k: k[0])

for L_mm, sand in keys_sorted:
    st = style_fine(L_mm, ms=5) if sand == "fine" else dict(
        marker="D", color=COLOR_MAP[L_mm], mfc="none",
        ms=5, mew=1.0, alpha=0.65, linestyle="None")

    for res in rms_idx.get((L_mm, sand), []):
        time, inv_r2 = inv_r2_curve(res)

        # look up matching Sigma from roi_idx (same key)
        roi_list = roi_idx.get((L_mm, sand), [])
        if not roi_list:
            continue
        _, Sigma, i0 = sigma_curve(roi_list[0])

        # interpolate Sigma onto rms time grid if needed
        valid_roi = np.isfinite(roi_list[0]["time"]) & np.isfinite(Sigma) & (Sigma > 0)
        f_sig = interp1d(roi_list[0]["time"][valid_roi], Sigma[valid_roi],
                         kind="linear", bounds_error=False, fill_value=np.nan)
        Sigma_interp = f_sig(time)


        valid = np.isfinite(inv_r2) & (inv_r2 > 0)
        # ax1.plot(2*time[valid][::3], Sigma_interp[valid][::3] / inv_r2[valid][::3],  **st)
        ax1.plot(2 * time[valid][::3], 1 / inv_r2[valid][::3], **st)

# axes
ax1.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)

# ax1.set_ylabel(r"$\Sigma \;/\; \Sigma_0$")
ax1.set_ylabel(r"$R^2/R_0^2 \cdot \Sigma \;/\; \Sigma_0$")
ax1.set_yscale("log")
ax1.set_xscale("log")
# ax1r.set_yscale("log")
# ax1r.set_xscale("log")
ax1.set_xlim(left=3, right=70)
# ax1.set_ylim(0.003, 1.2)
ax1.set_ylim(1, 30)
ax1.grid(True, ls="--", alpha=0.3)

# légende
legend_elements = []
for L_mm in sorted(COLOR_MAP.keys()):
    legend_elements.append(
        Line2D([0], [0], marker="o", linestyle="None",
               color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm],
               ms=4, mew=0.4, label=f"$d_2$ = {L_mm} mm — fine")
    )
    legend_elements.append(
        Line2D([0], [0], marker="D", linestyle="None",
               color=COLOR_MAP[L_mm], mfc="none",
               ms=4, mew=1.0, alpha=0.65, label=f"$d_2$ = {L_mm} mm — coarse")
    )
ax1.legend(handles=legend_elements, ncol=2, loc="lower left")
plt.show()
# ==========================================================
# FIGURE 2 — Σ / Σ_homo  vs  t/Ta   (sable fin, L_mm > 0)
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")
from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
# Référence homogène fine : on moyenne les réplicats si plusieurs
homo_rois = roi_idx.get((0, "fine"), [])
if len(homo_rois) == 0:
    print("[WARN] Pas d'expérience homogène fine dans data_roi.npy — Fig 2 vide")
else:
    # Grille de temps commune pour l'homo : on prend le premier réplicat
    # et on interpole les autres dessus avant de moyenner
    t_homo_ref = homo_rois[0]["time"]
    Sigma_homo_list = []
    for res in homo_rois:
        t_h, S_h, i0 = sigma_curve(res)
        valid = np.isfinite(t_h) & np.isfinite(S_h) & (S_h > 0)
        if valid.sum() < 5:
            continue
        f_interp = interp1d(t_h[valid], S_h[valid],
                            kind="linear", bounds_error=False,
                            fill_value=np.nan)
        Sigma_homo_list.append(f_interp(t_homo_ref))

    Sigma_homo_mean = np.nanmean(np.vstack(Sigma_homo_list), axis=0)

    import numpy as np
    from scipy.interpolate import interp1d

    # Définir une grille de temps commune (par exemple, la grille la plus fine)
    common_time = np.linspace(-1, 58, 200)  # Ajuste les bornes et le nombre de points selon tes besoins

    # Dictionnaire pour stocker les ratios interpolés par L_mm
    interpolated_ratios = defaultdict(list)

    for L_mm, sand in fine_keys_sorted:
        if sand != "fine" or L_mm == 0:
            continue

        for res in roi_idx.get((L_mm, sand), []):
            time, Sigma, i0 = sigma_curve(res)

            # Interpolation de Σ_homo
            valid_h = np.isfinite(t_homo_ref) & np.isfinite(Sigma_homo_mean)
            f_homo = interp1d(t_homo_ref[valid_h], Sigma_homo_mean[valid_h],
                              kind="linear", bounds_error=False,
                              fill_value=np.nan)
            Sigma_homo_interp = f_homo(time)
            ratio = Sigma / (Sigma_homo_interp)
            ratio /= ratio[i0]
            valid = (np.isfinite(time) & np.isfinite(ratio) &
                     (ratio > 0) & (time > -1))

            # Interpolation du ratio sur la grille commune
            f_ratio = interp1d(time[valid], ratio[valid],
                               kind="linear", bounds_error=False,
                               fill_value=np.nan)
            interpolated_ratio = f_ratio(common_time)
            interpolated_ratios[L_mm].append(interpolated_ratio)

    # Calcul des moyennes par temps pour chaque L_mm
    mean_ratios = {}
    for L_mm, ratios_list in interpolated_ratios.items():
        stacked_ratios = np.vstack(ratios_list)
        mean_ratios[L_mm] = np.nanmean(stacked_ratios, axis=0)

    # Tracé des courbes individuelles (optionnel)
    # Tracé : une expérience par L_mm (premier réplicat)
    for L_mm, sand in fine_keys_sorted:
        if sand != "fine" or L_mm == 0:
            continue

        res_list = roi_idx.get((L_mm, sand), [])
        if not res_list:
            continue
        res = res_list[1]  # premier réplicat uniquement

        time, Sigma, i0 = sigma_curve(res)

        valid_h = np.isfinite(t_homo_ref) & np.isfinite(Sigma_homo_mean)
        f_homo = interp1d(t_homo_ref[valid_h], Sigma_homo_mean[valid_h],
                          kind="linear", bounds_error=False,
                          fill_value=np.nan)
        Sigma_homo_interp = f_homo(time)
        ratio = Sigma / Sigma_homo_interp
        ratio /= ratio[i0]
        ratio = 1/ratio
        valid = (np.isfinite(time) & np.isfinite(ratio) &
                 (ratio > 0) & (time > -1))

        st = style_fine(L_mm, ms=6)
        ax2.plot(2 * time[valid][::3], ratio[valid][::3], **st)
    # Tracé des moyennes
    # for L_mm in sorted(mean_ratios.keys()):
    #     ax2.plot(2 * time[::2], ratio[L_mm][::2],
    #              color=COLOR_MAP[L_mm], linestyle="", marker='o',
    #              label=f"Moyenne L={L_mm} mm")

# axes
ax2.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
ax2.axhline(1, color="k", ls=":",  lw=0.7, alpha=0.4)
ax2.set_xlabel(r"$2t \;/\; T_a$")
ax2.set_ylabel(r"$ \Sigma_\mathrm{homo} \;/\; \Sigma $")
ax2.set_yscale("log")
ax2.set_xlim(left=-1, right=50)
ax2.set_ylim(bottom=0.95, top=50)
ax2.grid(True, ls="--", alpha=0.3)

legend2 = [
    Line2D([0], [0], marker="o", linestyle="None",
           color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm],
           ms=6, mew=0.4, label=f"$d_2$ = {L_mm} mm")
    for L_mm in sorted(COLOR_MAP.keys()) if L_mm > 0
]
ax2.legend(handles=legend2, ncol=1)
# ax2.set_title(r"Fine sand — $\Sigma / \Sigma_\mathrm{homo}$ vs $t/T_a$",
fig2.savefig("/home/chorus/enhancement.pdf")
plt.show()

style_map = {
    (10, "fine"): dict(marker="o", color="tab:blue", mfc="tab:blue", ms=6, alpha=0.8),
    (10, "coarse"): dict(marker="D", color="tab:blue", mfc="none", mew=1.2, ms=6, alpha=0.8),
    (6, "fine"): dict(marker="o", color="tab:orange", mfc="tab:orange", ms=6, alpha=0.8),
    (6, "coarse"): dict(marker="D", color="tab:orange", mfc="none", mew=1.2, ms=6, alpha=0.8),
    (3, "fine"): dict(marker="o", color="tab:red", mfc="tab:red", ms=6, alpha=0.8),
    (3, "coarse"): dict(marker="D", color="tab:red", mfc="none", mew=1.2, ms=6, alpha=0.8),
    (0,"fine"): dict(color="tab:green", marker="o", mfc="tab:green", ms=5, alpha=0.8),
    (0,"coarse"): dict(color="tab:olive", marker="D", mfc="none", mew=1.2, ms=5, alpha=0.8),
}


# ==========================================================
# FIGURE 3 — Σ vs t   (fine + coarse avec style_map)
# ==========================================================
fig3, ax3 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

for (L_mm, sand), style in style_map.items():

    for res in roi_idx.get((L_mm, sand), []):
        time, Sigma, i0 = sigma_curve(res)
        Sigma= Sigma
        valid = np.isfinite(time) & np.isfinite(Sigma) & (Sigma > 0)

        ax3.plot(2 * time[valid][::3],
                 Sigma[valid][::3],
                 linestyle="",
                 **style)

# axes
ax3.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
ax3.set_xlabel(r"$2t$")
ax3.set_ylabel(r"$\Sigma$")
ax3.set_yscale("log")
ax3.set_xscale("log")

ax3.set_xlim(left=5, right=70)
ax3.set_ylim(bottom=0.01, top=1.2)
ax3.grid(True, ls="--", alpha=0.3)

# légende
legend3 = []
for (L_mm, sand), style in style_map.items():
    label = f"$d_2$ = {L_mm} mm — {sand}"
    legend3.append(
        Line2D([0], [0],
               marker=style.get("marker", "o"),
               linestyle="None",
               color=style.get("color", "k"),
               mfc=style.get("mfc", "none"),
               mew=style.get("mew", 1.0),
               ms=style.get("ms", 6),
               alpha=style.get("alpha", 1.0),
               label=label)
    )

ax3.legend(handles=legend3, ncol=2, loc="lower left")
t_ref = np.logspace(np.log10(1), np.log10(60), 100)
y_ref = t_ref**-1  # pente 1 => y ~ t^1

# option : ajuster la position verticale (facteur multiplicatif)
factor = 0.5
ax3.plot(t_ref, factor * y_ref, 'k--', label=r"$\propto t^1$")
plt.show()

fig, ax = plt.subplots(figsize=(width / inches, height / inches),
                        layout="constrained")

D_dict    = {}   # (L_mm, sand) -> D moyen  [m²/s]
D_err_dict= {}   # (L_mm, sand) -> std(D)   [m²/s]
u_dict    = {}   # (L_mm, sand) -> u moyen  [m/s]  avec u = 0.01/Ta

# ==========================================================
# FIT PAR (L_mm, sand)
# t est en t/Ta  =>  R² = sigma_m²  [m²]  vs  t/Ta  [adim]
# slope de polyfit = dR²/d(t/Ta) = 4D*Ta   [m²]
# donc D = slope / (4*Ta)                  [m²/s]
# u = vitesse de Darcy = 0.01/Ta           [m/s]
# ==========================================================
for (L_mm, sand), style in style_map.items():

    Ds = []
    us = []

    for res in rms_idx.get((L_mm, sand), []):

        time    = res["time"]      # t/Ta  [adim]
        sigma_m = res["sigma_m"]   # [m]
        Ta      = res["Ta"]        # [s]

        R2 = sigma_m**2            # [m²]

        valid = np.isfinite(time) & np.isfinite(R2) & (time > A) & (time <= B)
        t = time[valid]
        y = R2[valid]

        if len(t) < 5:
            continue

        # slope en [m² / (t/Ta)] = [m²]  (t/Ta adimensionnel)
        slope, _ = np.polyfit(t, y, 1)

        # D [m²/s] : slope = 4D*Ta  =>  D = slope/(4*Ta)
        D = abs(slope) / (4.0 * Ta)
        u = 0.01 / Ta              # vitesse de Darcy [m/s]
        D = D/u
        if np.isfinite(D) and D > 0:
            Ds.append(D)
            us.append(u)

    if len(Ds) > 0:
        D_dict[(L_mm, sand)]    = np.mean(Ds)
        D_err_dict[(L_mm, sand)]= np.std(Ds)
        u_dict[(L_mm, sand)]    = np.mean(us)
    else:
        print(f"[WARN] No valid D for (L_mm={L_mm}, sand={sand})")

# ==========================================================
# CHARGEMENT alphaT homogène  [m]
# alphaT = D_homo / u  calculé dans le script diffusion
# ==========================================================
_diff_data   = np.load("../vieux codes/resultats_diffusion.npy", allow_pickle=True).item()
_alphaT_raw  = _diff_data["alphaT"]   # {"fine": [m, m, ...], "coarse": [...]}
_alphaT_mean = {sand: np.mean(vals) for sand, vals in _alphaT_raw.items()}

# ==========================================================
# PLOT
# axe X : Pe = d2 [m] / alphaT_homo [m]          [adim]
# axe Y : D_M / D_m  avec  D_m = alphaT_homo * u [m²/s]  [adim]
# ==========================================================
for (L_mm, sand), style in style_map.items():

    if (L_mm, sand) not in D_dict:
        continue

    alphaT_homo = _alphaT_mean[sand]            # [m]
    u_mean      = u_dict[(L_mm, sand)]          # [m/s]
    D_m         = alphaT_homo         # D référence sable seul [m²/s]
    d2_m        = L_mm * 1e-3                   # d2 en mètres

    Pe    = d2_m / alphaT_homo                  # adim
    Dnorm = D_dict[(L_mm, sand)] / D_m          # adim
    Dnorm_err = D_err_dict[(L_mm, sand)] / D_m  # adim

    ax.errorbar(
        Pe, Dnorm,
        yerr=Dnorm_err,
        fmt=style.get("marker", "o"),
        ms=style.get("ms", 6),
        color=style.get("color", "k"),
        mfc=style.get("mfc", "none"),
        mew=style.get("mew", 1.0),
        capsize=4,
        elinewidth=1.0,
        alpha=style.get("alpha", 0.8),
        linestyle="None"
    )

# ==========================================================
# AXES
# ==========================================================
ax.set_xlabel(r"$Pe = d_2 / \alpha_{\perp}$")
ax.set_ylabel(r"$D_{\perp M} / D_{\perp m}$")
ax.set_yscale("log")
ax.set_xscale("log")
ax.grid(True, ls="--", alpha=0.3)

# ==========================================================
# FIT MOINDRES CARRÉS EN LOG-LOG  (L_mm > 0 seulement)
# sigma(log10 D) = sigma_D / (D * ln10)  =>  w = D_norm / D_norm_err
# ==========================================================
Pe_all    = []
D_all     = []
D_err_all = []

for (L_mm, sand), style in style_map.items():
    if (L_mm, sand) not in D_dict or L_mm == 0:
        continue
    alphaT_homo = _alphaT_mean[sand]
    u_mean      = u_dict[(L_mm, sand)]
    D_m         = alphaT_homo
    d2_m        = L_mm * 1e-3

    Pe_all.append(d2_m / alphaT_homo)
    D_all.append(D_dict[(L_mm, sand)] / D_m)
    D_err_all.append(D_err_dict[(L_mm, sand)] / D_m)

Pe_all    = np.array(Pe_all)
D_all     = np.array(D_all)
D_err_all = np.array(D_err_all)

valid_fit = (Pe_all > 0) & (D_all > 0) & np.isfinite(D_all) & np.isfinite(Pe_all)
# points avec un seul réplicat ont D_err=0 : poids uniforme w=1 pour eux
w = np.where(D_err_all > 0, D_all / D_err_all, 1.0)

log_Pe = np.log10(Pe_all[valid_fit])
log_D  = np.log10(D_all[valid_fit])
w_fit  = w[valid_fit]

coeffs, cov = np.polyfit(log_Pe, log_D, 1, w=w_fit, cov=True)
slope     = coeffs[0]
intercept = coeffs[1]
slope_err = np.sqrt(cov[0, 0])

Pe_fit = np.logspace(np.log10(Pe_all[valid_fit].min()),
                     np.log10(Pe_all[valid_fit].max()), 200)
D_fit  = 10**intercept * Pe_fit**slope
ax.plot(Pe_fit, D_fit, "k-", lw=1.5,
        label=rf"fit : slope $= {slope:.2f} \pm {slope_err:.2f}$")

ax.legend()
plt.show()