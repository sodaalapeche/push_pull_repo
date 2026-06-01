"""
Post-traitement combiné : data_roi.npy  +  data_rms.npy

Figure 1 : R²/R0² · Σ/Σ0  vs  2t/Ta  (log-log)
Figure 2 : Σ_homo / Σ  vs  2t/Ta  (log, fine seulement)
Figure 3 : Σ  vs  2t/Ta  (log-log, fine + coarse)
Figure 4 : R²(t) brut avec fit linéaire superposé  (fine + coarse)
Figure 5 : D_perp_M / D_perp_m  vs  Pe = d2/alphaT  (log-log + fit puissance)

Unités :
  time  : t/Ta           [adim]
  sigma_m : [m]
  R²    : sigma_m²       [m²]
  fit linéaire R²(t/Ta) : slope = 4D·Ta  [m²]  =>  D = slope/(4·Ta)  [m²/s]
  u     = 0.01/Ta        [m/s]
  alphaT = D/u           [m]
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from collections import defaultdict

plt.style.use("science")

# ==========================================================
# PARAMÈTRES
# ==========================================================
A_FIT = 5    # début fenêtre fit  [t/Ta]
B_FIT = 18    # fin   fenêtre fit  [t/Ta]

width  = 15
height = width * 0.5
inches = 2.54

COLOR_MAP = {
    10: "tab:blue",
    6:  "tab:orange",
    3:  "tab:red",
    0:  "tab:green",
}

style_map = {
    (10, "fine"):   dict(marker="o", color="tab:blue",   mfc="tab:blue",   ms=6, alpha=0.8),
    # (10, "coarse"): dict(marker="D", color="tab:blue",   mfc="none", mew=1.2, ms=6, alpha=0.8),
    (6,  "fine"):   dict(marker="o", color="tab:orange", mfc="tab:orange", ms=6, alpha=0.8),
    (6,  "coarse"): dict(marker="D", color="tab:orange", mfc="none", mew=1.2, ms=6, alpha=0.8),
    (3,  "fine"):   dict(marker="o", color="tab:red",    mfc="tab:red",    ms=6, alpha=0.8),
    # (3,  "coarse"): dict(marker="D", color="tab:red",    mfc="none", mew=1.2, ms=6, alpha=0.8),
    (0,  "fine"):   dict(marker="o", color="tab:green",  mfc="tab:green",  ms=5, alpha=0.8),
    (0,  "coarse"): dict(marker="D", color="tab:olive",  mfc="none", mew=1.2, ms=5, alpha=0.8),
}

# ==========================================================
# CHEMINS
# ==========================================================
ROI_PATH  = "/home/chorus/data_roi.npy"
RMS_PATH  = "/home/chorus/data_rms.npy"
DIFF_PATH = "../vieux codes/resultats_diffusion.npy"

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

keys_sorted      = sorted(roi_idx.keys(), key=lambda k: k[0])
fine_keys_sorted = [k for k in keys_sorted if k[1] == "fine"]

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
    """Retourne (time [t/Ta], 1/R² normalisé à i0)."""
    time    = res["time"]
    sigma_m = res["sigma_m"]
    i0      = res["i0"]
    inv_r2  = 1.0 / sigma_m**2
    inv_r2  = inv_r2 / inv_r2[i0]
    return time, inv_r2

# ==========================================================
# FIGURE 1 — R²/R0² · Σ/Σ0  vs  2t/Ta
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

for L_mm, sand in keys_sorted:
    st = (dict(marker="o", color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm], ms=5, mew=0.4, alpha=0.85)
          if sand == "fine" else
          dict(marker="D", color=COLOR_MAP[L_mm], mfc="none", ms=5, mew=1.0, alpha=0.65))

    roi_list = roi_idx.get((L_mm, sand), [])
    if not roi_list:
        continue

    for res in rms_idx.get((L_mm, sand), []):
        time, inv_r2 = inv_r2_curve(res)
        _, Sigma, i0 = sigma_curve(roi_list[0])

        valid_roi = np.isfinite(roi_list[0]["time"]) & np.isfinite(Sigma) & (Sigma > 0)
        f_sig = interp1d(roi_list[0]["time"][valid_roi], Sigma[valid_roi],
                         kind="linear", bounds_error=False, fill_value=np.nan)
        Sigma_interp = f_sig(time)
        product = Sigma_interp / inv_r2
        valid = np.isfinite(product) & (product > 0)
        ax1.plot(2*time[valid][::3], product[valid][::3], linestyle="", **st)

ax1.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
ax1.set_xlabel(r"$2t\,/\,T_a$")
ax1.set_ylabel(r"$\frac{R^2}{R_0^2} \cdot \frac{\Sigma}{\Sigma_0}$")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xlim(3, 70)
ax1.set_ylim(0.1, 2)
ax1.grid(True, ls="--", alpha=0.3)

leg1 = []
for L_mm in sorted(COLOR_MAP):
    leg1 += [
        Line2D([0],[0], marker="o", ls="None", color=COLOR_MAP[L_mm],
               mfc=COLOR_MAP[L_mm], ms=4, label=f"$d_2$={L_mm} mm — fine"),
        Line2D([0],[0], marker="D", ls="None", color=COLOR_MAP[L_mm],
               mfc="none", mew=1, ms=4, label=f"$d_2$={L_mm} mm — coarse"),
    ]
ax1.legend(handles=leg1, ncol=2, loc="lower left")
plt.show()

# ==========================================================
# FIGURE 2 — Σ_homo / Σ  vs  2t/Ta  (fine seulement)
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

homo_rois = roi_idx.get((0, "fine"), [])
if not homo_rois:
    print("[WARN] Pas d'expérience homogène fine")
else:
    t_homo_ref = homo_rois[0]["time"]
    Sigma_homo_list = []
    for res in homo_rois:
        t_h, S_h, i0 = sigma_curve(res)
        valid = np.isfinite(t_h) & np.isfinite(S_h) & (S_h > 0)
        if valid.sum() < 5:
            continue
        f = interp1d(t_h[valid], S_h[valid], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        Sigma_homo_list.append(f(t_homo_ref))
    Sigma_homo_mean = np.nanmean(np.vstack(Sigma_homo_list), axis=0)

    for L_mm, sand in fine_keys_sorted:
        if L_mm == 0:
            continue
        res_list = roi_idx.get((L_mm, sand), [])
        if not res_list:
            continue
        res = res_list[1] if len(res_list) > 1 else res_list[0]
        time, Sigma, i0 = sigma_curve(res)

        valid_h = np.isfinite(t_homo_ref) & np.isfinite(Sigma_homo_mean)
        f_homo = interp1d(t_homo_ref[valid_h], Sigma_homo_mean[valid_h],
                          kind="linear", bounds_error=False, fill_value=np.nan)
        ratio = Sigma / f_homo(time)
        ratio /= ratio[i0]
        ratio = 1.0 / ratio
        valid = np.isfinite(time) & np.isfinite(ratio) & (ratio > 0) & (time > -1)

        st = dict(marker="o", color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm],
                  ms=6, mew=0.4, alpha=0.85)
        ax2.plot(2*time[valid][::3], ratio[valid][::3], linestyle="", **st)

ax2.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
ax2.axhline(1, color="k", ls=":",  lw=0.7, alpha=0.4)
ax2.set_xlabel(r"$2t\,/\,T_a$")
ax2.set_ylabel(r"$\Sigma_\mathrm{homo}\,/\,\Sigma$")
ax2.set_yscale("log")
ax2.set_xlim(-1, 50)
ax2.set_ylim(0.95, 50)
ax2.grid(True, ls="--", alpha=0.3)
leg2 = [Line2D([0],[0], marker="o", ls="None", color=COLOR_MAP[L],
               mfc=COLOR_MAP[L], ms=6, label=f"$d_2$={L} mm")
        for L in sorted(COLOR_MAP) if L > 0]
ax2.legend(handles=leg2)
plt.savefig("/home/chorus/enhancement.pdf")
plt.show()

# ==========================================================
# FIGURE 3 — Σ  vs  2t/Ta  (log-log, fine + coarse)
# ==========================================================
fig3, ax3 = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

for (L_mm, sand), style in style_map.items():
    for res in roi_idx.get((L_mm, sand), []):
        time, Sigma, i0 = sigma_curve(res)
        valid = np.isfinite(time) & np.isfinite(Sigma) & (Sigma > 0)
        ax3.plot(2*time[valid][::3], Sigma[valid][::3], linestyle="", **style)

ax3.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
ax3.set_xlabel(r"$2t\,/\,T_a$")
ax3.set_ylabel(r"$\Sigma\,/\,\Sigma_0$")
ax3.set_yscale("log")
ax3.set_xscale("log")
ax3.set_xlim(5, 70)
ax3.set_ylim(0.01, 1.2)
ax3.grid(True, ls="--", alpha=0.3)

t_ref = np.logspace(np.log10(1), np.log10(60), 100)
ax3.plot(t_ref, 0.5 * t_ref**-1, "k--", lw=1.2, label=r"$\propto t^{-1}$")

leg3 = [Line2D([0],[0], marker=s.get("marker","o"), ls="None",
               color=s.get("color","k"), mfc=s.get("mfc","none"),
               mew=s.get("mew",1.0), ms=s.get("ms",6),
               label=f"$d_2$={L} mm — {sd}")
        for (L,sd), s in style_map.items()]
ax3.legend(handles=leg3, ncol=2, loc="lower left")
plt.show()

# ==========================================================
# FIT LINÉAIRE  R²(t)  PAR RÉPLICAT
#
# Modèle : R²(t/Ta) = R0² + 4D·Ta · (t/Ta)
#   polyfit donne : slope = 4D·Ta  [m²]
#                   intercept = R0² [m²]
#   D = slope / (4·Ta)              [m²/s]
#   u = 0.01 / Ta                   [m/s]
# ==========================================================
D_dict     = {}
D_err_dict = {}
u_dict     = {}
R0sq_dict  = {}   # intercept moyen pour diagnostic

for (L_mm, sand), style in style_map.items():
    Ds = []; us = []; R0sqs = []

    for res in rms_idx.get((L_mm, sand), []):
        time    = res["time"]      # t/Ta [adim]
        sigma_m = res["sigma_m"]   # [m]
        Ta      = res["Ta"]        # [s]
        R2      = sigma_m**2       # [m²]

        valid = np.isfinite(time) & np.isfinite(R2) & (time > A_FIT) & (time <= B_FIT)
        t = time[valid];  y = R2[valid]
        if len(t) < 5:
            continue

        slope, intercept = np.polyfit(t, y, 1)
        # slope [m²] = 4D·Ta  =>  D [m²/s]
        D   = slope / (4.0 * Ta)
        u   = 0.01 / Ta
        R0sq = intercept            # [m²]

        if np.isfinite(D) and D > 0:
            Ds.append(D); us.append(u); R0sqs.append(R0sq)

    if Ds:
        D_dict[(L_mm, sand)]    = np.mean(Ds)
        D_err_dict[(L_mm, sand)]= np.std(Ds)
        u_dict[(L_mm, sand)]    = np.mean(us)
        R0sq_dict[(L_mm, sand)] = np.mean(R0sqs)
        print(f"({L_mm:2d} mm, {sand:6s}) "
              f"D={np.mean(Ds):.3e} m²/s  "
              f"R0={np.sqrt(max(np.mean(R0sqs),0))*1e3:.2f} mm  "
              f"n={len(Ds)}")
    else:
        print(f"[WARN] No valid D for ({L_mm}, {sand})")

# ==========================================================
# FIGURE 4 — R²(t) brut + droite de fit linéaire
# ==========================================================
fig4, ax4 = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

t_fit_range = np.linspace(A_FIT, B_FIT, 100)

for (L_mm, sand), style in style_map.items():
    if (L_mm, sand) not in D_dict:
        continue

    D_mean  = D_dict[(L_mm, sand)]
    R0sq    = R0sq_dict[(L_mm, sand)]
    Ta_mean = 0.01 / u_dict[(L_mm, sand)]   # Ta reconstruit depuis u=0.01/Ta

    # droite de fit : R²(t/Ta) = R0² + 4D·Ta · (t/Ta)
    R2_fit = R0sq + 4.0 * D_mean * Ta_mean * t_fit_range

    for res in rms_idx.get((L_mm, sand), []):
        time    = res["time"]
        sigma_m = res["sigma_m"]
        R2      = sigma_m**2
        valid   = np.isfinite(time) & np.isfinite(R2)
        ax4.plot(time[valid][::3], R2[valid][::3] * 1e6,
                 linestyle="", **style)

    ax4.plot(t_fit_range, R2_fit * 1e6,
             color=style.get("color", "k"), lw=1.5, ls="-", alpha=0.9)

ax4.axvline(A_FIT, color="gray", ls=":", lw=0.8, label=f"fenêtre fit [{A_FIT}, {B_FIT}]")
ax4.axvline(B_FIT, color="gray", ls=":", lw=0.8)
ax4.set_xlabel(r"$t\,/\,T_a$")
ax4.set_ylabel(r"$R^2$ [mm²]")
ax4.set_xlim(1, 35)
ax4.set_ylim(1, 95)
ax4.grid(True, ls="--", alpha=0.3)
ax4.set_yscale("log")
ax4.set_xscale("log")
leg4 = [Line2D([0],[0], marker=s.get("marker","o"), ls="None",
               color=s.get("color","k"), mfc=s.get("mfc","none"),
               mew=s.get("mew",1.0), ms=s.get("ms",6),
               label=f"$d_2$={L} mm — {sd}")
        for (L,sd), s in style_map.items() if (L,sd) in D_dict]
leg4.append(Line2D([0],[0], color="k", ls="-", lw=1.5, label="fit linéaire"))
ax4.legend(handles=leg4, ncol=2)
plt.show()

# ==========================================================
# FIGURE 5 — D_perp_M / D_0mm  vs  Pe = d2 / alphaT_0mm  (log-log + fit puissance)
#
# Référence cohérente : D_0mm mesuré de la même façon sur R²
# alphaT_0mm = D_0mm / u_0mm  (recalculé ici, pas depuis le fichier diffusion)
# Pe    = d2 [m] / alphaT_0mm [m]       [adim]
# Dnorm = D_M / D_0mm                   [adim]
# ==========================================================
fig5, ax5 = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

Pe_all = []; D_all = []; D_err_all = []

for sand in ["fine", "coarse"]:
    # référence 0mm pour ce type de sable
    if (0, sand) not in D_dict:
        print(f"[WARN] pas de référence 0mm pour {sand}, skip")
        continue
    D_0mm   = D_dict[(0, sand)]       # [m²/s]
    alphaT  = D_0mm           # [m]  — dispersivité transverse du sable seul

    for (L_mm, sd), style in style_map.items():
        if sd != sand or (L_mm, sd) not in D_dict:
            continue

        d2_m      = L_mm * 1e-3                          # [m]
        Pe        = d2_m / alphaT                        # [adim]  (Pe=0 pour L_mm=0)
        Dnorm     = D_dict[(L_mm, sd)] / D_0mm           # [adim]
        Dnorm_err = D_err_dict[(L_mm, sd)] / D_0mm       # [adim]

        ax5.errorbar(Pe if L_mm > 0 else 0.3 * alphaT / alphaT,  # décale 0mm pour visibilité
                     Dnorm,
                     yerr=Dnorm_err if Dnorm_err > 0 else None,
                     fmt=style.get("marker","o"), ms=style.get("ms",6),
                     color=style.get("color","k"), mfc=style.get("mfc","none"),
                     mew=style.get("mew",1.0), capsize=4, elinewidth=1.0,
                     alpha=style.get("alpha",0.8), linestyle="None")

        if L_mm > 0:
            Pe_all.append(Pe)
            D_all.append(Dnorm)
            D_err_all.append(Dnorm_err)

# fit puissance en log-log  (w = SNR en espace log)
Pe_all    = np.array(Pe_all)
D_all     = np.array(D_all)
D_err_all = np.array(D_err_all)

valid_fit = (Pe_all > 0) & (D_all > 0) & np.isfinite(D_all) & np.isfinite(Pe_all)
w = np.where(D_err_all > 0, D_all / D_err_all, 1.0)

coeffs, cov = np.polyfit(np.log10(Pe_all[valid_fit]),
                         np.log10(D_all[valid_fit]),
                         1, w=w[valid_fit], cov=True)
slope     = coeffs[0]
intercept = coeffs[1]
slope_err = np.sqrt(cov[0, 0])

Pe_fit = np.logspace(np.log10(Pe_all[valid_fit].min()),
                     np.log10(Pe_all[valid_fit].max()), 200)
ax5.plot(Pe_fit, 10**intercept * Pe_fit**slope, "k-", lw=1.5,
         label=rf"fit : slope $= {slope:.2f} \pm {slope_err:.2f}$")

# ligne D/D_0mm = 1 pour référence
ax5.axhline(1, color="gray", ls=":", lw=0.8, alpha=0.6)

ax5.set_xlabel(r"$Pe = d_2\,/\,\alpha_{\perp,0}$")
ax5.set_ylabel(r"$D_{\perp M}\,/\,D_{\perp,0\mathrm{mm}}$")
ax5.set_yscale("log")
ax5.set_xscale("log")
ax5.grid(True, ls="--", alpha=0.3)
ax5.legend()
plt.show()