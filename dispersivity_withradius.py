"""
Dispersivité transverse αT — R(t) issu de la mesure RMS.

Pour chaque expérience (L_mm, sand) :

  R(t) = sigma_m(t)   [m]  depuis data_rms  (rayon RMS du panache)
  Σ(t)               [—]  depuis data_roi  (variance normalisée)

  Sur la fenêtre [A, B] en t/Ta, point par point :

      αT(t) = |d log Σ / dt| · R(t)² / (l · β0² · d2)

  où d log Σ / dt est estimé par différences finies centrées sur Σ(t).
  R(t) est interpolé sur la grille temporelle de Σ(t).

  αT = moyenne temporelle de αT(t) sur [A, B]
  σ  = std temporel / sqrt(N)  (erreur sur la moyenne)

Figure :
  αT / d2  vs  Pe = d2 / αT_grain   log-log
  fit moindres carrés pondérés + bande ±1σ + courbe théorique Pe^{-0.15}
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import jn_zeros

from itertools import product as iproduct
import scienceplots

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")

width  = 15.5
height = width * 0.5
inches = 2.54

# ==========================================================
# PARAMÈTRES
# ==========================================================
ROI_PATH       = "/home/chorus/data_roi.npy"
RMS_PATH       = "/home/chorus/data_rms.npy"
DIFFUSION_PATH = "../vieux codes/resultats_diffusion.npy"

l     = 0.01               # longueur associée à Ta [m]
beta0 = jn_zeros(1, 1)[0]  # ≈ 3.8317

# Bornes de recherche de la fenêtre de fit [t/Ta]
A_MIN    = 13
A_MAX    = 14
DUR_MIN  = 6
DUR_MAX  = 10
A_STEP   = 0.5
DUR_STEP = 0.4

# Fenêtre fixe pour le calcul ponctuel de αT(t)
A_DERIV = 17
B_DERIV = 22

# ==========================================================
# STYLE MAP
# ==========================================================
style_map = {
    (10, "fine"):   dict(marker="o", color="tab:blue",   mfc="tab:blue",   ms=6, mew=0.4, alpha=0.8),
    (10, "coarse"): dict(marker="D", color="tab:blue",   mfc="none",       ms=6, mew=1.2, alpha=0.8),
    (6,  "fine"):   dict(marker="o", color="tab:orange", mfc="tab:orange", ms=6, mew=0.4, alpha=0.8),
    (6,  "coarse"): dict(marker="D", color="tab:orange", mfc="none",       ms=6, mew=1.2, alpha=0.8),
    (3,  "fine"):   dict(marker="o", color="tab:red",    mfc="tab:red",    ms=6, mew=0.4, alpha=0.8),
    (3,  "coarse"): dict(marker="D", color="tab:red",    mfc="none",       ms=6, mew=1.2, alpha=0.8),
}

EXCLUDE_FIT = {(3, "fine"), (10, "coarse")}

# ==========================================================
# CHARGEMENT
# ==========================================================
roi_results = np.load(ROI_PATH, allow_pickle=True)
rms_results = np.load(RMS_PATH, allow_pickle=True)
diff_data   = np.load(DIFFUSION_PATH, allow_pickle=True).item()

alphaT_grain = {sand: float(np.mean(vals))
                for sand, vals in diff_data["alphaT"].items()}

print("αT grain :")
for sand, val in alphaT_grain.items():
    print(f"  {sand:6s} : {val:.4e} m")

# ==========================================================
# INDEXATION
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
# ESTIMATION PONCTUELLE DE αT(t)
# ==========================================================
def estimate_alphaT(key, d2, A, B):
    """
    αT = |β| · R̄² / (l · β0² · d2)

    β   : pente du fit semi-log de Σ(t) sur [A, B]  (depuis data_roi)
    R̄  : moyenne de sigma_m(t) sur [A_DERIV, B_DERIV] (depuis data_rms)
          remplace le R fixe = 0.025 m par le rayon mesuré

    Retourne (alpha_mean, alpha_sigma) ou (nan, nan).
    """
    roi_list = roi_idx.get(key, [])
    rms_list = rms_idx.get(key, [])

    if not roi_list or not rms_list:
        return np.nan, np.nan

    alphas_rep = []

    for roi_res, rms_res in zip(roi_list, rms_list):

        # --- β depuis Σ(t) sur [A, B] ---
        time_roi = roi_res["time"]
        var      = roi_res["var"]
        mean_c   = roi_res["mean"]
        A0       = roi_res["A"]
        i0       = roi_res["i0"]

        Sigma  = var / (A0 * mean_c ** 2)
        Sigma /= Sigma[i0]

        mask_roi = (
            np.isfinite(time_roi) & np.isfinite(Sigma) & (Sigma > 0) &
            (time_roi >= A) & (time_roi <= B)
        )
        t_f, S_f = time_roi[mask_roi], Sigma[mask_roi]
        if t_f.size < 3:
            continue
        c    = np.polyfit(t_f, np.log(S_f), 1)
        beta = c[0]
        if beta >= 0:
            continue

        # --- R̄ = mean(sigma_m) sur [A_DERIV, B_DERIV] depuis data_rms ---
        time_rms = rms_res["time"]
        sigma_m  = rms_res["sigma_m"]

        mask_rms = (
            np.isfinite(time_rms) & np.isfinite(sigma_m) & (sigma_m > 0) &
            (time_rms >= A_DERIV) & (time_rms <= B_DERIV)
        )
        if mask_rms.sum() < 3:
            continue
        R_mean = float(np.mean(sigma_m[mask_rms]))

        # --- αT ---
        alpha = abs(beta) * R_mean ** 2 / (l * beta0 ** 2 * d2)
        alphas_rep.append(alpha)

    if len(alphas_rep) == 0:
        return np.nan, np.nan

    alphas_rep  = np.array(alphas_rep)
    alpha_mean  = float(np.mean(alphas_rep))
    alpha_sigma = float(np.std(alphas_rep)) if len(alphas_rep) > 1 \
                  else alpha_mean * 0.10

    return alpha_mean, alpha_sigma

# ==========================================================
# RECHERCHE DE LA FENÊTRE OPTIMALE
# critère : σ_pente du fit log-log αT/d2 vs Pe
# ==========================================================
print("\nRecherche de la fenêtre optimale...")

A_candidates   = np.arange(A_MIN, A_MAX,   A_STEP)
dur_candidates = np.arange(DUR_MIN, DUR_MAX, DUR_STEP)

best_se     = np.inf
best_A      = None
best_B      = None
best_points = None

for A_try, dur in iproduct(A_candidates, dur_candidates):
    B_try  = A_try + dur
    points = []

    for key in style_map:
        L_mm, sand = key
        if sand not in alphaT_grain:
            continue

        d2           = L_mm * 1e-3
        alpha, sigma = estimate_alphaT(key, d2, A_try, B_try)

        if not (np.isfinite(alpha) and alpha > 0 and sigma > 0):
            break   # expérience invalide → fenêtre rejetée

        Pe = d2 / alphaT_grain[sand]
        points.append(dict(key=key, Pe=Pe, alpha=alpha, sigma=sigma, d2=d2))
    else:
        fit_pts = [p for p in points if p["key"] not in EXCLUDE_FIT]
        if len(fit_pts) < 4:
            continue

        Pe_v  = np.array([p["Pe"]             for p in fit_pts])
        aTn_v = np.array([p["alpha"] / p["d2"] for p in fit_pts])

        if not (np.all(Pe_v > 0) and np.all(aTn_v > 0)):
            continue

        _, _, _, _, se = stats.linregress(np.log10(Pe_v), np.log10(aTn_v))
        if se < best_se:
            best_se     = se
            best_A      = A_try
            best_B      = B_try
            best_points = points

print(f"Fenêtre optimale : A={best_A} Ta, B={best_B} Ta  "
      f"(durée={best_B - best_A:.1f} Ta, σ_pente={best_se:.4f})")

# ==========================================================
# FIT FINAL log-log  αT/d2  vs  Pe
# ==========================================================
fit_pts = [p for p in best_points if p["key"] not in EXCLUDE_FIT]

Pe_fit  = np.array([p["Pe"]             for p in fit_pts])
aTn_fit = np.array([p["alpha"] / p["d2"] for p in fit_pts])
sig_fit = np.array([p["sigma"] / p["d2"] for p in fit_pts])

w_fit = aTn_fit / sig_fit
w_fit = np.where(np.isfinite(w_fit) & (w_fit > 0), w_fit, 1.0)

coeffs, cov = np.polyfit(np.log10(Pe_fit), np.log10(aTn_fit), 1,
                         w=w_fit, cov=True)
slope     = coeffs[0]
intercept = coeffs[1]
slope_err = np.sqrt(cov[0, 0])

print(f"\nFit log-log αT/d2 vs Pe :")
print(f"  pente = {slope:.3f} ± {slope_err:.3f}  (1σ)")
print(f"\nDétail :")
for p in best_points:
    L_mm, sand = p["key"]
    tag = " [exclu fit]" if p["key"] in EXCLUDE_FIT else ""
    print(f"  {L_mm:2d}mm {sand:6s}  Pe={p['Pe']:.2f}  "
          f"αT={p['alpha']:.4e} ± {p['sigma']:.4e}{tag}")

# ==========================================================
# FIGURE
# ==========================================================
fig, ax = plt.subplots(figsize=(width / inches, height / inches),
                       layout="constrained")

for p in best_points:
    style = style_map[p["key"]]
    ax.errorbar(
        p["Pe"], p["alpha"] / p["d2"],
        yerr=p["sigma"] / p["d2"],
        fmt=style["marker"],
        ms=style["ms"],
        color=style["color"],
        mfc=style["mfc"],
        mew=style.get("mew", 1.0),
        capsize=3,
        elinewidth=0.8,
        alpha=style["alpha"],
        linestyle="None",
    )

Pe_range   = np.logspace(np.log10(Pe_fit.min()) - 0.15,
                         np.log10(Pe_fit.max()) + 0.15, 300)
aT_curve   = 10 ** intercept * Pe_range ** slope
aT_curve_p = 10 ** intercept * Pe_range ** (slope + slope_err)
aT_curve_m = 10 ** intercept * Pe_range ** (slope - slope_err)

fit_label = rf"fit : slope $= {slope:.2f} \pm {slope_err:.2f}$"
ax.plot(Pe_range, aT_curve,   "k-",  lw=1.8, alpha=0.85, label=fit_label)
ax.fill_between(Pe_range, aT_curve_m, aT_curve_p,
                color="k", alpha=0.10, label=r"$\pm 1\sigma$ slope")

# Théorie Pe^{-0.15} calée à la médiane géométrique
Pe_med     = np.sqrt(Pe_fit.min() * Pe_fit.max())
aT_med     = 10 ** intercept * Pe_med ** slope
scale_theo = aT_med / Pe_med ** (-0.15)
ax.plot(Pe_range, scale_theo * Pe_range ** (-0.15),
        "k--", lw=1.0, alpha=0.5, label=r"$Pe^{-0.15}$ (théorie)")

legend_elements = []
for (L_mm, sand), style in style_map.items():
    tag = " (exclu fit)" if (L_mm, sand) in EXCLUDE_FIT else ""
    legend_elements.append(
        Line2D([0], [0],
               marker=style["marker"], linestyle="None",
               color=style["color"], markerfacecolor=style["mfc"],
               markeredgewidth=style.get("mew", 1.0),
               markersize=style["ms"], alpha=style["alpha"],
               label=f"$d_2$ = {L_mm} mm, {sand}{tag}")
    )
legend_elements.append(Line2D([0], [0], color="k", lw=1.8, label=fit_label))
legend_elements.append(Line2D([0], [0], color="k", lw=1.0, ls="--",
                               alpha=0.5, label=r"$Pe^{-0.15}$ (théorie)"))

ax.legend(handles=legend_elements, fontsize=7)
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel(r"$Pe = d_2 \;/\; \alpha_{\perp,\mathrm{grain}}$")
ax.set_ylabel(r"$\alpha_{\perp,M} \;/\; d_2$")
ax.minorticks_on()
ax.grid(True, which="major", ls="--", alpha=0.5)
ax.grid(True, which="minor", ls=":",  alpha=0.3)

fig.savefig("/home/chorus/dispersivity_combined.pdf")
plt.show()