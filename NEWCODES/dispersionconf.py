"""
Analyse de la dispersion en régime confiné.

Pour chaque expérience (L_mm > 0) :
  1. Fit semi-log de Σ(t/Ta) entre A et B  →  β [Ta⁻¹]
  2. Temps caractéristique confiné : τ = |β| * R² / (l * αT)

Recherche automatique de la fenêtre de fit [A, B] optimale :
  - A ∈ [A_MIN, A_MAX - DUR_MIN]
  - B = A + durée, durée ∈ [DUR_MIN, DUR_MAX]
  - critère : minimise l'erreur résiduelle du fit log-log τ vs Pe

Graphes :
  - Fig 1 : Σ(t/Ta) en semi-log, fenêtre optimale surlignée + droites de fit
  - Fig 2 : τ/Dm vs Pe en log-log, fit moindres carrés + bande ±1σ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from itertools import product as iproduct
import scienceplots
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")

width   = 15.5       # cm
height  = width * 0.5
inches  = 2.54

# ==========================================================
# PARAMÈTRES
# ==========================================================
DATA_PATH      = "/home/chorus/data_roi.npy"
DIFFUSION_PATH = "../vieux codes/resultats_diffusion.npy"

# Bornes de recherche de la fenêtre de fit [t/Ta]
A_MIN   = 13   # début minimum de la fenêtre
A_MAX   = 14    # début maximum
DUR_MIN =6 # durée minimale
DUR_MAX = 10     # durée maximale (A + DUR_MAX <= A_MAX + DUR_MAX, mais limité par les données)
A_STEP  = 0.5      # pas de balayage pour A
DUR_STEP = 0.4     # pas de balayage pour la durée
R  = 0.025    # rayon ROI [m]
l  = 0.01     # longueur associée à Ta [m]
Dm = 5e-9     # diffusion moléculaire [m²/s]

# ==========================================================
# STYLE MAP
# ==========================================================
style_map = {
    (10, "fine"):   dict(marker="o", color="tab:blue",   mfc="tab:blue",   ms=6, alpha=0.8),
    (10, "coarse"): dict(marker="D", color="tab:blue",   mfc="none",       mew=1.2, ms=6, alpha=0.8),
    (6,  "fine"):   dict(marker="o", color="tab:orange", mfc="tab:orange", ms=6, alpha=0.8),
    (6,  "coarse"): dict(marker="D", color="tab:orange", mfc="none",       mew=1.2, ms=6, alpha=0.8),
    (3,  "fine"):   dict(marker="o", color="tab:red",    mfc="tab:red",    ms=6, alpha=0.8),
    (3,  "coarse"): dict(marker="D", color="tab:red",    mfc="none",       mew=1.2, ms=6, alpha=0.8),
}

# ==========================================================
# CHARGEMENT
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
print(f"Chargé : {len(results)} expériences")

diff_data  = np.load(DIFFUSION_PATH, allow_pickle=True).item()
alphaT_map = {}
for sand, vals in diff_data["alphaT"].items():
    alphaT_map[sand] = float(np.mean(vals))
    print(f"  αT [{sand:6s}] = {alphaT_map[sand]:.4e} m")

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
# HELPER : fit semi-log sur une fenêtre donnée
# ==========================================================
def semilog_fit(time_red, Sigma, A, B):
    """Retourne (beta, log_S0) ou (nan, nan) si pas assez de points."""
    mask = (
        np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0) &
        (time_red >= A) & (time_red <= B)
    )
    t_f, S_f = time_red[mask], Sigma[mask]
    if t_f.size < 3:
        return np.nan, np.nan
    c = np.polyfit(t_f, np.log(S_f), 1)
    return c[0], c[1]

from scipy.special import jn_zeros

# ==========================================================
# FONCTION : calcule τ et Pe pour tous les records
# sur une fenêtre [A, B] donnée
# ==========================================================
def compute_records(A, B, exp_list):
    recs = []
    beta0 = jn_zeros(1, 1)[0]  # ≈ 3.8317  ← Neumann (zéros de J_1)
    for res, L_mm, sand, alphaT, style, time_red, Sigma, Ta in exp_list:
        beta, log_S0 = semilog_fit(time_red, Sigma, A, B)
        if np.isnan(beta):
            return None   # fenêtre invalide pour au moins une exp
        if sand == "coarse":
            sands=0.0006
        else:
            sands=0.00009
        d   = L_mm * 1e-3

        tau = abs(beta)* R**2 / (l *beta0**2* alphaT)
        # tau = abs(beta)
        u   = l / Ta
        Pe  = d / (alphaT)
        # Pe = d/sands
        recs.append(dict(L_mm=L_mm, sand=sand, beta=beta, tau=tau, Pe=Pe,
                         log_S0=log_S0, style=style, time_red=time_red,
                         Sigma=Sigma))
    return recs

# ==========================================================
# COLLECTE DES EXPÉRIENCES VALIDES
# ==========================================================
exp_list = []
for res in results:
    info = parse_label(res.get("label", ""))
    if info is None:
        continue
    L_mm, sand = info
    if L_mm == 0 :#or L_mm == 3:
        continue
    if sand not in alphaT_map:
        continue
    alphaT   = alphaT_map[sand]
    key      = (L_mm, sand)
    style    = style_map.get(key, dict(marker="s", color="gray", mfc="gray", ms=7, alpha=0.8))
    time_red = res["time"]
    mean     = res["mean"]
    var      = res["var"]
    i0       = res["i0"]
    A0       = res["A"]
    Ta       = res["Ta"]
    Sigma    = var / (A0 * mean**2)
    Sigma   /= Sigma[i0]
    exp_list.append((res, L_mm, sand, alphaT, style, time_red, Sigma, Ta))

print(f"\nExpériences retenues : {len(exp_list)}")

# ==========================================================
# RECHERCHE DE LA FENÊTRE OPTIMALE
# critère : résidu quadratique moyen du fit log-log τ vs Pe
# ==========================================================
A_candidates   = np.arange(A_MIN, A_MAX,   A_STEP)
dur_candidates = np.arange(DUR_MIN, DUR_MAX, DUR_STEP)

best_rmse   = np.inf
best_A      = None
best_B      = None
best_records = None

for A_try, dur in iproduct(A_candidates, dur_candidates):
    B_try = A_try + dur
    # recs  = compute_records(A_try, B_try, exp_list)

    recs = compute_records(A_try, B_try, exp_list)

    if recs is None or len(recs) < 4:
        continue

    recs = [
        r for r in recs
        if not (r['L_mm'] == 3 and r['sand'] == 'fine')
           and not (r['L_mm'] == 10 and r['sand'] == 'coarse')
    ]

    if len(recs) < 4:  # re-check after filtering
        continue

    Pe_v = np.array([r["Pe"] for r in recs])
    tau_v = np.array([r["tau"] for r in recs])

    _, _, _, _, se = stats.linregress(np.log10(Pe_v), np.log10(tau_v))
    # critère : erreur standard sur la pente (robuste au nombre de points)
    if se < best_rmse:
        best_rmse    = se
        best_A       = A_try
        best_B       = B_try
        best_records = recs

print(f"\nFenêtre optimale trouvée : A={best_A} Ta, B={best_B} Ta  "
      f"(durée={best_B-best_A} Ta, σ_pente={best_rmse:.4f})")

# Fit final sur la fenêtre optimale
Pe_all  = np.array([r["Pe"]  for r in best_records])
tau_all = np.array([r["tau"] for r in best_records])
log_Pe  = np.log10(Pe_all)
log_tau = np.log10(tau_all)
slope, intercept, r_val, p_val, se_slope = stats.linregress(log_Pe, log_tau)

print(f"\nFit log-log τ vs Pe :")
print(f"  pente  = {slope:.3f} ± {se_slope:.3f}  (1σ)")
print(f"  R²     = {r_val**2:.4f}")
for r in best_records:
    print(f"  {r['L_mm']:2d}mm {r['sand']:6s}  β={r['beta']:.4f}  τ={r['tau']:.4e}  Pe={r['Pe']:.4f}")

# ==========================================================
# FIGURE 1 — Σ vs t/Ta semi-log
# ==========================================================
fig1, ax1 =plt.subplots(figsize=(width/inches,height/inches),layout="constrained")

ax1.axvspan(2*best_A, 2*best_B, alpha=0.12, color="gray")
ax1.axvline(2*best_A, color="gray", ls=":", lw=0.8)
ax1.axvline(2*best_B, color="gray", ls=":", lw=0.8)

for r in best_records:
    time_red = r["time_red"]
    Sigma    = r["Sigma"]
    style    = r["style"]
    beta     = r["beta"]
    log_S0_r = r["log_S0"]

    valid = np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0)
    ax1.plot(2*time_red[valid][::3], Sigma[valid][::3], linestyle="None", **style)

    t_line = np.linspace(2*best_A, 2*best_B, 80)
    # ax1.plot(t_line, np.exp(log_S0_r + beta * t_line),
    #          color=style["color"], lw=1.6, ls="--", alpha=0.9)

legend_elements = []
for (L_mm, sand), style in style_map.items():
    if L_mm == 0:# or L_mm == 3:
        continue
    legend_elements.append(
        Line2D([0], [0], marker=style["marker"], linestyle="None",
               color=style["color"],
               markerfacecolor=style.get("mfc", style["color"]),
               markeredgewidth=style.get("mew", 1.0),
               markersize=style.get("ms", 6),
               alpha=style.get("alpha", 1.0),
               label=f"{L_mm} mm, {sand}")
    )
legend_elements.append(
    plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.15,
                  label=f"fit window : [{2*best_A}–{2*best_B}] $T_a$")
)

ax1.legend(handles=legend_elements, ncol=2)
ax1.axvline(0, color="k", ls="--", alpha=0.5)
ax1.set_xlabel(r"$t \;/\; T_a$")
ax1.set_ylabel(r"$\Sigma \;/\; \Sigma_0$")
ax1.set_yscale("log")
ax1.set_ylim(bottom=0.8e-2)
ax1.set_xlim(left=0.01, right=2*best_B + 8)
ax1.grid(True, ls="--", alpha=0.3)

fig1.tight_layout()
plt.savefig('/home/chorus/scaling2.pdf')

plt.show()
# ==========================================================
# FIGURE 2 — τ vs Pe  log-log
# ==========================================================
fig2, ax2=plt.subplots(figsize=(width/inches,height/inches),layout="constrained")

for rec in best_records:
    print(rec)

    style = {k: v for k, v in rec["style"].items()}
    ax2.plot(rec["Pe"], rec["tau"], linestyle="None", **style)

Pe_range  = np.logspace(np.log10(Pe_all.min()) - 0.15,
                        np.log10(Pe_all.max()) + 0.15, 300)
tau_fit   = 10**intercept * Pe_range**slope
tau_fit_p = 10**intercept * Pe_range**(slope + se_slope)
tau_fit_m = 10**intercept * Pe_range**(slope - se_slope)

fit_label = rf"trendline : slope $= {slope:.2f} \pm {se_slope:.2f}$ (1$\sigma$)"
ax2.plot(Pe_range, tau_fit, "k-", lw=1.8, alpha=0.85, label=fit_label)
ax2.fill_between(Pe_range, tau_fit_m, tau_fit_p,
                 color="k", alpha=0.12, label=r"$\pm 1\sigma$ slope")

legend_elements2 = []
for (L_mm, sand), style in style_map.items():
    if L_mm == 0 : #or L_mm == 3:
        continue
    if L_mm==3 and sand=='fine':
        continue
    if L_mm==10 and sand=='coarse':
        continue
    legend_elements2.append(
        Line2D([0], [0], marker=style["marker"], linestyle="None",
               color=style["color"],
               markerfacecolor=style.get("mfc", style["color"]),
               markeredgewidth=style.get("mew", 1.0),
               markersize=style.get("ms", 5),
               alpha=style.get("alpha", 1.0),
               label=f"{L_mm} mm, {sand}")
    )
legend_elements2.append(Line2D([0], [0], color="k", lw=1.8, label=fit_label))

ax2.legend(handles=legend_elements2)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$Pe = d_2 \;/\; \alpha_{\perp,m}$")
ax2.set_ylabel(r"$\frac{D_{\perp,M}}{D_{\perp,m}} = |\beta| \, R^2 \;/\; (l \, \alpha_{\perp,m} \beta_0^2)$")

ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='--', alpha=0.7)
ax2.grid(True, which='minor', linestyle=':', alpha=0.4)
fig2.tight_layout()
plt.savefig('/home/chorus/scaling.pdf')
plt.show()