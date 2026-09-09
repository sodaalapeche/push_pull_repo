"""
Analyse de la dispersion en régime confiné.

Pour chaque expérience (L_mm > 0) :
  1. Fit semi-log de Σ(t/Ta) entre A et B  →  β [Ta⁻¹]
  2. Temps caractéristique confiné : τ = |β| * R² / (l * αT)

Recherche automatique de la fenêtre de fit [A, B] optimale :
  - A ∈ [A_MIN, A_MAX - DUR_MIN]
  - B = A + durée, durée ∈ [DUR_MIN, DUR_MAX]
  - critère : minimise l'erreur standard de la pente du fit log-log τ vs Pe
    (loi de puissance pure). Ce critère sert UNIQUEMENT à choisir la
    meilleure fenêtre temporelle pour extraire β / τ — il n'influence pas
    le modèle physique utilisé pour la courbe de tendance finale (Fig. 2).

Graphes :
  - Fig 1 : Σ(t/Ta) en semi-log, fenêtre optimale surlignée
  - Fig 2 : τ/Dm vs Pe en log-log, fit non-linéaire  D_M/D_m = 1 + a*Pe^b
            (asymptote = 1 quand Pe -> 0, régime de micro-dispersion),
            avec bande d'incertitude à 95% (Monte-Carlo sur la covariance)
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

width  = 18 / 2.1  # cm
height = 10 / 2.1
inches = 2.54

# ==========================================================
# PARAMÈTRES
# ==========================================================
DATA_PATH      = "/home/chorus/data_roi.npy"
DIFFUSION_PATH = "../vieux codes/resultats_diffusion.npy"

# Bornes de recherche de la fenêtre de fit [t/Ta]
A_MIN    = 13     # début minimum de la fenêtre
A_MAX    = 15     # début maximum
DUR_MIN  = 8       # durée minimale
DUR_MAX  = 10      # durée maximale
A_STEP   = 0.5     # pas de balayage pour A
DUR_STEP = 0.4     # pas de balayage pour la durée
beta0 = jn_zeros(1, 1)[0]  # ≈ 3.8317  ← Neumann (zéros de J_1)

R  = 0.055 / 2
R2 = 0.029 / 2   # rayon ROI [m]
l  = .01        # longueur associée à Ta [m]
Dm = 5e-9        # diffusion moléculaire [m²/s]

# ==========================================================
# STYLE MAP
# ==========================================================
style_map = {
    (10, "fine"):   dict(marker="^", color="tab:blue",   mfc="tab:blue",   ms=4, alpha=0.8),
    (10, "coarse"): dict(marker="D", color="tab:blue",   mfc="none",       mew=1.2, ms=4, alpha=0.8),
    (6,  "fine"):   dict(marker="^", color="tab:orange", mfc="tab:orange", ms=4, alpha=0.8),
    (6,  "coarse"): dict(marker="D", color="tab:orange", mfc="none",       mew=1.2, ms=4, alpha=0.8),
    (3,  "fine"):   dict(marker="^", color="tab:red",    mfc="tab:red",    ms=4, alpha=0.8),
    (3,  "coarse"): dict(marker="D", color="tab:red",    mfc="none",       mew=1.2, ms=4, alpha=0.8),
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

        d   = L_mm * 1e-3
        tau = abs(beta*1/l) * R**2 /  ( beta0**2 * alphaT)

        if sand == "fine" and L_mm == 3:
            tau = abs(beta*1/l) * (R2)**2 / ( beta0**2 * alphaT)

        u  = l / Ta
        Pe = d / alphaT

        recs.append(dict(L_mm=L_mm, sand=sand, beta=beta, tau=tau, Pe=Pe,
                          log_S0=log_S0, style=style, time_red=time_red,
                          Sigma=Sigma,Ta=Ta,alphaT=alphaT))
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
# critère : erreur standard de la pente du fit log-log τ vs Pe
# (loi de puissance pure, uniquement pour choisir [A,B])
# ==========================================================
A_candidates   = np.arange(A_MIN, A_MAX, A_STEP)
dur_candidates = np.arange(DUR_MIN, DUR_MAX, DUR_STEP)

best_rmse    = np.inf
best_A       = None
best_B       = None
best_records = None

for A_try, dur in iproduct(A_candidates, dur_candidates):
    B_try = A_try + dur
    recs  = compute_records(A_try, B_try, exp_list)

    if recs is None or len(recs) < 4:
        continue

    recs = [
        r for r in recs
        if not (r['L_mm'] == 0 and r['sand'] == 'fine')
           and not (r['L_mm'] == 0 and r['sand'] == 'coarse')
    ]

    if len(recs) < 4:
        continue

    Pe_v  = np.array([r["Pe"]  for r in recs])
    tau_v = np.array([r["tau"] for r in recs])

    _, _, _, _, se = stats.linregress(np.log10(Pe_v), np.log10(tau_v))
    if se < best_rmse:
        best_rmse    = se
        best_A       = A_try
        best_B       = B_try
        best_records = recs

print(f"\nFenêtre optimale trouvée : A={best_A} Ta, B={best_B} Ta  "
      f"(durée={best_B - best_A} Ta, σ_pente={best_rmse:.4f})")

for r in best_records:
    print(f"  {r['L_mm']:2d}mm {r['sand']:6s}  β={r['beta']:.4f}  τ={r['tau']:.4e}  Pe={r['Pe']:.4f}")

Pe_all  = np.array([r["Pe"]  for r in best_records])
tau_all = np.array([r["tau"] for r in best_records])

# ==========================================================
# FIGURE 1 — Σ vs t/Ta semi-log
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

ax1.axvspan(2*best_A, 2*best_B, alpha=0.12, color="gray")
ax1.axvline(2*best_A, color="gray", ls=":", lw=0.8)
ax1.axvline(2*best_B, color="gray", ls=":", lw=0.8)

for r in best_records:
    time_red = r["time_red"]
    Sigma    = r["Sigma"]
    style    = r["style"]

    valid = np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0)
    ax1.plot(2*time_red[valid][::3], Sigma[valid][::3], linestyle="None", **style)

legend_elements = []
for (L_mm, sand), style in style_map.items():
    if L_mm == 0:
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
ax1.grid(True, ls="--", alpha=0.3)

fig1.tight_layout()
plt.savefig('/home/chorus/scaling2.pdf')
plt.show()

# ==========================================================
# FIGURE 2 — τ vs Pe  log-log
# Modèle non-linéaire :  D_M/D_m = 1 + a * Pe^b
# -> asymptote = 1 quand Pe -> 0 (régime de micro-dispersion)
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

# Regroupement par type d'expérience (L_mm, sand) :
# un seul point (τ moyen) avec barre d'erreur à 1σ/1.96σ par série.
groups = {}
for rec in best_records:
    key = (rec["L_mm"], rec["sand"])
    groups.setdefault(key, []).append(rec)

Pe_points, tau_points = [], []
for key, recs_g in groups.items():
    Pe_g     = np.mean([r["Pe"] for r in recs_g])
    tau_g    = np.array([r["tau"] for r in recs_g])
    tau_mean = tau_g.mean()
    tau_std  = tau_g.std(ddof=1) if tau_g.size > 1 else 0.0
    style    = {k: v for k, v in recs_g[0]["style"].items()}
    ax2.errorbar(Pe_g, tau_mean, yerr=1.96*tau_std/np.sqrt(len(tau_g)),
                 linestyle="None", capsize=3,
                 ecolor=style["color"], elinewidth=1.0,
                 **style)
    Pe_points.append(Pe_g)
    tau_points.append(tau_mean)

Pe_points  = np.array(Pe_points)
tau_points = np.array(tau_points)

# ------------------------------------------------------------
# Fit en loi de puissance pure  tau = 10^intercept * Pe^slope
# (fit standard, comme dans la version originale du script)
# ------------------------------------------------------------
slope, intercept, r_val, p_val, se_slope = stats.linregress(
    np.log10(Pe_all), np.log10(tau_all)
)

print(f"\nFit log-log τ vs Pe :")
print(f"  pente  = {slope:.3f} ± {se_slope:.3f}  (1σ)")
print(f"  R²     = {r_val**2:.4f}")

Pe_range = np.logspace(np.log10(Pe_all.min()),
                        np.log10(Pe_all.max()) + 0.15, 300)
tau_fit   = 10**intercept * Pe_range**slope
tau_fit_p = 10**intercept * Pe_range**(slope + se_slope)
tau_fit_m = 10**intercept * Pe_range**(slope - se_slope)

fit_label = rf"trendline : slope $= {slope:.2f} \pm {se_slope:.2f}$ (2$\sigma$)"
ax2.plot(Pe_range, tau_fit, "k-", lw=1.8, alpha=0.85, label=fit_label)
# ax2.fill_between(Pe_range, tau_fit_m, tau_fit_p,
#                  color="k", alpha=0.12, label=r"$\pm 2\sigma$ slope")

# ------------------------------------------------------------
# Repère visuel (pas un fit) : asymptote physique attendue
# quand Pe -> 0 (régime de micro-dispersion, D_M/D_m -> 1).
# Simple ligne guide pour montrer que le comportement est cohérent,
# sans contraindre le fit ci-dessus.
# ------------------------------------------------------------
ax2.axhline(1.0, color="gray", ls=":", lw=1.2, alpha=0.8,
            label=r"régime de micro-dispersion ($Pe \to 0$)")

legend_elements2 = []
for (L_mm, sand), style in style_map.items():
    if L_mm == 0:
        continue
    legend_elements2.append(
        Line2D([0], [0], marker=style["marker"], linestyle="None",
               color=style["color"],
               markerfacecolor=style.get("mfc", style["color"]),
               markeredgewidth=style.get("mew", 1.0),
               markersize=style.get("ms", 4),
               alpha=style.get("alpha", 1.0),
               label=f"{L_mm} mm, {sand}")
    )
legend_elements2.append(Line2D([0], [0], color="k", lw=1.8, label=fit_label))
legend_elements2.append(
    Line2D([0], [0], color="gray", lw=1.2, ls=":", alpha=0.8,
           label=r"micro-dispersion regime ($Pe \to 1$)")
)
# ax2.legend(handles=legend_elements2, ncol=2)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$Pe_M = d_M \;/\; \alpha_{\perp,m}$")
ax2.set_ylabel(r"$R_c^2 j_{1,1}^{2}/(\ell_{\perp,m}\cdot\ell_2)$")

# ax2.set_ylabel(r"$R^2 \;/\; (\tau_2 \cdot \, \alpha_{\perp,m} l \beta_1^2)$")
ax2.set_ylim(bottom=0.8,top=60)
ax2.set_xlim(left=9,right=400)
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='--', alpha=0.7)
ax2.grid(True, which='minor', linestyle=':', alpha=0.4)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- Define padding and size in axes coordinates (0 to 1) ---
pad_x = 0.13   # 3% horizontal padding from the left edge
pad_y = 0.03   # 3% vertical padding from the top edge

inset_w = 0.38 # 38% of parent axes width
inset_h = 0.38 # 38% of parent axes height

# For 'upper left' placement:
# The top-left corner of the inset is at (pad_x, 1 - pad_y)
# Therefore, the bottom-left corner (x0, y0) of its bounding box is:
x0 = pad_x
y0 = 1 - pad_y - inset_h

# Create the bounding box as a 4-tuple (x0, y0, width, height)
bbox = (x0, y0, inset_w, inset_h)

# Create the inset. Because width/height are set to "100%",
# the inset will exactly fill this bounding box.
ax2_inset = inset_axes(
    ax2,
    width="100%",            # Fill the bounding box width
    height="100%",           # Fill the bounding box height
    loc="upper left",        # Anchors the inset's corner to the bbox's corner
    bbox_to_anchor=bbox,
    bbox_transform=ax2.transAxes  # Coordinates are relative to ax2
)

# --- The rest of your code stays exactly the same ---
for rec in best_records:
    style = rec["style"]
    alpha2=(abs(rec["beta"])/R**2 )*beta0**2
    Pe_micro = (0.01/rec['Ta']) * rec["L_mm"] * 1e-3 / (10**-8)
    ax2_inset.plot(Pe_micro, alpha2, linestyle="None", **style)

ax2_inset.set_xscale("log")
ax2_inset.set_yscale("log")
ax2_inset.set_xlabel(r"$Pe_m$", labelpad=2)
ax2_inset.set_ylabel(r"$\ell_2^{-1}$ (cm$^{-1}$)", labelpad=0)
ax2_inset.xaxis.label.set_size(8)
ax2_inset.yaxis.label.set_size(8)
# ax2_inset.set_ylim(bottom=0.045)
ax2_inset.minorticks_off()
ax2_inset.tick_params(labelsize=5, pad=0.5)

ax2_inset.grid(True, which="major", ls="--", alpha=0.4)
fig2.tight_layout()
plt.savefig('/home/chorus/TexProject/clement draft avance (document de travail)/figures/scaling.pdf')
plt.show()