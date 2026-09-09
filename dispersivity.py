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
from scipy.special import jn_zeros

# ==========================================================
# STYLE - HOMOGENIZED
# ==========================================================
plt.style.use("science")

# Figure dimensions (homogenized)
FIG_WIDTH = 18/1.4      # cm
FIG_HEIGHT = 10/1.4
INCHES = 2.54

# Marker styles - homogenized
MARKER_STYLES = {
    (10, "fine"):   dict(marker="o", markersize=5, markeredgewidth=0.5, alpha=0.85),
    (10, "coarse"): dict(marker="D", markersize=5.5, markeredgewidth=1.0, alpha=0.85),
    (6,  "fine"):   dict(marker="o", markersize=5, markeredgewidth=0.5, alpha=0.85),
    (6,  "coarse"): dict(marker="D", markersize=5.5, markeredgewidth=1.0, alpha=0.85),
    (3,  "fine"):   dict(marker="o", markersize=5, markeredgewidth=0.5, alpha=0.85),
    (3,  "coarse"): dict(marker="D", markersize=5.5, markeredgewidth=1.0, alpha=0.85),
}

COLOR_MAP = {
    10: "tab:blue",
    6:  "tab:orange",
    3:  "tab:red",
}

# ==========================================================
# PARAMÈTRES
# ==========================================================
DATA_PATH      = "/home/chorus/data_roi.npy"
DIFFUSION_PATH = "../vieux codes/resultats_diffusion.npy"

# Bornes de recherche de la fenêtre de fit [t/Ta]
A_MIN   = 14
A_MAX   = 19
DUR_MIN = 4
DUR_MAX = 6
A_STEP  = 0.5
DUR_STEP = 0.4

R  = 0.057/2
R2 = 0.028/2
l  = 0.01

def get_style(L_mm, sand):
    """Get homogenized marker style."""
    style = MARKER_STYLES.get((L_mm, sand), dict(marker="o", markersize=5, alpha=0.8))
    color = COLOR_MAP.get(L_mm, "tab:gray")
    return {
        "marker": style["marker"],
        "markersize": style["markersize"],
        "markeredgewidth": style["markeredgewidth"],
        "alpha": style["alpha"],
        "color": color,
        "mfc": color if sand == "fine" else "none",
        "mec": color,
    }

def make_legend_element(L_mm, sand, label=None):
    """Create a legend element with homogenized style."""
    style = get_style(L_mm, sand)
    if label is None:
        label = f"{L_mm} mm, {sand}"
    return Line2D(
        [0], [0],
        marker=style["marker"],
        linestyle="None",
        color=style["color"],
        markerfacecolor=style["mfc"],
        markeredgecolor=style["mec"],
        markeredgewidth=style["markeredgewidth"],
        markersize=style["markersize"],
        alpha=style["alpha"],
        label=label
    )

# ==========================================================
# CHARGEMENT
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
print(f"Chargé : {len(results)} expériences")

diff_data = np.load(DIFFUSION_PATH, allow_pickle=True).item()
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
# ==========================================================
def compute_records(A, B, exp_list):
    recs = []
    beta0 = jn_zeros(1, 1)[0]
    for res, L_mm, sand, alphaT, time_red, Sigma, Ta in exp_list:
        beta, log_S0 = semilog_fit(time_red, Sigma, A, B)
        if np.isnan(beta):
            return None
        d = L_mm * 1e-3
        tau = abs(beta) * R**2 / (l * beta0**2 * d)
        if L_mm == 3 and sand == "fine":
            tau = abs(beta) * (R2)**2 / (l * beta0**2 * d)
        Pe = d / alphaT
        recs.append(dict(
            L_mm=L_mm, sand=sand, beta=beta, tau=tau, Pe=Pe,
            log_S0=log_S0, time_red=time_red, Sigma=Sigma, Ta=Ta
        ))
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
    if L_mm == 0:
        continue
    if sand not in alphaT_map:
        continue
    alphaT = alphaT_map[sand]
    time_red = res["time"]
    mean = res["mean"]
    var = res["var"]
    i0 = res["i0"]
    A0 = res["A"]
    Ta = res["Ta"]
    Sigma = var / (A0 * mean**2)
    Sigma /= Sigma[i0]
    exp_list.append((res, L_mm, sand, alphaT, time_red, Sigma, Ta))

print(f"\nExpériences retenues : {len(exp_list)}")

# ==========================================================
# RECHERCHE DE LA FENÊTRE OPTIMALE
# ==========================================================
A_candidates = np.arange(A_MIN, A_MAX, A_STEP)
dur_candidates = np.arange(DUR_MIN, DUR_MAX, DUR_STEP)

best_rmse = np.inf
best_A = None
best_B = None
best_records = None

for A_try, dur in iproduct(A_candidates, dur_candidates):
    B_try = A_try + dur
    recs = compute_records(A_try, B_try, exp_list)
    if recs is None or len(recs) < 4:
        continue
    recs = [r for r in recs if not (r['L_mm'] == 0 and r['sand'] == 'coarse')]
    if len(recs) < 4:
        continue
    Pe_v = np.array([r["Pe"] for r in recs])
    tau_v = np.array([r["tau"] for r in recs])
    _, _, _, _, se = stats.linregress(np.log10(Pe_v), np.log10(tau_v))
    if se < best_rmse:
        best_rmse = se
        best_A = A_try
        best_B = B_try
        best_records = recs

print(f"\nFenêtre optimale trouvée : A={best_A} Ta, B={best_B} Ta  "
      f"(durée={best_B-best_A} Ta, σ_pente={best_rmse:.4f})")

# Fit final sur la fenêtre optimale
Pe_all = np.array([r["Pe"] for r in best_records])
tau_all = np.array([r["tau"] for r in best_records])
log_Pe = np.log10(Pe_all)
log_tau = np.log10(tau_all)

# Forced slope
slope = -0.1
intercept = float(np.mean(log_tau - slope * log_Pe))
residuals = log_tau - (slope * log_Pe + intercept)

print(f"\nFit log-log τ vs Pe (pente forcée) :")
print(f"  pente     = {slope:.3f} (forcée)")
print(f"  intercept = {intercept:.3f}")
print(f"  RMSE      = {np.std(residuals):.3f}")

# ==========================================================
# GROUP RECORDS BY (L_mm, sand) FOR ERRORBARS
# ==========================================================
from collections import defaultdict

groups = defaultdict(list)
for r in best_records:
    key = (r["L_mm"], r["sand"])
    groups[key].append(r)

# Compute mean and std for each group
group_stats = {}
for key, recs in groups.items():
    Pe_mean = np.mean([r["Pe"] for r in recs])
    tau_vals = np.array([r["tau"] for r in recs])
    tau_mean = np.mean(tau_vals)
    tau_std = np.std(tau_vals, ddof=1) if len(tau_vals) > 1 else 0.0
    group_stats[key] = {
        "Pe": Pe_mean,
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "n": len(tau_vals),
        "records": recs
    }

print(f"\nGroupes avec {len(group_stats)} points pour les barres d'erreur:")

# ==========================================================
# FIGURE 1 — Σ vs t/Ta semi-log
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH/INCHES, FIG_HEIGHT/INCHES), layout="constrained")

ax1.axvspan(2*best_A, 2*best_B, alpha=0.12, color="gray")
ax1.axvline(2*best_A, color="gray", ls=":", lw=0.8)
ax1.axvline(2*best_B, color="gray", ls=":", lw=0.8)

for r in best_records:
    time_red = r["time_red"]
    Sigma = r["Sigma"]
    style = get_style(r["L_mm"], r["sand"])

    valid = np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0)
    ax1.plot(2*time_red[valid][::3], Sigma[valid][::3],
             marker=style["marker"], linestyle="None",
             color=style["color"],
             markerfacecolor=style["mfc"],
             markeredgecolor=style["mec"],
             markersize=style["markersize"],
             markeredgewidth=style["markeredgewidth"],
             alpha=style["alpha"])

legend_elements = []
for (L_mm, sand) in [(3,"fine"), (6,"fine"), (10,"fine"),
                     (3,"coarse"), (6,"coarse"), (10,"coarse")]:
    if (L_mm, sand) in group_stats:
        legend_elements.append(make_legend_element(L_mm, sand))
legend_elements.append(
    plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.15,
                  label=f"fit window : [{2*best_A}–{2*best_B}] $T_a$")
)

ax1.legend(handles=legend_elements, ncol=2, frameon=False)
ax1.axvline(0, color="k", ls="--", alpha=0.5)
ax1.set_xlabel(r"$t \,/\, T_a$")
ax1.set_ylabel(r"$\Sigma \,/\, \Sigma_0$")
ax1.set_yscale("log")
ax1.set_ylim(bottom=0.8e-2)
ax1.set_xlim(left=0.01, right=2*best_B + 8)
ax1.grid(True, ls="--", alpha=0.3)

fig1.tight_layout()
fig1.savefig('/home/chorus/scaling2.pdf', dpi=150, bbox_inches="tight")
plt.show()

# ==========================================================
# FIGURE 2 — τ vs Pe log-log WITH ERRORBARS BY GROUP
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH/INCHES, FIG_HEIGHT/INCHES), layout="constrained")

# Plot each group with errorbars
for key, stats in group_stats.items():
    L_mm, sand = key
    style = get_style(L_mm, sand)

    # Errorbar: mean ± std
    ax2.errorbar(
        stats["Pe"], stats["tau_mean"],
        yerr=stats["tau_std"] if stats["n"] > 1 else 0.0,
        marker=style["marker"],
        markersize=style["markersize"],
        color=style["color"],
        markerfacecolor=style["mfc"],
        markeredgecolor=style["mec"],
        markeredgewidth=style["markeredgewidth"],
        alpha=style["alpha"],
        linestyle="None",
        capsize=3,
        capthick=0.8,
        elinewidth=0.8,
        label=f"{L_mm} mm, {sand} (n={stats['n']})"
    )

    # Also plot individual points for reference (semi-transparent)
    for r in stats["records"]:
        ax2.plot(r["Pe"], r["tau"],
                 marker=style["marker"],
                 markersize=style["markersize"] * 0.6,
                 color=style["color"],
                 markerfacecolor=style["mfc"],
                 markeredgecolor=style["mec"],
                 alpha=0.3,
                 linestyle="None")

# Fit line
Pe_range = np.logspace(np.log10(Pe_all.min()) - 0.15,
                       np.log10(Pe_all.max()) + 0.15, 300)
tau_fit = 10**intercept * Pe_range**slope

fit_label = rf"trendline : slope $= {-slope:.1f}$"
ax2.plot(Pe_range, tau_fit, "k-", lw=1.2, alpha=0.85, label=fit_label)

# Legend
legend_elements2 = []
for key, stats in group_stats.items():
    L_mm, sand = key
    style = get_style(L_mm, sand)
    legend_elements2.append(
        Line2D([0], [0],
               marker=style["marker"],
               linestyle="None",
               color=style["color"],
               markerfacecolor=style["mfc"],
               markeredgecolor=style["mec"],
               markeredgewidth=style["markeredgewidth"],
               markersize=style["markersize"],
               alpha=style["alpha"],
               label=f"{L_mm} mm, {sand} (n={stats['n']})")
    )
legend_elements2.append(Line2D([0], [0], color="k", lw=1.2, label=fit_label))

ax2.legend(handles=legend_elements2,ncol=2, loc="lower right")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylim(0.01, 0.2)
ax2.set_xlabel(r"$Pe \propto d_2 / \alpha_{\perp,m}$", fontsize=9)
ax2.set_ylabel(r"$\alpha_{\perp,M} / d_2$", fontsize=9)
ax2.minorticks_on()
ax2.grid(True, which='major', linestyle='--', alpha=0.7)
ax2.grid(True, which='minor', linestyle=':', alpha=0.4)

fig2.tight_layout()
fig2.savefig('/home/chorus/dispersivity.pdf', dpi=150, bbox_inches="tight")
plt.show()

# ==========================================================
# PRINT SUMMARY TABLE
# ==========================================================
print("\n" + "=" * 70)
print("RÉSUMÉ - Points groupés pour barres d'erreur")
print("=" * 70)
print(f"{'L_mm':>6} {'sand':>10} {'Pe':>10} {'τ_mean':>12} {'τ_std':>12} {'n':>6}")
print("-" * 70)

for key, stats in sorted(group_stats.items()):
    L_mm, sand = key
    print(f"{L_mm:>6} {sand:>10} {stats['Pe']:>10.2f} "
          f"{stats['tau_mean']:>12.4e} {stats['tau_std']:>12.4e} {stats['n']:>6}")

print("=" * 70)

