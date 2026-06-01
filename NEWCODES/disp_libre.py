"""
Dispersion en régime libre (non confiné).

Pour chaque expérience (L_mm > 0, fine + coarse) :
  1. Fit semi-log de Σ(t/Ta) entre A et B  →  β [Ta⁻¹]
  2. R(t∈[A,B]) extrait de data_rms (sigma_m moyenné sur la fenêtre)
  3. Dispersivité mesurée : α_M = -β · R² / (4 · l)         [m]
  4. Rapport adimensionnel : α_M / α_T  =  D_M / D_m         [-]
  5. Pe = d2 / α_T                                            [-]

Dérivation :
  En milieu libre Fickien 2D, Σ ∝ 1/R² et R² = R0² + 4Dt
  => d(ln Σ)/dt = -4D/R²
  Avec β = d(ln Σ)/d(t/Ta) [adim] et u = l/Ta :
  D = -β R² / (4 Ta)             [m²/s]
  α_M = D/u = -β R² / (4 l)      [m]
  α_M / α_T = -β R² / (4 l α_T)  [-]

Graphes :
  - Fig 1 : Σ(t/Ta) semi-log, fenêtre [A,B] surlignée + droites de fit
  - Fig 2 : α_M/α_T vs Pe en log-log + fit moindres carrés ± 1σ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from itertools import product as iproduct
import scienceplots

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")

width  = 14      # cm
height = width * 0.5
inches = 2.54

# ==========================================================
# PARAMÈTRES — À AJUSTER
# ==========================================================
DATA_ROI_PATH  = "/home/chorus/data_roi.npy"
DATA_RMS_PATH  = "/home/chorus/data_rms.npy"
DIFFUSION_PATH = "../vieux codes/resultats_diffusion.npy"

# Bornes de recherche de la fenêtre de fit [t/Ta]
A_MIN    = 10.0    # début minimum de la fenêtre
A_MAX    = 25.0    # début maximum
DUR_MIN  = 1.0     # durée minimale
DUR_MAX  = 3.0     # durée maximale
A_STEP   = 0.2     # pas de balayage pour A
DUR_STEP = 0.2     # pas de balayage pour la durée

# Longueur de référence pour Ta : Ta = l/u
l = 0.01    # [m]

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
roi_results = np.load(DATA_ROI_PATH, allow_pickle=True)
rms_results = np.load(DATA_RMS_PATH, allow_pickle=True)
print(f"data_roi : {len(roi_results)} expériences")
print(f"data_rms : {len(rms_results)} expériences")

diff_data  = np.load(DIFFUSION_PATH, allow_pickle=True).item()
alphaT_map = {}
for sand, vals in diff_data["alphaT"].items():
    alphaT_map[sand] = float(np.mean(vals))
    print(f"  αT [{sand:6s}] = {alphaT_map[sand]:.4e} m")

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
# INDEXATION (L_mm, sand) -> liste de résultats
# ==========================================================
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
    """Retourne (time_red, Sigma) normalisé à i0."""
    time  = res["time"]
    var   = res["var"]
    mean  = res["mean"]
    A0    = res["A"]
    i0    = res["i0"]
    Sigma = var / (A0 * mean**2)
    Sigma = Sigma / Sigma[i0]
    return time, Sigma, i0

def R_window(rms_res, t_A, t_B):
    """Retourne R moyen sur la fenêtre [t_A, t_B] depuis sigma_m [m]."""
    time    = rms_res["time"]
    sigma_m = rms_res["sigma_m"]
    valid = (np.isfinite(time) & np.isfinite(sigma_m) &
             (sigma_m > 0) & (time >= t_A) & (time <= t_B))
    if valid.sum() < 2:
        return np.nan
    return float(np.mean(sigma_m[valid]))

def semilog_fit(time_red, Sigma, t_A, t_B):
    """Fit semi-log de Σ sur [t_A, t_B]. Retourne (beta, log_S0)."""
    mask = (
        np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0) &
        (time_red >= t_A) & (time_red <= t_B)
    )
    t_f, S_f = time_red[mask], Sigma[mask]
    if t_f.size < 3:
        return np.nan, np.nan
    c = np.polyfit(t_f, np.log(S_f), 1)
    return c[0], c[1]

# ==========================================================
# COLLECTE DES EXPÉRIENCES VALIDES (avant balayage)
# ==========================================================
exp_list = []
for (L_mm, sand), roi_list in roi_idx.items():
    if L_mm == 0:
        continue
    if sand not in alphaT_map:
        continue

    alphaT = alphaT_map[sand]
    style  = style_map.get((L_mm, sand),
                           dict(marker="s", color="gray", mfc="gray", ms=6, alpha=0.8))
    rms_list = rms_idx.get((L_mm, sand), [])

    for i, res_roi in enumerate(roi_list):
        time_red, Sigma, i0 = sigma_curve(res_roi)
        rms_paired = rms_list[i] if i < len(rms_list) else (rms_list[0] if rms_list else None)
        if rms_paired is None:
            continue
        Ta = float(res_roi["Ta"])
        exp_list.append((L_mm, sand, alphaT, style, time_red, Sigma,
                         rms_paired, Ta, res_roi, i))

print(f"\nExpériences retenues : {len(exp_list)}")

# ==========================================================
# FONCTION : calcule α_M/α_T et Pe pour tous les records
# sur une fenêtre [A, B] donnée
# ==========================================================
def compute_records(t_A, t_B, exp_list):
    recs = []
    for (L_mm, sand, alphaT, style, time_red, Sigma,
         rms_paired, Ta, res_roi, i) in exp_list:

        beta, log_S0 = semilog_fit(time_red, Sigma, t_A, t_B)
        if np.isnan(beta) or beta >= 0:
            return None   # fenêtre invalide pour au moins une exp

        R_w = R_window(rms_paired, t_A, t_B)
        if np.isnan(R_w) or R_w <= 0:
            return None

        alpha_M     = -beta * R_w**2 / (4.0 * l)
        alpha_ratio = alpha_M / alphaT
        Pe          = (L_mm * 1e-3) / alphaT
        u           = l / Ta
        D           = alpha_M * u
        Dm          = alphaT * u

        recs.append(dict(
            L_mm=L_mm, sand=sand, beta=beta, R_w=R_w,
            alpha_M=alpha_M, alpha_ratio=alpha_ratio, Pe=Pe,
            log_S0=log_S0, style=style,
            time_red=time_red, Sigma=Sigma, Ta=Ta,
            D=D, Dm=Dm
        ))
    return recs

# ==========================================================
# RECHERCHE DE LA FENÊTRE OPTIMALE
# critère : erreur standard (1σ) sur la pente du fit log-log
# ==========================================================
A_candidates   = np.arange(A_MIN, A_MAX,   A_STEP)
dur_candidates = np.arange(DUR_MIN, DUR_MAX, DUR_STEP)

best_se      = np.inf
best_A       = None
best_B       = None
best_records = None
best_slope   = None
best_R2      = None

print("\nBalayage de la fenêtre [A, B]...")
n_tested = 0
for A_try, dur in iproduct(A_candidates, dur_candidates):
    B_try = A_try + dur
    recs  = compute_records(A_try, B_try, exp_list)
    if recs is None or len(recs) < 4:
        continue

    Pe_v    = np.array([r["Pe"]          for r in recs])
    ratio_v = np.array([r["alpha_ratio"] for r in recs])
    if np.any(Pe_v <= 0) or np.any(ratio_v <= 0):
        continue

    slope, intercept, r_val, _, se = stats.linregress(
        np.log10(Pe_v), np.log10(ratio_v))
    n_tested += 1
    if se < best_se:
        best_se      = se
        best_A       = A_try
        best_B       = B_try
        best_records = recs
        best_slope   = slope
        best_R2      = r_val**2

print(f"  {n_tested} fenêtres valides testées")
print(f"\nFenêtre optimale : A={best_A:.2f} Ta, B={best_B:.2f} Ta  "
      f"(durée={best_B-best_A:.2f} Ta)")
print(f"  pente = {best_slope:.3f} ± {best_se:.3f} (1σ),  R² = {best_R2:.4f}")

# Récupère les records de la meilleure fenêtre pour la suite
records = best_records
A = best_A
B = best_B

print(f"\nDétail des points retenus :")
for r in records:
    print(f"  {r['L_mm']:2d}mm {r['sand']:6s} : "
          f"β={r['beta']:+.4f} Ta⁻¹, R={r['R_w']*1e3:.2f} mm, "
          f"α_M={r['alpha_M']*1e3:.3f} mm, α_M/α_T={r['alpha_ratio']:.2f}, "
          f"Pe={r['Pe']:.2f}")

if len(records) < 2:
    print("[ERREUR] Pas assez de points pour le fit — élargir la plage de balayage.")
    raise SystemExit

# ==========================================================
# RÉCUPÉRATION DES PARAMÈTRES DU MEILLEUR FIT
# (déjà calculés dans la boucle de balayage)
# ==========================================================
Pe_all     = np.array([r["Pe"]          for r in records])
ratio_all  = np.array([r["alpha_ratio"] for r in records])

log_Pe     = np.log10(Pe_all)
log_ratio  = np.log10(ratio_all)
slope, intercept, r_val, p_val, se_slope = stats.linregress(log_Pe, log_ratio)

print(f"\nFit log-log α_M/α_T vs Pe (fenêtre optimale) :")
print(f"  pente      = {slope:.3f} ± {se_slope:.3f}  (1σ)")
print(f"  R²         = {r_val**2:.4f}")
print(f"  intercept  = {intercept:.3f}")

# ==========================================================
# FIGURE 1 — Σ vs t/Ta semi-log + fenêtre de fit
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

ax1.axvspan(2*A, 2*B, alpha=0.12, color="gray")
ax1.axvline(2*A, color="gray", ls=":", lw=0.8)
ax1.axvline(2*B, color="gray", ls=":", lw=0.8)
ax1.axvline(0,   color="k",    ls="--", lw=0.7, alpha=0.5)

for r in records:
    time_red = r["time_red"]
    Sigma    = r["Sigma"]
    style    = r["style"]
    beta     = r["beta"]
    log_S0_r = r["log_S0"]

    valid = np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0)
    ax1.plot(2*time_red[valid][::3], Sigma[valid][::3],
             linestyle="None", **style)

    # Droite de fit sur la fenêtre (en variable 2t/Ta)
    t_line = np.linspace(2*A, 2*B, 80)
    ax1.plot(t_line, np.exp(log_S0_r + beta * t_line / 2),
             color=style["color"], lw=1.4, ls="--", alpha=0.75)

legend_elements = []
for (L_mm, sand), style in style_map.items():
    legend_elements.append(
        Line2D([0], [0], marker=style["marker"], linestyle="None",
               color=style["color"],
               markerfacecolor=style.get("mfc", style["color"]),
               markeredgewidth=style.get("mew", 1.0),
               markersize=style.get("ms", 5),
               alpha=style.get("alpha", 1.0),
               label=f"{L_mm} mm, {sand}")
    )
legend_elements.append(
    plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.15,
                  label=f"fit window : [{2*A}–{2*B}] $T_a$")
)

ax1.legend(handles=legend_elements, ncol=2, fontsize=8)
ax1.set_xlabel(r"$2t \;/\; T_a$")
ax1.set_ylabel(r"$\Sigma \;/\; \Sigma_0$")
ax1.set_yscale("log")
ax1.set_ylim(bottom=0.8e-2)
ax1.set_xlim(left=-1, right=2*B + 8)
ax1.grid(True, ls="--", alpha=0.3)
plt.show()

# ==========================================================
# FIGURE 2 — α_M / α_T  vs  Pe  log-log
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

for r in records:
    style = {k: v for k, v in r["style"].items()}
    ax2.plot(r["Pe"], r["alpha_ratio"], linestyle="None", **style)

Pe_range   = np.logspace(np.log10(Pe_all.min()) - 0.2,
                         np.log10(Pe_all.max()) + 0.2, 300)
ratio_fit   = 10**intercept * Pe_range**slope
ratio_fit_p = 10**intercept * Pe_range**(slope + se_slope)
ratio_fit_m = 10**intercept * Pe_range**(slope - se_slope)

fit_label = rf"slope $= {slope:.2f} \pm {se_slope:.2f}$ (1$\sigma$)"
ax2.plot(Pe_range, ratio_fit, "k-", lw=1.8, alpha=0.85, label=fit_label)
ax2.fill_between(Pe_range, ratio_fit_m, ratio_fit_p,
                 color="k", alpha=0.12, label=r"$\pm 1\sigma$")

legend_elements2 = []
for (L_mm, sand), style in style_map.items():
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
ax2.legend(handles=legend_elements2, fontsize=8)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"$Pe = d_2 \;/\; \alpha_T$")
ax2.set_ylabel(r"$\alpha_{\perp M} \;/\; \alpha_T \;=\; D_{\perp M}\,/\,D_{\perp m}$")
ax2.minorticks_on()
ax2.grid(True, which="major", ls="--", alpha=0.7)
ax2.grid(True, which="minor", ls=":",  alpha=0.4)
fig2.savefig("/home/chorus/dispersion_libre_scaling.pdf")

plt.show()