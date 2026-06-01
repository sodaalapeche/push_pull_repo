"""
Figure : beta (pente semi-log de Sigma vs t en secondes)  vs  R_0^2 / d_2^2

Pour chaque expérience (L_mm > 0) :
  1. Fit semi-log de Σ(t/Ta) entre A et B  →  β [Ta⁻¹]
     puis conversion : β_s = β / Ta  →  β_s [s⁻¹]
  2. R_0 = sigma_m[i0]  (rayon initial du blob au temps de référence i0,
                        depuis data_rms — c'est une longueur, pas une variance)
  3. Ratio adimensionnel  R_0^2 / d_2^2
  4. Tracé  β_s [s⁻¹]  vs  R_0^2 / d_2^2

Formalism conservé :
  - plt.style.use("science")
  - style_map (fine = cercles pleins, coarse = losanges creux)
  - COLOR_MAP par L_mm
  - sigma_curve() + semilog_fit() + parsing label identiques
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from scipy import stats

plt.style.use("science")

# ==========================================================
# PARAMÈTRES
# ==========================================================
ROI_PATH = "/home/chorus/data_roi.npy"
RMS_PATH = "/home/chorus/data_rms.npy"

# Fenêtre de fit semi-log [t/Ta]
A_FIT = 0
B_FIT = 3

width  = 14
height = width * 0.55
inches = 2.54

COLOR_MAP = {
    10: "tab:blue",
    6:  "tab:orange",
    3:  "tab:red",
    0:  "tab:green",
}

style_map = {
    (10, "fine"):   dict(marker="o", color="tab:blue",   mfc="tab:blue",   ms=7, alpha=0.85),
    (10, "coarse"): dict(marker="D", color="tab:blue",   mfc="none", mew=1.2, ms=7, alpha=0.85),
    (6,  "fine"):   dict(marker="o", color="tab:orange", mfc="tab:orange", ms=7, alpha=0.85),
    (6,  "coarse"): dict(marker="D", color="tab:orange", mfc="none", mew=1.2, ms=7, alpha=0.85),
    (3,  "fine"):   dict(marker="o", color="tab:red",    mfc="tab:red",    ms=7, alpha=0.85),
    (3,  "coarse"): dict(marker="D", color="tab:red",    mfc="none", mew=1.2, ms=7, alpha=0.85),
    (0,  "fine"):   dict(marker="o", color="tab:green",  mfc="tab:green",  ms=6, alpha=0.85),
    (0,  "coarse"): dict(marker="D", color="tab:olive",  mfc="none", mew=1.2, ms=6, alpha=0.85),
}

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
    return time, Sigma, i0

def semilog_fit(time_red, Sigma, t_A, t_B):
    """Fit semi-log de Σ sur [t_A, t_B]. Retourne (beta [Ta⁻¹], log_S0)."""
    mask = (
        np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0) &
        (time_red >= t_A) & (time_red <= t_B)
    )
    t_f, S_f = time_red[mask], Sigma[mask]
    if t_f.size < 3:
        return np.nan, np.nan
    c = np.polyfit(t_f, np.log(S_f), 1)
    return c[0], c[1]

def R0_from_rms(rms_res):
    """Retourne R_0 = sigma_m[i0] en mètres (longueur initiale du blob)."""
    sigma_m = rms_res["sigma_m"]
    i0      = rms_res["i0"]
    if np.isfinite(sigma_m[i0]) and sigma_m[i0] > 0:
        return float(sigma_m[i0])
    return np.nan

# ==========================================================
# COLLECTE DES POINTS  (beta, sigma0, d2)
# ==========================================================
records = []

for (L_mm, sand), roi_list in roi_idx.items():
    if L_mm == 0:
        # d_2 = 0 → ratio non défini ; on saute
        continue
    # if L_mm ==3 and sand =="coarse":
    #     continue
    style    = style_map.get((L_mm, sand),
                             dict(marker="s", color="gray", mfc="gray", ms=6, alpha=0))
    rms_list = rms_idx.get((L_mm, sand), [])

    for i, res_roi in enumerate(roi_list):
        # 1. fit beta en Ta⁻¹
        time_red, Sigma, _ = sigma_curve(res_roi)
        beta, _ = semilog_fit(time_red, Sigma, A_FIT, B_FIT)
        if np.isnan(beta):
            print(f"  [SKIP] {L_mm}mm {sand} exp#{i} : beta invalide")
            continue
        if sand == "coarse":
            sands = 0.0006
        elif sand == "fine":
            sands = 0.00009
        d1 = sands
        # conversion en s⁻¹ : β_s = β / Ta
        Ta     = float(res_roi["Ta"])
        beta_s = abs(beta)/Ta


        # 2. R_0 depuis rms apparié
        if i < len(rms_list):
            R_0 = R0_from_rms(rms_list[i])
        elif rms_list:
            R_0 = R0_from_rms(rms_list[0])
        else:
            R_0 = np.nan

        if np.isnan(R_0):
            print(f"  [SKIP] {L_mm}mm {sand} exp#{i} : R_0 invalide")
            continue

        # 3. ratio adimensionnel
        d2    = L_mm * 1e-3       # [m]
        ratio =10* d2/sands#/R_0   # [-]
        ratio = d1
        print(f"  {L_mm:2d}mm {sand:6s} exp#{i} : "
              f"beta_s={beta_s:+.4e} s⁻¹, "
              f"R_0={R_0*1e3:.2f} mm, "
              f"R_0²/d2²={ratio:.3f}")

        records.append(dict(
            L_mm=L_mm, sand=sand, beta=beta, beta_s=beta_s, Ta=Ta,
            R_0=R_0, d2=d2, ratio=ratio, style=style
        ))

print(f"\nNombre de points : {len(records)}")

if len(records) < 2:
    print("[ERREUR] Pas assez de points — vérifier A_FIT, B_FIT et les données.")
    raise SystemExit

# ==========================================================
# FIGURE — beta_s [s⁻¹] vs R_0^2 / d_2^2
# ==========================================================
fig, ax = plt.subplots(figsize=(width/inches, height/inches), layout="constrained")

for r in records:
    ax.plot(r["ratio"], r["beta_s"], linestyle="None", **r["style"])

# axe Y : beta_s est négatif pour de la décroissance — option de plot en |beta_s|
# Décommenter pour passer en log sur |beta_s|
# ax.plot(r["ratio"], -r["beta_s"], ...)
# ax.set_yscale("log")

# ax.axhline(0, color="k", ls=":", lw=0.6, alpha=0.5)

# ax.set_xlabel(r"$R_0^2\,/\,d_2^2$")
ax.set_xlabel(r"$Pe = 10 d_2\,/\,d_1$")
ax.set_ylabel(r"$\beta\ [\mathrm{Ta}^{-1}]$")
ax.grid(True, ls="--", alpha=0.3)

# ==========================================================
# LÉGENDE (style_map order)
# ==========================================================
legend_elements = []
for (L_mm, sand), style in style_map.items():
    if L_mm == 0:
        continue
    # n'afficher que les clés effectivement présentes dans les records
    if not any(r["L_mm"] == L_mm and r["sand"] == sand for r in records):
        continue
    legend_elements.append(
        Line2D([0], [0], marker=style["marker"], linestyle="None",
               color=style["color"],
               markerfacecolor=style.get("mfc", style["color"]),
               markeredgewidth=style.get("mew", 1.0),
               markersize=style.get("ms", 6),
               alpha=style.get("alpha", 1.0),
               label=fr"$d_2$ = {L_mm} mm — {sand}")
    )
ax.legend(handles=legend_elements, ncol=2, fontsize=8)
ax.set_yscale("log")
ax.set_xscale("log")
fig.savefig("/home/chorus/beta_vs_R0_over_d2.pdf")
plt.show()