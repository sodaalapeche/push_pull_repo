"""
Estimation du coefficient de diffusion D (m²/s)
depuis data_roi.npy, pour les expériences à 0 mm.

Théorie (t → ∞) :
    σ²_c / (A · μ²) = 1 / (8π D t)

En log-log, pente = -1 et ordonnée à l'origine = log(1 / (8π D)).
On force la pente à -1 et on ajuste uniquement l'amplitude C = 1/(8πD).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
# ==========================================================
# PARAMÈTRES
# ==========================================================
DATA_PATH = "/home/chorus/data_roi.npy"
s = 0.06 / 512          # taille pixel [m/px]

# Fenêtre de fit :
#   - on démarre quand t/Ta > T_MIN_REDUIT  (évite la bosse initiale)
#   - on ne prend que les points où Σ < FRAC_PEAK * Σ_max  (queue décroissante)
T_MIN_REDUIT = 10
T_MAX_REDUIT = 18
FRAC_PEAK    = 0.95      # on veut être clairement dans le régime décroissant

# ==========================================================
# CHARGEMENT
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
print(f"Chargé : {len(results)} expériences")

# ==========================================================
# FILTRE 0 mm  (fine + coarse)
# ==========================================================
def parse_label(label_text):
    lines = [l.strip().lower() for l in label_text.split("\n") if l.strip()]
    if len(lines) < 3:
        return None
    try:
        L_mm = int(lines[1].replace("mm", ""))
        sand = lines[2]
        return L_mm, sand
    except Exception:
        return None

exps_0mm = []
for r in results:
    info = parse_label(r.get("label", ""))
    if info is None:
        continue
    L_mm, sand = info
    if L_mm == 0:
        exps_0mm.append((r, sand))

print(f"Expériences 0 mm retenues : {len(exps_0mm)}")

# ==========================================================
# MODÈLE DE FIT  : Σ(t) = C / t_phys   (pente -1 forcée)
#   => D = 1 / (8π C)
#
# Fit log-log pente forcée à -1 :
#   log Σ = log C - log t_phys
#   => log C = mean( log Σ + log t_phys )
# ==========================================================

style_map = {
    "fine":   dict(color="tab:green", marker="o", mfc="tab:green",  ms=8, alpha=0.8),
    "coarse": dict(color="tab:olive", marker="D", mfc="none", mew=2, ms=8, alpha=1),
}

fig, ax = plt.subplots(figsize=(8, 5))

D_values = {}   # sand -> liste de D estimés
alphaT ={}
for res, sand in exps_0mm:
    time_red = res["time"]     # t/Ta  (temps réduit, 0 au pic de variance)
    mean     = res["mean"]
    var      = res["var"]
    i0       = res["i0"]
    A0       = res["A"]        # num_pixels * s²  [m²]
    Ta       = res["Ta"]       # [s]

    # --- Grandeur normalisée ---
    Sigma = var / (A0 * mean**2)
             # = 1 à t = t0
    u = 0.01/Ta
    u_values = {"fine": [], "coarse": []}
    u_values.setdefault(sand, ).append(u)
    # --- Temps physique RELATIF à i0 [s] ---
    t_phys_rel = time_red * Ta   # > 0 après i0
    if sand=="coarse":
        D_theo = 2*0.0006*0.2*0.01/Ta
    else:
        D_theo = 2*0.0001*0.2*0.01/Ta
    # --- Pic de Σ (peut légèrement dépasser i0) ---
    Sigma_peak = np.nanmax(Sigma)

    # --- Masque de fit ---
    mask_fit = (
        np.isfinite(time_red)    &
        np.isfinite(Sigma)       &
        (Sigma > 0)              &
        (t_phys_rel > 0)         &          # temps positif
        (time_red > T_MIN_REDUIT) &         # après la bosse
        (time_red < T_MAX_REDUIT)    # clairement en décroissance
    )
    # --- Courbe théorique 2D pour CETTE expérience ---
    t_range_phys = np.logspace(
        np.log10(1),
        np.log10(t_phys_rel.max()),
        300
    )

    Sigma_theo = 1.0 / (8.0 * np.pi * D_theo * t_range_phys)

    ax.plot(2*t_range_phys / Ta, Sigma_theo,
            color=style_map.get("color", "k"),
            lw=1.2, ls="-.",
            alpha=1,label=r"$D = 0.1\cdot u \cdot d_1$")
    t_fit   = t_phys_rel[mask_fit]
    Sig_fit = Sigma[mask_fit]

    D_est = np.nan
    C_fit = np.nan

    if t_fit.size >= 5:
        # Fit log-log pente -1 forcée
        log_C = np.mean(np.log(Sig_fit) + np.log(t_fit))
        C_fit = np.exp(log_C)
        D_est = 1.0 / (8.0 * np.pi * C_fit)
        print(f"  [{sand:6s}]  Ta={Ta:.2f}s  C={C_fit:.4e}  "
              f"D={D_est:.4e} m²/s  (n={t_fit.size} pts)")
        D_values.setdefault(sand, []).append(D_est)
        alphaT.setdefault(sand, []).append(D_est/u)
    else:
        print(f"  [{sand:6s}]  Pas assez de points ({t_fit.size}) — "
              f"ajuster T_MIN_REDUIT ou FRAC_PEAK")

    # --- Tracé des données ---
    style = style_map.get(sand, {})
    valid = np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0) & (time_red > 0)
    ax.plot(2*time_red[valid][::2], Sigma[valid][::2],
            linestyle="None", **style)

    # --- Tracé de la courbe de fit ---
    if np.isfinite(D_est) and t_fit.size >= 5:
        t_range_phys = np.logspace(
            np.log10(t_fit.min()),
            np.log10(t_phys_rel[valid].max()),
            300
        )
        Sigma_fit_curve = C_fit / t_range_phys
        # ax.plot(t_range_phys / Ta, Sigma_fit_curve,
        #         color=style.get("color", "k"), lw=1.8, ls="--", alpha=0.9)
# RÉSUMÉ CONSOLE
# ==========================================================
print("\n" + "="*50)
print("RÉSULTATS — D (expériences 0 mm)")
print("="*50)
for sand, vals in D_values.items():
    arr = np.array(vals)
    print(f"  {sand:6s} : D = {np.mean(arr):.4e} ± {np.std(arr):.2e} m²/s  "
          f"(n={len(arr)} réplicats)")

print("RÉSULTATS — ALPHA_T (expériences 0 mm)")
print("="*50)
for sand, vals in alphaT.items():
    arr = np.array(vals)
    print(f"  {sand:6s} : D = {np.mean(arr):.4e} ± {np.std(arr):.2e} m²/s  "
          f"(n={len(arr)} réplicats)")
# ==========================================================
# GRAPHE
# ==========================================================
legend_elements = [

    Line2D([0], [0], marker="o", linestyle="None",
           color="tab:green", markerfacecolor="tab:green",
           markersize=7, label="Homogeneous – fine sand"),
    Line2D([0], [0], marker="D", linestyle="None",
           color="tab:olive", markerfacecolor="none",
           markeredgewidth=1.2, markersize=7, label="Homogeneous – coarse sand"),
    Line2D([0], [0], marker=None, linestyle="-.",c="black",label=f"transverse dispersion model : $D = 0.2 \cdot d_1 u$")
]
# ==========================================================
# ANNOTATION DES DISPERSIVITÉS MOYENNES
# ==========================================================
y_text = 0.65  # position verticale (relative axes)
for i, (sand, vals) in enumerate(alphaT.items()):
    arr = np.array(vals)
    if len(arr) == 0:
        continue
    alpha_mean = np.mean(arr)
    alpha_std  = np.std(arr)

    ax.text(0.71, y_text - i*0.2,
            rf"$\alpha_T$ ({sand}) = {alpha_mean:.2e} m",
            transform=ax.transAxes,
            fontsize=13,color="tab:olive" if sand=="coarse" else "tab:green",)
ax.legend(handles=legend_elements, fontsize=12)
ax.axvline(1, color="k", ls=":", alpha=0.4, label="t/Ta = 1")
ax.set_xlabel(r"$2t \; / \; T_a$", fontsize=20,)
ax.set_ylabel(r"$\Sigma$", fontsize=20)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim(left=10,right=48)
ax.set_ylim(bottom=100,top=10**5)
ax.grid(True, ls="--", alpha=0.7)
fig.tight_layout()

plt.savefig('/home/chorus/homo.eps')
plt.show()

np.save("../vieux codes/resultats_diffusion.npy", {
    "D_values": D_values,
    "alphaT": alphaT
})