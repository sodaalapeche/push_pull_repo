"""
Estimation du coefficient de diffusion D (m²/s)
depuis data_roi.npy, pour les expériences à 0 mm.

Théorie (t → ∞) :
    Σ = σ²_c / (A · μ²) = 1 / (8π D t)

En log-log, pente = -1 et ordonnée à l'origine = log(1 / (8π D)).
On force la pente à -1 et on ajuste uniquement l'amplitude C = 1/(8πD).

--------------------------------------------------------------------------
CORRECTIONS par rapport à la version précédente :
  * la courbe ESTIMÉE (fit C/t) était commentée -> on la trace maintenant ;
  * elle était tracée en t/Ta alors que données + modèle sont en 2t/Ta
    -> facteur 2 corrigé, tout est sur le même axe ;
  * modèle ET fit sont tracés explicitement comme Σ = var/(A·μ²),
    donc directement comparables aux points expérimentaux ;
  * coefficient du modèle rendu explicite (C_MODEL) et reporté dans le label ;
  * nettoyage : import networkx parasite, bloc u_values cassé, doublons.
--------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use('science')

# ==========================================================
# PARAMÈTRES
# ==========================================================
DATA_PATH = "/home/chorus/data_roi.npy"
s = 0.06 / 512          # taille pixel [m/px]
l = 0.01                # longueur associée à Ta [m]  (u = l/Ta)

# Fenêtre de fit (en temps réduit, après la bosse initiale)
T_MIN_REDUIT = 10
T_MAX_REDUIT = 18

# Modèle de dispersion transverse :  D = C_MODEL * d1 * u
#   /!\ ton code calculait 0.05 (= 2*0.025), mais tes labels disaient 0.2 / 0.1.
#       À trancher : ici je conserve TON calcul (0.05) et je l'affiche tel quel.
C_MODEL = 2*0.12
D1 = {"fine": 0.00015, "coarse": 0.0006}   # taille de grain d1 [m]

# ==========================================================
# CHARGEMENT
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
print(f"Chargé : {len(results)} expériences")

# ==========================================================
# FILTRE 0 mm  (fine + coarse)
# ==========================================================
def parse_label(label_text):
    lines = [ln.strip().lower() for ln in label_text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return None
    try:
        return int(lines[1].replace("mm", "")), lines[2]
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
# STYLES
# ==========================================================
style_map = {
    "fine":   dict(color="tab:green", marker="o", mfc="tab:green", ms=8, alpha=0.8),
    "coarse": dict(color="tab:olive", marker="D", mfc="none", mew=2, ms=8, alpha=1),
}

fig, ax = plt.subplots(figsize=(8, 5))

D_values = {}   # sand -> liste de D estimés
alphaT   = {}   # sand -> liste de alpha_T = D/u
model_drawn = set()   # pour ne tracer le modèle qu'une fois par sable

for res, sand in exps_0mm:
    time_red = res["time"]      # t/Ta  (0 au pic de variance, i0)
    mean     = res["mean"]
    var      = res["var"]
    i0       = res["i0"]
    A0       = res["A"]         # [m²]
    Ta       = res["Ta"]        # [s]

    # --- Grandeur normalisée Σ (IDENTIQUE pour données / fit / théorie) ---
    Sigma = var  / (A0 * mean**2)

    # --- Temps physique relatif à i0 [s] ---
    t_phys_rel = time_red * Ta

    # --- Vitesse et modèle de D ---
    u = l / Ta
    d1 = D1[sand]
    D_theo = C_MODEL * d1 * u            # [m²/s]   (= 0.05 * d1 * u, ton calcul)

    style = style_map.get(sand, {})
    valid = (np.isfinite(time_red) & np.isfinite(Sigma) &
             (Sigma > 0) & (time_red > 0))

    # ======================================================
    # (1) COURBE THÉORIQUE (modèle)  ->  Σ_theo = 1/(8π D_theo t)
    #     une seule fois par sable, sur le MÊME axe 2t/Ta
    # ======================================================
    if sand not in model_drawn and valid.any():
        t_theo = np.logspace(np.log10(1.0),
                             np.log10(t_phys_rel[valid].max()), 300)
        Sigma_theo = 1.0 / (8.0 * np.pi * D_theo * t_theo)
        ax.plot(2 * t_theo / Ta, Sigma_theo,
                color="k", lw=1.2, ls="-.", alpha=0.9)
        model_drawn.add(sand)

    # ======================================================
    # (2) FIT : Σ ≈ C / t_phys  (pente -1 forcée)  =>  D = 1/(8π C)
    # ======================================================
    mask_fit = (valid &
                (time_red > T_MIN_REDUIT) &
                (time_red < T_MAX_REDUIT))
    t_fit   = t_phys_rel[mask_fit]
    Sig_fit = Sigma[mask_fit]

    D_est = np.nan
    C_fit = np.nan
    if t_fit.size >= 5:
        log_C = np.mean(np.log(Sig_fit) + np.log(t_fit))   # pente -1 forcée
        C_fit = np.exp(log_C)
        D_est = 1.0 / (8.0 * np.pi * C_fit)
        print(f"  [{sand:6s}]  Ta={Ta:.2f}s  C={C_fit:.4e}  "
              f"D={D_est:.4e} m²/s  (n={t_fit.size} pts)")
        D_values.setdefault(sand, []).append(D_est)
        alphaT.setdefault(sand, []).append(D_est / u)
    else:
        print(f"  [{sand:6s}]  Pas assez de points ({t_fit.size}) — "
              f"ajuster T_MIN_REDUIT / T_MAX_REDUIT")

    # --- Points expérimentaux (Σ) ---
    ax.plot(2 * time_red[valid][::2], Sigma[valid][::2],
            linestyle="None", **style)

    # ======================================================
    # (3) COURBE ESTIMÉE (le fit)  ->  Σ_est = C_fit / t_phys
    #     même grandeur Σ, même axe 2t/Ta  (c'était le couac : décommentée
    #     + facteur 2 corrigé)
    # ======================================================
    if np.isfinite(D_est):
        t_curve = np.logspace(np.log10(t_fit.min()),
                              np.log10(t_phys_rel[valid].max()), 300)
        Sigma_est = C_fit / t_curve
        ax.plot(2 * t_curve / Ta, Sigma_est,
                color=style.get("color", "k"), lw=1.8, ls="--", alpha=0.9)
    print("alpha_calculé="+str(D_est/u) + "    alpha_théorique= "+str(0.25*d1))
# ==========================================================
# RÉSUMÉ CONSOLE
# ==========================================================
print("\n" + "=" * 50)
print("RÉSULTATS — D (expériences 0 mm)")
print("=" * 50)
for sand, vals in D_values.items():
    arr = np.array(vals)
    print(f"  {sand:6s} : D = {np.mean(arr):.4e} ± {np.std(arr):.2e} m²/s  "
          f"(n={len(arr)})")

print("\nRÉSULTATS — ALPHA_T (expériences 0 mm)")
print("=" * 50)
for sand, vals in alphaT.items():
    arr = np.array(vals)
    print(f"  {sand:6s} : alpha_T = {np.mean(arr):.4e} ± {np.std(arr):.2e} m  "
          f"(n={len(arr)})")

# ==========================================================
# LÉGENDE + ANNOTATIONS
# ==========================================================
legend_elements = [
    Line2D([0], [0], marker="o", linestyle="None",
           color="tab:green", markerfacecolor="tab:green",
           markersize=7, label="Homogeneous – fine sand"),
    Line2D([0], [0], marker="D", linestyle="None",
           color="tab:olive", markerfacecolor="none",
           markeredgewidth=1.2, markersize=7, label="Homogeneous – coarse sand"),
    Line2D([0], [0], color="k", ls="--", lw=1.8,
           label=r"estimated decay : $\Sigma = C/t$"),
    Line2D([0], [0], color="k", ls="-.", lw=1.2,
           label=rf"model : $D = {C_MODEL}\,d_1\,u$"),
]

y_text = 0.65
for i, (sand, vals) in enumerate(alphaT.items()):
    arr = np.array(vals)
    if arr.size == 0:
        continue
    ax.text(0.71, y_text - i * 0.2,
            rf"$\alpha_T$ ({sand}) = {np.mean(arr):.2e} m",
            transform=ax.transAxes,
            color="tab:olive" if sand == "coarse" else "tab:green")

ax.legend(handles=legend_elements)
ax.set_xlabel(r"$2t \; / \; T_a$")
ax.set_ylabel(r"$\Sigma$")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim(left=10, right=48)
ax.set_ylim(bottom=100, top=10**5)
ax.grid(True, ls="--", alpha=0.7)
fig.tight_layout()

plt.savefig('/home/chorus/homo.eps')
plt.show()

np.save("../vieux codes/resultats_diffusion.npy", {
    "D_values": D_values,
    "alphaT": alphaT,
})