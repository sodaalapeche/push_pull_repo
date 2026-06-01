"""
Mesure du temps de mélange T_mix :
  abscisse de l'intersection entre
    - pente initiale  : fit semi-log de Σ sur [0, T_INIT]
    - pente finale    : fit semi-log de Σ sur [A, B]

Plot : T_mix/Ta vs d2  (une couleur par (L_mm, sand))
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import scienceplots

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")

width  = 14
height = width * 0.6
inches = 2.54

# ==========================================================
# PARAMÈTRES — À AJUSTER
# ==========================================================
DATA_PATH = "/home/chorus/data_roi.npy"

T_INIT = 3.0    # fin de la fenêtre de la pente initiale  [Ta]
A      = 15.0   # début de la fenêtre de la pente finale  [Ta]
B      = 25.0   # fin   de la fenêtre de la pente finale  [Ta]

# ==========================================================
# STYLE MAP
# ==========================================================
style_map = {
    (10, "fine"):   dict(marker="o", color="tab:blue",   mfc="tab:blue",   ms=7,  alpha=0.85),
    (10, "coarse"): dict(marker="D", color="tab:blue",   mfc="none",       mew=1.2, ms=7, alpha=0.85),
    (6,  "fine"):   dict(marker="o", color="tab:orange", mfc="tab:orange", ms=7,  alpha=0.85),
    (6,  "coarse"): dict(marker="D", color="tab:orange", mfc="none",       mew=1.2, ms=7, alpha=0.85),
    (3,  "fine"):   dict(marker="o", color="tab:red",    mfc="tab:red",    ms=7,  alpha=0.85),
    (3,  "coarse"): dict(marker="D", color="tab:red",    mfc="none",       mew=1.2, ms=7, alpha=0.85),
}

# ==========================================================
# CHARGEMENT
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
print(f"Chargé : {len(results)} expériences")

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
# HELPER : fit semi-log sur une fenêtre
# retourne (pente, ordonnée à l'origine) dans l'espace (t/Ta, logΣ)
# ==========================================================
def semilog_fit(time_red, Sigma, t_start, t_end):
    mask = (
        np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0) &
        (time_red >= t_start) & (time_red <= t_end)
    )
    t_f, S_f = time_red[mask], Sigma[mask]
    if t_f.size < 3:
        return np.nan, np.nan
    c = np.polyfit(t_f, np.log(S_f), 1)
    return c[0], c[1]   # (pente, intercept) dans (t/Ta, logΣ)

# ==========================================================
# CALCUL T_MIX POUR CHAQUE EXPÉRIENCE
# Intersection de deux droites en espace semi-log :
#   logΣ = β_ini * t + b_ini
#   logΣ = β_fin * t + b_fin
#   → t_mix = (b_fin - b_ini) / (β_ini - β_fin)
# ==========================================================
records = []

for res in results:
    info = parse_label(res.get("label", ""))
    if info is None:
        continue
    L_mm, sand = info
    if L_mm == 0:
        continue

    time  = res["time"]
    var   = res["var"]
    mean  = res["mean"]
    A0    = res["A"]
    i0    = res["i0"]
    Ta=res["Ta"]
    Sigma = var / (A0 * mean**2)
    Sigma = Sigma / Sigma[i0]

    # --- pente initiale : [0, T_INIT] ---
    beta_ini, b_ini = semilog_fit(time, Sigma, 0.0, T_INIT)

    # --- pente finale : [A, B] ---
    beta_fin, b_fin = semilog_fit(time, Sigma, A, B)

    if np.isnan(beta_ini) or np.isnan(beta_fin):
        print(f"  [SKIP] {L_mm}mm {sand} : fit invalide "
              f"(β_ini={beta_ini:.4f}, β_fin={beta_fin:.4f})")
        continue

    if np.isclose(beta_ini, beta_fin, rtol=1e-3):
        print(f"  [SKIP] {L_mm}mm {sand} : pentes trop proches, pas d'intersection")
        continue

    # intersection des deux droites semi-log
    t_mix = (b_fin - b_ini) / (beta_ini - beta_fin)   # [Ta]

    # vérification : t_mix doit être dans un intervalle raisonnable
    if not (T_INIT <= t_mix <= A + 2*(B - A)):
        print(f"  [WARN] {L_mm}mm {sand} : t_mix={t_mix:.2f} Ta hors fenêtre attendue")

    print(f"  {L_mm:2d}mm {sand:6s} : β_ini={beta_ini:.4f}, β_fin={beta_fin:.4f}, "
          f"t_mix={t_mix:.2f} Ta")

    style = style_map.get((L_mm, sand),
                          dict(marker="s", color="gray", mfc="gray", ms=6, alpha=0.8))

    records.append(dict(
        L_mm=L_mm, sand=sand,
        beta_ini=beta_ini, b_ini=b_ini,
        beta_fin=beta_fin, b_fin=b_fin,
        t_mix=t_mix,
        time=time, Sigma=Sigma,
        style=style,Ta=Ta
    ))

print(f"\nNombre de points retenus : {len(records)}")

# ==========================================================
# FIGURE 1 — Σ vs t/Ta semi-log : vérification visuelle des fits
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

ax1.axvspan(0,   2*T_INIT, alpha=0.10, color="tab:blue",  label=f"pente ini [0–{2*T_INIT}] $T_a$")
ax1.axvspan(2*A, 2*B,      alpha=0.10, color="tab:red",   label=f"pente fin [{2*A}–{2*B}] $T_a$")
ax1.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)

t_plot_ini = np.linspace(0,   2*T_INIT, 60)
t_plot_fin = np.linspace(2*A, 2*B,      60)

for r in records:
    valid = np.isfinite(r["time"]) & np.isfinite(r["Sigma"]) & (r["Sigma"] > 0)
    ax1.plot(2*r["time"][valid][::3], r["Sigma"][valid][::3],
             linestyle="None", **r["style"])

    c = r["style"]["color"]
    # droite initiale
    ax1.plot(t_plot_ini,
             np.exp(r["b_ini"] + r["beta_ini"] * t_plot_ini / 2),
             color=c, lw=1.4, ls="--", alpha=0.8)
    # droite finale
    ax1.plot(t_plot_fin,
             np.exp(r["b_fin"] + r["beta_fin"] * t_plot_fin / 2),
             color=c, lw=1.4, ls="-",  alpha=0.8)
    # intersection
    ax1.axvline(2*r["t_mix"], color=c, lw=0.8, ls=":", alpha=0.7)

legend_elements = []
for (L_mm, sand), st in style_map.items():
    legend_elements.append(
        Line2D([0], [0], marker=st["marker"], linestyle="None",
               color=st["color"],
               markerfacecolor=st.get("mfc", st["color"]),
               markeredgewidth=st.get("mew", 1.0),
               markersize=st.get("ms", 5),
               alpha=st.get("alpha", 1.0),
               label=f"{L_mm} mm, {sand}")
    )
ax1.legend(handles=legend_elements, ncol=2, fontsize=8)
ax1.set_xlabel(r"$2t \;/\; T_a$")
ax1.set_ylabel(r"$\Sigma \;/\; \Sigma_0$")
ax1.set_yscale("log")
ax1.set_ylim(bottom=1e-2)
ax1.set_xlim(left=-1,right=60)
ax1.grid(True, ls="--", alpha=0.3)
fig1.savefig("/home/chorus/t_mix_verification.pdf")
plt.show()
# ==========================================================
# FIGURE 2 — T_mix/Ta vs d2
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

for r in records:
    if r['sand']=="coarse":
        sands=6
    else :
        sands=0.9
    ax2.plot(r["L_mm"], 2*r["t_mix"]*r['Ta'],
             linestyle="None", **r["style"])

legend_elements2 = []
for (L_mm, sand), st in style_map.items():
    legend_elements2.append(
        Line2D([0], [0], marker=st["marker"], linestyle="None",
               color=st["color"],
               markerfacecolor=st.get("mfc", st["color"]),
               markeredgewidth=st.get("mew", 1.0),
               markersize=st.get("ms", 5),
               alpha=st.get("alpha", 1.0),
               label=f"{L_mm} mm, {sand}")
    )
ax2.legend(handles=legend_elements2, ncol=1, fontsize=8)
ax2.set_xlabel(r"$d_2$ [mm]")
ax2.set_ylabel(r"$T_\mathrm{mix} \;/\; T_a$")
ax2.grid(True, ls="--", alpha=0.3)
ax2.set_yscale("log")
ax2.set_xscale("log")
fig2.savefig("/home/chorus/t_mix_vs_d2.pdf")

plt.show()