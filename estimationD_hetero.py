import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots

plt.style.use('science')

# ==========================================================
# PARAMÈTRES
# ==========================================================
DATA_PATH = "/home/chorus/data_roi.npy"

# constantes (mêmes que ton script initial)
s = 0.06 / 512  # m/px

T_MIN = 5
T_MAX = 22

# densités caractéristiques (adapter si besoin)
d1_map = {
    "fine": 0.9e-4,
    "coarse": 6e-4,
    "contactless": 0.9e-4
}

# ==========================================================
# STYLE (fourni par toi)
# ==========================================================
style_map = {
    (10, "fine"): dict(marker="o", color="tab:blue", mfc="tab:blue", ms=6, alpha=0.8),
    (10, "coarse"): dict(marker="D", color="tab:blue", mfc="none", mew=1.2, ms=6, alpha=0.8),
    (6, "fine"): dict(marker="o", color="tab:orange", mfc="tab:orange", ms=6, alpha=0.8),
    (6, "coarse"): dict(marker="D", color="tab:orange", mfc="none", mew=1.2, ms=6, alpha=0.8),
    (3, "fine"): dict(marker="o", color="tab:red", mfc="tab:red", ms=6, alpha=0.8),
    (3, "coarse"): dict(marker="D", color="tab:red", mfc="none", mew=1.2, ms=6, alpha=0.8),
    (0, "fine"): dict(color="tab:green", marker="o", mfc="tab:green", ms=5, alpha=0.8),
    (0, "coarse"): dict(color="tab:olive", marker="D", mfc="none", mew=1.2, ms=5, alpha=0.8),
}

# ==========================================================
# CHARGEMENT
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
print(f"Chargé : {len(results)} expériences")

# ==========================================================
# PARSING LABEL
# ==========================================================
def parse_label(label):
    lines = [l.strip().lower() for l in label.split("\n") if l.strip()]
    if len(lines) < 3:
        return None
    try:
        L_mm = int(lines[1].replace("mm", ""))
        sand = lines[2]
        return L_mm, sand
    except:
        return None

# ==========================================================
# STOCKAGE
# ==========================================================
D_values = {}
alphaT_values = {}

# ==========================================================
# FIT D (pente forcée -1)
# ==========================================================
fig, ax = plt.subplots(figsize=(8, 5))

for res in results:

    info = parse_label(res.get("label", ""))
    if info is None:
        continue

    L_mm, sand = info

    time_red = res["time"]
    mean = res["mean"]
    var = res["var"]
    Ta = res["Ta"]

    Sigma = var / (res["A"] * mean**2)
    t_phys = time_red * Ta

    valid = (
        np.isfinite(time_red) &
        np.isfinite(Sigma) &
        (Sigma > 0)
        # (time_red >= T_MIN) &
        # (time_red <= T_MAX)
    )

    t_fit = t_phys[valid]
    s_fit = Sigma[valid]

    if len(t_fit) < 5:
        continue

    # ======================================================
    # FIT : Sigma = C / t
    # ======================================================
    logC = np.mean(np.log(s_fit) + np.log(t_fit))
    C = np.exp(logC)

    D = 2.0 / (8.0 * np.pi * C)

    D_values.setdefault((L_mm, sand), []).append(D)

    # vitesse caractéristique
    u = 0.01 / Ta
    alphaT = D / u
    alphaT_values.setdefault((L_mm, sand), []).append(alphaT)

    # ======================================================
    # PLOT DATA
    # ======================================================
    style = style_map.get((L_mm, sand), dict(marker="o"))

    ax.plot(2 * time_red[valid][::3],
            Sigma[valid][::3],
            linestyle="",
            **style)

    # courbe fit
    t_line = np.logspace(np.log10(t_fit.min()),
                         np.log10(t_fit.max()), 200)
    ax.plot(2 * t_line / Ta,
            C / t_line,
            color=style.get("color", "k"),
            lw=1)

# ==========================================================
# AXES
# ==========================================================
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$2t$")
ax.set_ylim(30,5*10**4)
ax.set_xlim(3,60)
ax.set_ylabel(r"$\Sigma$")
ax.grid(True, ls="--", alpha=0.3)

# ligne de référence t^-1
t_ref = np.logspace(0, 2, 100)
ax.plot(t_ref, 0.5 * t_ref**-1, "k--", label=r"$t^{-1}$")

# ==========================================================
# LEGEND
# ==========================================================
legend = []
for (L_mm, sand), style in style_map.items():
    legend.append(
        Line2D([0], [0],
               marker=style.get("marker", "o"),
               linestyle="None",
               color=style.get("color", "k"),
               mfc=style.get("mfc", "none"),
               mew=style.get("mew", 1),
               ms=6,
               label=f"{L_mm} mm — {sand}")
    )

ax.legend(handles=legend, ncol=2, fontsize=10)
plt.tight_layout()
plt.show()

# ==========================================================
# PLOT alpha_T vs d2/d1
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(6, 4))

for (L_mm, sand), vals in alphaT_values.items():

    if len(vals) == 0:
        continue

    alpha_mean = np.mean(vals)/d1_map[sand]

    d1 = d1_map[sand]
    x = L_mm / (d1 * 1e3)  # ratio sans dimension (mm / m -> cohérent relatif)

    style = style_map.get((L_mm, sand), dict(marker="o"))

    ax2.plot(x, alpha_mean, **style)

ax2.set_xlabel(r"$d_2 / d_1$")
ax2.set_ylabel(r"$\alpha_T = D/u$")
ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================================
# RÉSUMÉ
# ==========================================================
print("\n===== D MOYEN =====")
for k, v in D_values.items():
    v = np.array(v)
    print(f"{k} : {v.mean():.3e} ± {v.std():.2e}")

print("\n===== alpha_T MOYEN =====")
for k, v in alphaT_values.items():
    v = np.array(v)
    print(f"{k} : {v.mean():.3e} ± {v.std():.2e}")