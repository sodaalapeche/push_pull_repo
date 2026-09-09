import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros

# ==========================================================
# CONFIG
# ==========================================================
DATA_PATH = "/home/chorus/data_full.npy"

# Fit windows
FIT_TMIN_0MM = 5
FIT_TMAX_0MM = 20
FIT_TMIN_OTHERS = 16
FIT_TMAX_OTHERS = 23

VALID_SANDS = {"fine", "coarse"}
D1_MM = {"fine": 0.1, "coarse": 0.6}  # mm

R = 0.055 / 2
ELL = 0.01
DM = 5e-9  # diffusion moléculaire [m^2/s]

# ==========================================================
# HELPERS
# ==========================================================
def parse_label(label_text):
    lines = [l.strip().lower() for l in str(label_text).split("\n") if l.strip()]
    if len(lines) < 3: return None, None
    try:
        d2_mm = int(lines[1].replace("mm", ""))
    except:
        return None, None
    sand = lines[2]
    return d2_mm, sand

def sigma_from_result(res):
    time = np.asarray(res["time"], dtype=float)
    mean = np.asarray(res["mean"], dtype=float)
    var = np.asarray(res["var"], dtype=float)
    sigma_norm = var / (mean**2)
    # Dénormalisation : les données sont stockées divisées par Sigma_max
    sigma_max = float(res["Sigma_max"]) if "Sigma_max" in res else np.nanmax(sigma_norm)
    sigma = sigma_norm * sigma_max          # σ physique réel
    valid = (time > 0) & (sigma > 0) & np.isfinite(time) & np.isfinite(sigma)
    return time, sigma, valid

def fit_loglog_minus1(time, sigma, tmin, tmax):
    mask = (time >= tmin) & (time <= tmax)
    if np.sum(mask) < 2: return np.nan
    t_fit = time[mask]
    s_fit = sigma[mask]
    A = np.mean(s_fit * t_fit)
    return A

def compute_D0_from_A(A):
    """D0 (cas 0mm) : σ(t) ~ A/t  →  D0 = 1/(8π A)"""
    if not np.isfinite(A) or A <= 0: return np.nan
    return 1.0 / (8 * np.pi * A)

def compute_loglog_slope(time, sigma, tmin, tmax):
    """Cas d2≠0mm : fit semi-log  log(σ) vs t  →  retourne beta = -slope [s⁻¹]"""
    m = (np.isfinite(time) & np.isfinite(sigma) &
         (time > 0) & (sigma > 0) &
         (time >= tmin) & (time <= tmax))
    if np.count_nonzero(m) < 2: return np.nan
    x = time[m]
    y = np.log(sigma[m])
    if np.unique(x).size < 2: return np.nan
    slope, _ = np.polyfit(x, y, 1)
    return -slope   # beta > 0

def compute_D_total(beta, u):
    """D_total = beta * (u * R²) / (2 * b0² * ELL)"""
    b0 = jn_zeros(0, 1)[0]   # ≈ 2.4048
    if not np.isfinite(beta) or beta <= 0: return np.nan
    if not np.isfinite(u)    or u    <= 0: return np.nan
    return beta * (u * R**2) / (2 * b0**2 * ELL)

# ==========================================================
# LOAD
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)

# ==========================================================
# CALCUL D0MM — moyenne sur toutes les expériences 0mm par sable
# ==========================================================
D0_runs   = {}   # {sand: [(Ta, u, A0, D), ...]}  toutes les runs 0mm
D0_raw    = {}   # pour la figure 2 : liste de dicts par sand
t_fit_ref = np.linspace(FIT_TMIN_0MM, FIT_TMAX_0MM, 100)

for res in results:
    d2_mm, sand = parse_label(res.get("label", ""))
    if d2_mm != 0 or sand not in VALID_SANDS:
        continue
    time, sigma, valid = sigma_from_result(res)
    if not np.any(valid): continue

    A0  = fit_loglog_minus1(time[valid], sigma[valid], FIT_TMIN_0MM, FIT_TMAX_0MM)
    D_i = compute_D0_from_A(A0)
    Ta  = float(res.get("Ta", np.nan))
    u_i = 0.01 / Ta if np.isfinite(Ta) and Ta > 0 else np.nan

    if sand not in D0_runs:
        D0_runs[sand] = []
        D0_raw[sand]  = []
    D0_runs[sand].append({"Ta": Ta, "u": u_i, "A0": A0, "D": D_i})
    D0_raw[sand].append({"time": time, "sigma": sigma, "valid": valid,
                          "A0": A0, "Ta": Ta, "u": u_i})

# Moyenne des A0 → D0 moyen par sable
D0           = {}
D0_fit_curves = {}
for sand, runs in D0_runs.items():
    A0_vals = [r["A0"] for r in runs if np.isfinite(r["A0"]) and r["A0"] > 0]
    A0_mean = np.mean(A0_vals)
    D0[sand] = compute_D0_from_A(A0_mean)
    D0_fit_curves[sand] = A0_mean / t_fit_ref

print("\n=== D0mm — détail par run ===")
for sand, runs in D0_runs.items():
    for r in runs:
        print(f"  {sand:6s}  Ta={r['Ta']:.2f}s  u={r['u']:.4f}m/s  "
              f"A0={r['A0']:.4g}  D={r['D']:.3e} m²/s")
    print(f"  → D0[{sand}] moyen = {D0[sand]:.3e} m²/s  "
          f"(sur {len(runs)} run(s))\n")

# ==========================================================
# CALCUL POINTS EXPERIMENTAUX  (inchangé)
# ==========================================================
data = []
for res in results:
    d2_mm, sand = parse_label(res.get("label", ""))
    if d2_mm is None or sand not in VALID_SANDS or d2_mm == 0:
        continue
    time, sigma, valid = sigma_from_result(res)
    if not np.any(valid): continue

    Ta  = res.get("Ta", np.nan)
    u   = 0.01 / Ta if np.isfinite(Ta) and Ta > 0 else np.nan
    d2_m = d2_mm * 1e-3

    # Fit semi-log : log(σ) vs t  →  beta [s⁻¹]
    beta    = compute_loglog_slope(time[valid], sigma[valid],
                                   FIT_TMIN_OTHERS, FIT_TMAX_OTHERS)
    D_total = compute_D_total(beta, u)
    Pe      = u * d2_m / D0[sand] if np.isfinite(u) else np.nan

    data.append({
        "d2":     d2_mm,
        "sand":   sand,
        "Pe":     Pe,
        "beta":   beta,
        "D_total": D_total,
    })

# ==========================================================
# STYLE MAP
# ==========================================================
style_map = {
    (10, "fine"):   dict(marker="o", color="tab:blue",
                         mfc="tab:blue",  ms=4, alpha=0.8),
    (10, "coarse"): dict(marker="D", color="tab:blue",
                         mfc="none", mew=1.2, ms=5, alpha=0.8),
    (10, "stokes"): dict(marker="^", color="purple",
                         mfc="none", mew=1.2, ms=5, alpha=0.8),

    (6, "fine"):    dict(marker="o", color="tab:orange",
                         mfc="tab:orange", ms=4, alpha=0.8),
    (6, "coarse"):  dict(marker="D", color="tab:orange",
                         mfc="none", mew=1.2, ms=5, alpha=0.8),

    (3, "fine"):    dict(marker="o", color="tab:red",
                         mfc="tab:red",  ms=4, alpha=0.8),
    (3, "coarse"):  dict(marker="D", color="tab:red",
                         mfc="none", mew=1.2, ms=5, alpha=0.8),

    (0, "fine"):    dict(marker="o", color="tab:green",
                         mfc="tab:green", ms=4, alpha=0.8),
    (0, "coarse"):  dict(marker="D", color="tab:green",
                         mfc="none", mew=1.2, ms=5, alpha=0.8),
}

def get_style(d2_mm, sand):
    """Retourne le style pour (d2_mm, sand), ou un style par défaut."""
    return style_map.get((d2_mm, sand), dict(marker="x", color="gray", ms=5, alpha=0.7))



# FIGURE 1 — originale
# ==========================================================
# Edit this set to show/hide combinations — None means "all"
SELECTION = None  # e.g. {(10, "fine"), (6, "coarse"), (3, "fine")}
# SELECTION = {(10, "fine"), (6, "fine"), (3, "coarse")}
fig, ax = plt.subplots(figsize=(7, 5))

legend_seen = set()

data_sel = [row for row in data if SELECTION is None or (row["d2"], row["sand"]) in SELECTION]

for row in data_sel:
    sand = row["sand"]
    d2 = row["d2"]
    Pe = row["Pe"]
    ratio = row["D_total"] / D0[sand]

    sty = get_style(d2, sand)
    key = (d2, sand)
    label = f"d₂={d2} mm — {sand}" if key not in legend_seen else ""
    legend_seen.add(key)

    ax.plot(Pe, ratio, sty["marker"],
            color=sty["color"], mfc=sty.get("mfc", sty["color"]),
            mew=sty.get("mew", 1.0), ms=sty["ms"], alpha=sty["alpha"],
            label=label)

# (les courbes fit D0mm sont affichées en Figure 2 uniquement)

x = np.array([row["Pe"] for row in data_sel])
y = np.array([row["D_total"] / D0[row["sand"]] for row in data_sel])
mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
xfit = x[mask];
yfit = y[mask]

logx = np.log10(xfit)
logy = np.log10(yfit)
coeffs, cov = np.polyfit(logx, logy, 1, cov=True)
a, b = coeffs
da = np.sqrt(cov[0, 0])
y_pred = 10 ** b * xfit ** a
ax.plot(xfit, y_pred, "--", color="black", label=f"fit slope={a:.2f} ± {da:.2f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Pe")
ax.set_ylabel(r"$D / D_{0mm}$")
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
# ==========================================================
# FIGURE 2 — courbes de variance 0mm + fit reconstruit en log-log
# Une colonne par sable, une courbe par run (vitesse différente)
# ==========================================================
sands_sorted = sorted(D0_raw.keys())
fig2, axes = plt.subplots(1, len(sands_sorted), figsize=(6 * len(sands_sorted), 5), squeeze=False)

# Palette pour distinguer les runs d'une même colonne
run_colors = ["black", "dimgray", "silver"]

for col, sand in enumerate(sands_sorted):
    ax2 = axes[0][col]
    runs = D0_raw[sand]
    sty  = get_style(0, sand)

    t_dense = np.linspace(FIT_TMIN_0MM, FIT_TMAX_0MM, 200)

    for i, run in enumerate(runs):
        time  = run["time"]
        sigma = run["sigma"]
        valid = run["valid"]
        A0    = run["A0"]
        u_i   = run["u"]
        rc    = run_colors[i % len(run_colors)]
        u_label = f"u={u_i:.4f} m/s" if np.isfinite(u_i) else f"run {i+1}"

        # Données brutes
        ax2.plot(time[valid], sigma[valid],
                 sty["marker"], color=rc,
                 mfc=rc, mew=sty.get("mew", 1.0),
                 ms=sty["ms"], alpha=0.5,
                 label=f"σ(t) {u_label}")

        # Points dans la fenêtre de fit
        mask_win = valid & (time >= FIT_TMIN_0MM) & (time <= FIT_TMAX_0MM)
        ax2.plot(time[mask_win], sigma[mask_win],
                 sty["marker"], color=rc,
                 mfc=rc, mew=sty.get("mew", 1.0),
                 ms=sty["ms"] + 3, alpha=1.0, zorder=5)

        # Fit individuel A0_i / t
        ax2.plot(t_dense, A0 / t_dense,
                 "--", color=rc, lw=1.5,
                 label=f"fit A₀/t  (A₀={A0:.3g})")

    # Fit moyen (A0 moyen utilisé pour D0)
    A0_mean = np.mean([r["A0"] for r in runs if np.isfinite(r["A0"]) and r["A0"] > 0])
    ax2.plot(t_dense, A0_mean / t_dense,
             "-", color=sty["color"], lw=2.5, zorder=6,
             label=rf"fit moyen $\bar{{A_0}}$/t  (D₀={D0[sand]:.3e})")

    D_val = D0[sand]
    ax2.set_title(f"{sand.capitalize()} sand — 0 mm\n"
                  r"$D_0$" + f" = {D_val:.3e} m²/s  ({len(runs)} run(s))", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("t  (s)", fontsize=11)
    ax2.set_ylabel(r"$\sigma(t) = \mathrm{Var}/\langle s \rangle^2$", fontsize=10)

    # Zone de fit en grisé
    ax2.axvspan(FIT_TMIN_0MM, FIT_TMAX_0MM, alpha=0.10, color="gray",
                label=f"fenêtre fit [{FIT_TMIN_0MM}, {FIT_TMAX_0MM}] s")

    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend(fontsize=9)

fig2.suptitle("Courbes de variance σ(t) — cas 0 mm — fit en t⁻¹ (log-log)",
              fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()