"""
Dispersion en régime initial (temps courts, autour de i0).

Pour chaque expérience (L_mm > 0, fine + coarse) :
  1. s0 = sigma_m[i0]  (taille initiale du panache, depuis data_rms)
  2. Fit semi-log de Σ(t/Ta) sur la fenêtre [C, D] autour de i0
     → pente β_init  [Ta⁻¹]
  3. Coefficient de diffusion initial :
        D_init = - β_init · s0² / (2 · Ta)
     Dérivation : d(log Σ)/d(t/Ta)|_{t=0} = -2D·Ta / s0²
                  donc β_init = -2·D·Ta/s0²
                  donc D = -β_init · s0² / (2·Ta)

Graphes :
  - Fig 1 : Σ(t/Ta) semi-log, fenêtre initiale surlignée + droite de fit
  - Fig 2 : D_init vs L_mm, par type de sable
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import scienceplots

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")

width  = 18/2.1# cm
height = 10/2.1
inches = 2.54

# ==========================================================
# PARAMÈTRES — À AJUSTER
# ==========================================================
DATA_ROI_PATH  = "/home/chorus/data_roi.npy"
DATA_RMS_PATH  = "/home/chorus/data_rms.npy"
DIFFUSION_PATH = "../vieux codes/resultats_diffusion.npy"

# Fenêtre de fit autour de i0, en unités de t/Ta
# Le fit commence à C (peut être légèrement négatif pour inclure i0)
# et se termine à D.
C = 0.1   # début de fenêtre [Ta] — typiquement 0 (= i0)
D = 2.3   # fin   de fenêtre [Ta] — à ajuster selon la durée du régime initial

Dm = 5e-9  # diffusion moléculaire [m²/s] (pour normalisation éventuelle)

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
# INDEXATION  (L_mm, sand) -> liste de résultats
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
    """Retourne (time_red, Sigma, i0) — Sigma normalisée à 1 en i0."""
    time  = res["time"]
    var   = res["var"]
    mean  = res["mean"]
    A0    = res["A"]
    i0    = res["i0"]
    Sigma = var / (A0 * mean**2)
    Sigma = Sigma / Sigma[i0]
    return time, Sigma, i0

def s0_from_rms(rms_res):
    """
    Retourne s0 = sigma_m[i0] en mètres (ou unité native du tableau).
    C'est la taille initiale du panache au moment du lâcher.
    """
    sigma_m = rms_res["sigma_m"]
    i0      = rms_res["i0"]
    val     = sigma_m[i0]
    if not np.isfinite(val) or val <= 0:
        return np.nan
    return float(val)

def semilog_fit_window(time_red, Sigma, t_A, t_B):
    """
    Fit semi-log de Σ sur [t_A, t_B] (en unités de t/Ta).
    Retourne (beta [Ta⁻¹], log_S0, n_points).
    """
    mask = (
        np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0) &
        (time_red >= t_A) & (time_red <= t_B)
    )
    t_f, S_f = time_red[mask], Sigma[mask]
    n = t_f.size
    if n < 3:
        return np.nan, np.nan, n
    c = np.polyfit(t_f, np.log(S_f), 1)
    return c[0], c[1], n

# ==========================================================
# CALCUL POUR CHAQUE EXPÉRIENCE
# ==========================================================
records = []

diff_data  = np.load(DIFFUSION_PATH, allow_pickle=True).item()
alphaT_map = {}
for sand, vals in diff_data["alphaT"].items():
    alphaT_map[sand] = float(np.mean(vals))
    print(f"  αT [{sand:6s}] = {alphaT_map[sand]:.4e} m")

for (L_mm, sand), roi_list in roi_idx.items():
    if L_mm == 0:
        continue

    style    = style_map.get((L_mm, sand),
                              dict(marker="s", color="gray", mfc="gray", ms=6, alpha=0.8))
    rms_list = rms_idx.get((L_mm, sand), [])

    for i, res_roi in enumerate(roi_list):

        # --- courbe Σ(t/Ta) ---
        time_red, Sigma, i0 = sigma_curve(res_roi)
        Ta = float(res_roi["Ta"])

        # --- s0 depuis rms apparié ---
        if i < len(rms_list):
            rms_res = rms_list[i]
        elif rms_list:
            rms_res = rms_list[0]
        else:
            print(f"  [SKIP] {L_mm}mm {sand} exp#{i} : pas de rms apparié")
            continue

        s0 = s0_from_rms(rms_res)   # [mm] ou [m] selon l'unité de sigma_m
        if np.isnan(s0):
            print(f"  [SKIP] {L_mm}mm {sand} exp#{i} : s0 invalide")
            continue

        # --- fit β sur la fenêtre initiale [C, D] (en t/Ta) ---
        beta_init, log_S0, n_pts = semilog_fit_window(time_red, Sigma, C, D)
        print(beta_init)
        if np.isnan(beta_init) or beta_init >= 0:
            print(f"  [SKIP] {L_mm}mm {sand} exp#{i} : "
                  f"β_init={beta_init} invalide ({n_pts} pts)")
            continue

        # --- Exclusion manuelle : 6mm fine avec beta_init < -1 (outlier) ---
        if L_mm == 6 and sand == "fine" and beta_init > -0.1:
            print(f"  [EXCLU] {L_mm}mm {sand} exp#{i} : "
                  f"β_init={beta_init:.4f} < -1, exclu manuellement")
            continue
        if np.isnan(beta_init) or beta_init >= 0:
            print(f"  [SKIP] {L_mm}mm {sand} exp#{i} : "
                  f"β_init={beta_init} invalide ({n_pts} pts)")
            continue

        # --- Exclusion manuelle : 6mm fine avec beta_init < -1 (outlier) ---
        if L_mm == 6 and sand == "fine" and beta_init < -1:
            print(f"  [EXCLU] {L_mm}mm {sand} exp#{i} : "
                  f"β_init={beta_init:.4f} < -1, exclu manuellement")
            continue

        if sand == "coarse":
            sands = 0.0006
        else:
            sands = 0.00009
        # --- D_init = -β_init · s0² / (2 · Ta) ---
        # Unités : β_init [Ta⁻¹], s0 [m], Ta [s]  →  D [m²/s]
        # Si sigma_m est en mm, convertir s0 en m : s0_m = s0 * 1e-3
        s0_m   = s0 * 1e-3          # ← ajuster si sigma_m déjà en mètres
        D_init = (-beta_init * (s0_m*1.5)**2) / (sands *(2.0 ))#) [m²/s]
        #D_init = -beta_init
        print(f"  {L_mm:2d}mm {sand:6s} exp#{i} : "
              f"β_init={beta_init:.4f} Ta⁻¹, "
              f"s0={s0_m*1e3:.2f} mm, "
              f"Ta={Ta:.3f} s, "
              f"D_init={D_init:.3e} m²/s  ({n_pts} pts)")

        records.append(dict(
            L_mm=L_mm, sand=sand,
            beta_init=beta_init, log_S0=log_S0,
            s0=s0_m, Ta=Ta,
            D_init=D_init,
            style=style,
            time_red=time_red, Sigma=Sigma,
            alphaT=alphaT_map[sand],
        ))

print(f"\nNombre de points retenus : {len(records)}")

if len(records) == 0:
    print("[ERREUR] Aucun point retenu — vérifier C, D et les données.")
    raise SystemExit

# ==========================================================
# FIGURE 1 — Σ vs t/Ta semi-log, fenêtre initiale + droite de fit
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

ax1.axvspan(C, D, alpha=0.12, color="steelblue",
            label=rf"fenêtre initiale $[{C:.1f},\,{D:.1f}]\,T_a$")
ax1.axvline(C, color="steelblue", ls=":", lw=0.8)
ax1.axvline(D, color="steelblue", ls=":", lw=0.8)
ax1.axvline(0, color="k",         ls="--", lw=0.7, alpha=0.5)

for r in records:
    time_red = r["time_red"]
    Sigma    = r["Sigma"]
    style    = r["style"]
    beta     = r["beta_init"]
    log_S0_r = r["log_S0"]

    valid = np.isfinite(time_red) & np.isfinite(Sigma) & (Sigma > 0)
    ax1.plot(time_red[valid][::3], Sigma[valid][::3],
             linestyle="None", **style)

    # Droite de fit sur la fenêtre initiale
    t_line = np.linspace(C, D, 80)
    ax1.plot(t_line, np.exp(log_S0_r + beta * t_line),
             color=style["color"], lw=1.4, ls="--", alpha=0.85)

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
    plt.Rectangle((0, 0), 1, 1, fc="steelblue", alpha=0.15,
                  label=rf"fenêtre initiale $[{C:.1f},\,{D:.1f}]\,T_a$")
)

ax1.legend(handles=legend_elements, ncol=2, fontsize=8)
ax1.set_xlabel(r"$t \;/\; T_a$")
ax1.set_ylabel(r"$\Sigma \;/\; \Sigma_0$")
ax1.set_yscale("log")
ax1.set_ylim(bottom=1e-2, top=2.0)
ax1.set_xlim(left=-0.5, right=D + 5)
ax1.grid(True, ls="--", alpha=0.3)
fig1.savefig("/home/chorus/disp_initiale_sigma.pdf")
plt.show()

# ==========================================================
# FIGURE 2 — D_init vs L_mm, par type de sable
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

for r in records:
    if r['sand'] =="coarse":
        sands= 0.0006
    else :
        sands= 0.00009
    style = {k: v for k, v in r["style"].items()}
    Pe = (10**-3*r['L_mm']/r['alphaT'])
    # Pe= r['s0']**2
    ax2.plot(Pe, -1/r["beta_init"], linestyle="None", **style)

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
ax2.legend(handles=legend_elements2, fontsize=8)
ax2.set_xlabel(r"$Pe_M$")
# ax2.set_ylabel(r"$D_{\mathrm{init}}$ [m$^2$/s]")
ax2.set_ylabel(r"$\ell_1^{-1}$ (cm$^{-1}$)")
ax2.grid(True, ls="--", alpha=0.4)
ax2.set_yscale("log")
ax2.set_xscale("log")
fig2.savefig("/home/chorus/disp_initiale_D.pdf")
plt.show()

# ==========================================================
# FIGURE 2 — D_init vs L_mm, par type de sable (avec error bars)
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

# --- Regroupement par (L_mm, sand) ---
groups = {}
for r in records:
    key = (r["L_mm"], r["sand"])
    tau1 = -1 / r["beta_init"]
    Pe   = 1e-3 * r["L_mm"] / r["alphaT"]
    groups.setdefault(key, dict(Pe=[], tau1=[], style=r["style"]))
    groups[key]["Pe"].append(Pe)
    groups[key]["tau1"].append(tau1)

for (L_mm, sand), g in groups.items():
    Pe_arr   = np.array(g["Pe"])
    tau1_arr = np.array(g["tau1"])

    Pe_mean   = np.mean(Pe_arr)
    tau1_mean = np.mean(tau1_arr)
    tau1_err  = np.std(tau1_arr, ddof=1) if len(tau1_arr) > 1 else 0.0

    style = g["style"]
    ax2.errorbar(Pe_mean, tau1_mean, yerr=tau1_err,
                 linestyle="None", capsize=3, elinewidth=1.0,
                 marker=style["marker"], color=style["color"],
                 mfc=style.get("mfc", style["color"]),
                 mew=style.get("mew", 1.0),
                 ms=style.get("ms", 6),
                 alpha=style.get("alpha", 0.8))

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
# ax2.legend(handles=legend_elements2, fontsize=8, ncols=2)
ax2.set_xlabel(r"$Pe_M$")
ax2.set_ylabel(r"$\ell_1^{-1}$ (cm$^{-1}$)")
ax2.grid(True, ls="--", alpha=0.4)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_ylim(bottom=1, top=15)
fig2.savefig("/home/chorus/TexProject/clement draft avance (document de travail)/figures/disp_initiale_D.pdf")
plt.show()


# ==========================================================
# FIGURE 2 — D_init vs L_mm, par type de sable (avec error bars)
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

# --- Regroupement par (L_mm, sand) ---
groups = {}
for r in records:
    key = (r["L_mm"], r["sand"])
    tau1 = -1 / r["beta_init"]
    Pe   = 1e3 * r["s0"] / r["L_mm"]
    # Pe = r["alphaT"]
    groups.setdefault(key, dict(Pe=[], tau1=[], style=r["style"]))
    groups[key]["Pe"].append(Pe)
    groups[key]["tau1"].append(tau1)

for (L_mm, sand), g in groups.items():
    Pe_arr   = np.array(g["Pe"])
    tau1_arr = np.array(g["tau1"])

    Pe_mean   = np.mean(Pe_arr)
    tau1_mean = np.mean(tau1_arr)
    tau1_err  = np.std(tau1_arr, ddof=1) if len(tau1_arr) > 1 else 0.0

    style = g["style"]
    ax2.errorbar(Pe_mean, tau1_mean, yerr=tau1_err,
                 linestyle="None", capsize=3, elinewidth=1.0,
                 marker=style["marker"], color=style["color"],
                 mfc=style.get("mfc", style["color"]),
                 mew=style.get("mew", 1.0),
                 ms=style.get("ms", 6),
                 alpha=style.get("alpha", 0.8))

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
# ax2.legend(handles=legend_elements2, fontsize=8, ncols=2)
ax2.set_xlabel(r"$R_0 / d_M$")
# ax2.set_xlabel(r"$alpha_T$")


ax2.set_ylabel(r"$\ell_1^{-1}$ (cm$^{-1}$)")
ax2.grid(True, ls="--", alpha=0.4)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_ylim(bottom=1, top=15)
fig2.savefig("/home/chorus/TexProject/clement draft avance (document de travail)/figures/tau1scalingrdm.pdf")
plt.show()