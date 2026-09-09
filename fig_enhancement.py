"""
Post-traitement combiné : data_roi.npy + data_rms.npy
Sable fin uniquement.

Figure 1 (deux axes Y) :
  axe gauche  — Σ = σ²_c / (A · μ²_c)  normalisé à t=0  [log]
  axe droit   — 1 / R²  [m⁻²]  (R = sigma_m du script rms.py)
  couleur = L_mm,  marqueur = rond plein (fine)

Figure 2 :
  Σ(L_mm) / Σ_homo  vs t/Ta   (enhancement)
  couleur = L_mm,  marqueur = rond plein (fine)

Vous pouvez maintenant sélectionner le réplicat de chaque expérience fine
à afficher dans la Figure 2 via les variables SELECTED_*_REPLICATE.
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from collections import defaultdict

# ==========================================================
# PARAMÈTRES DE SÉLECTION DES RÉPLICATS POUR LA FIGURE 2
# ==========================================================
# Choisissez quel réplicat de chaque expérience fine doit apparaître
# dans la Figure 2 (enhancement). 0 = premier, 1 = deuxième, etc.
SELECTED_3MM_REPLICATE  = 1   # ← MODIFIEZ ICI pour 3 mm fine
SELECTED_6MM_REPLICATE  = 0   # ← MODIFIEZ ICI pour 6 mm fine
SELECTED_10MM_REPLICATE = 2   # ← MODIFIEZ ICI pour 10 mm fine

# Si vous voulez exclure un mauvais réplicat pour une taille donnée,
# définissez-le ci-dessous. None = pas d'exclusion.
BAD_3MM_REPLICATE  = None   # ou 2 par exemple
BAD_6MM_REPLICATE  = None   # ou 2 par exemple
BAD_10MM_REPLICATE = None   # ou 2 par exemple

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")
A = 7
B = 15
width = 18 /2.1         # cm
height = 10    /2.1     # cm
inches = 2.54

COLOR_MAP = {
    10: "tab:blue",
    6:  "tab:orange",
    3:  "tab:red",
    0:  "tab:green",
}
SAND = "fine"
L_VALUES = [0, 3, 6, 10]

def style_fine(L_mm, ms=6):
    return dict(marker="o", color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm],
                ms=ms, mew=0.4, alpha=0.85, linestyle="None")

# Marqueurs dédiés à la Figure 2 : un par courbe hétérogène (3/6/10 mm).
# Les ronds restent réservés au cas homogène (non tracé ici).
MARKER_MAP_FIG2 = {3: "D", 6: "s", 10: "^"}

def style_fig2(L_mm, ms=6):
    return dict(marker=MARKER_MAP_FIG2[L_mm], color=COLOR_MAP[L_mm],
                mfc=COLOR_MAP[L_mm], ms=ms, mew=0.4, alpha=0.85,
                linestyle="None")

# ==========================================================
# CHEMINS
# ==========================================================
ROI_PATH = "/home/chorus/data_roi.npy"
RMS_PATH = "/home/chorus/data_rms.npy"

# ==========================================================
# CHARGEMENT
# ==========================================================
roi_results = np.load(ROI_PATH, allow_pickle=True)
rms_results = np.load(RMS_PATH, allow_pickle=True)

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

def index_results(results):
    idx = {}
    for r in results:
        info = parse_label(r.get("label", ""))
        if info is None:
            continue
        key = info
        idx.setdefault(key, []).append(r)
    return idx

roi_idx = index_results(roi_results)
rms_idx = index_results(rms_results)

# ==========================================================
# HELPERS
# ==========================================================
def sigma_curve(res):
    time = res["time"]
    var = res["var"]
    mean = res["mean"]
    A0 = res["A"]
    i0 = res["i0"]
    Sigma = var / (A0 * mean**2)
    Sigma = Sigma / Sigma[i0]
    return time, Sigma, i0

def inv_r2_curve(res):
    time = res["time"]
    sigma_m = res["sigma_m"]
    i0 = res["i0"]
    inv_r2 = 1.0 / sigma_m**2
    inv_r2 = inv_r2 / inv_r2[i0]
    return time, inv_r2

# ==========================================================
# FONCTION POUR SÉLECTIONNER LE RÉPLICAT D'UNE EXPÉRIENCE
# ==========================================================
def select_replicate(res_list, selected_idx, bad_idx=None):
    """
    Sélectionne un réplicat dans une liste.

    Parameters
    ----------
    res_list : list
        Liste des résultats pour une expérience donnée
    selected_idx : int
        Index du réplicat souhaité
    bad_idx : int or None
        Index du réplicat à exclure (si None, aucun exclu)

    Returns
    -------
    res : dict or None
        Le réplicat sélectionné, ou None si aucun disponible
    used_idx : int
        L'index réellement utilisé
    """
    if not res_list:
        return None, -1

    # Si l'index demandé est invalide, prendre le premier
    if selected_idx >= len(res_list):
        print(f"  Attention : réplicat {selected_idx} non trouvé (max {len(res_list)-1}), utilisation du premier.")
        selected_idx = 0

    # Si ce réplicat est marqué comme "mauvais" et qu'il y en a d'autres, passer au suivant
    if bad_idx is not None and selected_idx == bad_idx and len(res_list) > 1:
        print(f"  Le réplicat {selected_idx} est exclu, utilisation du suivant.")
        # Chercher le premier réplicat qui n'est pas mauvais
        for idx in range(len(res_list)):
            if idx != bad_idx:
                return res_list[idx], idx
        # Si tous sont mauvais, prendre le premier quand même
        return res_list[0], 0

    return res_list[selected_idx], selected_idx

# ==========================================================
# FIGURE 1 — Σ et 1/R² vs t/Ta (double axe Y)
# ==========================================================
fig1, ax1 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

keys_sorted = sorted([k for k in roi_idx], key=lambda k: k[0])

for L_mm, sand in keys_sorted:
    st = style_fine(L_mm, ms=5) if sand == "fine" else dict(
        marker="D", color=COLOR_MAP[L_mm], mfc="none",
        ms=5, mew=1.0, alpha=0.65, linestyle="None")

    for res in rms_idx.get((L_mm, sand), []):
        time, inv_r2 = inv_r2_curve(res)
        roi_list = roi_idx.get((L_mm, sand), [])
        if not roi_list:
            continue
        _, Sigma, i0 = sigma_curve(roi_list[0])
        valid_roi = np.isfinite(roi_list[0]["time"]) & np.isfinite(Sigma) & (Sigma > 0)
        f_sig = interp1d(roi_list[0]["time"][valid_roi], Sigma[valid_roi],
                         kind="linear", bounds_error=False, fill_value=np.nan)
        Sigma_interp = f_sig(time)
        valid = np.isfinite(inv_r2) & (inv_r2 > 0)
        ax1.plot(2 * time[valid][::3], 1 / inv_r2[valid][::3], **st)

ax1.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
ax1.set_ylabel(r"$R^2/R_0^2 \cdot \Sigma \;/\; \Sigma_0$")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xlim(left=3, right=70)
ax1.set_ylim(1, 30)
ax1.grid(True, ls="--", alpha=0.3)

legend_elements = []
for L_mm in sorted(COLOR_MAP.keys()):
    legend_elements.append(
        Line2D([0], [0], marker="o", linestyle="None",
               color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm], mew=0.4,
               label=f"$d_2$ = {L_mm} mm — fine")
    )
    legend_elements.append(
        Line2D([0], [0], marker="D", linestyle="None",
               color=COLOR_MAP[L_mm], mfc="none",
               mew=1.0, alpha=0.65,
               label=f"$d_2$ = {L_mm} mm — coarse")
    )
ax1.legend(handles=legend_elements, ncol=2, loc="lower left")
plt.show()

# ==========================================================
# FIGURE 2 — Σ / Σ_homo vs t/Ta  (enhancement) AVEC SÉLECTION
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

homo_rois = roi_idx.get((0, "fine"), [])
if len(homo_rois) == 0:
    print("[WARN] Pas d'expérience homogène fine — Fig 2 vide")
else:
    t_homo_ref = homo_rois[0]["time"]
    Sigma_homo_list = []
    for res in homo_rois:
        t_h, S_h, i0 = sigma_curve(res)
        valid = np.isfinite(t_h) & np.isfinite(S_h) & (S_h > 0)
        if valid.sum() < 5:
            continue
        f_interp = interp1d(t_h[valid], S_h[valid],
                            kind="linear", bounds_error=False,
                            fill_value=np.nan)
        Sigma_homo_list.append(f_interp(t_homo_ref))
    Sigma_homo_mean = np.nanmean(np.vstack(Sigma_homo_list), axis=0)

    # Dictionnaire des sélections par taille
    SELECTION_MAP = {
        3:  {"idx": SELECTED_3MM_REPLICATE,  "bad": BAD_3MM_REPLICATE},
        6:  {"idx": SELECTED_6MM_REPLICATE,  "bad": BAD_6MM_REPLICATE},
        10: {"idx": SELECTED_10MM_REPLICATE, "bad": BAD_10MM_REPLICATE},
    }

    used_replicates = {}  # Pour stocker les index utilisés

    # Parcourir les expériences fine (hors homogène)
    for L_mm, sand in sorted([k for k in roi_idx if k[1] == "fine"], key=lambda x: x[0]):
        if L_mm == 0:
            continue

        res_list = roi_idx.get((L_mm, sand), [])
        if not res_list:
            continue

        # ---- Sélection du réplicat selon la taille ----
        if L_mm in SELECTION_MAP:
            sel = SELECTION_MAP[L_mm]
            res, used_idx = select_replicate(res_list, sel["idx"], sel["bad"])
            used_replicates[L_mm] = used_idx
        else:
            res = res_list[0]
            used_replicates[L_mm] = 0

        if res is None:
            print(f"  Attention : aucun réplicat valide pour {L_mm} mm fine")
            continue

        time, Sigma, i0 = sigma_curve(res)
        valid_h = np.isfinite(t_homo_ref) & np.isfinite(Sigma_homo_mean)
        f_homo = interp1d(t_homo_ref[valid_h], Sigma_homo_mean[valid_h],
                          kind="linear", bounds_error=False,
                          fill_value=np.nan)
        Sigma_homo_interp = f_homo(time)
        ratio = Sigma / Sigma_homo_interp
        ratio /= ratio[i0]
        ratio = 1.0 / ratio   # on veut Σ_homo / Σ

        valid = (np.isfinite(time) & np.isfinite(ratio) &
                 (ratio > 0) & (time > -1))

        st = style_fig2(L_mm, ms=6)
        label = f"$d_M$ = {L_mm} mm"
        if L_mm in used_replicates:
            label += f" (rép. {used_replicates[L_mm]})"
        ax2.plot(2 * time[valid][::3], ratio[valid][::3], **st, label=label)

# Mise en forme de la Figure 2
ax2.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
ax2.axhline(1, color="k", ls=":", lw=0.7, alpha=0.4)
ax2.set_xlabel(r"$2t \cdot u $ (cm)")
ax2.set_ylabel(r"$\frac{\sigma^2}{\mu^2} / \frac{\sigma_m^2}{\mu_m^2}$")
ax2.set_yscale("log")
ax2.set_xlim(left=-1, right=50)
ax2.set_ylim(bottom=0.95, top=50)
ax2.grid(True, ls="--", alpha=0.3)

# Légende personnalisée avec les numéros de réplicats
legend2 = []
for L_mm in sorted(COLOR_MAP.keys()):
    if L_mm > 0:
        if L_mm==3:
            Pe=70
        elif L_mm==6:
            Pe=140
        elif L_mm==10:
            Pe=245
        label = f"$Pe_M$ = {Pe}"
        #if L_mm in used_replicates:
            #label += f" (rép. {used_replicates[L_mm]})"
        legend2.append(
            Line2D([0], [0], marker=MARKER_MAP_FIG2[L_mm], linestyle="None",
                   color=COLOR_MAP[L_mm], mfc=COLOR_MAP[L_mm],
                   mew=0.4, label=label)
        )
ax2.legend(handles=legend2, ncol=1, fontsize=8, frameon=False)

fig2.savefig("/home/chorus/TexProject/clement draft avance (document de travail)/figures/enhancement.pdf", dpi=150, bbox_inches="tight")
plt.show()

# ==========================================================
# SUITE : Figure 3, D, etc. (inchangé)
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
# FIGURE 3 — Σ vs t (fine + coarse avec style_map)
# ==========================================================
fig3, ax3 = plt.subplots(figsize=(width / inches, height / inches),
                         layout="constrained")

for (L_mm, sand), style in style_map.items():
    for res in roi_idx.get((L_mm, sand), []):
        time, Sigma, i0 = sigma_curve(res)
        valid = np.isfinite(time) & np.isfinite(Sigma) & (Sigma > 0)
        ax3.plot(2 * time[valid][::3], Sigma[valid][::3],
                 linestyle="", **style)

ax3.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
ax3.set_xlabel(r"$2t$")
ax3.set_ylabel(r"$\Sigma$")
ax3.set_yscale("log")
ax3.set_xscale("log")
ax3.set_xlim(left=5, right=70)
ax3.set_ylim(bottom=0.01, top=1.2)
ax3.grid(True, ls="--", alpha=0.3)

legend3 = []
for (L_mm, sand), style in style_map.items():
    label = f"$d_2$ = {L_mm} mm — {sand}"
    legend3.append(
        Line2D([0], [0],
               marker=style.get("marker", "o"),
               linestyle="None",
               color=style.get("color", "k"),
               mfc=style.get("mfc", "none"),
               mew=style.get("mew", 1.0),
               ms=style.get("ms", 6),
               alpha=style.get("alpha", 1.0),
               label=label)
    )
ax3.legend(handles=legend3, ncol=2, loc="lower left")
t_ref = np.logspace(np.log10(1), np.log10(60), 100)
y_ref = t_ref**-1
factor = 0.5
ax3.plot(t_ref, factor * y_ref, 'k--', label=r"$\propto t^1$")
plt.show()

# ==========================================================
# FIT D ET PLOT (inchangé)
# ==========================================================
fig, ax = plt.subplots(figsize=(width / inches, height / inches),
                        layout="constrained")

D_dict = {}
D_err_dict = {}
u_dict = {}

for (L_mm, sand), style in style_map.items():
    Ds = []
    us = []
    for res in rms_idx.get((L_mm, sand), []):
        time = res["time"]
        sigma_m = res["sigma_m"]
        Ta = res["Ta"]
        R2 = sigma_m**2
        valid = np.isfinite(time) & np.isfinite(R2) & (time > A) & (time <= B)
        t = time[valid]
        y = R2[valid]
        if len(t) < 5:
            continue
        slope, _ = np.polyfit(t, y, 1)
        D = abs(slope) / (4.0 * Ta)
        u = 0.01 / Ta
        D = D / u
        if np.isfinite(D) and D > 0:
            Ds.append(D)
            us.append(u)
    if len(Ds) > 0:
        D_dict[(L_mm, sand)] = np.mean(Ds)
        D_err_dict[(L_mm, sand)] = np.std(Ds)
        u_dict[(L_mm, sand)] = np.mean(us)
    else:
        print(f"[WARN] No valid D for (L_mm={L_mm}, sand={sand})")

_diff_data = np.load("../vieux codes/resultats_diffusion.npy", allow_pickle=True).item()
_alphaT_raw = _diff_data["alphaT"]
_alphaT_mean = {sand: np.mean(vals) for sand, vals in _alphaT_raw.items()}

for (L_mm, sand), style in style_map.items():
    if (L_mm, sand) not in D_dict:
        continue
    alphaT_homo = _alphaT_mean[sand]
    u_mean = u_dict[(L_mm, sand)]
    D_m = alphaT_homo
    d2_m = L_mm * 1e-3
    Pe = d2_m / alphaT_homo
    Dnorm = D_dict[(L_mm, sand)] / D_m
    Dnorm_err = D_err_dict[(L_mm, sand)] / D_m
    ax.errorbar(
        Pe, Dnorm,
        yerr=Dnorm_err,
        fmt=style.get("marker", "o"),
        ms=style.get("ms", 6),
        color=style.get("color", "k"),
        mfc=style.get("mfc", "none"),
        mew=style.get("mew", 1.0),
        capsize=4,
        elinewidth=1.0,
        alpha=style.get("alpha", 0.8),
        linestyle="None"
    )

ax.set_xlabel(r"$Pe = d_2 / \alpha_{\perp}$")
ax.set_ylabel(r"$D_{\perp M} / D_{\perp m}$")
ax.set_yscale("log")
ax.set_xscale("log")
ax.grid(True, ls="--", alpha=0.3)

Pe_all = []
D_all = []
D_err_all = []

for (L_mm, sand), style in style_map.items():
    if (L_mm, sand) not in D_dict or L_mm == 0:
        continue
    alphaT_homo = _alphaT_mean[sand]
    u_mean = u_dict[(L_mm, sand)]
    D_m = alphaT_homo
    d2_m = L_mm * 1e-3
    Pe_all.append(d2_m / alphaT_homo)
    D_all.append(D_dict[(L_mm, sand)] / D_m)
    D_err_all.append(D_err_dict[(L_mm, sand)] / D_m)

Pe_all = np.array(Pe_all)
D_all = np.array(D_all)
D_err_all = np.array(D_err_all)

valid_fit = (Pe_all > 0) & (D_all > 0) & np.isfinite(D_all) & np.isfinite(Pe_all)
w = np.where(D_err_all > 0, D_all / D_err_all, 1.0)

log_Pe = np.log10(Pe_all[valid_fit])
log_D = np.log10(D_all[valid_fit])
w_fit = w[valid_fit]

coeffs, cov = np.polyfit(log_Pe, log_D, 1, w=w_fit, cov=True)
slope = coeffs[0]
intercept = coeffs[1]
slope_err = np.sqrt(cov[0, 0])

Pe_fit = np.logspace(np.log10(Pe_all[valid_fit].min()),
                     np.log10(Pe_all[valid_fit].max()), 200)
D_fit = 10**intercept * Pe_fit**slope
ax.plot(Pe_fit, D_fit, "k-", lw=1.5,
        label=rf"fit : slope $= {slope:.2f} \pm {slope_err:.2f}$")

ax.legend()
plt.show()

# ==========================================================
# RÉSUMÉ DES SÉLECTIONS
# ==========================================================
print("\n" + "=" * 60)
print("RÉSUMÉ DES RÉPLICATS UTILISÉS DANS LA FIGURE 2")
print("=" * 60)
print(f"  3 mm fine  : réplicat {used_replicates.get(3, 'N/A')}")
print(f"  6 mm fine  : réplicat {used_replicates.get(6, 'N/A')}")
print(f" 10 mm fine  : réplicat {used_replicates.get(10, 'N/A')}")
print("=" * 60)