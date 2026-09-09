"""
Test : Σ(t) suit-il une LOI DE PUISSANCE  Σ ∼ (t − t0)^(−α) ?

Miroir du test exponentiel :
  - exponentielle  -> droite en semilog   (ln Σ vs t)
  - loi de puissance -> droite en LOG-LOG (ln Σ vs ln(t − t0))

Pour chaque expérience on ajuste  ln Σ = c − α·ln(t − t0)  et on regarde :
  - la rectitude en log-log (Fig 1),
  - la PLATITUDE des résidus (Fig 2) — courbure = pas une loi de puissance,
  - le R² comparé à celui du fit exponentiel sur LES MÊMES points (table),
    les deux prédisant ln Σ -> RSS / ΔBIC directement comparables.

Origine t0 :
  AUTO_T0 = True  -> balayage borné de t0 (juste sous le 1er point) maximisant
                     la linéarité log-log. Borné car t0 → −∞ ramène ln(t−t0)
                     à une fonction linéaire de t (= on retombe sur l'expo).
  Si le t0 optimal colle au bord du scan -> avertissement (résultat douteux).

Rappel physique : en milieu CONFINÉ (masse bornée, modes propres), la queue
attendue est exponentielle, pas en puissance. Une vraie loi de puissance serait
droite sur tout le log-log ; une somme d'exponentielles paraît, elle, COURBÉE
(concave) en log-log. Ce script sert justement à départager.
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# ==========================================================
# STYLE
# ==========================================================
plt.style.use("science")
inches = 2.54

# ==========================================================
# PARAMÈTRES
# ==========================================================
DATA_PATH    = "/home/chorus/data_roi.npy"

TIME_SCALE   = 1.0     # 2.0 pour coller à scaling fig1
START_AT_I0  = True    # commence à t[i0] (après la référence d'injection)
TMAX_TA      = 25.0    # coupure physique en t/Ta (sur le temps réduit brut)

AUTO_T0      = True    # ajuste t0 en maximisant la linéarité log-log
T0           = 0.0     # t0 imposé si AUTO_T0 = False
T0_SCAN_SPAN = 5.0     # amplitude du scan de t0 SOUS le premier point (en t/Ta)
T0_SCAN_N    = 80

style_map = {
    (10, "fine"):   "tab:blue",  (10, "coarse"): "tab:blue",
    (6,  "fine"):   "tab:orange",(6,  "coarse"): "tab:orange",
    (3,  "fine"):   "tab:red",   (3,  "coarse"): "tab:red",
}

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

# ==========================================================
# OUTILS
# ==========================================================
def linfit(x, y):
    """Régression linéaire -> slope, intercept, R², rss, se_slope, yhat."""
    n = x.size
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    rss = np.sum((y - yhat) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    se = np.nan
    if n > 2 and rss > 0:
        se = np.sqrt(rss / (n - 2) * np.linalg.inv(X.T @ X)[1, 1])
    return coef[1], coef[0], r2, rss, se, yhat

def bic(rss, n, k):
    return n * np.log(rss / n) + k * np.log(n)

def best_T0(t, y, span, n):
    """Balayage borné de t0 < t_min maximisant le R² du fit log-log."""
    tmin = t.min()
    cands = np.linspace(tmin - span, tmin - 1e-3, n)
    best = dict(r2=-np.inf)
    for t0 in cands:
        u = np.log(t - t0)
        _, _, r2, *_ = linfit(u, y)
        if r2 > best["r2"]:
            best = dict(r2=r2, t0=t0)
    at_edge = abs(best["t0"] - (tmin - span)) < 1e-6
    return best["t0"], at_edge

# ==========================================================
# CHARGEMENT + COLLECTE
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
print(f"Chargé : {len(results)} expériences\n")

exps = []
for res in results:
    info = parse_label(res.get("label", ""))
    if info is None:
        continue
    L_mm, sand = info
    if L_mm == 0 or (L_mm, sand) not in style_map:
        continue

    time = np.asarray(res["time"], dtype=float)
    mean = np.asarray(res["mean"], dtype=float)
    var  = np.asarray(res["var"],  dtype=float)
    A0   = res["A"]
    i0   = res["i0"]

    Sigma = var / (A0 * mean ** 2)
    t = TIME_SCALE * time
    lo = TIME_SCALE * time[i0] if START_AT_I0 else -np.inf

    ok = (np.isfinite(t) & np.isfinite(Sigma) & (Sigma > 0) &
          (t >= lo) & (time <= TMAX_TA))
    t, Sigma = t[ok], Sigma[ok]
    order = np.argsort(t)
    t, Sigma = t[order], Sigma[order]
    if t.size < 6:
        continue
    exps.append(dict(L_mm=L_mm, sand=sand, t=t, lnS=np.log(Sigma),
                     color=style_map[(L_mm, sand)]))

print(f"Expériences analysées : {len(exps)}\n")

# ==========================================================
# ANALYSE
# ==========================================================
header = (f"{'exp':>14} | {'t0':>7} | {'α':>7} {'σα':>6} | "
          f"{'R²_pow':>7} {'R²_exp':>7} | {'ΔBIC(pow-exp)':>13}")
print(header)
print("-" * len(header))

fitted = []
edge_warns = []
for e in exps:
    t, y = e["t"], e["lnS"]
    n = t.size

    # origine t0
    if AUTO_T0:
        t0, at_edge = best_T0(t, y, T0_SCAN_SPAN, T0_SCAN_N)
        if at_edge:
            edge_warns.append(f"{e['L_mm']}mm {e['sand']}")
        k_pow = 3   # t0 estimé compte comme paramètre
    else:
        t0 = min(T0, t.min() - 1e-3)
        k_pow = 2

    u = np.log(t - t0)                       # log-log
    slope_pl, ic_pl, r2_pl, rss_pl, se_pl, yhat_pl = linfit(u, y)
    alpha = -slope_pl

    # exponentielle (semilog) sur les mêmes points / même y = ln Σ
    _, _, r2_exp, rss_exp, _, _ = linfit(t, y)

    dbic = bic(rss_pl, n, k_pow) - bic(rss_exp, n, 2)

    e.update(t0=t0, u=u, yhat_pl=yhat_pl, alpha=alpha,
             r2_pl=r2_pl, r2_exp=r2_exp)
    fitted.append(e)

    tag = f"{e['L_mm']}mm {e['sand']}"
    print(f"{tag:>14} | {t0:7.2f} | {alpha:7.3f} {se_pl:6.3f} | "
          f"{r2_pl:7.4f} {r2_exp:7.4f} | {dbic:13.1f}")

print("\nLecture :")
print("  - α est l'exposant de la loi de puissance Σ ∼ (t−t0)^(−α).")
print("  - R²_pow > R²_exp ET résidus plats (Fig 2) -> loi de puissance plausible.")
print("  - ΔBIC < 0 favorise la puissance (indicatif).")
print("  - Le JUGE reste la platitude des résidus, pas le R² seul.")
if edge_warns:
    print("\n[!] t0 optimal collé au bord du scan pour : "
          + ", ".join(edge_warns)
          + "\n    -> résultat douteux (dégénérescence vers l'exponentielle).")

# ==========================================================
# FIGURES
# ==========================================================
def grid(n):
    nc = min(3, max(n, 1))
    return int(np.ceil(n / nc)), nc

n = len(fitted)
if n == 0:
    raise SystemExit("Aucune expérience exploitable.")
nr, nc = grid(n)

# Fig 1 : log-log  ln Σ vs ln(t − t0)  + droite de puissance
fig1, axes1 = plt.subplots(nr, nc, figsize=(18 / inches, 5.5 * nr / inches),
                           layout="constrained", squeeze=False)
for ax, e in zip(axes1.flat, fitted):
    ax.plot(e["u"], e["lnS"], ".", ms=2.5, color=e["color"], alpha=0.5)
    ax.plot(e["u"], e["yhat_pl"], "-", color="k", lw=1.3)
    ax.set_title(rf"{e['L_mm']} mm {e['sand']} : "
                 rf"$\alpha={e['alpha']:.3f},\ R^2={e['r2_pl']:.4f}$", fontsize=7)
    ax.set_xlabel(r"$\ln(t/T_a - t_0)$")
    ax.set_ylabel(r"$\ln \Sigma$")
    ax.grid(True, ls="--", alpha=0.3)
for ax in axes1.flat[n:]:
    ax.axis("off")
fig1.savefig("/home/chorus/powerlaw_loglog.pdf")
plt.show()
# Fig 2 : résidus du fit puissance (doivent être plats)
fig2, axes2 = plt.subplots(nr, nc, figsize=(18 / inches, 5.5 * nr / inches),
                           layout="constrained", squeeze=False)
for ax, e in zip(axes2.flat, fitted):
    resid = e["lnS"] - e["yhat_pl"]
    ax.plot(e["u"], resid, ".", ms=2.5, color=e["color"], alpha=0.6)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(f"{e['L_mm']} mm {e['sand']} — résidus puissance", fontsize=7)
    ax.set_xlabel(r"$\ln(t/T_a - t_0)$")
    ax.set_ylabel(r"$\ln\Sigma - \widehat{\ln\Sigma}$")
    ax.grid(True, ls="--", alpha=0.3)
for ax in axes2.flat[n:]:
    ax.axis("off")
fig2.savefig("/home/chorus/powerlaw_residuals.pdf")

plt.show()