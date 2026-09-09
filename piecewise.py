"""
Test de DEUX régimes exponentiels sur Σ(t) : fit piecewise-linéaire continu de ln Σ.

Modèle (régression segmentée continue, "broken-stick") :
    ln Σ(t) = a + β1·t + (β2 − β1)·(t − t_b)_+
  -> pente avant rupture = β1, pente après = β2, raccord continu en t_b.

Le point de rupture t_b est trouvé par balayage (minimisation du RSS).
Pour chaque expérience on reporte : t_b, β1±σ, β2±σ, R² PAR SEGMENT,
et la comparaison au fit mono-segment (R², ΔBIC) pour juger si la rupture
est justifiée. Les résidus par segment (Fig 2) doivent être PLATS si le
modèle à deux exponentielles est correct.

Avertissement statistique : t_b étant estimé sur les données (problème de
Davies), ΔBIC / significativité sont INDICATIFS, pas un test exact. La vraie
preuve = résidus plats par segment + rupture reproductible entre expériences.

NB : pense à régler T_MAX pour tronquer le plancher de bruit de fin (là où Σ
a trop décru et où d(ln Σ)/dt n'a plus de sens).
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
DATA_PATH   = "/home/chorus/data_roi.npy"

TIME_SCALE  = 1.0     # 2.0 pour coller à scaling fig1
START_AT_I0 = True    # commence le fit à t[i0] (après la référence d'injection)
T_MIN       = None    # borne basse supplémentaire (unités de l'axe), ou None
T_MAX       = None    # borne haute en unités de l'axe (None = pas de coupure ici)
TMAX_TA     = 25.0    # coupure PHYSIQUE en t/Ta (disparition de la masse au-delà),
                      # appliquée sur le temps réduit brut, indépendante de TIME_SCALE
MIN_PTS_SEG = 5       # nb min de points de chaque côté de la rupture
N_BREAKS    = 120     # nb de positions de rupture testées

style_map = {
    (10, "fine"):   dict(color="tab:blue"),
    (10, "coarse"): dict(color="tab:blue"),
    (6,  "fine"):   dict(color="tab:orange"),
    (6,  "coarse"): dict(color="tab:orange"),
    (3,  "fine"):   dict(color="tab:red"),
    (3,  "coarse"): dict(color="tab:red"),
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
# AJUSTEMENTS
# ==========================================================
def fit_line(t, y):
    """Régression linéaire simple -> slope, intercept, R², rss."""
    n = t.size
    X = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    rss = np.sum((y - yhat) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    return coef[1], coef[0], r2, rss

def segmented_fit(t, y, t_b):
    """Continu : y = a + β1 t + (β2-β1)(t-t_b)+. Retourne coef, yhat, rss, cov."""
    hinge = np.maximum(0.0, t - t_b)
    X = np.column_stack([np.ones_like(t), t, hinge])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    rss = np.sum((y - yhat) ** 2)
    n = t.size
    cov = None
    if n > 3 and rss > 0:
        s2 = rss / (n - 3)
        try:
            cov = s2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            cov = None
    return coef, yhat, rss, cov

def find_break(t, y, n_breaks, min_pts):
    cands = np.linspace(t.min(), t.max(), n_breaks)
    best = dict(rss=np.inf)
    for tb in cands:
        if (np.count_nonzero(t < tb) < min_pts or
                np.count_nonzero(t >= tb) < min_pts):
            continue
        coef, yhat, rss, cov = segmented_fit(t, y, tb)
        if rss < best["rss"]:
            best = dict(rss=rss, tb=tb, coef=coef, yhat=yhat, cov=cov)
    return best

def bic(rss, n, k):
    return n * np.log(rss / n) + k * np.log(n)

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
    t_i0 = TIME_SCALE * time[i0]

    lo = -np.inf
    if START_AT_I0:
        lo = max(lo, t_i0)
    if T_MIN is not None:
        lo = max(lo, T_MIN)
    hi = T_MAX if T_MAX is not None else np.inf

    ok = (np.isfinite(t) & np.isfinite(Sigma) & (Sigma > 0) &
          (t >= lo) & (t <= hi) & (time <= TMAX_TA))
    t, Sigma = t[ok], Sigma[ok]
    order = np.argsort(t)
    t, Sigma = t[order], Sigma[order]
    if t.size < 2 * MIN_PTS_SEG:
        print(f"  [skip] {L_mm}mm {sand} : trop peu de points ({t.size})")
        continue

    exps.append(dict(L_mm=L_mm, sand=sand, t=t, lnS=np.log(Sigma),
                     color=style_map[(L_mm, sand)]["color"]))

print(f"Expériences analysées : {len(exps)}\n")

# ==========================================================
# ANALYSE
# ==========================================================
header = (f"{'exp':>14} | {'t_b':>6} | {'β1':>8} {'σβ1':>7} | "
          f"{'β2':>8} {'σβ2':>7} | {'R²s1':>6} {'R²s2':>6} | "
          f"{'R²seg':>6} {'R²lin':>6} | {'ΔBIC':>7}")
print(header)
print("-" * len(header))

fitted = []
for e in exps:
    t, y = e["t"], e["lnS"]
    n = t.size

    # mono-segment
    _, _, r2lin, rss1 = fit_line(t, y)

    # segmenté
    best = find_break(t, y, N_BREAKS, MIN_PTS_SEG)
    if "tb" not in best:
        print(f"  [skip] {e['L_mm']}mm {e['sand']} : rupture introuvable")
        continue
    coef, yhat, rss2, cov, tb = (best["coef"], best["yhat"],
                                 best["rss"], best["cov"], best["tb"])
    b1 = coef[1]
    b2 = coef[1] + coef[2]
    if cov is not None:
        seb1 = np.sqrt(max(cov[1, 1], 0.0))
        seb2 = np.sqrt(max(cov[1, 1] + cov[2, 2] + 2 * cov[1, 2], 0.0))
    else:
        seb1 = seb2 = np.nan

    # R² par segment (résidus du modèle segmenté, points de chaque côté)
    mleft = t < tb

    def seg_r2(mask):
        if np.count_nonzero(mask) < 2:
            return np.nan
        yy, yh = y[mask], yhat[mask]
        tss = np.sum((yy - yy.mean()) ** 2)
        return 1 - np.sum((yy - yh) ** 2) / tss if tss > 0 else np.nan

    r2s1, r2s2 = seg_r2(mleft), seg_r2(~mleft)

    tss = np.sum((y - y.mean()) ** 2)
    r2seg = 1 - rss2 / tss if tss > 0 else np.nan
    dbic = bic(rss2, n, 4) - bic(rss1, n, 2)

    e.update(tb=tb, b1=b1, b2=b2, yhat=yhat, r2seg=r2seg, r2lin=r2lin)
    fitted.append(e)

    tag = f"{e['L_mm']}mm {e['sand']}"
    print(f"{tag:>14} | {tb:6.2f} | {b1:8.4f} {seb1:7.4f} | "
          f"{b2:8.4f} {seb2:7.4f} | {r2s1:6.3f} {r2s2:6.3f} | "
          f"{r2seg:6.4f} {r2lin:6.4f} | {dbic:7.1f}")

print("\nΔBIC < 0 favorise le modèle segmenté (indicatif, cf. en-tête).")
print("Verdict réel : résidus plats par segment (Fig 2) + t_b reproductible.")

# ==========================================================
# FIGURES
# ==========================================================
def grid(n):
    ncols = min(3, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols

n = len(fitted)
if n == 0:
    raise SystemExit("Aucune expérience exploitable.")
nr, nc = grid(n)

# Fig 1 : ln Σ + fit segmenté
fig1, axes1 = plt.subplots(nr, nc, figsize=(18 / inches, 5.5 * nr / inches),
                           layout="constrained", squeeze=False)
for ax, e in zip(axes1.flat, fitted):
    ax.plot(e["t"], e["lnS"], ".", ms=2.5, color=e["color"], alpha=0.5)
    ax.plot(e["t"], e["yhat"], "-", color="k", lw=1.3)
    ax.axvline(e["tb"], color="gray", ls=":", lw=1.0)
    ax.set_title(rf"{e['L_mm']} mm {e['sand']} : "
                 rf"$\beta_1={e['b1']:.3f},\ \beta_2={e['b2']:.3f}$", fontsize=7)
    ax.set_xlabel(r"$t/T_a$")
    ax.set_ylabel(r"$\ln \Sigma$")
    ax.grid(True, ls="--", alpha=0.3)
for ax in axes1.flat[n:]:
    ax.axis("off")
fig1.savefig("/home/chorus/piecewise_fit.pdf")
plt.show()
# Fig 2 : résidus du modèle segmenté (doivent être plats)
fig2, axes2 = plt.subplots(nr, nc, figsize=(18 / inches, 5.5 * nr / inches),
                           layout="constrained", squeeze=False)
for ax, e in zip(axes2.flat, fitted):
    resid = e["lnS"] - e["yhat"]
    ax.plot(e["t"], resid, ".", ms=2.5, color=e["color"], alpha=0.6)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(e["tb"], color="gray", ls=":", lw=1.0)
    ax.set_title(f"{e['L_mm']} mm {e['sand']} — résidus", fontsize=7)
    ax.set_xlabel(r"$t/T_a$")
    ax.set_ylabel(r"$\ln\Sigma - \widehat{\ln\Sigma}$")
    ax.grid(True, ls="--", alpha=0.3)
for ax in axes2.flat[n:]:
    ax.axis("off")
fig2.savefig("/home/chorus/piecewise_residuals.pdf")

plt.show()