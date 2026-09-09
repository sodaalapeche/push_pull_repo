"""
Crossover macrodispersion -> confinement, via la pente log-log LOCALE.

    α_loc(t) = − d(ln Σ) / d ln(t − t0)

Signatures :
  - loi de puissance (macro, plume libre)  -> α_loc PLAT  (= exposant α0)
  - exponentielle (confiné, mode propre)   -> α_loc RAMPE linéairement
                                               en (t − t0), de pente β

Hypothèse testée : α_loc plat à α0 aux temps courts (le traceur n'a pas vu la
paroi), puis rampe une fois le bord atteint. Le coude = temps de confinement t*.

On ajuste α_loc(t) par un modèle CONTINU "plat-puis-rampe" :
    α_loc(t) = α0 + s · (t − t*)_+
-> α0 = exposant macro, t* = entrée en confinement, s = pente de rampe (≈ β).
Cross-check : β estimé directement en semilog sur t > t* doit ≈ s.

NB : α_loc est une grandeur dérivée/lissée -> l'ajustement plat-puis-rampe est
INDICATIF (lecture du coude), pas un test statistique rigoureux.
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

T0           = 0.0     # origine du temps pour ln(t − t0)
AUTO_T0      = False    # True : balaie t0 (borné) en maximisant la linéarité log-log globale
T0_SCAN_SPAN = 5.0
T0_SCAN_N    = 80

HALF_WIDTH   = 2.0     # demi-fenêtre de régression locale (en t/Ta) pour α_loc
MIN_PTS      = 5       # nb min de points dans la fenêtre locale
MIN_PTS_SEG  = 5       # nb min de points de chaque côté de t* (fit plat-puis-rampe)
N_TSTAR      = 100     # positions de coude testées

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
def best_T0(t, y, span, n):
    tmin = t.min()
    cands = np.linspace(tmin - span, tmin - 1e-3, n)
    best = dict(r2=-np.inf, t0=0.0)
    for t0 in cands:
        u = np.log(t - t0)
        c = np.polyfit(u, y, 1)
        yhat = np.polyval(c, u)
        ss = np.sum((y - yhat) ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss / tss if tss > 0 else -np.inf
        if r2 > best["r2"]:
            best = dict(r2=r2, t0=t0)
    return best["t0"]

def alpha_loc(t, lnS, t0, half_width, min_pts):
    """Pente log-log locale : régression de ln Σ vs ln(t−t0) sur [t−h, t+h]."""
    u = np.log(t - t0)
    a = np.full(t.size, np.nan)
    for i in range(t.size):
        m = (t >= t[i] - half_width) & (t <= t[i] + half_width)
        if np.count_nonzero(m) >= min_pts:
            a[i] = -np.polyfit(u[m], lnS[m], 1)[0]
    return a

def flat_then_ramp(t, a, n_tstar, min_pts):
    """Ajuste α_loc = α0 + s·(t−t*)_+ ; balayage de t*. Retourne α0, s, t*."""
    good = np.isfinite(a)
    tt, aa = t[good], a[good]
    if tt.size < 2 * min_pts:
        return np.nan, np.nan, np.nan
    cands = np.linspace(tt.min(), tt.max(), n_tstar)
    best = dict(rss=np.inf, a0=np.nan, s=np.nan, tstar=np.nan)
    for ts in cands:
        if (np.count_nonzero(tt < ts) < min_pts or
                np.count_nonzero(tt >= ts) < min_pts):
            continue
        X = np.column_stack([np.ones_like(tt), np.maximum(0.0, tt - ts)])
        coef, *_ = np.linalg.lstsq(X, aa, rcond=None)
        rss = np.sum((aa - X @ coef) ** 2)
        if rss < best["rss"]:
            best = dict(rss=rss, a0=coef[0], s=coef[1], tstar=ts)
    return best["a0"], best["s"], best["tstar"]

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
    if t.size < 2 * MIN_PTS:
        continue
    exps.append(dict(L_mm=L_mm, sand=sand, t=t, lnS=np.log(Sigma),
                     color=style_map[(L_mm, sand)]))

print(f"Expériences analysées : {len(exps)}\n")

# ==========================================================
# ANALYSE
# ==========================================================
header = (f"{'exp':>14} | {'α0 (macro)':>10} | {'t* (conf.)':>10} | "
          f"{'s (rampe)':>10} | {'β (t>t*)':>10}")
print(header)
print("-" * len(header))

fitted = []
for e in exps:
    t, y = e["t"], e["lnS"]
    t0 = best_T0(t, y, T0_SCAN_SPAN, T0_SCAN_N) if AUTO_T0 else min(T0, t.min() - 1e-3)

    a = alpha_loc(t, y, t0, HALF_WIDTH, MIN_PTS)
    a0, s, tstar = flat_then_ramp(t, a, N_TSTAR, MIN_PTS_SEG)

    # cross-check : β semilog direct sur t > t*
    beta_late = np.nan
    if np.isfinite(tstar):
        late = t > tstar
        if np.count_nonzero(late) >= 3:
            beta_late = np.polyfit(t[late], y[late], 1)[0]

    e.update(t0=t0, a=a, a0=a0, s=s, tstar=tstar, beta_late=beta_late)
    fitted.append(e)

    tag = f"{e['L_mm']}mm {e['sand']}"
    print(f"{tag:>14} | {a0:10.3f} | {tstar:10.2f} | {s:10.4f} | {beta_late:10.4f}")

print("\nLecture :")
print("  - α0      : exposant de la loi de puissance (régime macro, plateau plat).")
print("  - t*      : entrée en confinement (coude où α_loc se met à ramper).")
print("  - s       : pente de la rampe ; pour un confinement exponentiel, s ≈ |β|.")
print("  - β (t>t*): β semilog direct aux temps longs — doit ≈ s si la rampe = expo.")
print("  Hypothèse macro→confiné OK si : α0 ~ cste plausible, coude net, s ≈ |β|.")
print("  Si α_loc rampe DÈS le début (pas de plateau) -> exponentiel partout, pas de macro.")

# ==========================================================
# FIGURE : α_loc(t) + ajustement plat-puis-rampe
# ==========================================================
def grid(n):
    nc = min(3, max(n, 1))
    return int(np.ceil(n / nc)), nc

n = len(fitted)
if n == 0:
    raise SystemExit("Aucune expérience exploitable.")
nr, nc = grid(n)

fig, axes = plt.subplots(nr, nc, figsize=(18 / inches, 5.5 * nr / inches),
                         layout="constrained", squeeze=False)
for ax, e in zip(axes.flat, fitted):
    t, a = e["t"], e["a"]
    ax.plot(t, a, ".", ms=2.5, color=e["color"], alpha=0.55)

    # modèle plat-puis-rampe
    if np.isfinite(e["tstar"]):
        tl = np.linspace(t.min(), t.max(), 200)
        model = e["a0"] + e["s"] * np.maximum(0.0, tl - e["tstar"])
        ax.plot(tl, model, "-", color="k", lw=1.2)
        ax.axvline(e["tstar"], color="gray", ls=":", lw=1.0)

    for guide in (1.0, 2.0):
        ax.axhline(guide, color="gray", ls="--", lw=0.5, alpha=0.4)

    finite = a[np.isfinite(a)]
    if finite.size:
        ax.set_ylim(min(0.0, finite.min() - 0.2),
                    min(4.0, np.nanpercentile(finite, 98) + 0.4))
    ax.set_title(rf"{e['L_mm']} mm {e['sand']} : "
                 rf"$\alpha_0={e['a0']:.2f},\ t^*={e['tstar']:.1f}$", fontsize=7)
    ax.set_xlabel(r"$t/T_a$")
    ax.set_ylabel(r"$\alpha_\mathrm{loc}=-\,\mathrm{d}\ln\Sigma/\mathrm{d}\ln(t-t_0)$",
                  fontsize=6)
    ax.grid(True, ls="--", alpha=0.3)
for ax in axes.flat[n:]:
    ax.axis("off")

fig.savefig("/home/chorus/alpha_loc_crossover.pdf")
plt.show()