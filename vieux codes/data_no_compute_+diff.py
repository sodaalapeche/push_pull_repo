import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================================
# STYLE
# ==========================================================
plt.rcParams['font.family'] = 'Ubuntu'
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'

# ==========================================================
# CONFIG
# ==========================================================
DATA_PATH = "/home/chorus/testu.npy"

REQUEST = [
    (0, 'fine'),
    (0, 'coarse'),
    (10, "fine"),
    (10, "coarse"),
    (3, "coarse"),
    (6, "coarse"),
    (3, "fine"),
    (6, "fine"),

]

SIZE_COLOR_MAP = {
    10: "tab:blue",
    6: "tab:orange",
    3: "tab:red",
    0 : "tab:green",
}
FIT_TMIN = 13
FIT_TMAX = 24

D1_MM = {
    "fine": 0.1,
    "coarse": 0.6,
}

D1_M = {
    "fine": 0.1e-3,
    "coarse": 0.6e-3,
}

R = 0.055 / 2
ELL = 0.01
DM = 5e-9 # diffusion moléculaire [m^2/s]

CURVE_SUBSAMPLE = 2
DEFAULT_ERRORBAR = "std"   # "std" ou "sem"
# ==========================================================
# LOAD DELGADO DATA (WebPlotDigitizer)
# ==========================================================
delgado_data = np.loadtxt(
    "/home/chorus/Downloads/webdataset.csv",
    delimiter=",",
)

Pe_delgado = delgado_data[:, 0]
D_over_Dm_delgado = delgado_data[:, 1]
# ==========================================================
# LABELS AUTOMATIQUES
# ==========================================================
X_INFO = {
"Pe_geom": {
    "label": r"$Pe_1 \, (d_2/d_1)^{1/2}$",
    "log": True,
},

    "Pe": {
        "label": r"$Pe = 10\cdot d_2/d_1$",
        "log": True,
    },
    "d2_mm": {
        "label": r"$d_2\;(\mathrm{mm})$",
        "log": True,
    },
    "d2_m": {
        "label": r"$d_2\;(\mathrm{m})$",
        "log": True,
    },
    "beta_mean": {
        "label": r"$\beta$",
        "log": False,
    },
    "d1_mm": {
        "label": r"$d_1(mm)$",
        "log": True,
    },
}

Y_INFO = {
"ratio_micro": {
    "mean": "ratio_micro_mean",
    "std": "ratio_micro_std",
    "sem": "ratio_micro_sem",
    "label": r"$D / D_{\mathrm{micro}}$",
    "log": True,
},
    "beta": {
        "mean": "beta_mean",
        "std": "beta_std",
        "sem": "beta_sem",
        "label": r"$\beta$",
        "log": False,
    },
    "q_d1": {
        "mean": "q_d1_mean",
        "std": "q_d1_std",
        "sem": "q_d1_sem",
        "label": r"$\beta R^2 / (\ell d_1)$",
        "log": True,
    },
    "q_d1_plus_dm": {
        "mean": "q_d1_plus_dm_mean",
        "std": "q_d1_plus_dm_std",
        "sem": "q_d1_plus_dm_sem",
        "label": r"$\beta R^2 / \left(\ell (1 d_1 + D_m/u)\right)$",
        "log": True,
    },
    "q_d2": {
        "mean": "q_d2_mean",
        "std": "q_d2_std",
        "sem": "q_d2_sem",
        "label": r"$\gamma R^2 / (2 \beta_0^2 \ell d_2)$",
        "log": True,
    },
    "q_d1d2": {
        "mean": "q_d1d2_mean",
        "std": "q_d1d2_std",
        "sem": "q_d1d2_sem",
        "label": r"$\beta /(d_1**-0.4 d_2**0.4)$",
        "log": True,
    },
    "beta_Pe": {
        "mean": "beta_Pe_mean",
        "std": "beta_Pe_std",
        "sem": "beta_Pe_sem",
        "label": r"$\beta/d_2 \cdot Pe_{\mathrm{Darcy}}$",
        "log": True,
    },
    "beta_d2m02": {
        "mean": "beta_d2m02_mean",
        "std": "beta_d2m02_std",
        "sem": "beta_d2m02_sem",
        "label": r"$\beta\, d_2^{-0.2}$",
        "log": True,
    },
}

# ==========================================================
# HELPERS DONNÉES
# ==========================================================
def parse_label(label_text):
    lines = [l.strip().lower() for l in str(label_text).split("\n") if l.strip()]
    if len(lines) < 3:
        return None, None

    try:
        d2_mm = int(lines[1].replace("mm", ""))
    except Exception:
        return None, None

    sand = lines[2]
    return d2_mm, sand


def sigma_from_result(res):
    time = np.asarray(res["time"], dtype=float)
    mean = np.asarray(res["mean"], dtype=float)
    var = np.asarray(res["var"], dtype=float)

    sigma = var / (mean ** 2)
    smax = np.nanmax(sigma)
    if np.isfinite(smax) and smax > 0:
        sigma = sigma / smax

    valid = np.isfinite(time) & np.isfinite(sigma) & (time > 0) & (sigma > 0)
    return time, sigma, valid


def compute_peclet(d2_mm, sand):
    d1_mm = D1_MM.get(sand)
    if d1_mm is None:
        return np.nan
    return 10 * float(d2_mm) / float(d1_mm)

def D_micro(Pe, Dm):
    return Dm * (1 + 0.01 * Pe)
def compute_loglog_slope(time, sigma, tmin, tmax):
    m = np.isfinite(time) & np.isfinite(sigma) & (time > 0) & (sigma > 0)
    m &= (time >= tmin) & (time <= tmax)

    if np.count_nonzero(m) < 2:
        return np.nan

    x = time[m]
    y = np.log(sigma[m])

    if np.unique(x).size < 2:
        return np.nan

    slope, _ = np.polyfit(x, y, 1)
    return -slope


def build_selected_results(results, request):
    request_set = {(int(d2), str(sand).lower()) for d2, sand in request}
    selected = []

    for res in results:
        d2_mm, sand = parse_label(res.get("label", ""))
        if d2_mm is None:
            continue
        if (d2_mm, sand) in request_set:
            selected.append(((d2_mm, sand), res))

    return selected


def build_summary(selected_results, tmin, tmax):
    grouped = {}

    for (d2_mm, sand), res in selected_results:
        time, sigma, valid = sigma_from_result(res)
        if not np.any(valid):
            continue

        beta = compute_loglog_slope(time[valid], sigma[valid], tmin, tmax)
        if not np.isfinite(beta):
            continue

        grouped.setdefault((d2_mm, sand), []).append(beta)

    from scipy.special import jn_zeros

    summary = []
    for (d2_mm, sand), betas in grouped.items():
        betas = np.asarray(betas, dtype=float)

        d1_m = D1_M.get(sand, np.nan)
        d1_mm = D1_MM.get(sand, np.nan)
        d2_m = d2_mm * 1e-3
        pe = compute_peclet(d2_mm, sand)

        Ta = res.get("Ta", np.nan)
        u = 0.01 / Ta if np.isfinite(Ta) and Ta > 0 else np.nan
        factor_d1d2 = 1.0 / (d1_m**-0.4 * d2_m**0.5) if np.isfinite(d1_m) and d1_m > 0 and np.isfinite(d2_m) and d2_m > 0 else np.nan
        q_d1d2 = betas * factor_d1d2
        Pe_1 = u * d1_m / DM if np.isfinite(u) else np.nan

        # variable combinée Pe_1 * (d2/d1)**1/2
        Pe_geom = Pe_1 * np.sqrt(d2_m / d1_m) if np.isfinite(Pe_1) else np.nan
        j0_first_zero = jn_zeros(0, 1)[0]

        factor_d1 = R**2 / (ELL * d1_m) if np.isfinite(d1_m) and d1_m > 0 else np.nan
        factor_d2 = R**2 / (ELL * d2_m) if np.isfinite(d2_m) and d2_m > 0 else np.nan

        d1_eff = d1_m + DM / u
        # d1_eff = DM / u
        factor_d1_plus_dm = R**2 / (ELL*d1_eff) if np.isfinite(d1_eff) and d1_eff > 0 else np.nan

        q_d1 = betas * factor_d1
        q_d1_plus_dm = betas * factor_d1_plus_dm
        q_d2 = betas * factor_d2
        beta_Pe = betas * pe / d2_mm
        Dm = 5e-9
        from scipy.special import jn_zeros

        b0 = jn_zeros(0, 1)[0]  # ≈ 2.4048
        pe_p = u * d1_m/Dm
        D_total = betas * (u * R ** 2) / (2 * b0 ** 2 * ELL)
        D_micro = Dm * (1+ 0.025 * pe_p)

        ratio_micro = D_total / D_micro
        summary.append({
            "beta_Pe": beta_Pe,
            "beta_Pe_mean": np.mean(beta_Pe),
            "beta_Pe_std": np.std(beta_Pe, ddof=1) if len(beta_Pe) > 1 else 0.0,
            "beta_Pe_sem": np.std(beta_Pe, ddof=1) / np.sqrt(len(beta_Pe)) if len(beta_Pe) > 1 else 0.0,
            "ratio_micro_mean": np.mean(ratio_micro),
            "ratio_micro_std": np.std(ratio_micro, ddof=1),
            "ratio_micro_std": np.std(ratio_micro, ddof=1),
            "ratio_micro_sem": np.std(ratio_micro, ddof=1) / np.sqrt(len(ratio_micro)),
            "d2_mm": d2_mm,
            "d2_m": d2_m,
            "sand": sand,
            "d1_mm": d1_mm,
            "d1_m": d1_m,
            "Pe": pe,
            "n_exp": len(betas),
            "Pe_1": Pe_1,
            "Pe_geom": Pe_geom,
            "q_d1d2_mean": np.mean(q_d1d2),
            "q_d1d2_std": np.std(q_d1d2, ddof=1) if len(q_d1d2) > 1 else 0.0,
            "q_d1d2_sem": np.std(q_d1d2, ddof=1) / np.sqrt(len(q_d1d2)) if len(q_d1d2) > 1 else 0.0,

            "beta_mean": np.mean(betas),
            "beta_std": np.std(betas, ddof=1) if len(betas) > 1 else 0.0,
            "beta_sem": np.std(betas, ddof=1) / np.sqrt(len(betas)) if len(betas) > 1 else 0.0,


            "q_d1_mean": np.mean(q_d1),
            "q_d1_std": np.std(q_d1, ddof=1) if len(q_d1) > 1 else 0.0,
            "q_d1_sem": np.std(q_d1, ddof=1) / np.sqrt(len(q_d1)) if len(q_d1) > 1 else 0.0,

            "q_d1_plus_dm_mean": np.mean(q_d1_plus_dm),
            "q_d1_plus_dm_std": np.std(q_d1_plus_dm, ddof=1) if len(q_d1_plus_dm) > 1 else 0.0,
            "q_d1_plus_dm_sem": np.std(q_d1_plus_dm, ddof=1) / np.sqrt(len(q_d1_plus_dm)) if len(q_d1_plus_dm) > 1 else 0.0,

            "q_d2_mean": np.mean(q_d2),
            "q_d2_std": np.std(q_d2, ddof=1) if len(q_d2) > 1 else 0.0,
            "q_d2_sem": np.std(q_d2, ddof=1) / np.sqrt(len(q_d2)) if len(q_d2) > 1 else 0.0,
        })

    return sorted(summary, key=lambda r: (r["d2_mm"], r["sand"]))

# ==========================================================
# HELPERS STYLE / LÉGENDE
# ==========================================================
def make_style(d2_mm, sand):
    color = SIZE_COLOR_MAP.get(d2_mm, "k")

    if sand == "fine":
        return dict(marker="o", color=color, mfc=color, mec=color, mew=1.0, ms=5, alpha=0.9)
    if sand == "coarse":
        return dict(marker="D", color=color, mfc="none", mec=color, mew=1.2, ms=6, alpha=0.9)

    return dict(marker="o", color="k", mfc="k", mec="k", mew=1.0, ms=5, alpha=0.9)


def legend_from_request(request):
    handles = []
    done = set()

    for d2_mm, sand in request:
        key = (int(d2_mm), str(sand).lower())
        if key in done:
            continue
        done.add(key)
        style = make_style(*key)
        handles.append(
            Line2D(
                [0], [0],
                marker=style["marker"],
                linestyle="None",
                color=style["color"],
                markerfacecolor=style["mfc"],
                markeredgecolor=style["mec"],
                markeredgewidth=style["mew"],
                markersize=style["ms"],
                alpha=style["alpha"],
                label=f"{d2_mm} mm, {sand}"
            )
        )
    return handles

# ==========================================================
# PLOT DE COURBES
# ==========================================================
def plot_sigma_curves(selected_results, request, tmin, tmax, subsample=4):
    fig, ax = plt.subplots(figsize=(7, 5))

    for (d2_mm, sand), res in selected_results:
        time, sigma, valid = sigma_from_result(res)
        if np.any(valid):
            style = make_style(d2_mm, sand)
            ax.plot(time[valid][::subsample], sigma[valid][::subsample],linestyle="None",**style)

    # ax.axvspan(tmin, tmax, color="0.85", alpha=0.5, zorder=0, label="fit window")
    ax.legend(handles=legend_from_request(request))
    ax.set_xlabel(r"$t/T_a$")
    ax.set_ylabel(r"$\sigma_c^2(t)/\mu_c^2$")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlim(-0.1, 32)
    ax.set_ylim(0.004, 1.1)
    fig.tight_layout()
    return fig, ax
def plot_variance_ratio_mean_fine(selected_results, n_grid=200):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7,5))

    # ======================================================
    # 1. regrouper les courbes
    # ======================================================
    groups = {}

    for (d2_mm, sand), res in selected_results:
        if sand != "coarse":
            continue

        time, sigma, valid = sigma_from_result(res)
        if not np.any(valid):
            continue

        t = time[valid]
        s = sigma[valid]

        # 🔥 NORMALISATION PAR LE MAX (clé)
        s_max = np.nanmax(s)
        if s_max <= 0:
            continue

        s_norm = s / s_max

        groups.setdefault(d2_mm, []).append((t, s_norm))

    # ======================================================
    # 2. grille commune
    # ======================================================
    t_min = np.inf
    t_max = 0

    for curves in groups.values():
        for t, _ in curves:
            t_min = min(t_min, np.nanmin(t))
            t_max = max(t_max, np.nanmax(t))

    t_grid = np.linspace(t_min, t_max, n_grid)

    # ======================================================
    # 3. interpolation + moyenne
    # ======================================================
    mean_curves = {}

    for d2_mm, curves in groups.items():
        all_interp = []

        for t, s in curves:
            s_interp = np.interp(t_grid, t, s, left=np.nan, right=np.nan)
            all_interp.append(s_interp)

        all_interp = np.array(all_interp)

        mean_curve = np.nanmean(all_interp, axis=0)
        mean_curves[d2_mm] = mean_curve

    # ======================================================
    # 4. référence 0 mm
    # ======================================================
    if 0 not in mean_curves:
        raise RuntimeError("Pas de 0 mm fine")

    sigma_ref = mean_curves[0]

    # ======================================================
    # 5. ratio
    # ======================================================
    for d2_mm, sigma_mean in mean_curves.items():
        if d2_mm == 0:
            continue

        ratio = sigma_ref/sigma_mean

        mask = np.isfinite(ratio) & (ratio > 0)

        style = make_style(d2_mm, "coarse")

        ax.plot(
            t_grid[mask],
            ratio[mask],
            linestyle="None",
            marker=style["marker"],
            color=style["color"],
            mfc=style["mfc"],
            mec=style["mec"],
            ms=style["ms"],
            label=f"{d2_mm} mm"
        )

    # ======================================================
    # STYLE
    # ======================================================
    ax.axhline(1, color="k", linestyle="--", alpha=0.5)

    ax.set_xlabel(r"$t/T_a$")
    ax.set_ylabel(r"$\langle \sigma^2/\max \rangle \;/\; \langle \sigma^2_{0}/\max \rangle$")

    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    ax.legend(title="d2 (fine)")

    fig.tight_layout()
    return fig, ax
# ==========================================================
# FONCTIONS ULTRA-SIMPLES POUR PLOTS RÉSUMÉS
# ==========================================================
def add_reference_slope(ax, slope, x0, y0, label=None, n=200, **kwargs):
    xlim = ax.get_xlim()
    xmin, xmax = xlim

    if xmin <= 0 or xmax <= 0:
        return

    xline = np.logspace(np.log10(xmin), np.log10(xmax), n)
    A = y0 / (x0 ** slope)
    yline = A * xline ** slope

    if label is None:
        label = f"slope = {slope}"

    ax.plot(xline, yline, label=label, **kwargs)


def make_plot(
    summary,
    x="Pe",
    y="q_d1",
    err=DEFAULT_ERRORBAR,
    ref_slope=None,
    ref_x0=1,
    ref_y0=1,
    ref_kwargs=None,
    show_legend=True,
    figsize=(7, 5),
):
    fig, ax = plt.subplots(figsize=figsize)

    xinfo = X_INFO[x]
    yinfo = Y_INFO[y]

    ymean_key = yinfo["mean"]
    yerr_key = yinfo[err]

    for row in summary:
        style = make_style(row["d2_mm"], row["sand"])
        ax.errorbar(
            row[x],
            row[ymean_key],
            yerr=row[yerr_key],
            fmt=style["marker"],
            color=style["color"],
            mfc=style["mfc"],
            mec=style["mec"],
            mew=style["mew"],
            ms=style["ms"],
            capsize=4,
            linestyle="None",
            alpha=style["alpha"],
        )

    if xinfo["log"]:
        ax.set_xscale("log")
    if yinfo["log"]:
        ax.set_yscale("log")

    ax.set_xlabel(xinfo["label"])
    ax.set_ylabel(yinfo["label"])
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if ref_slope is not None:
        if ref_kwargs is None:
            ref_kwargs = dict(color="black", linestyle="-", linewidth=2)
        add_reference_slope(
            ax,
            slope=ref_slope,
            x0=ref_x0,
            y0=ref_y0,
            **ref_kwargs
        )

    if show_legend:
        ax.legend()

    fig.tight_layout()
    return fig, ax

# ==========================================================
# LOAD + PREP
# ==========================================================
results = np.load(DATA_PATH, allow_pickle=True)
selected_results = build_selected_results(results, REQUEST)

if len(selected_results) == 0:
    raise RuntimeError("No experiment in data2.npy matches REQUEST.")

summary = build_summary(selected_results, FIT_TMIN, FIT_TMAX)
# summary = [row for row in summary if row["d2_mm"] != 3]
print(f"Experiments loaded   : {len(results)}")
print(f"Experiments selected : {len(selected_results)}")

print("\n=== Summary ===")
for row in summary:
    print(
        f"d2={row['d2_mm']:>2} mm | sand={row['sand']:<6} | "
        f"Pe={row['Pe']:>6.2f} | n={row['n_exp']} | "
        f"beta={row['beta_mean']:.4f} | "
        f"q_d1={row['q_d1_mean']:.4f} | "
        f"q_d1+Dm={row['q_d1_plus_dm_mean']:.4f} | "
        f"q_d2={row['q_d2_mean']:.4f}"
    )

# ==========================================================
# EXEMPLES D'UTILISATION
# ==========================================================
plot_sigma_curves(
    selected_results,
    request=REQUEST,
    tmin=FIT_TMIN,
    tmax=FIT_TMAX,
    subsample=CURVE_SUBSAMPLE,
)
plt.show()

# plot 1 : comme avant
# make_plot(summary, x="Pe", y="betas", err="std")
# plt.show()

# fig, ax = make_plot(
#     summary,
#     x="Pe",
#     y="ratio_micro",
#     err="std",
# )

fig, ax = make_plot(
    summary,
    x="Pe",
    y="q_d2",
    err="std",
)
# ==========================================================
# AJOUT FIT LOG-LOG
# ==========================================================
# extraction des données
# x = np.array([row["Pe"] for row in summary])
x = np.array([row["Pe"] for row in summary])
y = np.array([row["q_d2_mean"] for row in summary])

# filtre log
mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

# fit log-log : log(y) = a log(x) + b
logx = np.log10(x)
logy = np.log10(y)

a, b = np.polyfit(logx, logy, 1)
# résidus
y_pred = a * logx + b
residuals = logy - y_pred

# variance résiduelle
N = len(logx)
s2 = np.sum(residuals**2) / (N - 2)

# variance de la pente
Sxx = np.sum((logx - np.mean(logx))**2)

sigma_a = np.sqrt(s2 / Sxx)
# droite de fit
xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
yfit = 10**b * xfit**a

# ax.plot(xfit, yfit, linestyle="--")

# coefficient de corrélation
R = np.corrcoef(logx, logy)[0, 1]

# affichage équation
eq_text = (
    rf"$y = {10**b:.2e} \cdot x^{{{a:.2f} \pm {sigma_a:.2f}}}$"
    + "\n"
    + rf"$R = {R:.3f}$"
)
ax.text(
    0.05, 0.95,
    eq_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)
# ==========================================================
# OVERLAY DELGADO
# ==========================================================
# ax.scatter(
#     Pe_delgado,
#     D_over_Dm_delgado,
#     marker="s",
#     facecolors="none",
#     edgecolors="black",
#     s=40,
#     label="Delgado (2007)"
# )


plt.yscale("log")



plt.show()



fig, ax = plot_variance_ratio_mean_fine(selected_results)
ax.set_xlim(left=0,right=30)
plt.show()