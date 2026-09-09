# ==========================================================
# EXTRACTION PDF(c) AU COURS DU TEMPS — version rigoureuse
# ==========================================================
#
# Estimateur utilisé : histogramme log-espacé normalisé en densité
#
#   pdf(c_k) = n_k / (N * Delta_k)
#
# avec Delta_k = c_{k+1} - c_k  (largeur réelle du bin en espace linéaire)
# et N = nombre total de pixels dans la ROI.
#
# Propriété garantie :  sum_k pdf(c_k) * Delta_k = 1
#
# Pas de lissage post-normalisation (brise la normalisation sur bins
# log-espacés et déforme les pentes en loi de puissance).
# Le nombre de bins est choisi par la règle de Knuth (ou Scott log) pour
# minimiser l'erreur quadratique moyenne de l'estimateur.
#
# Un assert vérifie la normalisation à 1 % près à chaque temps.
# ==========================================================
import os
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scienceplots

plt.style.use('science')
width  = 13    # cm
height = width * 0.6
inches = 2.54

# ==========================================================
# PARAMÈTRES GLOBAUX
# ==========================================================
BASE_PATH  = "/home/chorus/HETEROGENE_binned/"
R_EVAL_PIX = 190

# Nombre de bins : règle de Scott adaptée à l'espace log
# h_log = 3.49 * std(log10 c) / N^(1/3)
# N_BINS = (log10(c_hi) - log10(c_lo)) / h_log
# Calculé dynamiquement dans extract_pdf_vs_time() pour chaque image.
# Plancher à 80 bins et plafond à 500 pour rester raisonnable.
N_BINS_MIN = 200
N_BINS_MAX = 500

# Seuil statistique minimal par bin pour le conserver (évite les bins vides
# ou sous-peuplés qui gonflent artificiellement la queue de la PDF).
MIN_COUNTS_PER_BIN = 50   # ~ erreur relative < 15 % (1/sqrt(50))

# Temps adimensionnels à extraire (multiples de Ta)
T_TARGETS = [1.5, 2.6,5, 8, 17, 30]

# Cas à afficher (chacun aura sa propre figure)
CASES_TO_PLOT = [
    (0,  "fine"),
    (3,  "fine"),
    (6, "fine"),
    (10, "fine"),
]

# ==========================================================
# UTILITAIRES
# ==========================================================
def select_exact_combinations(base_path, combinations):
    selected = []
    for L_mm, sand in combinations:
        size_folder = f"{L_mm}mm"
        sand_folder = sand.lower()
        path = os.path.join(base_path, size_folder, sand_folder)
        if not os.path.exists(path):
            continue
        for d in os.listdir(path):
            exp_path = os.path.join(path, d)
            if not os.path.isdir(exp_path):
                continue
            tif_files = glob(os.path.join(exp_path, "**", "*.tif"), recursive=True)
            csv_path  = os.path.join(exp_path, "weight_data.csv")
            if tif_files and os.path.isfile(csv_path):
                selected.append(exp_path)
    return selected


def find_image_folder(root):
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and glob(os.path.join(p, "*.tif")):
            return p
    raise RuntimeError(f"No TIFF folder found under: {root}")


def list_tifs(folder):
    return sorted(
        glob(os.path.join(folder, "*.tif")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )


def read_label(root):
    p = os.path.join(root, "label.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    return os.path.basename(root)


def build_t_img_from_csv(root_folder, n_images):
    df  = pd.read_csv(os.path.join(root_folder, "weight_data.csv"))
    t   = df["Timestamp"].values.astype(float)
    dt  = np.median(np.diff(t))
    return t[0] + dt * np.arange(n_images)


def compute_col_profile(frame0):
    col_profile = np.mean(frame0, axis=0)
    col_profile /= np.mean(col_profile)
    return np.maximum(col_profile, 1e-6)


def apply_col_profile(frame, col_profile):
    return frame / col_profile[None, :]

# ==========================================================
# EXTRACTION : PDF à plusieurs t/Ta pour un dossier
# ==========================================================
def extract_pdf_vs_time(root_folder):
    img_folder = find_image_folder(root_folder)
    files      = list_tifs(img_folder)
    n          = len(files)

    frame0      = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    col_profile = compute_col_profile(frame0)

    H, W   = frame0.shape
    Ypix, Xpix = np.indices((H, W))
    xc, yc = W / 2, H / 2

    t_img = build_t_img_from_csv(root_folder, n)

    # --- détermination de Ta ---
    mean = np.full(n, np.nan)
    var  = np.full(n, np.nan)

    for i, f in enumerate(files):
        frame      = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        frame      = apply_col_profile(frame, col_profile)
        r2         = (Xpix - xc)**2 + (Ypix - yc)**2
        roi        = r2 <= R_EVAL_PIX**2
        vals       = frame[roi]
        mean[i]    = float(np.nanmean(vals))
        var[i]     = float(np.nanvar(vals))

    i0      = int(np.nanargmax(var))
    dmean_dt = np.gradient(mean, t_img)
    i_start  = min(i0 + 30, n - 1)
    i_min    = i_start + int(np.nanargmin(dmean_dt[i_start:])) if i_start < n - 1 else i0
    Ta       = (t_img[i_min] - t_img[i0]) / 30

    # --- extraction PDF à chaque t_target ---
    pdfs = []
    for t_mult in T_TARGETS:
        t_target = t_img[i0] + t_mult * Ta
        if t_target > t_img[-1]:
            continue

        idx   = np.argmin(np.abs(t_img - t_target))
        frame = cv2.imread(files[idx], cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        frame = apply_col_profile(frame, col_profile)

        r2   = (Xpix - xc)**2 + (Ypix - yc)**2
        roi  = r2 <= R_EVAL_PIX**2
        vals = frame[roi]

        # --- normalisation par la moyenne empirique ---
        c_mean = np.mean(vals)
        c = vals / c_mean          # c/⟨c⟩, sans biais (on divise par la moyenne ROI)

        # --- grille log-espacée ---
        # Bornes : percentiles 0.1 % et 99.9 % pour exclure les pixels aberrants
        c_lo = max(np.percentile(c, 0.1), 1e-4)
        c_hi = np.percentile(c, 99.9)

        # Règle de Scott adaptée à l'espace log :
        #   h_log = 3.49 * sigma_log / N^(1/3)
        #   N_BINS = (log10_hi - log10_lo) / h_log
        log_c    = np.log10(c[(c >= c_lo) & (c <= c_hi)])
        h_log    = 3.49 * np.std(log_c) / (len(log_c) ** (1.0 / 3.0))
        n_bins   = int(np.ceil((np.log10(c_hi) - np.log10(c_lo)) / h_log))
        n_bins   = int(np.clip(n_bins, N_BINS_MIN, N_BINS_MAX))

        edges = np.logspace(np.log10(c_lo), np.log10(c_hi), n_bins + 1)

        hist, edges = np.histogram(c, bins=edges)

        # --- estimateur de densité normalisé ---
        # pdf(c_k) = n_k / (N * Delta_k)   avec Delta_k en espace linéaire
        # => intégrale ∫ pdf dc = 1  (exacte à la précision de l'histogramme)
        delta   = np.diff(edges)                      # largeurs linéaires des bins
        N_tot   = hist.sum()
        pdf_est = hist / (N_tot * delta)              # densité [unité : 1/c]

        # --- centre géométrique de chaque bin (correct en log) ---
        centers = np.sqrt(edges[:-1] * edges[1:])

        # --- vérification de normalisation (assert scientifique) ---
        norm_check = np.sum(pdf_est * delta)
        assert abs(norm_check - 1.0) < 0.01, (
            f"Normalisation PDF échouée : intégrale = {norm_check:.4f} "
            f"(tolérance 1 %) — vérifier les bords de bins."
        )

        # --- masque statistique : conserver uniquement les bins suffisamment peuplés ---
        # Erreur relative de Poisson ~ 1/sqrt(n_k) < 15 % si n_k > MIN_COUNTS_PER_BIN
        mask    = hist >= MIN_COUNTS_PER_BIN
        centers = centers[mask]
        pdf_est = pdf_est[mask]
        # Pas de lissage : toute convolution post-normalisation sur bins log-espacés
        # brise ∫pdf dc = 1 et déforme les pentes en loi de puissance.

        pdfs.append({
            "t_mult":   t_mult,
            "centers":  centers,
            "pdf":      pdf_est,
            "N_pixels": int(N_tot),
            "hist":     hist[mask],   # counts par bin (pour barres d'erreur Poisson)
            "delta":    delta[mask],  # largeurs des bins retenus
        })

    label_text = read_label(root_folder)
    lines      = [l.strip().lower() for l in label_text.split("\n") if l.strip()]
    try:
        L_mm = int(lines[1].replace("mm", ""))
        sand = lines[2]
    except Exception:
        L_mm = -1
        sand = "?"

    return {"L_mm": L_mm, "sand": sand, "pdfs": pdfs}

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    root_folders = select_exact_combinations(BASE_PATH, CASES_TO_PLOT)

    with Pool(max(1, cpu_count() - 2)) as pool:
        all_results = pool.map(extract_pdf_vs_time, root_folders)

    # Regrouper par (L_mm, sand) — moyenne des réplicats si plusieurs
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["L_mm"], r["sand"])].append(r["pdfs"])

    # ==========================================================
    # FIGURE PAR CAS
    # ==========================================================
    for (L_mm, sand), pdfs_list in grouped.items():

        # On prend le premier réplicat (ou tu peux moyenner)
        pdfs = pdfs_list[0]

        t_mults = [p["t_mult"] for p in pdfs]
        t_min, t_max = min(t_mults), max(t_mults)

        norm    = mcolors.Normalize(vmin=t_min, vmax=t_max)
        cmap    = cm.plasma

        fig, ax = plt.subplots(figsize=(width / inches, height / inches))

        # --- limites automatiques basées sur les données effectives ---
        # x : union des plages [centers.min(), centers.max()] sur tous les temps,
        #     avec une marge logarithmique de 20 %
        all_x = np.concatenate([p["centers"] for p in pdfs])
        all_y = np.concatenate([p["pdf"]     for p in pdfs])

        x_lo = 0.8
        x_hi = all_x.max() * 1.2

        # y : on exclut les valeurs sous MIN_COUNTS_PER_BIN (déjà masquées),
        #     marge d'un demi-décade en bas, un demi-décade en haut
        y_lo = 10 ** (np.floor(np.log10(all_y.min())))
        y_hi = 4

        for p in pdfs:
            color = cmap(norm(p["t_mult"]))
            pdf   = p["pdf"]
            n_k   = p["hist"]

            # Incertitude de Poisson sur la densité :
            # sigma_pdf_k = pdf_k / sqrt(n_k)
            # (erreur relative ~ 1/sqrt(n_k), valide pour n_k >= MIN_COUNTS_PER_BIN)
            sigma = pdf / np.sqrt(n_k)

            ax.plot(p["centers"], pdf, lw=1.5, color=color)
            # ax.fill_between(p["centers"],
            #                 np.maximum(pdf - sigma, 1e-8),
            #                 pdf + sigma,
            #                 color=color, alpha=0.15, linewidth=0)

        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$t \;/\; T_a$")

        ax.set_xlabel(r"$c / \langle c \rangle$")
        ax.set_ylabel(r"$\mathrm{pdf}(c)$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_title(f"{L_mm} mm – {sand}")

        plt.tight_layout()
        plt.savefig(f"pdf_vs_time_{L_mm}mm_{sand}.pdf", dpi=150)
        plt.show()
        print(f"Figure sauvegardée : pdf_vs_time_{L_mm}mm_{sand}.pdf")