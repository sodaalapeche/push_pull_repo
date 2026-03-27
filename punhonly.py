import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Config ---
DOSSIER = "/media/chorus/T7/inj simple/exp_09_09_1erpoint/serie1/"  # chemin du dossier d'images
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SORT_BY = "name"  # "name" (par nom de fichier) ou "ratio" (variance/moyenne)
ANNOTATE_VALUES = False  # Passe à True pour annoter chaque point avec la valeur num.

# --- Fonctions utilitaires ---
def load_gray16_to_float(path: str) -> np.ndarray:
    """Charge une image (16 bits niveaux de gris si dispo) et renvoie un array float64.
    Si l'image est RGB, on moyenne les canaux pour obtenir du gris.
    """
    with Image.open(path) as im:
        # Convertit en mode qui préserve la dynamique 16 bits si possible
        if im.mode in ("I;16", "I;16L", "I;16B", "I"):
            arr = np.array(im)
        else:
            # Si l'image est RGB/RGBA, convertit en L (8 bits) puis étend dyna.
            # On essaie d'abord en L pour compat uniformité; si vous souhaitez
            # préserver strictement 16 bits pour ces cas, adaptez selon vos fichiers.
            im = im.convert("L")
            arr = np.array(im)
        # Convertit en float pour calculs robustes
        arr = arr.astype(np.float64, copy=False)
        # Si l'image est RGB (rare ici), on aurait déjà converti; mais par sécurité :
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return arr

# --- Collecte des mesures ---
results = []  # liste de dicts: {name, mean, var, ratio}

for fname in sorted(os.listdir(DOSSIER)):
    if not fname.lower().endswith(EXTS):
        continue
    fpath = os.path.join(DOSSIER, fname)
    if not os.path.isfile(fpath):
        continue

    arr = load_gray16_to_float(fpath)

    # Calculs (en ignorant les NaN/Inf potentiels si présents)
    mean_val = float(np.nanmean(arr))
    var_val = float(np.nanvar(arr))
    ratio = float(var_val / mean_val) if mean_val != 0 else np.nan  # variance / moyenne

    results.append({
        "name": fname,
        "mean": mean_val,
        "var": var_val,
        "ratio": ratio,
    })

# Tri optionnel
if SORT_BY == "ratio":
    results.sort(key=lambda d: (np.isnan(d["ratio"]), d["ratio"]))
else:  # "name"
    results.sort(key=lambda d: d["name"].lower())

# --- Préparation des données pour le tracé ligne 2D ---
labels = [os.path.splitext(r["name"])[0] for r in results]
x = np.arange(len(results))
y = np.array([r["ratio"] for r in results], dtype=np.float64)

# --- Tracé (ligne 2D) ---
plt.figure(figsize=(12, 6))
plt.plot(x, y, marker='o')  # ligne 2D avec marqueurs

# Ticks: noms de fichiers (sans extension) sur l'axe X
plt.xticks(ticks=x, labels=labels, rotation=45, ha='right')

# Annotations numériques optionnelles près de chaque point
if ANNOTATE_VALUES:
    for xi, yi in zip(x, y):
        if np.isfinite(yi):
            plt.annotate(f"{yi:.3g}", (xi, yi), xytext=(0, 6), textcoords='offset points', fontsize=8)

plt.ylabel("Variance / Moyenne")
plt.xlabel("Image")
plt.title("Fano (variance/moyenne) par image")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
