import numpy as np
import cv2
import tifffile
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# =========================
# Paramètres
# =========================

base_path = "/home/chorus/test/"
angle_range = (-15, 15)   # degrés
angle_step = 0.5          # précision angulaire
translation_bounds = 100  # pixels
roi_radius_frac = 0.45    # fraction du rayon image utilisée

# =========================
# Chargement des images
# =========================

img_exp = tifffile.imread(base_path + "1.tif").astype(np.float32)
mask = cv2.imread(base_path + "masque.jpg", cv2.IMREAD_GRAYSCALE)

mask = (mask > 128).astype(np.uint8)

H, W = img_exp.shape

# resize UNIQUE du masque (échelle déjà correcte)
mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

# =========================
# Prétraitement : bords
# =========================

# normalisation image exp pour Canny
img_norm = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

edges_exp = cv2.Canny(img_norm, 50, 150)
edges_mask = cv2.Canny(mask * 255, 50, 150)

# =========================
# Masque radial (évite biais du disque)
# =========================

Y, X = np.ogrid[:H, :W]
r = np.sqrt((X - W / 2)**2 + (Y - H / 2)**2)
roi = (r < roi_radius_frac * min(H, W)).astype(np.uint8)

edges_exp *= roi
edges_mask *= roi

# =========================
# Étape 1 — Recherche robuste de l'angle
# =========================
def autoscale(img, low=2, high=98):
    """
    Rescale automatiquement les niveaux de gris pour l'affichage
    (basé sur les percentiles)
    """
    vmin, vmax = np.percentile(img, (low, high))
    img_clip = np.clip(img, vmin, vmax)
    return (img_clip - vmin) / (vmax - vmin)

def estimate_rotation_by_edges(mask_edges, img_edges):
    best_angle = 0.0
    best_score = -np.inf

    center = (W / 2, H / 2)

    for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        mask_rot = cv2.warpAffine(
            mask_edges,
            M,
            (W, H),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        score = np.sum(mask_rot * img_edges)

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle

angle_fixed = estimate_rotation_by_edges(edges_mask, edges_exp)
print(f"Angle optimal (bords) : {angle_fixed:.2f}°")

# =========================
# Transformation masque (angle fixé)
# =========================

def transform_mask(mask, tx, ty, angle):
    center = (W / 2, H / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty

    return cv2.warpAffine(
        mask,
        M,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

# =========================
# Étape 2 — Optimisation translation
# =========================

def cost_function(params):
    tx, ty = params
    mask_t = transform_mask(mask, tx, ty, angle_fixed)
    return -np.sum(img_exp * mask_t)

result = minimize(
    cost_function,
    x0=[0.0, 0.0],
    method="Powell",
    bounds=[
        (-translation_bounds, translation_bounds),
        (-translation_bounds, translation_bounds)
    ],
    options={"disp": True}
)

tx, ty = result.x
print(f"Translation optimale : tx = {tx:.2f}, ty = {ty:.2f}")

# =========================
# Résultat final
# =========================

mask_aligned = transform_mask(mask, tx, ty, angle_fixed)

# =========================
# Visualisation
# =========================

img_masked = img_exp * mask_aligned
img_exp_disp = autoscale(img_exp)
img_masked_disp = autoscale(img_masked)

# =========================
# Visualisation
# =========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Image_exp")
plt.imshow(img_exp_disp, cmap="viridis")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Masque rotation+translation")
plt.imshow(mask_aligned, cmap="viridis")

plt.subplot(1, 3, 3)
plt.title("Image × masque")
plt.imshow(img_masked_disp, cmap="viridis")
plt.colorbar()

plt.tight_layout()
plt.show()

