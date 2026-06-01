"""
Segmentation streaming de billes de verre dans un stack tomographique
volumineux + reconstruction 3D isométrique + coupe 2D.

Principe : aucune copie complète du volume plein résolution en RAM.
  - Pass 1 : échantillonne quelques coupes pour calculer un seuil Otsu global.
  - Pass 2 : streaming par fenêtre glissante de coupes ->
              seuillage + morpho 3D locale -> écriture du mask coupe par coupe
              + accumulation d'un volume sous-échantillonné en RAM pour la viz 3D.
  - Rendu 3D : marching cubes sur le volume réduit uniquement.

Dépendances :
    pip install numpy scipy scikit-image tifffile matplotlib tqdm
    pip install plotly                  # optionnel
"""

from pathlib import Path
import numpy as np
import tifffile
from skimage import filters, morphology, measure, exposure
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
INPUT_DIR       = Path("/home/chorus/20250707_COLONNE_PETITJEAN_01_SlicesY/")
OUTPUT_DIR      = Path("/home/chorus/tomo3D/")

VOXEL_SIZE      = (1.0, 1.0, 1.0)   # (dz, dy, dx) — physique si connu
DOWNSAMPLE      = 32 # facteur de réduction pour le volume de viz 3D
                                    #   2 → 8x plus léger, 4 → 64x, 8 → 512x.
                                    #   Augmente si la RAM coince ou si marching cubes rame.
SLICE_INDEX     = None              # None = milieu du stack
INVERT          = False             # True si billes SOMBRES sur fond clair

# Seuillage
OTSU_SAMPLE_N   = 30                # nombre de coupes échantillonnées pour estimer Otsu global
PERCENTILES     = (1, 99)           # pour la normalisation (robuste aux outliers)

# Morphologie 3D streaming
WINDOW_Z        = 5                 # fenêtre glissante en z (coupes traitées par bloc)
                                    #   doit être >= 2*MORPHO_RADIUS+1 pour éviter les artefacts
MORPHO_RADIUS   = 1                 # rayon des balls pour ouverture/fermeture (en voxels)
MIN_BEAD_AREA   = 50                # px² — supprime les petits objets en 2D par tranche

# Crop 2D (utile si l'image contient des bords/halos hors colonne) — None = pas de crop
CROP_XY         = None              # ex: (slice(50, -50), slice(50, -50))

SAVE_MASK       = True              # écrit le mask plein résolution (multi-page TIFF)
INTERACTIVE_3D  = True              # tente plotly
# =============================================================================


def list_tif_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
        return [path]
    files = sorted(p for p in path.iterdir()
                   if p.suffix.lower() in (".tif", ".tiff"))
    if not files:
        raise FileNotFoundError(f"Aucun .tif dans {path}")
    return files


def open_stack(path: Path):
    """
    Retourne (n_slices, shape_yx, dtype, reader)
    reader(i) -> coupe 2D numpy d'index i.
    Gère :
      - dossier de .tif individuels (un fichier = une coupe)
      - .tif multi-page unique
    """
    files = list_tif_files(path)

    if len(files) == 1:
        # multi-page
        f = files[0]
        with tifffile.TiffFile(str(f)) as tf:
            n = len(tf.pages)
            shape_yx = tf.pages[0].shape
            dtype = tf.pages[0].dtype
        # On rouvre à chaque lecture — TiffFile n'aime pas être partagé entre threads
        def reader(i, _f=f):
            return tifffile.imread(str(_f), key=i)
        return n, shape_yx, dtype, reader

    # série de fichiers
    first = tifffile.imread(str(files[0]))
    def reader(i, _files=files):
        return tifffile.imread(str(_files[i]))
    return len(files), first.shape, first.dtype, reader


def normalize_slice(s: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    out = exposure.rescale_intensity(s, in_range=(vmin, vmax), out_range=(0, 1))
    if INVERT:
        out = 1.0 - out
    return out


def estimate_intensity_range(reader, n_slices: int) -> tuple[float, float]:
    """Percentiles (1, 99) estimés sur un sous-échantillonnage de coupes."""
    print("[norm] estimation des percentiles d'intensité sur échantillon...")
    idx = np.linspace(0, n_slices - 1, min(OTSU_SAMPLE_N, n_slices)).astype(int)
    lows, highs = [], []
    for i in tqdm(idx, desc="percentiles"):
        s = reader(i)
        if CROP_XY is not None:
            s = s[CROP_XY]
        lo, hi = np.percentile(s, PERCENTILES)
        lows.append(lo); highs.append(hi)
    vmin, vmax = float(np.median(lows)), float(np.median(highs))
    print(f"[norm] vmin={vmin:.3f}, vmax={vmax:.3f}")
    return vmin, vmax


def estimate_threshold(reader, n_slices: int, vmin: float, vmax: float) -> float:
    """Seuil Otsu calculé sur l'histogramme cumulé d'un sous-échantillon de coupes."""
    print("[seg] estimation du seuil Otsu global...")
    idx = np.linspace(0, n_slices - 1, min(OTSU_SAMPLE_N, n_slices)).astype(int)
    # On accumule les pixels normalisés (downsamplés en xy pour ne pas exploser la RAM)
    samples = []
    for i in tqdm(idx, desc="échantillons Otsu"):
        s = reader(i)
        if CROP_XY is not None:
            s = s[CROP_XY]
        s = normalize_slice(s, vmin, vmax)
        # downsample 4x pour l'estimation du seuil — Otsu n'a pas besoin de la pleine résolution
        s = s[::4, ::4]
        samples.append(s.ravel())
    pooled = np.concatenate(samples)
    th = float(filters.threshold_otsu(pooled))
    print(f"[seg] seuil Otsu = {th:.3f}")
    return th


def downsample_slice_xy(binary_slice: np.ndarray, factor: int) -> np.ndarray:
    """Sous-échantillonne une coupe binaire 2D par bloc 'any'."""
    if factor <= 1:
        return binary_slice
    h, w = binary_slice.shape
    h2, w2 = h // factor, w // factor
    cropped = binary_slice[:h2 * factor, :w2 * factor]
    return cropped.reshape(h2, factor, w2, factor).any(axis=(1, 3))


def stream_segment(reader, n_slices: int, shape_yx: tuple, dtype,
                   vmin: float, vmax: float, threshold: float,
                   mask_writer, ds_factor: int):
    """
    Streaming par fenêtre glissante :
      - lit WINDOW_Z coupes,
      - normalise + seuille,
      - applique morpho 3D sur la fenêtre (ouverture, fermeture),
      - écrit la coupe centrale,
      - accumule la version downsamplée pour le rendu 3D.
    Retourne le volume binaire sous-échantillonné (en RAM, petit).
    """
    pad = MORPHO_RADIUS  # halo nécessaire pour la morpho 3D
    window = WINDOW_Z
    assert window >= 2 * pad + 1, "WINDOW_Z trop petit pour MORPHO_RADIUS"

    # Shape après crop xy
    if CROP_XY is not None:
        sample = reader(0)[CROP_XY]
        h, w = sample.shape
    else:
        h, w = shape_yx

    # Volume réduit pour la viz 3D
    z_small = n_slices // ds_factor
    h_small = h // ds_factor
    w_small = w // ds_factor
    vol_small = np.zeros((z_small, h_small, w_small), dtype=bool)

    # Buffer circulaire des coupes binaires brutes (avant morpho 3D)
    # On garde une fenêtre de [window] coupes en mémoire
    buf = np.zeros((window, h, w), dtype=bool)

    selem = morphology.ball(MORPHO_RADIUS).astype(bool)

    # Indices : on traite la coupe i quand on a chargé jusqu'à i+pad
    pbar = tqdm(total=n_slices, desc="streaming segmentation")

    # Pré-charge les premières coupes
    for k in range(min(window, n_slices)):
        s = reader(k)
        if CROP_XY is not None:
            s = s[CROP_XY]
        s = normalize_slice(s, vmin, vmax)
        buf[k] = s > threshold

    # Index dans le buffer du centre courant
    # On commence par traiter k = pad (premières coupes traitées avec halo droit valide,
    # halo gauche réplique).
    for z in range(n_slices):
        # S'assurer que le buffer contient les coupes [z-pad ... z+pad]
        # On charge à la demande la coupe (z+pad) si elle n'est pas déjà dans le buffer.
        target = min(z + pad, n_slices - 1)
        # Décale le buffer si nécessaire pour qu'il pointe sur [z-pad .. z+pad]
        # Stratégie simple : on garde buf[0..2*pad] = coupes [z-pad .. z+pad]
        start = max(0, z - pad)
        end = min(n_slices - 1, z + pad)
        # Recharge la fenêtre — coût modéré : (2*pad+1) lectures par coupe.
        # Pour pad=1 → 3 lectures par coupe centrale → tolérable.
        local_window = []
        for kk in range(start, end + 1):
            s = reader(kk)
            if CROP_XY is not None:
                s = s[CROP_XY]
            s = normalize_slice(s, vmin, vmax)
            local_window.append(s > threshold)
        # Padding par réplication aux bords
        while len(local_window) < 2 * pad + 1:
            if z - pad < 0:
                local_window.insert(0, local_window[0])
            else:
                local_window.append(local_window[-1])
        local = np.stack(local_window, axis=0)  # (2*pad+1, h, w)

        # Morpho 3D locale : ouverture puis fermeture
        local = morphology.binary_opening(local, selem)
        local = morphology.binary_closing(local, selem)

        center = local[pad]

        # Nettoyage 2D : petits objets, trous internes
        center = morphology.remove_small_objects(center, min_size=MIN_BEAD_AREA)
        center = ndi.binary_fill_holes(center)

        # Écriture du mask plein résolution
        mask_writer(z, center)

        # Accumulation dans le volume réduit (any sur les blocs)
        z_idx_small = z // ds_factor
        if z_idx_small < z_small:
            slice_small = downsample_slice_xy(center, ds_factor)
            vol_small[z_idx_small] |= slice_small

        pbar.update(1)
    pbar.close()

    return vol_small


class TiffStreamWriter:
    """Écrit un mask binaire coupe par coupe dans un TIFF multi-page."""
    def __init__(self, path: Path):
        self.path = path
        self._writer = tifffile.TiffWriter(str(path), bigtiff=True)

    def __call__(self, z: int, slice_2d: np.ndarray):
        self._writer.write((slice_2d.astype(np.uint8) * 255),
                           photometric="minisblack", contiguous=True)

    def close(self):
        self._writer.close()


class NullWriter:
    def __call__(self, z, s): pass
    def close(self): pass


def render_3d_matplotlib(mask_small: np.ndarray, out_path: Path):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if mask_small.sum() == 0:
        print("[3D] mask vide, abandon.")
        return
    print("[3D] marching cubes sur volume réduit...")
    spacing = tuple(v * DOWNSAMPLE for v in VOXEL_SIZE)
    verts, faces, _, _ = measure.marching_cubes(mask_small.astype(np.uint8),
                                                level=0.5, spacing=spacing)
    print(f"[3D] mesh : {len(verts)} sommets, {len(faces)} faces")

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.85, linewidths=0.0)
    mesh.set_facecolor("#cfd6e1")
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    z, y, x = mask_small.shape
    ax.set_xlim(0, x * spacing[2]); ax.set_ylim(0, y * spacing[1]); ax.set_zlim(0, z * spacing[0])
    ax.set_box_aspect((x * spacing[2], y * spacing[1], z * spacing[0]))
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Reconstruction 3D des billes (isométrique)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[3D] figure : {out_path}")


def render_3d_plotly(mask_small: np.ndarray, out_path: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[3D] plotly non installé, on saute.")
        return
    if mask_small.sum() == 0:
        return
    spacing = tuple(v * DOWNSAMPLE for v in VOXEL_SIZE)
    verts, faces, _, _ = measure.marching_cubes(mask_small.astype(np.uint8),
                                                level=0.5, spacing=spacing)
    x, y, z = verts.T
    i, j, k = faces.T
    fig = go.Figure(data=[go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color="#9aa6b8", opacity=1.0, flatshading=True, name="billes",
    )])
    fig.update_layout(
        title="Billes de verre — vue 3D interactive",
        scene=dict(aspectmode="data",
                   camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_path))
    print(f"[3D] interactif : {out_path}")


def render_2d_slice(reader, mask_path: Path, z_idx: int, vmin, vmax, threshold,
                    out_path: Path, n_slices: int):
    """Coupe 2D : brute / mask (relue depuis le tiff écrit) / contours."""
    raw = reader(z_idx)
    if CROP_XY is not None:
        raw = raw[CROP_XY]

    # Recharge la coupe correspondante du mask depuis le fichier streamé
    if mask_path.exists():
        mask_slice = tifffile.imread(str(mask_path), key=z_idx).astype(bool)
    else:
        # Recalcul à la volée si SAVE_MASK=False
        s = normalize_slice(raw, vmin, vmax)
        mask_slice = s > threshold
        mask_slice = morphology.remove_small_objects(mask_slice, min_size=MIN_BEAD_AREA)
        mask_slice = ndi.binary_fill_holes(mask_slice)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw, cmap="gray"); axes[0].set_title(f"Brute (z={z_idx})"); axes[0].axis("off")
    axes[1].imshow(mask_slice, cmap="gray"); axes[1].set_title("Mask"); axes[1].axis("off")
    axes[2].imshow(raw, cmap="gray")
    for c in measure.find_contours(mask_slice.astype(float), level=0.5):
        axes[2].plot(c[:, 1], c[:, 0], color="#e24b4a", linewidth=0.6)
    axes[2].set_title("Contours"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[2D] coupe : {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ouvre le stack en mode lazy
    n_slices, shape_yx, dtype, reader = open_stack(INPUT_DIR)
    print(f"[load] {n_slices} coupes, shape coupe = {shape_yx}, dtype = {dtype}")

    # 2. Estimation des stats globales sur échantillon
    vmin, vmax = estimate_intensity_range(reader, n_slices)
    threshold = estimate_threshold(reader, n_slices, vmin, vmax)

    # 3. Streaming segmentation
    mask_path = OUTPUT_DIR / "mask_segmente.tif"
    if SAVE_MASK:
        writer = TiffStreamWriter(mask_path)
    else:
        writer = NullWriter()

    try:
        vol_small = stream_segment(reader, n_slices, shape_yx, dtype,
                                   vmin, vmax, threshold,
                                   writer, DOWNSAMPLE)
    finally:
        writer.close()

    print(f"[3D] volume réduit : shape = {vol_small.shape}, "
          f"occupation = {vol_small.mean():.2%}")

    # 4. Coupe 2D de contrôle
    z_idx = SLICE_INDEX if SLICE_INDEX is not None else n_slices // 2
    render_2d_slice(reader, mask_path, z_idx, vmin, vmax, threshold,
                    OUTPUT_DIR / "coupe_2d.png", n_slices)

    # 5. Rendus 3D
    render_3d_matplotlib(vol_small, OUTPUT_DIR / "vue_3d_isometrique.png")
    if INTERACTIVE_3D:
        render_3d_plotly(vol_small, OUTPUT_DIR / "vue_3d_interactive.html")

    print("\n[done] Tout est dans :", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()