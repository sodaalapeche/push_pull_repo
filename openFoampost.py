from pathlib import Path
import re
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import csv

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
vtk_dir = Path("/home/chorus/OpenFOAM/chorus-13/run/cylinderDiffusion2D/VTK")
field_name = "T"
s0=0.0015
D=6e-8
# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def extract_time(path: Path):
    """
    Try to extract simulation time from common OpenFOAM VTK naming patterns.
    """
    txt = path.as_posix()

    patterns = [
        r'_([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)\.(vtk|vtu)$',
        r'/([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)/[^/]+\.(vtk|vtu)$',
        r'/([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)\.(vtk|vtu)$'
    ]

    for pat in patterns:
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            return float(m.group(1))

    return None


def get_cell_scalar(mesh, name):
    """
    Return scalar field on cells.
    """
    if name in mesh.cell_data:
        arr = np.asarray(mesh.cell_data[name]).reshape(-1)
        return mesh, arr

    if name in mesh.point_data:
        mesh2 = mesh.point_data_to_cell_data()
        arr = np.asarray(mesh2.cell_data[name]).reshape(-1)
        return mesh2, arr

    return None, None


def get_cell_weights(mesh):
    """
    For a 2D OpenFOAM case extruded with one cell in z, cell volume is fine
    as the weighting measure. If volume is unavailable, try area.
    """
    sized = mesh.compute_cell_sizes(length=False, area=True, volume=True)

    if "Volume" in sized.cell_data:
        w = np.asarray(sized.cell_data["Volume"]).reshape(-1)
        if np.any(w > 0):
            return w

    if "Area" in sized.cell_data:
        w = np.asarray(sized.cell_data["Area"]).reshape(-1)
        if np.any(w > 0):
            return w

    raise RuntimeError("Could not compute cell weights (Volume/Area).")


# -------------------------------------------------------------------
# Collect files
# -------------------------------------------------------------------
files = sorted(list(vtk_dir.rglob("*.vtk")) + list(vtk_dir.rglob("*.vtu")))

if not files:
    raise FileNotFoundError(f"No .vtk or .vtu files found under: {vtk_dir}")

rows = []

for f in files:
    t = extract_time(f)
    if t is None:
        continue

    try:
        mesh = pv.read(f)
    except Exception as e:
        print(f"Skipping unreadable file: {f}\n  {e}")
        continue

    mesh, T = get_cell_scalar(mesh, field_name)
    if T is None:
        print(f"Skipping file without field '{field_name}': {f}")
        continue

    w = get_cell_weights(mesh)

    if len(T) != len(w):
        print(f"Skipping file with mismatched field/weight lengths: {f}")
        continue

    wsum = np.sum(w)
    mean_T = np.sum(w * T) / wsum
    mean_T2 = np.sum(w * T * T) / wsum
    var_T = mean_T2 - mean_T**2

    if np.isclose(mean_T, 0.0):
        sigma = np.nan
    else:
        sigma = var_T / (mean_T**2)

    rows.append((t, mean_T, mean_T2, var_T, sigma, str(f)))

if not rows:
    raise RuntimeError("No usable VTK files found with field 'T'.")

# Remove duplicate times if multiple VTK files exist for the same time.
# Keep the first usable one.
rows.sort(key=lambda x: x[0])
unique = {}
for row in rows:
    t = row[0]
    if t not in unique:
        unique[t] = row

rows = [unique[t] for t in sorted(unique.keys())]

times = np.array([r[0] for r in rows], dtype=float)
means = np.array([r[1] for r in rows], dtype=float)
vars_ = np.array([r[3] for r in rows], dtype=float)
sigmas = np.array([r[4] for r in rows], dtype=float)

# Find sigma0 from first valid time
valid = np.isfinite(sigmas)
if not np.any(valid):
    raise RuntimeError("All sigma values are invalid. Mean(T) may be zero at every time.")

sigma0 = sigmas[valid][0]
sigma_norm = sigmas

# -------------------------------------------------------------------
# Save CSV
# -------------------------------------------------------------------
out_csv = vtk_dir.parent / "sigma_decay.csv"
with open(out_csv, "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(["time", "mean_T", "variance_T", "sigma", "sigma_over_sigma0"])
    for t, m, v, s, sn in zip(times, means, vars_, sigmas, sigma_norm):
        writer.writerow([t, m, v, s, sn])

print(f"Wrote: {out_csv}")

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(times+s0**2/D, sigma_norm, marker="o")
plt.xlabel("time")
plt.ylabel(r"$\sigma / \sigma_0$")
plt.title(r"$\sigma = \mathrm{Var}(C)/\langle C\rangle^2$")
plt.grid(True)
plt.tight_layout()

out_png = vtk_dir.parent / "sigma_decay.png"
plt.savefig(out_png, dpi=200)
print(f"Wrote: {out_png}")
plt.yscale('log')
plt.xscale('log')
plt.show()

# Optional console summary
print("\nFirst few values:")
for t, s, sn in list(zip(times, sigmas, sigma_norm))[:10]:
    print(f"time={t:12.6g}   sigma={s:12.6e}   sigma/sigma0={sn:12.6e}")