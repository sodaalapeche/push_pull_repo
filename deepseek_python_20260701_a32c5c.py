"""
Centralized plotting style configuration for all figures.
Ensures consistent sizing, markers, colors, and typography across the paper.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

# ==========================================================
# GLOBAL FIGURE DIMENSIONS
# ==========================================================
# Single column figure (for paper)
FIG_WIDTH_SINGLE = 8.5      # cm
FIG_HEIGHT_SINGLE = 5.5     # cm

# Double column figure (for paper)
FIG_WIDTH_DOUBLE = 17.5     # cm
FIG_HEIGHT_DOUBLE = 9.0     # cm

# Full page width
FIG_WIDTH_FULL = 18.0       # cm
FIG_HEIGHT_FULL = 10.0      # cm

INCHES = 2.54

# ==========================================================
# MARKER STYLES - HOMOGENIZED
# ==========================================================
MARKER_STYLES = {
    # Fine sand: filled circles (same size for all)
    (10, "fine"):   dict(marker="o", markersize=5, markeredgewidth=0.5, alpha=0.85, mfc="tab:blue", mec="tab:blue"),
    (6,  "fine"):   dict(marker="o", markersize=5, markeredgewidth=0.5, alpha=0.85, mfc="tab:orange", mec="tab:orange"),
    (3,  "fine"):   dict(marker="o", markersize=5, markeredgewidth=0.5, alpha=0.85, mfc="tab:red", mec="tab:red"),
    (0,  "fine"):   dict(marker="o", markersize=5, markeredgewidth=0.5, alpha=0.85, mfc="tab:green", mec="tab:green"),
    
    # Coarse sand: open diamonds (same size for all)
    (10, "coarse"): dict(marker="D", markersize=5.5, markeredgewidth=1.0, alpha=0.85, mfc="none", mec="tab:blue"),
    (6,  "coarse"): dict(marker="D", markersize=5.5, markeredgewidth=1.0, alpha=0.85, mfc="none", mec="tab:orange"),
    (3,  "coarse"): dict(marker="D", markersize=5.5, markeredgewidth=1.0, alpha=0.85, mfc="none", mec="tab:red"),
    (0,  "coarse"): dict(marker="D", markersize=5.5, markeredgewidth=1.0, alpha=0.85, mfc="none", mec="tab:green"),
    
    # Special cases
    (10, "stokes"):      dict(marker="x", markersize=5, markeredgewidth=1.0, alpha=0.85, mfc="tab:purple", mec="tab:purple"),
    (10, "contactless"): dict(marker="s", markersize=5, markeredgewidth=0.5, alpha=0.85, mfc="tab:pink", mec="tab:pink"),
}

# ==========================================================
# COLOR MAP
# ==========================================================
COLOR_MAP = {
    10: "tab:blue",
    6:  "tab:orange",
    3:  "tab:red",
    0:  "tab:green",
}

# ==========================================================
# LINE STYLES - HOMOGENIZED
# ==========================================================
LINE_STYLES = {
    "fit": dict(color="k", linestyle="-", linewidth=1.2, alpha=0.85),
    "guide": dict(color="gray", linestyle="--", linewidth=0.8, alpha=0.5),
    "theory": dict(color="k", linestyle="-.", linewidth=1.0, alpha=0.6),
    "grid": dict(linestyle="--", alpha=0.3, linewidth=0.5),
}

# ==========================================================
# LEGEND STYLES - HOMOGENIZED
# ==========================================================
LEGEND_STYLES = {
    "fontsize": 7,
    "frameon": False,
    "handlelength": 1.8,
    "handletextpad": 0.5,
    "columnspacing": 1.2,
}

# ==========================================================
# AXIS STYLES - HOMOGENIZED
# ==========================================================
AXIS_STYLES = {
    "labelsize": 9,
    "titlesize": 10,
    "ticklabelsize": 8,
}

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================
def setup_figure(width_cm=FIG_WIDTH_SINGLE, height_cm=None, aspect_ratio=0.65):
    """Create a figure with consistent sizing."""
    if height_cm is None:
        height_cm = width_cm * aspect_ratio
    fig, ax = plt.subplots(figsize=(width_cm / INCHES, height_cm / INCHES), layout="constrained")
    return fig, ax

def setup_figure_double(aspect_ratio=0.55):
    """Create a double-column figure."""
    return setup_figure(FIG_WIDTH_DOUBLE, aspect_ratio=aspect_ratio)

def apply_axis_style(ax, xlabel=None, ylabel=None, title=None,
                     xscale="linear", yscale="linear",
                     xlim=None, ylim=None, grid=True):
    """Apply consistent axis styling."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_STYLES["labelsize"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_STYLES["labelsize"])
    if title:
        ax.set_title(title, fontsize=AXIS_STYLES["titlesize"])
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if grid:
        ax.grid(**LINE_STYLES["grid"])
    
    ax.tick_params(labelsize=AXIS_STYLES["ticklabelsize"])
    return ax

def get_marker_style(L_mm, sand):
    """Get marker style for a given (L_mm, sand) combination."""
    return MARKER_STYLES.get((L_mm, sand), dict(marker="o", markersize=5, alpha=0.8))

def get_color(L_mm):
    """Get color for a given L_mm value."""
    return COLOR_MAP.get(L_mm, "tab:gray")

def make_legend_element(L_mm, sand, label=None):
    """Create a Line2D legend element."""
    style = get_marker_style(L_mm, sand)
    color = COLOR_MAP.get(L_mm, "tab:gray")
    if label is None:
        label = f"{L_mm} mm, {sand}"
    return plt.Line2D(
        [0], [0],
        marker=style["marker"],
        linestyle="None",
        color=color,
        markerfacecolor=style.get("mfc", color) if sand == "fine" else "none",
        markeredgecolor=style.get("mec", color),
        markeredgewidth=style.get("markeredgewidth", 0.5),
        markersize=style.get("markersize", 5),
        alpha=style.get("alpha", 0.85),
        label=label
    )

def create_legend(ax, handles, **kwargs):
    """Create a legend with consistent styling."""
    legend_kwargs = {**LEGEND_STYLES, **kwargs}
    return ax.legend(handles=handles, **legend_kwargs)

def apply_global_style():
    """Apply global matplotlib style settings."""
    plt.style.use("science")