"""Shared figure style for the paper (static, print-first).

Palette: the dataviz reference palette, categorical slots 1-3 (blue, orange,
aqua) plus the muted ink for the string baseline; the slot order is the
validated one. Every series also carries a distinct marker, so identity never
rests on colour alone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# one visual identity per method, used by every figure
SERIES = {
    "strings":   dict(color=MUTED,  marker="x", label="String TF-IDF",       ls=":"),
    "dense":     dict(color=ORANGE, marker="o", label="Dense",               ls="-"),
    "dense_pc1": dict(color=ORANGE, marker="o", label="Dense, PC1 removed",  ls="--", mfc="white"),
    "sae_mean":  dict(color=AQUA,   marker="s", label="SAE, unweighted",     ls="-"),
    "sae_idf":   dict(color=BLUE,   marker="D", label="SAE-IDF (ours)",      ls="-"),
}

COLUMN_IN, TEXT_IN = 3.15, 6.5   # ACL column and text widths


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 4,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def style(key, **over):
    d = dict(SERIES[key]); d.pop("label")
    d.setdefault("mec", d["color"]); d.setdefault("mfc", d["color"]); d.setdefault("mew", 0.9)
    d.update(over)
    return d


def marker(key, **over):
    """Marker-only style (no line), for dot plots."""
    return style(key, ls="none", **over)
