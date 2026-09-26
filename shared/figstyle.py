"""Shared figure style for the paper's figures — print-first, validated palette.

Design contract (see also the figure scripts that import this):

* **Canvas in millimetres.** Figures are built at their final printed size and included in LaTeX
  with ``width=\\linewidth`` and no scaling, so a 7 pt label in the figure is 7 pt on the page.
  ``elsarticle`` preprint text width = 390 pt = 137.4 mm.
* **Explicit layout.** Axes are placed with :func:`add_axes_mm` from a millimetre layout table;
  panel headers are figure-level text at fixed millimetre positions (never ``ax.set_title``,
  which overflows its slot and is what makes panels collide).
* **Typography.** LaTeX-typeset serif (Computer Modern), identical to the body text, so the
  symbols in the figure are the same glyphs as the symbols in the running text.
* **Colour is assigned by the job it does, and the palette is validated, not eyeballed**
  (``dataviz`` skill, ``scripts/validate_palette.js``):
    - *identity* (which method): two slots only, ``SG`` and ``STFT`` below. Validated all-pairs on
      a light surface: CVD ΔE 29.5 (protan), normal-vision ΔE 37.6, both ≥ 3:1 contrast.
      (green+orange was rejected by the validator: protan ΔE 3.2.)
    - *polarity* (signed attribution): one diverging map, :func:`divmap`, blue ↔ neutral ↔ red,
      red = positive (same orientation as the LRP convention and as Fig. 2).
    - *text* never wears a data colour: labels use the ink tokens; identity is carried by the
      coloured mark beside the text.
* **Marks.** Hairline recessive axes; dashing is reserved for a real threshold, never for grids.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1.0 / 25.4          # mm -> inch
TEXT_W_MM = 137.4        # \linewidth of elsarticle preprint (390 pt)

# ── palette: documented hexes only ────────────────────────────────────────────
INK = {
    "primary": "#0b0b0b",     # data strokes that are not identity-coded (waveforms)
    "secondary": "#52514e",   # annotations, values
    "muted": "#898781",       # axis labels, ticks, non-data notes
    "grid": "#e1e0d9",        # hairline grid (solid, recessive)
    "axis": "#c3c2b7",        # spines / baselines
    "surface": "#ffffff",     # paper
}
ID = {
    "sg": "#4a3aa7",          # identity slot 1 — Spectral Gradients   (violet)
    "stft": "#eb6834",        # identity slot 2 — STFT windows         (orange)
}
DIV = {
    "neg": "#1c5cab",         # blue,  sequential ramp step 550
    "mid": "#ffffff",         # neutral = the surface (the paper), so "zero" reads as blank
    "pos": "#e34948",         # red,   categorical slot 8
}
EVENT = dict(color=INK["muted"], alpha=0.12, lw=0)   # ground-truth event wash


def divmap(name="sg_div"):
    """Signed-attribution colormap: blue (negative) -> neutral -> red (positive)."""
    return LinearSegmentedColormap.from_list(name, [DIV["neg"], DIV["mid"], DIV["pos"]], N=256)


def use_paper_style(usetex=True):
    """rcParams for a print figure typeset by LaTeX in the document's own font."""
    plt.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}\usepackage{newtxtext,newtxmath}",
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        # hairline, recessive chrome
        "axes.linewidth": 0.4,
        "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["secondary"],
        "axes.facecolor": "none",
        "figure.facecolor": INK["surface"],
        "grid.color": INK["grid"],
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 1.8,
        "ytick.major.size": 1.8,
        "xtick.major.pad": 1.6,
        "ytick.major.pad": 1.6,
        "lines.linewidth": 0.7,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.5,
        "savefig.facecolor": INK["surface"],
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "pdf.compression": 6,
        "figure.dpi": 300,
    })


def new_figure(w_mm=TEXT_W_MM, h_mm=118.0):
    """Figure at exactly its printed size; nothing is scaled by LaTeX afterwards."""
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig.set_layout_engine("none")          # no auto-layout: the mm table is the layout
    fig._paper_size_mm = (w_mm, h_mm)      # read back by add_axes_mm / fig_text_mm
    return fig


def _size(fig):
    return getattr(fig, "_paper_size_mm", (TEXT_W_MM, 118.0))


def add_axes_mm(fig, x, y, w, h, **kw):
    """Axes at (x, y) mm with size (w, h) mm, y measured **from the top** of the canvas."""
    W, H = _size(fig)
    return fig.add_axes([x / W, (H - y - h) / H, w / W, h / H], **kw)


def fig_text_mm(fig, x, y, s, size=6.5, color=None, ha="left", va="baseline", **kw):
    """Figure-level text at (x, y) mm, y from the top. Used for panel headers and notes."""
    W, H = _size(fig)
    return fig.text(x / W, (H - y) / H, s, fontsize=size,
                    color=color or INK["secondary"], ha=ha, va=va, **kw)


def panel_header(fig, x, y, letter, title, size=7.0):
    """``(a)`` in primary ink plus a short sentence-case title in secondary ink, on one line."""
    fig_text_mm(fig, x, y, r"\textbf{(%s)}" % letter, size=size, color=INK["primary"])
    fig_text_mm(fig, x + 4.6, y, title, size=size, color=INK["secondary"])


def strip_spines(ax, keep=("bottom",)):
    for name, sp in ax.spines.items():
        sp.set_visible(name in keep)


def heat(ax, m, extent, vmax, cmap=None, dpi=900):
    """A signed attribution map on the shared symmetric scale, rasterised inside a vector PDF."""
    im = ax.imshow(m, aspect="auto", origin="lower", extent=extent,
                   cmap=cmap or divmap(), vmin=-vmax, vmax=vmax,
                   interpolation="nearest", rasterized=True)
    im.set_zorder(0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
        sp.set_color(INK["axis"])
    return im
