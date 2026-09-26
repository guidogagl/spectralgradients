"""Figure 1 — how Spectral Gradients works, on a schematic synthetic case.

This figure is a SCHEMATIC, not an experiment: four low-frequency bands, four ablation steps,
and invented scores. No number here is a measurement; the quantitative results are Tables 2-4.
What the figure does guarantee is internal consistency -- the asserts below fail the build if
any of the relations it draws stops holding.

One synthetic case runs through all three panels:
  (a) one path pi_1: bands enter one at a time, each step scores a band importance phi^i
  (b) that same path's phi is redistributed over time by the path-integrated gradient,
      exactly: sum_t I^i_pi(t) = phi^i (Lemma 1)
  (c) a different order yields a different phi; the expectation over orders is stable, and
      completeness survives the averaging

Design contract, palette and layout helpers live in ``shared/figstyle.py``. The layout is an
explicit millimetre table, never an auto-layout.

    python3 shared/fig1_schema.py [--no-tex]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--no-tex", action="store_true", help="draft render without LaTeX")
args = ap.parse_args()
fs.use_paper_style(usetex=not args.no_tex)
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch      # noqa: E402

INK, ID, SMALL = fs.INK, fs.ID, 6.5
plt.rcParams.update({"font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.5})

# ════════════════════════════════════════════════════════════════════════════
# the one synthetic case
# ════════════════════════════════════════════════════════════════════════════
EDGES = np.array([0.5, 1.5, 2.5, 3.5, 4.5])         # four 1 Hz bands
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])            # 1, 2, 3, 4 Hz
STAR = 2                                            # the discriminative band, 3 Hz
NCOMP = 3                                           # a band is a GROUP of sinusoids
T = np.linspace(0, 2, 1200)

PERMS = [[0, 1, 2, 3], [2, 0, 3, 1], [1, 3, 0, 2]]
PHI = np.array([[0.11, 0.17, 0.52, 0.12],           # pi_1 -- the path shown in (a) and (b)
                [0.05, 0.11, 0.70, 0.06],
                [0.08, 0.14, 0.58, 0.12]])
PHI_BAR = PHI.mean(axis=0)
TOTAL = 0.92                                        # g(x) - g(x_empty), path-invariant
G0 = 0.02
assert np.allclose(PHI.sum(axis=1), TOTAL)          # telescoping: same total on every path
assert np.allclose(PHI_BAR, [0.08, 0.14, 0.60, 0.10])

G = np.concatenate([[G0], G0 + np.cumsum(PHI[0][PERMS[0]])])   # the score along pi_1

# the signals of (a): each step adds one band's group of sinusoids
rng = np.random.default_rng(7)
SIG = [np.zeros_like(T)]
for k in range(4):
    lo, hi = EDGES[k], EDGES[k + 1]
    f = lo + (np.arange(NCOMP) + 0.5) / NCOMP * (hi - lo)
    a = rng.uniform(0.75, 1.25, NCOMP)
    ph = rng.uniform(0, 2 * np.pi, NCOMP)
    band = sum(ai * np.sin(2 * np.pi * fi * T + pi) for ai, fi, pi in zip(a, f, ph))
    SIG.append(SIG[-1] + 0.62 * band / band.std())  # equal energy: every step equally visible
YMAX = np.abs(SIG[-1]).max() * 1.08

ENV = [(0.60, 0.30), (1.45, 0.27), (1.20, 0.24), (0.95, 0.32)]

def time_profiles(path):
    """I^i(t) for permutation `path`.

    The gradient is taken between x^(i-1) and x^i, which depend on the whole route, so a
    different order reshapes the profile -- it does not merely rescale it. Each row is then
    rescaled so that sum_t I^i(t) == phi^i exactly: Lemma 1 holds for the data drawn.
    """
    r = np.random.default_rng(11 + path)
    out = np.zeros((4, T.size))
    for i in range(4):
        c, w = ENV[i]
        c = c + r.uniform(-0.26, 0.26)              # where the band lands shifts with the route
        w = w * r.uniform(0.75, 1.35)
        env = np.exp(-0.5 * ((T - c) / w) ** 2)
        osc = 1.0 + 1.55 * np.sin(2 * np.pi * CENTRES[i] * T + r.uniform(0, 2 * np.pi))
        prof = env * osc
        out[i] = prof / prof.sum() * PHI[path][i]
    return out

I_ALL = np.stack([time_profiles(r) for r in range(3)])
I_PI = I_ALL[0]                                     # (b): the single path of (a)
I_BAR = I_ALL.mean(axis=0)                          # (c): the expectation over routes
assert np.allclose(I_PI.sum(axis=1), PHI[0])        # Lemma 1, per path
assert np.allclose(I_BAR.sum(axis=1), PHI_BAR)      # ... and it survives the averaging

# ════════════════════════════════════════════════════════════════════════════
W_MM, H_MM = fs.TEXT_W_MM, 133.0
fig = fs.new_figure(w_mm=W_MM, h_mm=H_MM)
OFF_A, OFF_B, OFF_C = 0.0, 43.0, 86.0

# ═══ (a) the ablation path ══════════════════════════════════════════════════
O = OFF_A
fs.panel_header(fig, 9, O + 4.6, "a",
                r"Spectral ablation path along $\pi_1$ and the resulting band importances")
X0, CW, CG = 9.0, 16.0, 2.6
YSIG, HSIG = O + 11.0, 11.0
ROWW = 5 * CW + 4 * CG
cx = [X0 + k * (CW + CG) for k in range(5)]
labels = [r"$\mathbf{x}^{0}\!=\!\mathbf{x}^{\emptyset}$", r"$\mathbf{x}^{1}$",
          r"$\mathbf{x}^{2}$", r"$\mathbf{x}^{3}$", r"$\mathbf{x}^{4}\!=\!\mathbf{x}$"]
for k in range(5):
    ax = fs.add_axes_mm(fig, cx[k], YSIG, CW, HSIG)
    hot = k == STAR + 1
    ax.plot(T, SIG[k], color=ID["sg"] if hot else INK["primary"], lw=0.6)
    ax.set_xlim(0, 2); ax.set_ylim(-YMAX, YMAX)
    ax.set_xticks([]); ax.set_yticks([]); fs.strip_spines(ax, keep=())
    ax.text(0.5, -0.13, labels[k], transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color=INK["primary"])
    ax.text(0.5, -0.46, r"$g=%.2f$" % G[k], transform=ax.transAxes, ha="center", va="top",
            fontsize=SMALL, color=INK["secondary"])

YB = YSIG + HSIG + 8.4
axb = fs.add_axes_mm(fig, 0, YB, W_MM, 9.0)
axb.set_xlim(0, W_MM); axb.set_ylim(0, 9.0); axb.set_axis_off()
for i, band in enumerate(PERMS[0]):
    a, b = cx[i] + CW / 2 + 0.9, cx[i + 1] + CW / 2 - 0.9
    hot = band == STAR
    col = ID["sg"] if hot else INK["muted"]
    axb.plot([a, a, b, b], [9.0, 7.4, 7.4, 9.0], color=col, lw=0.8 if hot else 0.5,
             solid_joinstyle="miter", clip_on=False)
    axb.text((a + b) / 2, 6.6, r"$\phi^{%d}=%+.2f$" % (i + 1, PHI[0][band]),
             ha="center", va="top", fontsize=SMALL, color=col)
axb.text(X0 + ROWW / 2, 1.0, r"$\phi^{i}=g(\mathbf{x}^{i})-g(\mathbf{x}^{i-1})$",
         ha="center", va="top", fontsize=SMALL, color=INK["muted"])

HX = X0 + ROWW + 9.0
axh = fs.add_axes_mm(fig, HX, YSIG, W_MM - HX - 4.0, HSIG + 3.0)
axh.bar(CENTRES, PHI[0], width=0.74, edgecolor="none",
        color=[ID["sg"] if i == STAR else INK["muted"] for i in range(4)])
axh.axhline(0, color=INK["axis"], lw=0.4)
axh.set_xlim(EDGES[0], EDGES[-1]); axh.set_xticks(CENTRES)
axh.set_ylim(0, 0.62); axh.set_yticks([0, 0.3, 0.6])
fs.strip_spines(axh, keep=("left", "bottom"))
axh.tick_params(labelsize=SMALL, pad=1.0)
axh.set_xlabel("frequency (Hz)", labelpad=3.4)
axh.set_ylabel(r"$\phi^i$", labelpad=3.4)

# ═══ (b) the spread over time ═══════════════════════════════════════════════
O = OFF_B
fs.panel_header(fig, 9, O + 4.6, "b",
                r"Redistribution of each band importance over time by the path-integrated gradient")
YT, HT = O + 15.0, 18.0
XH2, WH2 = 15.0, 19.0
XA0, XA1 = XH2 + WH2, 52.0
XM, WM = XA1, 63.0
XC, WC = XM + WM + 2.2, 2.4
VMAX_B = np.abs(I_PI).max()

axh2 = fs.add_axes_mm(fig, XH2, YT, WH2, HT)
axh2.barh(CENTRES, PHI[0], height=0.70, edgecolor="none",
          color=[ID["sg"] if i == STAR else INK["muted"] for i in range(4)])
axh2.axvline(0, color=INK["axis"], lw=0.4)
axh2.set_ylim(EDGES[0], EDGES[-1]); axh2.set_yticks(CENTRES)
axh2.set_yticklabels([r"$%g$" % c for c in CENTRES])
axh2.set_xlim(0, 0.60); axh2.set_xticks([0, 0.25, 0.5])
fs.strip_spines(axh2, keep=("left", "bottom"))
axh2.tick_params(labelsize=SMALL, pad=1.0)
axh2.set_xlabel(r"band importance $\phi^i$", labelpad=3.2)
axh2.set_ylabel("frequency (Hz)", labelpad=3.4)

axm = fs.add_axes_mm(fig, XM, YT, WM, HT)
fs.heat(axm, I_PI, extent=(0, 2, EDGES[0], EDGES[-1]), vmax=VMAX_B)
axm.set_ylim(EDGES[0], EDGES[-1]); axm.set_yticks(CENTRES); axm.set_yticklabels([])
axm.set_xlim(0, 2); axm.set_xticks([0, 1, 2])
axm.tick_params(labelsize=SMALL, pad=1.0)
axm.set_xlabel("time (s)", labelpad=3.4)
for e in EDGES[1:-1]:
    axm.axhline(e, color=INK["surface"], lw=0.5)
cb = fig.colorbar(axm.images[0], cax=fs.add_axes_mm(fig, XC, YT, WC, HT))
cb.set_ticks([-VMAX_B, 0, VMAX_B]); cb.set_ticklabels([r"$-$", r"$0$", r"$+$"])
cb.ax.tick_params(labelsize=SMALL, pad=1.0)
cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(INK["axis"])

def yfrac(c, ytop, h):
    return 1 - (ytop + h - (c - EDGES[0]) / (EDGES[-1] - EDGES[0]) * h) / H_MM

for i in range(4):
    hot = i == STAR
    fig.add_artist(FancyArrowPatch(
        ((XA0 + 1.0) / W_MM, yfrac(CENTRES[i], YT, HT)),
        ((XA1 - 1.0) / W_MM, yfrac(CENTRES[i], YT, HT)),
        transform=fig.transFigure, arrowstyle="-|>,head_length=2.6,head_width=1.3",
        color=ID["sg"] if hot else INK["muted"], lw=0.9 if hot else 0.45,
        shrinkA=0, shrinkB=0))
midb = (XA0 + XA1) / 2
fs.fig_text_mm(fig, midb, YT - 3.4, r"$\sum_t I^i(t)=\phi^i$ {\color{gray}(completeness)}",
               size=SMALL + 0.3, color=ID["sg"], ha="center")
fs.fig_text_mm(fig, XM + WM / 2, YT - 2.0, r"per-band temporal attribution $I^i(t)$",
               size=SMALL, color=INK["secondary"], ha="center")

# ═══ (c) the order dependence and its expectation ═══════════════════════════
O = OFF_C
fs.panel_header(fig, 9, O + 4.6, "c",
                r"Path dependence of the band importances and their expectation over $\pi$")
XG = 15.0
NX = np.linspace(XG, XG + 64.0, 5)
YR = [O + 16.5, O + 26.0, O + 35.5]
GW, GH = 3.6, 5.2
axg = fs.add_axes_mm(fig, 0, 0, W_MM, H_MM)
axg.set_xlim(0, W_MM); axg.set_ylim(H_MM, 0); axg.set_axis_off()

def node(xc, yc, present):
    for k in range(4):
        y = yc + GH / 2 - (k + 1) * GH / 4
        on = k in present
        axg.add_patch(Rectangle((xc - GW / 2, y), GW, GH / 4,
                      facecolor=(ID["sg"] if k == STAR else INK["secondary"]) if on
                      else INK["surface"], edgecolor=INK["axis"], lw=0.3, zorder=3))

for r, perm in enumerate(PERMS):
    y = YR[r]
    for j in range(5):
        node(NX[j], y, set(perm[:j]))
    for j, band in enumerate(perm):
        hot = band == STAR
        a, b = NX[j] + GW / 2 + 0.7, NX[j + 1] - GW / 2 - 0.7
        col = ID["sg"] if hot else INK["muted"]
        axg.annotate("", xy=(b, y), xytext=(a, y),
                     arrowprops=dict(arrowstyle="-|>,head_length=2.6,head_width=1.3",
                                     mutation_scale=2.4, color=col,
                                     lw=0.9 if hot else 0.45, shrinkA=0, shrinkB=0))
        axg.text((a + b) / 2, y - 1.0, r"$%g$ Hz" % CENTRES[band], ha="center", va="bottom",
                 fontsize=SMALL, color=col)
        axg.text((a + b) / 2, y + 1.0, r"$%+.2f$" % PHI[r][band], ha="center", va="top",
                 fontsize=SMALL, color=col)
    axg.text(XG - GW / 2 - 2.6, y, r"$\pi_%d$" % (r + 1), ha="right", va="center",
             fontsize=SMALL, color=INK["secondary"])
axg.text(NX[0], YR[0] - 5.6, r"$\mathbf{x}^{\emptyset}$", ha="center", va="center",
         fontsize=6.5, color=INK["primary"])
axg.text(NX[-1], YR[0] - 5.6, r"$\mathbf{x}$", ha="center", va="center",
         fontsize=6.5, color=INK["primary"])
axg.text((NX[0] + NX[-1]) / 2, YR[0] - 5.6, "node: set of restored bands",
         ha="center", va="center", fontsize=SMALL, color=INK["muted"])
axg.text((NX[0] + NX[-1]) / 2, YR[-1] + 7.4,
         r"$3$ Hz band: $\phi=+0.52,\,+0.70,\,+0.58$ \qquad "
         r"$\phi^1\!+\!\dots\!+\!\phi^4=%.2f$ for every $\pi$" % TOTAL,
         ha="center", va="center", fontsize=SMALL, color=INK["secondary"])

XM2, WM2, YM2, HM2 = 92.0, 36.0, O + 16.0, 17.0
VMAX_C = np.abs(I_BAR).max()
axm2 = fs.add_axes_mm(fig, XM2, YM2, WM2, HM2)
fs.heat(axm2, I_BAR, extent=(0, 2, EDGES[0], EDGES[-1]), vmax=VMAX_C)
axm2.set_ylim(EDGES[0], EDGES[-1]); axm2.set_yticks(CENTRES)
axm2.set_yticklabels([r"$%g$" % c for c in CENTRES])
axm2.set_xlim(0, 2); axm2.set_xticks([0, 1, 2])
axm2.tick_params(labelsize=SMALL, pad=1.0)
axm2.set_xlabel("time (s)", labelpad=3.4)
axm2.set_ylabel("frequency (Hz)", labelpad=3.4)
for e in EDGES[1:-1]:
    axm2.axhline(e, color=INK["surface"], lw=0.5)
axm2.set_title(r"$\mathbb{E}_\pi\!\left[I^i(t)\right]$",
               fontsize=SMALL + 0.3, color=INK["primary"], pad=2.5)
cb2 = fig.colorbar(axm2.images[0], cax=fs.add_axes_mm(fig, XM2 + WM2 + 1.6, YM2, 2.2, HM2))
cb2.set_ticks([-VMAX_C, 0, VMAX_C]); cb2.set_ticklabels([r"$-$", r"$0$", r"$+$"])
cb2.ax.tick_params(labelsize=SMALL, pad=1.0)
cb2.outline.set_linewidth(0.4); cb2.outline.set_edgecolor(INK["axis"])

out = HERE.parent / "figures" / "fig1_schema"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".pdf"), dpi=900)
fig.savefig(out.with_suffix(".png"), dpi=600)
print(f"saved {out.with_suffix('.pdf')}  (canvas {W_MM:.1f} x {H_MM:.1f} mm, "
      f"g={np.round(G, 2).tolist()}, phi(pi1)={PHI[0].tolist()}, "
      f"E[phi]={PHI_BAR.tolist()})")
