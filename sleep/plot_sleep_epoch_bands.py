"""Qualitative figure: SG map on one real sleep epoch with the clinical EEG bands annotated.

Produces ``figures/sleep_epoch_bands.pdf`` (vector) and a ``.png`` preview:
  top    – raw EEG of the central 30-s epoch
  middle – SG time-frequency attribution (0–30 Hz) with δ/θ/α/σ/β spans and labels
  bottom – frequency marginal of the map, one bar per band, coloured by band

Model, dataset and SG configuration are exactly those of the paper's Sleep-EDF case study
(TsinalisCNN from the PhysioEx hub, Δf = 2 Hz, s = 10, K = 20), reusing the helpers of
``generate_qualitative_figures.py``. The epoch shown is chosen deterministically: among the
first ``--n-candidates`` correctly classified epochs of the requested stage with confidence
≥ ``--conf-thresh`` (seed-ordered scan), the one whose SG frequency marginal puts the largest
share of attribution in the stage's defining band (σ for N2, δ for N3).

Usage::
    python sleep/plot_sleep_epoch_bands.py --gpu 0 [--stage N2] [--n-candidates 40]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_qualitative_figures as gq  # noqa: E402  (shared constants/helpers)

from physioex.models import load_from_pretrained  # noqa: E402
from physioex.data.datasets import get_dataset  # noqa: E402
from physioex.data.presets import get_preset  # noqa: E402
from physioex.explain.posthoc.spectralgradients import SpectralGradients  # noqa: E402

# ── Style (fonts slightly smaller than body text) ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
MM2IN = 1 / 25.4
FULL_W = 137 * MM2IN  # elsarticle preprint text width (390 pt)

DEFINING_BAND = {"N2": "σ", "N3": "δ", "Wake": "α", "REM": "θ", "N1": "θ"}
FMAX = 30.0


def collect_stage_candidates(ds, score_fn, device, stage_idx, n_candidates, conf_thresh, seed):
    """First n_candidates correctly classified epochs of one stage with conf >= thresh."""
    torch.manual_seed(seed)
    order = torch.randperm(len(ds)).tolist()
    out = []
    for i, idx in enumerate(order):
        if len(out) >= n_candidates:
            break
        item = ds[idx]
        if item is None:
            continue
        labels = item["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        if int(labels[gq.L // 2]) != stage_idx:
            continue
        sigs = [item["signals"][k] for k in sorted(item["signals"].keys())]
        x_flat = torch.stack(sigs, dim=-2)[:, 0, :].reshape(-1)
        if x_flat.shape[0] != gq.TOTAL_LEN:
            continue
        x_dev = x_flat.to(device)
        with torch.no_grad():
            p = score_fn(x_dev.unsqueeze(0))
        conf, pred = p[0, stage_idx].item(), p.argmax(dim=-1).item()
        if pred == stage_idx and conf >= conf_thresh:
            out.append((x_dev, conf, idx))
        if i % 2000 == 0:
            print(f"  scanned {i}, found {len(out)}", flush=True)
    return out


def band_share(freqs, marginal, band):
    lo, hi = gq.BANDS[band]
    m = (freqs >= lo) & (freqs < hi)
    tot = np.abs(marginal).sum()
    return float(np.abs(marginal[m]).sum() / tot) if tot > 0 else 0.0


def plot(signal, sg_attr, stage_name, conf, out_pdf):
    c0 = (gq.L // 2) * gq.N_TIMES
    c1 = c0 + gq.N_TIMES
    t = np.arange(gq.N_TIMES) / gq.FS
    sig = signal[c0:c1].cpu().numpy()
    sg = sg_attr.cpu().numpy()[:, c0:c1]  # (n_bands, N_TIMES)
    n_bands = sg.shape[0]
    freqs = np.arange(n_bands) * gq.SG_FREQ_STEP + gq.SG_FREQ_STEP / 2
    fmask = freqs <= FMAX
    sg_v = sg[fmask]
    marg = np.abs(sg).sum(axis=1)
    marg_n = marg / (marg.sum() + 1e-12)

    fig, axes = plt.subplots(
        3, 1, figsize=(FULL_W, 118 * MM2IN),
        gridspec_kw={"height_ratios": [0.8, 2.0, 1.0], "hspace": 0.85},
    )

    # (a) raw EEG
    ax = axes[0]
    ax.plot(t, sig, color="#333333", linewidth=0.5)
    ax.set_xlim(0, 30)
    ax.set_ylabel("EEG (µV)")
    ax.set_title(f"(a) Raw EEG, central 30-s epoch — scored {stage_name}, "
                 f"predicted {stage_name} (confidence {conf:.2f})", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) SG map with band annotation
    ax = axes[1]
    vmax = np.percentile(np.abs(sg_v), 99.5) or 1.0
    im = ax.imshow(sg_v, aspect="auto", origin="lower",
                   extent=[0, 30, 0, freqs[fmask][-1] + gq.SG_FREQ_STEP / 2],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_ylim(0, FMAX)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("(b) Spectral Gradients attribution "
                 f"(Δf = {gq.SG_FREQ_STEP:g} Hz, K = {gq.SG_PERMS}, s = {gq.SG_STEPS})", loc="left")
    ax.set_xlabel("Time (s)")
    for name, (lo, hi) in gq.BANDS.items():
        hi_c = min(hi, FMAX)
        ax.axhline(lo, color="black", linewidth=0.4, linestyle=(0, (2, 2)), alpha=0.7)
        ax.text(30.3, (lo + hi_c) / 2, f"{name} {lo:g}–{hi:g} Hz", va="center", ha="left",
                fontsize=7, color=gq.BAND_COLORS[name], fontweight="bold", clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.17)
    cb.set_label("attribution", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # (c) frequency marginal, coloured by band
    ax = axes[2]
    colors = []
    for f in freqs:
        c = "#999999"
        for name, (lo, hi) in gq.BANDS.items():
            if lo <= f < hi:
                c = gq.BAND_COLORS[name]
                break
        colors.append(c)
    ax.bar(freqs[fmask], marg_n[fmask], width=gq.SG_FREQ_STEP * 0.9,
           color=[c for c, m in zip(colors, fmask) if m], edgecolor="none")
    ax.set_xlim(0, FMAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Share of |attribution|")
    shares = {b: band_share(freqs, marg, b) for b in gq.BANDS}
    txt = " · ".join(f"{b} {100 * s:.0f}%" for b, s in shares.items())
    ax.set_title(f"(c) Frequency marginal — {txt}", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=600)  # rasterised heatmap >= 300 dpi (journal artwork rule)
    fig.savefig(out_pdf.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_pdf} (+ .png)")
    return shares


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stage", default="N2", choices=gq.STAGE_NAMES)
    ap.add_argument("--n-candidates", type=int, default=40)
    ap.add_argument("--conf-thresh", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=gq.FIGURES_DIR / "sleep_epoch_bands.pdf")
    ap.add_argument("--freq-step", type=float, default=None,
                    help="SG band width in Hz (default: gq.SG_FREQ_STEP; the Sleep-EDF benchmark uses 0.5)")
    ap.add_argument("--chunk", type=int, default=4,
                    help="candidates explained per forward batch (memory only; results unchanged)")
    ap.add_argument("--epoch-idx", type=int, default=None,
                    help="force this dataset index instead of scanning candidates (reproduce a chosen epoch)")
    args = ap.parse_args()
    if args.freq_step is not None:
        gq.SG_FREQ_STEP = args.freq_step  # read at call time by plot()/band_share()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    stage_idx = gq.STAGE_NAMES.index(args.stage)

    model = load_from_pretrained(gq.MODEL_NAME, device=str(device))
    model.eval()
    score_fn = gq.build_score_fn(model)

    if not os.environ.get("PHYSIOEX_DATA"):
        sys.exit("PHYSIOEX_DATA is not set (expects <dir>/physionet-sleep-data/*.edf)")
    DatasetCls = get_dataset(gq.DATASET)
    ds = DatasetCls(channels=["EEG"], pipelines=get_preset(gq.PIPELINE),
                    sequence_length=gq.L)

    if args.epoch_idx is not None:
        item = ds[args.epoch_idx]
        sigs = [item["signals"][k] for k in sorted(item["signals"].keys())]
        x_dev = torch.stack(sigs, dim=-2)[:, 0, :].reshape(-1).to(device)
        with torch.no_grad():
            p = score_fn(x_dev.unsqueeze(0))
        lab = int(item["labels"][gq.L // 2]); pred = int(p.argmax(dim=-1))
        assert lab == stage_idx == pred, f"epoch {args.epoch_idx}: label {lab}, pred {pred}, requested {stage_idx}"
        cands = [(x_dev, float(p[0, stage_idx]), args.epoch_idx)]
        print(f"Using forced epoch idx {args.epoch_idx} (conf {cands[0][1]:.3f})")
    else:
        print(f"Collecting {args.n_candidates} {args.stage} candidates (conf ≥ {args.conf_thresh})...")
        cands = collect_stage_candidates(ds, score_fn, device, stage_idx,
                                         args.n_candidates, args.conf_thresh, args.seed)
    if not cands:
        sys.exit("No candidate epochs found; lower --conf-thresh")
    print(f"  {len(cands)} candidates")

    x_batch = torch.stack([c[0] for c in cands])
    exp = SpectralGradients(f=score_fn, fs=gq.FS, freq_step=gq.SG_FREQ_STEP, steps=gq.SG_STEPS,
                            path="shapley", n_perms=gq.SG_PERMS, target=stage_idx,
                            expects_batch=True)
    # explain in chunks: SG is per-sample, so the result is identical to one batch,
    # but 40 candidates x 101 bands (Δf = 0.5 Hz) do not fit a 24 GB GPU at once
    attr = torch.cat([exp(x_batch[i:i + args.chunk]) for i in range(0, len(x_batch), args.chunk)])
    if torch.is_complex(attr):
        attr = attr.abs()
    attr = attr.detach().cpu()  # (n, n_bands, TOTAL_LEN)

    # pick the epoch whose map is most concentrated in the stage's defining band
    band = DEFINING_BAND[args.stage]
    n_bands = attr.shape[1]
    freqs = np.arange(n_bands) * gq.SG_FREQ_STEP + gq.SG_FREQ_STEP / 2
    c0 = (gq.L // 2) * gq.N_TIMES
    shares = [band_share(freqs, a[:, c0:c0 + gq.N_TIMES].abs().sum(dim=1).numpy(), band)
              for a in attr]
    best = int(np.argmax(shares))
    print(f"Selected candidate {best} (dataset idx {cands[best][2]}, conf {cands[best][1]:.3f}, "
          f"{band}-share {shares[best]:.2f}; median {band}-share over candidates {np.median(shares):.2f})")

    out_shares = plot(cands[best][0], attr[best], args.stage, cands[best][1], args.out)
    print("Band shares of the shown epoch:", {k: round(v, 3) for k, v in out_shares.items()})


if __name__ == "__main__":
    main()
