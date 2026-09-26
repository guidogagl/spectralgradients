"""Generate qualitative sleep staging figures for supplementary materials.

Produces:
  - figures/sleep_classwise_profiles.pdf  (global freq-marginal profiles)
  - figures/sleep_example_N3.pdf           (instance-level N3)
  - figures/sleep_example_Wake.pdf         (instance-level Wake)
  - figures/sleep_example_N2.pdf           (instance-level N2)
  - figures/sleep_examples.pdf             (raw EEG epochs, 1 per class)

Usage::
    python sleep/generate_qualitative_figures.py [--gpu 0] [--n-per-class 30]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

from physioex.models import load_from_pretrained
from physioex.data.datasets import get_dataset
from physioex.data.presets import get_preset
from physioex.explain.posthoc.spectralgradients import SpectralGradients
from physioex.explain.posthoc.vistdft import STFTIntegratedGradients

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# Clinical EEG bands (Hz)
BANDS = {
    "δ": (0.5, 4.0),
    "θ": (4.0, 8.0),
    "α": (8.0, 13.0),
    "σ": (12.0, 16.0),
    "β": (16.0, 30.0),
}
BAND_COLORS = {
    "δ": "#4878CF",  # blue
    "θ": "#6ACC65",  # green
    "α": "#D65F5F",  # orange-red
    "σ": "#B47CC7",  # purple
    "β": "#C44E52",  # red
}

# Model / data config (TsinalisCNN on SleepEDF)
MODEL_NAME = "tsinalis-2016"
DATASET = "sleepedf"
PIPELINE = "identity"
FS = 100.0
L = 5          # sequence length (epochs)
N_TIMES = 3000  # samples per epoch
TOTAL_LEN = L * N_TIMES  # 15000

# SG config
SG_FREQ_STEP = 2.0
SG_STEPS = 10
SG_PERMS = 20

# STFT-IG-bal config
STFT_NFFT = 128
STFT_HOP = 32

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_score_fn(model):
    """Build a flat-signal score function."""
    def score_fn(x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, L, 1, N_TIMES)
        out = model(x)
        return out.squeeze(1).softmax(dim=-1)
    return score_fn


# ---------------------------------------------------------------------------
# Balanced sample collection
# ---------------------------------------------------------------------------

def collect_balanced_samples(dataset, score_fn, device, n_per_class=30,
                             conf_thresh=0.6, seed=42):
    """Collect n_per_class high-confidence correctly-classified samples per class."""
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset)).tolist()

    class_samples = defaultdict(list)  # class -> [(x, conf, ds_idx), ...]
    target = n_per_class * len(STAGE_NAMES)

    print(f"Collecting {n_per_class} samples per class (conf > {conf_thresh})...")

    for i, idx in enumerate(indices):
        # Check if all classes are full
        if sum(len(v) for v in class_samples.values()) >= target:
            break
        if i % 1000 == 0:
            counts = {STAGE_NAMES[c]: len(v) for c, v in class_samples.items()}
            print(f"  scanned {i}/{len(indices)}, found {counts}")

        item = dataset[idx]
        if item is None:
            continue

        labels = item["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        central_label = int(labels[L // 2])

        if central_label < 0 or central_label > 4:
            continue
        if len(class_samples[central_label]) >= n_per_class:
            continue

        sigs = []
        for k in sorted(item["signals"].keys()):
            sigs.append(item["signals"][k])
        x_seq = torch.stack(sigs, dim=-2)
        x_flat = x_seq[:, 0, :].reshape(-1)

        if x_flat.shape[0] != TOTAL_LEN:
            continue

        x_dev = x_flat.to(device)
        with torch.no_grad():
            p = score_fn(x_dev.unsqueeze(0))
            conf = p[0, central_label].item()
            pred = p.argmax(dim=-1).item()

        if conf >= conf_thresh and pred == central_label:
            class_samples[central_label].append((x_dev, conf, idx))

    # Report final counts
    print("Final sample counts:")
    for c in range(5):
        n = len(class_samples[c])
        print(f"  {STAGE_NAMES[c]}: {n}/{n_per_class}")
        if n < n_per_class:
            print(f"    WARNING: only {n} samples found for {STAGE_NAMES[c]}")

    return class_samples


# ---------------------------------------------------------------------------
# Attribution computation
# ---------------------------------------------------------------------------

def compute_sg_attributions(class_samples, score_fn, device):
    """Compute SG attributions for all samples, grouped by class."""
    print("\nComputing SG attributions (Shapley, K=20, Δf=2 Hz)...")
    sg_attrs = {}  # class -> (n, n_bands, total_len)

    for y in sorted(class_samples.keys()):
        samples = class_samples[y]
        x_batch = torch.stack([s[0] for s in samples])  # (n, total_len)
        print(f"  {STAGE_NAMES[y]}: {len(samples)} samples...", end=" ", flush=True)

        exp = SpectralGradients(
            f=score_fn, fs=FS, freq_step=SG_FREQ_STEP,
            steps=SG_STEPS, path="shapley", n_perms=SG_PERMS,
            target=y, expects_batch=True,
        )
        attr = exp(x_batch)  # (n, n_bands, total_len)
        if torch.is_complex(attr):
            attr = attr.abs()
        sg_attrs[y] = attr.cpu()
        print(f"shape={attr.shape}", flush=True)

    return sg_attrs


def compute_ig_bal_attributions(class_samples, score_fn, device):
    """Compute STFT-IG-bal attributions for all samples, grouped by class."""
    print("\nComputing STFT-IG-bal attributions (n_fft=128)...")
    ig_attrs = {}

    exp = STFTIntegratedGradients(
        f=score_fn, n_fft=STFT_NFFT, length=TOTAL_LEN,
        hop_length=STFT_HOP, real_valued=True,
        expects_batch=True, steps=SG_STEPS,
    )

    for y in sorted(class_samples.keys()):
        samples = class_samples[y]
        x_batch = torch.stack([s[0] for s in samples])
        print(f"  {STAGE_NAMES[y]}: {len(samples)} samples...", end=" ", flush=True)

        # Set target class
        exp.target = y
        attr = exp(x_batch)  # (n, F, T)
        ig_attrs[y] = attr.detach().cpu()
        print(f"shape={attr.shape}", flush=True)

    return ig_attrs


# ---------------------------------------------------------------------------
# Frequency marginal computation
# ---------------------------------------------------------------------------

def sg_freq_marginal(attr, fs=FS, freq_step=SG_FREQ_STEP):
    """Compute frequency-marginal attribution from SG.

    Args:
        attr: (n_bands, total_len) or (n, n_bands, total_len)

    Returns:
        freqs: (n_bands,) frequency centers
        marginal: (n_bands,) or (n, n_bands) sum over time of absolute attributions
    """
    n_bands = attr.shape[-2]
    freqs = torch.arange(n_bands) * freq_step + freq_step / 2
    marginal = attr.abs().sum(dim=-1)  # sum over time
    return freqs.numpy(), marginal.detach().numpy()


def stft_freq_marginal(attr, fs=FS, n_fft=STFT_NFFT):
    """Compute frequency-marginal attribution from STFT-IG.

    Args:
        attr: (F, T) or (n, F, T)

    Returns:
        freqs: (F,) frequency centers
        marginal: (F,) or (n, F) sum over time of absolute attributions
    """
    F = attr.shape[-2]
    freqs = torch.linspace(0, fs / 2, F)
    marginal = attr.abs().sum(dim=-1)
    return freqs.numpy(), marginal.detach().numpy()


def compute_band_percentages(freqs, marginal):
    """Compute % of total attribution in each clinical EEG band.

    Args:
        freqs: (n_freq,) frequency bin centers
        marginal: (n_freq,) or (n, n_freq) frequency-marginal attribution

    Returns:
        dict: band_name -> percentage (averaged over samples if batched)
    """
    if marginal.ndim == 1:
        marginal = marginal[np.newaxis, :]

    pcts = {}
    for band_name, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        band_sum = np.abs(marginal[:, mask]).sum(axis=1)
        total = np.abs(marginal).sum(axis=1)
        total = np.where(total == 0, 1, total)
        pcts[band_name] = float((band_sum / total * 100).mean())
    return pcts


# ---------------------------------------------------------------------------
# Plotting: classwise profiles
# ---------------------------------------------------------------------------

def plot_classwise_profiles(sg_attrs, ig_attrs, output_path):
    """Generate the classwise frequency-marginal profiles figure."""
    fig, axes = plt.subplots(5, 2, figsize=(12, 14), constrained_layout=True)

    for row, y in enumerate(range(5)):
        # SG column
        ax_sg = axes[row, 0]
        sg_freqs, sg_marg = sg_freq_marginal(sg_attrs[y])
        _plot_band_bars(ax_sg, sg_freqs, sg_marg, f"{STAGE_NAMES[y]} — SG")

        # IG-bal column
        ax_ig = axes[row, 1]
        ig_freqs, ig_marg = stft_freq_marginal(ig_attrs[y])
        _plot_band_bars(ax_ig, ig_freqs, ig_marg, f"{STAGE_NAMES[y]} — STFT-IG-bal")

    axes[0, 0].set_title("SG (Shapley, K=20, Δf=2 Hz)", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("STFT-IG-bal (n_fft=128)", fontsize=12, fontweight="bold")

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_band_bars(ax, freqs, marginal, title, max_freq=30.0):
    """Plot frequency-marginal bars colored by clinical band with SEM error bars.

    Args:
        ax: matplotlib Axes
        freqs: (n_freq,) frequency bin centers
        marginal: (n_samples, n_freq) frequency-marginal attributions
    """
    # Filter to max_freq
    mask = freqs <= max_freq
    freqs = freqs[mask]
    marginal = marginal[:, mask] if marginal.ndim > 1 else marginal[mask]

    # Mean and SEM
    if marginal.ndim > 1:
        mean = marginal.mean(axis=0)
        sem = marginal.std(axis=0) / np.sqrt(marginal.shape[0])
    else:
        mean = marginal
        sem = np.zeros_like(mean)

    # Normalize to sum=1 for comparability
    total = np.abs(mean).sum()
    if total > 0:
        mean = mean / total
        sem = sem / total

    # Color each bar by its band
    colors = []
    for f in freqs:
        color = "#999999"  # default gray
        for band_name, (flo, fhi) in BANDS.items():
            if flo <= f < fhi:
                color = BAND_COLORS[band_name]
                break
        colors.append(color)

    bar_width = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    ax.bar(freqs, mean, width=bar_width * 0.9, color=colors, edgecolor="none",
           alpha=0.8)
    ax.errorbar(freqs, mean, yerr=sem, fmt="none", ecolor="black", elinewidth=0.5,
                capsize=1.5, capthick=0.5)

    ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized attribution")
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Plotting: instance-level examples
# ---------------------------------------------------------------------------

def plot_instance_example(signal, sg_attr, ig_attr, stage_name, output_path):
    """Generate a single instance-level example figure.

    Layout:
      Row 0: Raw EEG signal (central epoch)
      Row 1: SG heatmap (time-frequency)
      Row 2: STFT-IG-bal heatmap
      Row 3: Frequency-marginal profiles (SG and IG-bal side by side)
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10),
                             gridspec_kw={"height_ratios": [1, 1.5, 1.5, 1]},
                             constrained_layout=True)

    # Central epoch boundaries (in seconds)
    central_start = (L // 2) * N_TIMES
    central_end = central_start + N_TIMES
    t_signal = np.arange(N_TIMES) / FS

    # Row 0: Raw signal (central epoch)
    ax = axes[0]
    sig_epoch = signal[central_start:central_end].cpu().numpy()
    ax.plot(t_signal, sig_epoch, color="black", linewidth=0.5)
    ax.set_xlim(0, 30)
    ax.set_ylabel("Amplitude (μV)")
    ax.set_title(f"{stage_name} — Raw EEG (central epoch)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Row 1: SG heatmap
    ax = axes[1]
    sg_2d = sg_attr.cpu().numpy()  # (n_bands, total_len)
    n_bands = sg_2d.shape[0]
    # Show only central epoch
    sg_epoch = sg_2d[:, central_start:central_end]
    freq_extent = n_bands * SG_FREQ_STEP
    im = ax.imshow(sg_epoch, aspect="auto", origin="lower",
                   extent=[0, 30, 0, freq_extent],
                   cmap="RdBu_r", interpolation="bilinear")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("SG attribution", fontsize=10)
    ax.set_ylim(0, min(30, freq_extent))
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    # Row 2: IG-bal heatmap
    ax = axes[2]
    ig_2d = ig_attr.cpu().numpy()  # (F, T)
    F_bins = ig_2d.shape[0]
    T_frames = ig_2d.shape[1]
    # Compute time extent for the full signal
    t_max_ig = TOTAL_LEN / FS  # 150s for L=5
    # Central epoch in IG time frames
    frame_start = int(central_start / TOTAL_LEN * T_frames)
    frame_end = int(central_end / TOTAL_LEN * T_frames)
    ig_epoch = ig_2d[:, frame_start:frame_end]
    im = ax.imshow(ig_epoch, aspect="auto", origin="lower",
                   extent=[0, 30, 0, FS / 2],
                   cmap="RdBu_r", interpolation="bilinear")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("STFT-IG-bal attribution", fontsize=10)
    ax.set_ylim(0, 30)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    # Row 3: Frequency marginal comparison
    ax = axes[3]
    sg_freqs, sg_marg = sg_freq_marginal(sg_attr.unsqueeze(0))
    ig_freqs, ig_marg = stft_freq_marginal(ig_attr.unsqueeze(0))
    # Normalize
    sg_marg_n = sg_marg[0] / (sg_marg[0].sum() + 1e-12)
    ig_marg_n = ig_marg[0] / (ig_marg[0].sum() + 1e-12)

    sg_mask = sg_freqs <= 30
    ig_mask = ig_freqs <= 30

    ax.bar(sg_freqs[sg_mask] - 0.4, sg_marg_n[sg_mask], width=0.8,
           alpha=0.7, label="SG", color="#4878CF")
    ax.bar(ig_freqs[ig_mask] + 0.4, ig_marg_n[ig_mask], width=0.8,
           alpha=0.7, label="IG-bal", color="#D65F5F")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized attribution")
    ax.set_title("Frequency-marginal profile", fontsize=10)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plotting: raw EEG examples
# ---------------------------------------------------------------------------

def plot_raw_examples(class_samples, output_path):
    """Plot one raw EEG epoch per sleep stage."""
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), constrained_layout=True)

    for y in range(5):
        ax = axes[y]
        # Pick the highest-confidence sample
        samples = class_samples[y]
        best = max(samples, key=lambda s: s[1])
        x = best[0].cpu().numpy()

        # Central epoch
        central_start = (L // 2) * N_TIMES
        central_end = central_start + N_TIMES
        epoch = x[central_start:central_end]
        t = np.arange(N_TIMES) / FS

        ax.plot(t, epoch, color="black", linewidth=0.5)
        ax.set_xlim(0, 30)
        ax.set_ylabel("μV")
        ax.set_title(STAGE_NAMES[y], fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Band percentage table
# ---------------------------------------------------------------------------

def print_band_table(sg_attrs, ig_attrs):
    """Print the band percentage table for tab:band_pct."""
    print("\n" + "=" * 80)
    print("BAND PERCENTAGE TABLE (for tab:band_pct in supplementary)")
    print("=" * 80)
    header = f"{'Class':<6} {'Method':<8}"
    for band in BANDS:
        header += f" {band:>8}"
    print(header)
    print("-" * 80)

    for y in range(5):
        # SG
        sg_freqs, sg_marg = sg_freq_marginal(sg_attrs[y])
        sg_pcts = compute_band_percentages(sg_freqs, sg_marg)
        row = f"{STAGE_NAMES[y]:<6} {'SG':<8}"
        for band in BANDS:
            row += f" {sg_pcts[band]:>7.1f}%"
        print(row)

        # IG-bal
        ig_freqs, ig_marg = stft_freq_marginal(ig_attrs[y])
        ig_pcts = compute_band_percentages(ig_freqs, ig_marg)
        row = f"{'':6} {'IG-bal':<8}"
        for band in BANDS:
            row += f" {ig_pcts[band]:>7.1f}%"
        print(row)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate sleep qualitative figures")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--n-per-class", type=int, default=30,
                        help="Samples per class for classwise analysis")
    parser.add_argument("--conf-thresh", type=float, default=0.6,
                        help="Minimum confidence threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = load_from_pretrained(MODEL_NAME, device=str(device))
    model.eval()
    score_fn = build_score_fn(model)

    # Load dataset
    print(f"Loading dataset: {DATASET}")
    DatasetCls = get_dataset(DATASET)
    pipeline = get_preset(PIPELINE)
    ds = DatasetCls(
        channels=["EEG"],
        pipelines=pipeline,
        sequence_length=L,
    )

    # Collect balanced samples
    class_samples = collect_balanced_samples(
        ds, score_fn, device,
        n_per_class=args.n_per_class,
        conf_thresh=args.conf_thresh,
        seed=args.seed,
    )

    # Compute attributions
    sg_attrs = compute_sg_attributions(class_samples, score_fn, device)
    ig_attrs = compute_ig_bal_attributions(class_samples, score_fn, device)

    # Generate figures
    print("\nGenerating figures...")

    # 1. Classwise profiles
    plot_classwise_profiles(
        sg_attrs, ig_attrs,
        FIGURES_DIR / "sleep_classwise_profiles.pdf"
    )

    # 2. Instance-level examples (N3, Wake, N2)
    for stage_idx, stage_name in [(3, "N3"), (0, "Wake"), (2, "N2")]:
        if stage_idx not in class_samples or not class_samples[stage_idx]:
            print(f"  Skipping {stage_name}: no samples")
            continue
        # Pick highest-confidence sample
        best_idx = max(range(len(class_samples[stage_idx])),
                       key=lambda i: class_samples[stage_idx][i][1])
        signal = class_samples[stage_idx][best_idx][0]
        sg_attr = sg_attrs[stage_idx][best_idx]
        ig_attr = ig_attrs[stage_idx][best_idx]
        plot_instance_example(
            signal, sg_attr, ig_attr, stage_name,
            FIGURES_DIR / f"sleep_example_{stage_name}.pdf"
        )

    # 3. Raw EEG examples
    plot_raw_examples(class_samples, FIGURES_DIR / "sleep_examples.pdf")

    # 4. Print band percentages
    print_band_table(sg_attrs, ig_attrs)

    print("\nDone!")


if __name__ == "__main__":
    main()
