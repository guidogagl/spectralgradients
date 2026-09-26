"""Generate K convergence plot: freq_infidelity vs K for 3 synthetic setups.

Outputs: figures/k_convergence.pdf (1x3 subplot)

Usage::
    python synt/k_convergence.py [--gpu 0] [--n-samples 50]
"""

import argparse
from pathlib import Path

import importlib.util
import matplotlib.pyplot as plt

# Figure drawn at elsarticle text width (137 mm)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 0.8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
MM2IN = 1 / 25.4
import numpy as np
import torch

# Import from synt/benchmark.py (file, not package synt/benchmark/)
_spec = importlib.util.spec_from_file_location(
    "synt_benchmark_main",
    str(Path(__file__).parent / "benchmark.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_model = _mod.load_model
build_score_fn = _mod.build_score_fn

from synt.benchmark.config import MODEL_CONFIGS
from synt.benchmark.samples import SampleCollector
from physioex.explain.posthoc.spectralgradients import SpectralGradients

SETUPS = ["synt-setup0", "synt-setup1", "synt-setup2"]
K_VALUES = [1, 2, 5, 10, 20, 50, 100, 200]
FREQ_STEP = 2.0
STEPS = 10
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def compute_freq_infidelity(attr, x_batch, score_fn, fs, freq_patch_size=10):
    """Compute frequency infidelity for a batch of attributions."""
    results = []
    for j in range(attr.shape[0]):
        a = attr[j]  # (n_bands, n_times)
        x = x_batch[j:j+1]
        # Frequency-marginal attribution
        freq_marginal = a.abs().sum(dim=-1)  # (n_bands,)

        def f_ablate(x_in):
            return score_fn(x_in)

        # Use the infidelity function from physioex
        inf_val = infidelity(
            freq_marginal, x, f_ablate,
            fs=fs, domain="frequency",
            patch_size=freq_patch_size,
        )
        results.append(inf_val)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-json", action="store_true",
                        help="re-plot from figures/k_convergence.json without recomputing")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for setup_name in ([] if args.from_json else SETUPS):

        print(f"\n{'='*60}")
        print(f"Setup: {setup_name}")
        print(f"{'='*60}")

        model_cfg = MODEL_CONFIGS[setup_name]
        model = load_model(setup_name, model_cfg, device)
        score_fn = build_score_fn(model, model_cfg["n_times"], device)

        # Collect samples
        config = {"max_samples": args.n_samples}
        collector = SampleCollector(model_cfg, config)
        samples = collector.collect(
            score_fn=score_fn,
            max_samples=args.n_samples,
            seed=args.seed,
        )

        # Group by class for target-aware SG
        from collections import defaultdict
        class_groups = defaultdict(list)
        for x, y in samples:
            class_groups[y].append(x)

        setup_results = {}

        for K in K_VALUES:
            print(f"  K={K}...", end=" ", flush=True)
            infidelity_values = []

            for y, x_list in class_groups.items():
                x_batch = torch.stack(x_list)

                exp = SpectralGradients(
                    f=score_fn, fs=model_cfg["fs"],
                    freq_step=FREQ_STEP, steps=STEPS,
                    path="shapley", n_perms=K,
                    target=y, expects_batch=True,
                )
                attr = exp(x_batch)
                if torch.is_complex(attr):
                    attr = attr.abs()

                # Compute freq infidelity per sample
                for j in range(attr.shape[0]):
                    a = attr[j]  # (n_bands, n_times)
                    freq_marg = a.abs().sum(dim=-1)  # (n_bands,)
                    # ROAR-style frequency ablation
                    xj = x_batch[j:j+1]
                    n = xj.shape[-1]
                    xdft = torch.fft.rfft(xj, dim=-1)
                    n_freqs = xdft.shape[-1]
                    bin_step = max(1, round(FREQ_STEP / (model_cfg["fs"] / n)))
                    n_bands = len(range(0, n_freqs, bin_step))

                    # Rank bands by importance (descending)
                    rank = torch.argsort(freq_marg, descending=True)

                    # Compute degradation curve
                    with torch.no_grad():
                        out_orig = score_fn(xj)
                        if out_orig.dim() == 1:
                            f_orig = out_orig[y].item()
                        else:
                            f_orig = out_orig[0, y].item()
                        curve = [1.0]
                        xdft_ablated = xdft.clone()
                        for idx in range(n_bands):
                            band_i = rank[idx].item()
                            si = band_i * bin_step
                            ei = min(si + bin_step, n_freqs)
                            xdft_ablated[..., si:ei] = 0.0
                            x_abl = torch.fft.irfft(xdft_ablated, n=n, dim=-1)
                            out_abl = score_fn(x_abl)
                            if out_abl.dim() == 1:
                                f_abl = out_abl[y].item()
                            else:
                                f_abl = out_abl[0, y].item()
                            curve.append(f_abl / (f_orig + 1e-12))
                        # AUC via trapezoidal rule
                        curve = np.array(curve)
                        curve = np.clip(curve, 0, 1)
                        auc = getattr(np, "trapezoid", getattr(np, "trapz", None))(curve, dx=1.0/n_bands)

                    infidelity_values.append(auc)

            median_inf = np.median(infidelity_values)
            setup_results[K] = {
                "values": infidelity_values,
                "median": median_inf,
                "mean": np.mean(infidelity_values),
                "std": np.std(infidelity_values),
            }
            print(f"median={median_inf:.4f} (n={len(infidelity_values)})")

        all_results[setup_name] = setup_results

    import json
    json_path = FIGURES_DIR / "k_convergence.json"
    if args.from_json:
        with open(json_path) as fh:
            raw = json.load(fh)
        all_results = {s_: {int(k): v for k, v in r.items()} for s_, r in raw.items()}
    else:
        with open(json_path, "w") as fh:
            json.dump({s_: {str(k): {"median": float(v["median"]), "values": [float(x) for x in v["values"]]}
                            for k, v in r.items()} for s_, r in all_results.items()}, fh)
        print(f"Saved raw results: {json_path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(137 * MM2IN, 48 * MM2IN), constrained_layout=True, sharey=True)

    for ax, setup_name in zip(axes, SETUPS):
        results = all_results[setup_name]
        medians = [results[K]["median"] for K in K_VALUES]
        q25 = [np.percentile(results[K]["values"], 25) for K in K_VALUES]
        q75 = [np.percentile(results[K]["values"], 75) for K in K_VALUES]

        ax.plot(K_VALUES, medians, "o-", color="#4878CF", linewidth=1.0, markersize=3)
        ax.fill_between(K_VALUES, q25, q75, alpha=0.2, color="#4878CF")
        ax.set_xlabel("$K$ (permutations)")
        ax.set_title("Setup " + setup_name.replace("synt-", "").replace("setup", ""), fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xscale("log")
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.minorticks_off()

    axes[0].set_ylabel("Freq. Infidelity (median)")

    output_path = FIGURES_DIR / "k_convergence.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(str(output_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
