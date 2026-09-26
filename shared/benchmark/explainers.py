"""Shared explainer execution logic for benchmarking across domains.

Provides run_sg() and run_stft() functions used by synt, audio, arrhythmia, and sleep benchmarks.
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from physioex.explain.posthoc.spectralgradients import SpectralGradients
from physioex.explain.posthoc.vistdft import (
    STFTIntegratedGradients, STFTSaliency, STFTInputXGradient,
)

from shared.benchmark.metrics import compute_all_metrics


STFT_CLASSES = {
    "IG": STFTIntegratedGradients,
    "Sal": STFTSaliency,
    "IxG": STFTInputXGradient,
}


def _save_attributions(
    attributions: List[torch.Tensor],
    labels: List[int],
    output_file: Path,
    metadata: dict
):
    """Save attributions to disk for later analysis.

    Args:
        attributions: List of attribution tensors (each: n_bands x total_len or similar)
        labels: List of corresponding labels
        output_file: Path to save attributions (.pt format)
        metadata: Additional metadata (config, etc.)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Stack attributions into single tensor (pad if needed)
    # For now, assume all have same shape
    attrs_stacked = torch.stack([a.cpu() for a in attributions])

    payload = {
        "attributions": attrs_stacked,
        "labels": torch.tensor(labels),
        **metadata
    }

    torch.save(payload, output_file)
    print(f"  Saved attributions to {output_file}", flush=True)


def run_sg(
    samples: List[Tuple[torch.Tensor, int]],
    score_fn: callable,
    config: dict,
    model_cfg: dict,
    device: torch.device,
    save_attributions: bool = False,
    attribution_file: Optional[Path] = None,
    checkpoint_file: Optional[Path] = None
) -> dict:
    """Run SpectralGradients explainer on samples.

    Args:
        samples: List of (x_tensor, label) tuples
        score_fn: Score function
        config: SG configuration dict
        model_cfg: Model configuration (fs, n_times, etc.)
        device: Device for computation
        save_attributions: Whether to save raw attributions to disk
        attribution_file: Path to save attributions (required if save_attributions=True)
        checkpoint_file: Explicit per-run checkpoint path. MUST be unique per
            parallel job (e.g. per chunk) — otherwise concurrent chunk jobs
            share one checkpoint and corrupt each other's class-skip state.

    Returns:
        Dict with metric results for each sample
    """
    fs = model_cfg["fs"]
    n_times = model_cfg["n_times"]
    total_len = model_cfg["sequence_length"] * n_times

    # Pre-allocate results
    n_total = len(samples)
    metric_names = ["freq_infidelity", "freq_complexity",
                    "time_infidelity", "time_complexity", "tfc_tf_spread"]
    results = {m: [None] * n_total for m in metric_names}

    # Storage for attributions if saving
    all_attributions = [None] * n_total
    all_labels = [None] * n_total

    t0 = time.time()

    # Group samples by class for batched processing
    class_groups = defaultdict(list)
    for si, (x, y) in enumerate(samples):
        class_groups[y].append((si, x))

    # Precompute max STFT freq size for interpolation
    max_stft_freq = max(w["n_fft"] // 2 + 1 for w in model_cfg["stft_windows"].values())

    # Checkpoint: load partial results if available.
    # Prefer an explicit per-run path (unique per chunk). Fall back to the
    # attribution dir, then cwd — the fallbacks are NOT safe for concurrent
    # chunk jobs, so callers running chunks must pass checkpoint_file.
    if checkpoint_file is not None:
        checkpoint_file = Path(checkpoint_file)
    elif attribution_file:
        checkpoint_file = attribution_file.parent / "_checkpoint_SG.pt"
    else:
        # Fallback: use current directory
        checkpoint_file = Path("_checkpoint_SG.pt")
    completed_classes = set()
    if checkpoint_file and checkpoint_file.exists():
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        results = ckpt["results"]
        completed_classes = set(ckpt["completed_classes"])
        print(f"  Resumed from checkpoint: {len(completed_classes)} classes done", flush=True)

    processed = sum(len(g) for y, g in class_groups.items() if y in completed_classes)
    for y, group in class_groups.items():
        if y in completed_classes:
            print(f"  class {y}: SKIPPED (checkpoint)", flush=True)
            continue

        indices = [g[0] for g in group]
        x_batch = torch.stack([g[1] for g in group])  # (Gc, total_len)

        # Split into sub-batches if needed
        sg_bs = config["batch_size"] if config["batch_size"] > 0 else len(group)

        for chunk_start in range(0, len(group), sg_bs):
            chunk_end = min(chunk_start + sg_bs, len(group))
            xc = x_batch[chunk_start:chunk_end]
            chunk_indices = indices[chunk_start:chunk_end]

            # Create explainer
            exp = SpectralGradients(
                f=score_fn, fs=fs, freq_step=config["freq_step"],
                steps=config["steps"], path=config["path"],
                n_perms=config["perms"], target=y, expects_batch=True,
            )
            attr = exp(xc)  # (Bc, n_bands, total_len)
            if torch.is_complex(attr):
                attr = attr.abs()

            band_freqs = exp.band_frequencies(total_len)

            # Compute metrics per sample in chunk
            for j in range(len(chunk_indices)):
                si = chunk_indices[j]
                a2 = attr[j]  # (n_bands, total_len)
                xb = xc[j:j+1]

                # Store attribution if saving (detach and move to CPU to avoid GPU OOM)
                if save_attributions:
                    all_attributions[si] = a2.detach().cpu()
                    all_labels[si] = y

                def f_target(xin, _y=y):
                    return score_fn(xin)[:, _y]

                metric_results = compute_all_metrics(
                    a2, xb, f_target, fs,
                    band_freqs=band_freqs,
                    total_len=total_len,
                    max_stft_freq=max_stft_freq,
                    config=config.get("metrics", {})
                )

                for m in metric_names:
                    results[m][si] = metric_results[m]

            processed += len(chunk_indices)
            del attr, exp
            torch.cuda.empty_cache()

        completed_classes.add(y)
        print(f"  class {y}: {len(group)} samples ({processed}/{n_total} total) {time.time()-t0:.1f}s",
              flush=True)

        # Save checkpoint after each class
        if checkpoint_file:
            torch.save({"results": results, "completed_classes": list(completed_classes)},
                       checkpoint_file)
            print(f"  [checkpoint saved: {len(completed_classes)} classes]", flush=True)

    # Save attributions if requested
    if save_attributions and attribution_file:
        metadata = {
            "explainer": "SG",
            "n_bands": all_attributions[0].shape[0] if all_attributions[0] is not None else None,
            "total_len": total_len,
            "fs": fs,
            "config": config,
        }
        _save_attributions(all_attributions, all_labels, attribution_file, metadata)

    # Clean up checkpoint after successful completion
    if checkpoint_file and checkpoint_file.exists():
        checkpoint_file.unlink()
        print("  [checkpoint cleaned up]", flush=True)

    return results


def run_stft(
    samples: List[Tuple[torch.Tensor, int]],
    score_fn: callable,
    config: dict,
    window_cfg: dict,
    model_cfg: dict,
    device: torch.device,
    save_attributions: bool = False,
    attribution_file: Optional[Path] = None
) -> dict:
    """Run STFT-based explainer on samples.

    Args:
        samples: List of (x_tensor, label) tuples
        score_fn: Score function
        config: STFT configuration dict
        window_cfg: Window configuration (n_fft, hop)
        model_cfg: Model configuration (fs, n_times, etc.)
        device: Device for computation
        save_attributions: Whether to save raw attributions to disk
        attribution_file: Path to save attributions (required if save_attributions=True)

    Returns:
        Dict with metric results for each sample
    """
    fs = model_cfg["fs"]
    n_times = model_cfg["n_times"]
    total_len = model_cfg["sequence_length"] * n_times

    # Pre-allocate results
    n_total = len(samples)
    metric_names = ["freq_infidelity", "freq_complexity",
                    "time_infidelity", "time_complexity", "tfc_tf_spread"]
    results = {m: [None] * n_total for m in metric_names}

    # Storage for attributions if saving
    all_attributions = [None] * n_total
    all_labels = [None] * n_total

    t0 = time.time()

    # Precompute max STFT freq size for interpolation
    max_stft_freq = max(w["n_fft"] // 2 + 1 for w in model_cfg["stft_windows"].values())

    for si, (x, y) in enumerate(samples):
        xb = x.unsqueeze(0)

        def f_target(xin, _y=y):
            return score_fn(xin)[:, _y]

        # Get explainer class from config
        method_name = config.get("method", "IG")  # Default to IG
        cls = STFT_CLASSES[method_name]

        kw = dict(
            f=score_fn, n_fft=window_cfg["n_fft"], length=total_len,
            hop_length=window_cfg["hop"], win_length=window_cfg["n_fft"],
            target=y, expects_batch=True,
            real_valued=config.get("real_valued", True)
        )
        if cls is STFTIntegratedGradients:
            kw["steps"] = config.get("steps", 10)

        exp = cls(**kw)
        attr = exp(xb)
        if torch.is_complex(attr):
            attr = attr.abs()

        attr_2d = attr.squeeze(0)

        # Store attribution if saving (detach and move to CPU to avoid GPU OOM)
        if save_attributions:
            all_attributions[si] = attr_2d.detach().cpu()
            all_labels[si] = y

        # Compute metrics
        freq_attr = attr.sum(dim=-1).squeeze(0)  # (n_freq,)
        time_attr = attr.sum(dim=-2).squeeze(0)  # (total_len,)

        # Interpolation is now handled by compute_all_metrics
        freq_attr_c = freq_attr
        time_attr_c = time_attr

        # Compute metrics
        metric_results = compute_all_metrics(
            attr_2d, xb, f_target, fs,
            band_freqs=None,  # STFT computes its own
            total_len=total_len,
            max_stft_freq=max_stft_freq,
            config=config.get("metrics", {})
        )

        for m in metric_names:
            results[m][si] = metric_results[m]

        if (si + 1) % 10 == 0 or si == 0:
            print(f"  [{si+1}/{n_total}] {time.time()-t0:.1f}s", flush=True)
        # Periodically clear GPU cache to prevent fragmentation on long runs
        if (si + 1) % 100 == 0:
            torch.cuda.empty_cache()

    # Save attributions if requested
    if save_attributions and attribution_file:
        metadata = {
            "explainer": f"STFT-{config.get('method', 'IG')}",
            "window": window_cfg,
            "total_len": total_len,
            "fs": fs,
            "config": config,
        }
        _save_attributions(all_attributions, all_labels, attribution_file, metadata)

    return results
