"""Benchmark SG vs STFT explainers on pretrained sleep models.

Usage::
    # Using YAML config (recommended)
    python sleep/benchmark.py --config configs/sleep/tsinalis-2016_sg.yaml

    # Using CLI arguments
    python sleep/benchmark.py --model tsinalis-2016 --explainer SG --max-samples 200

    # Override specific config values
    python sleep/benchmark.py --config configs/sleep/tsinalis-2016_sg.yaml --max-samples 50
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

from physioex.models import load_from_pretrained
from physioex.data.datasets import get_dataset
from physioex.data.presets import get_preset

from sleep.benchmark.config import (
    MODEL_CONFIGS, ALL_EXPLAINERS, load_config
)
from sleep.benchmark.samples import SampleCollector
from shared.benchmark import run_sg, run_stft, STFT_CLASSES


def build_score_fn(model, L, C, n_times):
    """Build a flat-signal score_fn for SG/STFT explainers.

    Args:
        model: Pretrained model
        L: Sequence length
        C: Number of channels (should be 1)
        n_times: Number of time steps

    Returns:
        (score_fn, total_len) tuple
    """
    total = L * n_times

    def score_fn(x):
        # x: (B, total) flat concatenated signal
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, L, C, n_times)
        out = model(x)  # (B, 1, n_classes)
        return out.squeeze(1).softmax(dim=-1)  # (B, n_classes)

    return score_fn, total


def parse_args():
    """Parse command line arguments.

    Most configuration should be in YAML files. CLI args are for overrides.
    """
    parser = argparse.ArgumentParser(
        description="Sleep SG vs STFT benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (recommended way to configure)
    parser.add_argument(
        "--config", type=Path,
        help="YAML configuration file"
    )

    # Common overrides
    parser.add_argument("--model", help="Model name (overrides config)")
    parser.add_argument(
        "--explainer", default=None,
        help="Explainer name or 'all' (overrides config)"
    )
    parser.add_argument("--max-samples", type=int, help="Override max_samples")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--data-root", type=Path, help="Override data root")

    # Chunking: split the (shared) sample set into N strided partitions and run
    # only partition `chunk-idx`. Enables parallel per-chunk jobs for slow
    # explainers (SG/IG on the full tsinalis test set); merge with merge_chunks.py.
    parser.add_argument("--chunk-idx", type=int, default=0, help="Chunk index (0..chunk-n-1)")
    parser.add_argument("--chunk-n", type=int, default=1, help="Number of chunks (1 = no chunking)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration from YAML and merge with CLI overrides
    cli_overrides = {
        "model": args.model,
        "explainers": [args.explainer] if args.explainer else None,
        "max_samples": args.max_samples,
        "gpu": args.gpu,
        "seed": args.seed,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "data_root": str(args.data_root) if args.data_root else None,
    }

    cfg = load_config(args.config, cli_overrides)

    # Determine output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(os.environ.get("SG_RESULTS", "results")) / cfg["model"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine explainers to run
    if cfg["explainers"]:
        explainers_to_run = cfg["explainers"]
    else:
        explainers_to_run = ALL_EXPLAINERS

    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    print(f"Model: {cfg['model']}")
    print(f"Explainers: {explainers_to_run}")
    print(f"Samples: {cfg['max_samples']}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # Load model
    model = load_from_pretrained(cfg["model"], device=str(device))
    model.eval()

    model_cfg = MODEL_CONFIGS[cfg["model"]]
    L = model_cfg["sequence_length"]
    fs = model_cfg["fs"]
    n_times = model_cfg["n_times"]
    score_fn, total_len = build_score_fn(model, L, 1, n_times)

    # Load dataset
    DatasetCls = get_dataset(model_cfg["dataset"])
    pipeline = get_preset(model_cfg["pipeline"], **model_cfg.get("pipeline_kwargs", {}))
    ds_kwargs = dict(model_cfg["dataset_kwargs"])

    # Dataset location: --data-root / data_root in the YAML, otherwise physioex resolves
    # $PHYSIOEX_DATA/<dataset subdir>. Nothing is downloaded automatically.
    if cfg.get("data_root"):
        ds_kwargs["root"] = cfg["data_root"]
    elif not os.environ.get("PHYSIOEX_DATA"):
        sys.exit("PHYSIOEX_DATA is not set. Point it to the directory holding the raw datasets "
                 "(<dir>/physionet-sleep-data/*.edf for Sleep-EDF, <dir>/MASS/Original/SS03/ for MASS) "
                 "or pass --data-root <dataset dir>.")

    ds = DatasetCls(
        channels=model_cfg["channels"],
        pipelines=pipeline,
        sequence_length=L,
        **ds_kwargs,
    )

    # Collect samples
    cache_file = out_dir / f"samples_cache_n{cfg['max_samples']}_s{cfg['seed']}.json"

    collector = SampleCollector(ds, score_fn, model_cfg)
    samples = collector.collect(
        max_samples=cfg["max_samples"],
        seed=cfg["seed"],
        device=device,
        cache_file=cache_file,
        conf_thresh=cfg["metrics"]["conf_thresh"]
    )

    if not samples:
        print("No valid samples found!")
        return

    # Save samples if requested (save once, shared by all explainers)
    if cfg.get("output", {}).get("save_samples", False):
        samples_file = out_dir / f"samples_n{cfg['max_samples']}_s{cfg['seed']}.pt"
        collector.save_samples(samples, samples_file)

    # Optional chunking: keep only a strided partition of the shared sample set.
    # Strided slicing keeps the per-chunk class mix balanced; merge_chunks.py
    # concatenates the per-sample metric arrays back into the full result.
    suffix = ""
    if args.chunk_n and args.chunk_n > 1:
        n_full = len(samples)
        samples = samples[args.chunk_idx::args.chunk_n]
        suffix = f"_chunk{args.chunk_idx}of{args.chunk_n}"
        print(f"Chunk {args.chunk_idx}/{args.chunk_n}: {len(samples)}/{n_full} samples", flush=True)

    # Get output settings
    output_cfg = cfg.get("output", {})
    save_attr = output_cfg.get("save_attributions", False)

    # Run each explainer
    for ename in explainers_to_run:
        print(f"\nRunning {ename}...", flush=True)
        t0 = time.time()

        # Generate attribution file path if saving
        attr_file = None
        if save_attr:
            if ename == "SG":
                tag = f"SG-{cfg['sg']['path']}-fs{cfg['sg']['freq_step']}-s{cfg['sg']['steps']}-p{cfg['sg']['perms']}"
                attr_file = out_dir / f"{tag}_attributions.pt"
            else:
                attr_file = out_dir / f"{ename}_attributions.pt"

        if ename == "SG":
            results = run_sg(samples, score_fn, cfg["sg"], model_cfg, device,
                           save_attributions=save_attr, attribution_file=attr_file,
                           checkpoint_file=out_dir / f"_checkpoint_SG{suffix}.pt")
            tag = f"SG-{cfg['sg']['path']}-fs{cfg['sg']['freq_step']}-s{cfg['sg']['steps']}-p{cfg['sg']['perms']}"
            out_file = out_dir / f"{tag}{suffix}.json"
        else:
            # Parse explainer name: "method-window" e.g., "IG-bal"
            method_name, window_name = ename.split("-")
            window_cfg = model_cfg["stft_windows"][window_name]

            # Update STFT config with method name
            stft_cfg = {**cfg["stft"], "method": method_name}

            results = run_stft(samples, score_fn, stft_cfg, window_cfg, model_cfg, device,
                             save_attributions=save_attr, attribution_file=attr_file)
            out_file = out_dir / f"{ename}{suffix}.json"

        dt = time.time() - t0

        # Save results
        payload = {
            "model": cfg["model"],
            "explainer": ename,
            "n_samples": len(samples),
            "config": {
                "fs": fs, "L": L, "n_times": n_times, "total_len": total_len,
                "seed": cfg["seed"],
            },
            "metrics": results,
            "time_seconds": dt,
        }

        # Add explainer-specific config
        if ename == "SG":
            payload["config"].update({
                "sg_path": cfg["sg"]["path"],
                "sg_steps": cfg["sg"]["steps"],
                "sg_perms": cfg["sg"]["perms"],
                "freq_step": cfg["sg"]["freq_step"],
            })
        else:
            payload["config"].update({
                "window": window_cfg,
                "real_valued": cfg["stft"]["real_valued"],
            })

        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)

        # Print summary
        metric_names = ["freq_infidelity", "freq_complexity",
                        "time_infidelity", "time_complexity", "tfc_tf_spread"]
        medians = {m: float(np.median([v for v in results[m] if v is not None])) for m in metric_names}
        print(f"  {ename}: " + " ".join(f"{m.split('_')[1][:3]}={medians[m]:.4f}" for m in metric_names)
              + f"  ({dt:.1f}s)")
        print(f"  Saved to {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
