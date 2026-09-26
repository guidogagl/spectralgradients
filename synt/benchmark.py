"""Benchmark SG vs STFT explainers on synt models.

Usage::
    # Using YAML config (recommended)
    python synt/benchmark.py --config configs/synt/synt-setup0_sg.yaml

    # Using CLI arguments
    python synt/benchmark.py --model synt-setup0 --explainer SG --max-samples 200

    # Override specific config values
    python synt/benchmark.py --config configs/synt/synt-setup0_sg.yaml --max-samples 50
"""

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch

from synt.benchmark.config import (
    MODEL_CONFIGS, ALL_EXPLAINERS, load_config
)
from synt.benchmark.samples import SampleCollector
from shared.benchmark import run_sg, run_stft, STFT_CLASSES


def build_score_fn(model, n_times, device):
    """Build a flat-signal score_fn for SG/STFT explainers.

    Args:
        model: TimeConvNet model
        n_times: Number of time steps
        device: Device to run on

    Returns:
        score_fn function
    """
    def score_fn(x):
        # x: (B, total) flat signal
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)  # Add channel dimension
        x = x.to(device)
        out = model(x)  # (B, n_classes)
        return out.softmax(dim=-1)

    return score_fn


def load_model(model_name: str, model_cfg: dict, device: torch.device):
    """Load pretrained synt model.

    Args:
        model_name: Model identifier
        model_cfg: Model configuration dict
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    from synt.train import TimeConvNet

    model_path = Path(model_cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create model
    model = TimeConvNet(
        input_shape=(model_cfg["n_times"],),
        fs=int(model_cfg["fs"]),
        n_classes=model_cfg["n_classes"],
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"[Info] Loaded model from {model_path}")
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark SG vs STFT explainers on synt models"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Model name (synt-setup0, synt-setup1, synt-setup2)"
    )

    parser.add_argument(
        "--explainer", "-e",
        type=str,
        choices=ALL_EXPLAINERS,
        default=None,
        help="Explainer name (SG, IG-time, IG-bal, IG-freq, etc.)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to benchmark"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )

    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Results directory (default: $SG_RESULTS/<model> or results/<model>)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(
        config_path=args.config,
        model=args.model,
        explainers=args.explainer,
        max_samples=args.max_samples,
        gpu=args.gpu,
        seed=args.seed,
    )

    # Setup device
    gpu_id = config["gpu"]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Set seed
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = config["model"]
    model_cfg = config["model_cfg"]

    print(f"\n{'='*60}")
    print(f"Synt Benchmark: {model_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Explainers: {config['explainers']}")
    print(f"Max samples: {config['max_samples']}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(model_name, model_cfg, device)
    score_fn = build_score_fn(model, model_cfg["n_times"], device)

    # Collect samples
    collector = SampleCollector(
        model_cfg=model_cfg,
        config=config,
    )

    # Check for cached samples
    output_dir = args.output_dir or Path(os.environ.get("SG_RESULTS", "results")) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / f"samples_n{config['max_samples']}_s{config['seed']}.npz"

    if cache_file.exists():
        print(f"[Info] Loading cached samples from {cache_file}")
        samples = collector.load_samples(cache_file)
    else:
        samples = collector.collect(
            score_fn=score_fn,
            max_samples=config["max_samples"],
        )

        collector.save_samples(samples, cache_file)


    # Run explainers
    for explainer in config["explainers"]:
        print(f"\n{'='*60}")
        print(f"Running: {explainer}")
        print(f"{'='*60}")

        t_start = time.time()

        if explainer == "SG":
            results = run_sg(
                samples=samples,
                score_fn=score_fn,
                config=config["sg"],
                model_cfg=model_cfg,
                device=device,
                save_attributions=config["output"]["save_attributions"],
                attribution_file=output_dir / f"{explainer}_attributions.pt" if config["output"]["save_attributions"] else None,
            )

        elif explainer in ALL_EXPLAINERS:
            # Parse STFT explainer (e.g., "IG-time", "Sal-bal")
            parts = explainer.split("-")
            method = parts[0]
            window = parts[1] if len(parts) > 1 else "bal"

            stft_config = config["stft"].copy()
            stft_config["method"] = method

            window_cfg = model_cfg["stft_windows"][window]

            results = run_stft(
                samples=samples,
                score_fn=score_fn,
                config=stft_config,
                window_cfg=window_cfg,
                model_cfg=model_cfg,
                device=device,
                save_attributions=config["output"]["save_attributions"],
                attribution_file=output_dir / f"{explainer}_attributions.pt" if config["output"]["save_attributions"] else None,
            )

        # Save results
        elapsed = time.time() - t_start

        output_file = output_dir / f"{explainer}_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "model": model_name,
                "explainer": explainer,
                "n_samples": len(samples),
                "elapsed_sec": elapsed,
                "results": results,
            }, f, indent=2)

        print(f"[Info] Saved results to {output_file}")
        print(f"[Info] Elapsed time: {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"Benchmark complete!")
    print(f"Results in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
