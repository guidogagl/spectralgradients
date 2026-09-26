"""Benchmark SG vs STFT explainers on audio model.

Usage::
    # Using YAML config (recommended)
    python audio/benchmark.py --config configs/audio/audio-wavcnn_sg.yaml

    # Using CLI arguments
    python audio/benchmark.py --model audio-wavcnn --explainer SG --max-samples 100

    # Override specific config values
    python audio/benchmark.py --config configs/audio/audio-wavcnn_sg.yaml --max-samples 50
"""

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch

from audio.benchmark.config import (
    MODEL_CONFIGS, ALL_EXPLAINERS, load_config
)
from audio.benchmark.samples import SampleCollector
from shared.benchmark import run_sg, run_stft, STFT_CLASSES


def build_score_fn(model, device):
    """Build a score_fn for SG/STFT explainers.

    Args:
        model: WaveCNNClassifier model
        device: Device to run on

    Returns:
        score_fn function
    """
    def score_fn(x):
        # x: (B, T) audio waveform - add channel dim to get (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, T) -> (B, 1, T)
        x = x.to(device)
        out = model(x, is_train=False)  # (B, n_classes)
        return out.softmax(dim=-1)

    return score_fn


def load_model(model_name: str, model_cfg: dict, device: torch.device):
    """Load pretrained audio model.

    Args:
        model_name: Model identifier
        model_cfg: Model configuration dict
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    model_path = Path(model_cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Import model class
    from audio.train import WaveCNNClassifier

    # Load checkpoint to get num_classes
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        n_classes = len(checkpoint.get("labels", []))
    else:
        state_dict = checkpoint
        n_classes = model_cfg["n_classes"]

    # Create model
    model = WaveCNNClassifier(num_classes=n_classes)

    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"[Info] Loaded model from {model_path} (n_classes={n_classes})")
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark SG vs STFT explainers on audio model"
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
        help="Model name (audio-wavcnn)"
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

    parser.add_argument("--chunk-idx", type=int, default=0, help="Chunk index (0..chunk-n-1)")
    parser.add_argument("--chunk-n", type=int, default=1, help="Number of chunks (1 = no chunking)")

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
    print(f"Audio Benchmark: {model_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Explainers: {config['explainers']}")
    print(f"Max samples: {config['max_samples']}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(model_name, model_cfg, device)
    score_fn = build_score_fn(model, device)

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
        # Always persist the sample set so every explainer/chunk job reuses the
        # identical selection (paired SG-vs-STFT comparison).
        collector.save_samples(samples, cache_file)

    # Optional chunking: strided partition of the shared sample set.
    suffix = ""
    if args.chunk_n and args.chunk_n > 1:
        n_full = len(samples)
        samples = samples[args.chunk_idx::args.chunk_n]
        suffix = f"_chunk{args.chunk_idx}of{args.chunk_n}"
        print(f"[Info] Chunk {args.chunk_idx}/{args.chunk_n}: {len(samples)}/{n_full} samples")


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
                checkpoint_file=output_dir / f"_checkpoint_SG{suffix}.pt",
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

        output_file = output_dir / f"{explainer}_results{suffix}.json"
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
