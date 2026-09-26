"""Shared benchmark utilities for all domains.

Exports:
- compute_all_metrics: 5-metric computation (freq/time infidelity+complexity, tfc_tf_spread)
- run_sg: SpectralGradients explainer execution
- run_stft: STFT-based explainer execution (IG, Sal, IxG)
- STFT_CLASSES: Dict mapping method names to explainer classes
"""

from shared.benchmark.metrics import compute_all_metrics
from shared.benchmark.explainers import run_sg, run_stft, STFT_CLASSES

__all__ = ["compute_all_metrics", "run_sg", "run_stft", "STFT_CLASSES"]
