"""Synt benchmark package.

Modularized benchmarking for synthetic time-series with SG and STFT explainers.
Uses shared/benchmark for metrics and explainer execution.
"""

from synt.benchmark.config import MODEL_CONFIGS, STFT_METHODS, load_config, merge_with_defaults
from synt.benchmark.samples import SampleCollector
from shared.benchmark import compute_all_metrics, run_sg, run_stft

__all__ = [
    "MODEL_CONFIGS",
    "STFT_METHODS",
    "load_config",
    "merge_with_defaults",
    "SampleCollector",
    "compute_all_metrics",
    "run_sg",
    "run_stft",
]
