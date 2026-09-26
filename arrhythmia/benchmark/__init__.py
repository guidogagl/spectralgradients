"""Arrhythmia benchmark package.

Modularized benchmarking for MIT-BIH with SG and STFT explainers.
Uses shared/benchmark for metrics and explainer execution.
"""

from arrhythmia.benchmark.config import MODEL_CONFIGS, STFT_METHODS, load_config, merge_with_defaults
from arrhythmia.benchmark.samples import SampleCollector
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
