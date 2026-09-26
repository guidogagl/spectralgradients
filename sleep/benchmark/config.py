"""Configuration for sleep benchmarking."""

from pathlib import Path
from typing import Any

import yaml


MODEL_CONFIGS = {
    "chambon2018": {
        "dataset": "mass",
        "dataset_kwargs": {"cohort": 3},
        "channels": ["EEG"],
        "pipeline": "raw",
        "pipeline_kwargs": {"target_fs": 128.0},
        "sequence_length": 3,
        "fs": 128.0,
        "n_times": 3840,  # 30s * 128Hz
        "stft_windows": {
            "time": {"n_fft": 32, "hop": 8},
            "bal":  {"n_fft": 128, "hop": 32},
            "freq": {"n_fft": 512, "hop": 128},
        },
    },
    "tsinalis-2016": {
        "dataset": "sleepedf",
        "dataset_kwargs": {},
        "channels": ["EEG"],
        "pipeline": "identity",
        "pipeline_kwargs": {},
        "sequence_length": 5,
        "fs": 100.0,
        "n_times": 3000,  # 30s * 100Hz
        "stft_windows": {
            "time": {"n_fft": 32, "hop": 8},
            "bal":  {"n_fft": 128, "hop": 32},
            "freq": {"n_fft": 512, "hop": 128},
        },
    },
}

STFT_METHODS = {
    "IG": "STFTIntegratedGradients",
    "Sal": "STFTSaliency",
    "IxG": "STFTInputXGradient",
}

ALL_EXPLAINERS = ["SG"] + [
    f"{m}-{w}" for m in STFT_METHODS for w in ["time", "bal", "freq"]
]

METRIC_NAMES = [
    "freq_infidelity", "freq_complexity",
    "time_infidelity", "time_complexity",
    "tfc_tf_spread",
]

# Default configuration values
DEFAULTS = {
    "SG": {
        "path": "shapley",
        "steps": 10,
        "perms": 20,
        "freq_step": 2.0,
        "batch_size": 0,
    },
    "STFT": {
        "real_valued": True,
        "steps": 10,
    },
    "METRICS": {
        "conf_thresh": 0.6,
        "freq_patch_size": 1,
        "time_patch_size": 50,
    },
    "common": {
        "seed": 42,
        "gpu": 0,
    },
    "output": {
        "save_samples": False,      # Save collected samples to disk
        "save_attributions": False, # Save raw attributions to disk
    }
}


def merge_with_defaults(cfg: dict) -> dict:
    """Merge user config with defaults.

    Args:
        cfg: User configuration dict (may be partial)

    Returns:
        Complete configuration with all defaults filled in
    """
    result = {}

    # Common defaults
    for key, value in DEFAULTS["common"].items():
        result[key] = cfg.get(key, value)

    # SG defaults
    if "sg" not in result:
        result["sg"] = {}
    if "sg" in cfg:
        result["sg"] = {**DEFAULTS["SG"], **cfg["sg"]}
    else:
        result["sg"] = DEFAULTS["SG"].copy()

    # STFT defaults
    if "stft" not in result:
        result["stft"] = {}
    if "stft" in cfg:
        result["stft"] = {**DEFAULTS["STFT"], **cfg["stft"]}
    else:
        result["stft"] = DEFAULTS["STFT"].copy()

    # Metrics defaults
    if "metrics" not in result:
        result["metrics"] = {}
    if "metrics" in cfg:
        result["metrics"] = {**DEFAULTS["METRICS"], **cfg["metrics"]}
    else:
        result["metrics"] = DEFAULTS["METRICS"].copy()

    # Output defaults
    if "output" not in result:
        result["output"] = {}
    if "output" in cfg:
        result["output"] = {**DEFAULTS["output"], **cfg["output"]}
    else:
        result["output"] = DEFAULTS["output"].copy()

    # Copy remaining user config
    for key in ["model", "explainers", "max_samples", "output_dir", "data_root"]:
        if key in cfg:
            result[key] = cfg[key]

    return result


def load_config(config_path: Path = None, cli_overrides: dict = None) -> dict:
    """Load configuration from YAML file and merge with CLI overrides.

    Args:
        config_path: Path to YAML config file (optional)
        cli_overrides: Dict of CLI argument overrides

    Returns:
        Complete configuration dict
    """
    cfg = {}

    # Load from YAML if provided
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                cfg[key] = value

    # Merge with defaults
    return merge_with_defaults(cfg)
