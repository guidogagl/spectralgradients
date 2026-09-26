"""Audio benchmark configuration.

Model config for SpeechCommands with WaveCNNClassifier.
"""

import os
from pathlib import Path


# Repository root; SG_MODELS / SG_DATA override the shipped checkpoints and the data location
REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.environ.get("SG_MODELS", REPO / "models"))
DATA_ROOT = Path(os.environ.get("SG_DATA", REPO / "data")) / "speech"


# Model configuration
MODEL_CONFIGS = {
    "audio-wavcnn": {
        "dataset": "SpeechCommands",
        "model_path": str(MODEL_ROOT / "audio-wavcnn.pt"),
        "model_class": "WaveCNNClassifier",
        "fs": 16000.0,
        "length": 1,  # 1 second (variable length in practice)
        "n_classes": 35,  # SpeechCommands num_classes
        "n_times": 16000,  # 1s * 16000Hz (reference length)
        "sequence_length": 1,
        "n_channels": 1,
        "stft_windows": {
            "time": {"n_fft": 256, "hop": 64},
            "bal":  {"n_fft": 1024, "hop": 256},
            "freq": {"n_fft": 4096, "hop": 1024},
        },
        "data_root": str(DATA_ROOT),
    },
}


# STFT method variants
STFT_METHODS = ["IG", "Sal", "IxG"]
STFT_WINDOWS = ["time", "bal", "freq"]

# All explainers: SG + 3x3 STFT variants
ALL_EXPLAINERS = ["SG"] + [
    f"{method}-{window}" for method in STFT_METHODS for window in STFT_WINDOWS
]

# Metric names
METRIC_NAMES = [
    "freq_infidelity",
    "freq_complexity",
    "time_infidelity",
    "time_complexity",
    "tfc_tf_spread",
]


# Default configuration values
DEFAULTS = {
    "max_samples": 100,
    "gpu": 0,
    "seed": 42,
    "sg": {
        "path": "shapley",
        "steps": 10,
        "perms": 20,
        "freq_step": 100.0,  # Higher for audio (16kHz sampling)
        "batch_size": 5,  # Lower for longer audio sequences
        "metrics": {
            "conf_thresh": 0.6,
            "freq_patch_size": 20,
            "time_patch_size": 500,
        },
    },
    "stft": {
        "real_valued": True,
        "steps": 10,
        "metrics": {
            "conf_thresh": 0.6,
            "freq_patch_size": 20,
            "time_patch_size": 500,
        },
    },
    "output": {
        "save_samples": False,
        "save_attributions": True,
    },
}


def merge_with_defaults(config: dict) -> dict:
    """Merge user config with defaults.

    Args:
        config: User configuration dict (from YAML or CLI)

    Returns:
        Merged configuration dict
    """
    result = DEFAULTS.copy()

    # Merge sg config
    if "sg" in config:
        sg_default = result["sg"].copy()
        sg_default.update(config["sg"])
        result["sg"] = sg_default

    # Merge stft config
    if "stft" in config:
        stft_default = result["stft"].copy()
        stft_default.update(config["stft"])
        result["stft"] = stft_default

    # Merge output config
    if "output" in config:
        output_default = result["output"].copy()
        output_default.update(config["output"])
        result["output"] = output_default

    # Merge metrics config if present
    if "metrics" in config:
        if "sg" not in result:
            result["sg"] = DEFAULTS["sg"].copy()
        if "stft" not in result:
            result["stft"] = DEFAULTS["stft"].copy()
        result["sg"]["metrics"] = {**DEFAULTS["sg"]["metrics"], **config["metrics"]}
        result["stft"]["metrics"] = {**DEFAULTS["stft"]["metrics"], **config["metrics"]}

    # Top-level overrides
    for key in ["max_samples", "gpu", "seed"]:
        if key in config:
            result[key] = config[key]

    return result


def load_config(config_path: str = None, **kwargs) -> dict:
    """Load configuration from YAML file or CLI arguments.

    Args:
        config_path: Path to YAML config file
        **kwargs: CLI argument overrides (only non-None values override YAML)

    Returns:
        Configuration dict with model, explainer, and params
    """
    yaml_config = {}

    if config_path:
        import yaml
        # Resolve relative path from repo root
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = REPO / config_path
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

    # CLI overrides take precedence, but only if not None
    for key, value in kwargs.items():
        if value is not None:
            yaml_config[key] = value

    # Merge with defaults
    config = merge_with_defaults(yaml_config)

    # Get model name
    model = yaml_config.get("model")
    if model is None:
        raise ValueError("Must specify 'model' in config or CLI")

    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Choose from: {list(MODEL_CONFIGS.keys())}")

    # Add model config to result
    config["model"] = model
    config["model_cfg"] = MODEL_CONFIGS[model]

    # Get explainers list
    explainers = yaml_config.get("explainers", ["SG"])
    if isinstance(explainers, str):
        explainers = [explainers]
    config["explainers"] = explainers

    return config
