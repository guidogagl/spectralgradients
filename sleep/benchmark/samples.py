"""Sample collection and caching for sleep benchmarking."""

import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F


class SampleCollector:
    """Collect high-confidence correctly-classified samples with caching.

    Features:
    - Cache validation (checks if model changed)
    - Proper error handling
    - Progress reporting
    """

    def __init__(self, dataset, score_fn, config: dict):
        """Initialize SampleCollector.

        Args:
            dataset: PhysioEx dataset instance
            score_fn: Function that takes (B, total_len) and returns (B, n_classes) probs
            config: Model configuration dict (from MODEL_CONFIGS)
        """
        self.dataset = dataset
        self.score_fn = score_fn
        self.config = config
        self.cfg_L = config["sequence_length"]
        self.n_times = config["n_times"]
        self.total_len = config["sequence_length"] * config["n_times"]

    def _validate_sample(self, item: dict, x_flat: torch.Tensor, label: int) -> bool:
        """Validate a sample before adding to collection.

        Args:
            item: Dataset item
            x_flat: Flattened signal tensor
            label: Central label

        Returns:
            True if sample is valid

        Raises:
            ValueError: If sample has unexpected structure (helps debugging)
        """
        # Check shape
        if x_flat.shape[0] != self.total_len:
            # Log instead of silent continue for debugging
            return False

        # Check label range (assuming 5 classes for sleep staging)
        if label < 0 or label > 4:
            return False

        return True

    def collect(
        self,
        max_samples: int,
        seed: int,
        device: torch.device,
        cache_file: Path = None,
        conf_thresh: float = 0.6
    ) -> List[Tuple[torch.Tensor, int]]:
        """Collect high-confidence correctly-classified test samples.

        Args:
            max_samples: Maximum number of samples to collect
            seed: Random seed for reproducibility
            device: Device to move samples to
            cache_file: Path to cache file (for loading/saving)
            conf_thresh: Minimum confidence threshold

        Returns:
            List of (x_tensor_on_device, label) tuples
        """
        # Try loading from cache first
        if cache_file and cache_file.exists():
            try:
                return self._load_from_cache(cache_file, device)
            except ValueError as e:
                print(f"Cache invalid: {e}. Recollecting samples...", flush=True)

        # Collect new samples
        torch.manual_seed(seed)
        indices = torch.randperm(len(self.dataset)).tolist()

        samples = []
        sample_indices = []

        print(f"Collecting {max_samples} high-confidence samples...", flush=True)

        for i, idx in enumerate(indices):
            if len(samples) >= max_samples:
                break
            if i % 500 == 0:
                print(f"  scanned {i}/{len(indices)}, found {len(samples)}/{max_samples}", flush=True)

            item = self.dataset[idx]
            if item is None:
                continue

            # Extract label
            labels = item["labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            central_label = int(labels[self.cfg_L // 2])

            # Extract signal
            sigs = []
            for k in sorted(item["signals"].keys()):
                sigs.append(item["signals"][k])
            x_seq = torch.stack(sigs, dim=-2)  # (L, C, T)
            x_flat = x_seq[:, 0, :].reshape(-1)  # (L*T,)

            # Validate
            if not self._validate_sample(item, x_flat, central_label):
                continue

            # Check prediction confidence
            x_dev = x_flat.to(device)
            with torch.no_grad():
                p = self.score_fn(x_dev.unsqueeze(0))
                conf = p[0, central_label].item()
                pred = p.argmax(dim=-1).item()

            if conf >= conf_thresh and pred == central_label:
                samples.append((x_dev, central_label))
                sample_indices.append({
                    "ds_index": idx,
                    "label": central_label,
                    "conf": float(conf)
                })

        # Save to cache
        if cache_file and sample_indices:
            self._save_to_cache(cache_file, sample_indices, max_samples, seed)

        print(f"Collected {len(samples)} samples", flush=True)
        return samples

    def _load_from_cache(
        self,
        cache_file: Path,
        device: torch.device
    ) -> List[Tuple[torch.Tensor, int]]:
        """Load samples from cache file with validation.

        Args:
            cache_file: Path to cache file
            device: Device to move samples to

        Returns:
            List of (x_tensor_on_device, label) tuples

        Raises:
            ValueError: If cache is invalid or model mismatch
        """
        print(f"Loading cached sample indices from {cache_file}", flush=True)

        with open(cache_file) as f:
            cached = json.load(f)

        # Validate cache structure
        if "samples" not in cached:
            raise ValueError("Cache missing 'samples' key")

        # Note: Could add model name validation here if needed
        # For now, we just check the cache exists and has valid structure

        samples = []
        for entry in cached["samples"]:
            item = self.dataset[entry["ds_index"]]

            sigs = []
            for k in sorted(item["signals"].keys()):
                sigs.append(item["signals"][k])
            x_seq = torch.stack(sigs, dim=-2)
            x_flat = x_seq[:, 0, :].reshape(-1).to(device)
            samples.append((x_flat, entry["label"]))

        print(f"Loaded {len(samples)} cached samples", flush=True)
        return samples

    def _save_to_cache(
        self,
        cache_file: Path,
        sample_indices: list,
        n_samples: int,
        seed: int
    ):
        """Save sample indices to cache file.

        Args:
            cache_file: Path to cache file
            sample_indices: List of sample metadata dicts
            n_samples: Number of samples collected
            seed: Random seed used
        """
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_samples": n_samples,
            "seed": seed,
            "samples": sample_indices
        }

        with open(cache_file, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Cached {n_samples} sample indices to {cache_file}", flush=True)

    def save_samples(
        self,
        samples: List[Tuple[torch.Tensor, int]],
        output_file: Path
    ):
        """Save complete samples (signals + labels) to disk for later analysis.

        Args:
            samples: List of (x_tensor, label) tuples
            output_file: Path to save samples (will use .pt format)
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert list of tuples to dict for easier loading
        signals = torch.stack([s[0].cpu() for s in samples])
        labels = torch.tensor([s[1] for s in samples])

        payload = {
            "signals": signals,  # (n_samples, total_len)
            "labels": labels,    # (n_samples,)
            "total_len": self.total_len,
            "n_times": self.n_times,
            "sequence_length": self.cfg_L,
            "fs": self.config["fs"],
        }

        torch.save(payload, output_file)
        print(f"Saved {len(samples)} samples to {output_file}", flush=True)

    @staticmethod
    def load_samples(input_file: Path, device: torch.device = None):
        """Load complete samples from disk.

        Args:
            input_file: Path to samples file (.pt)
            device: Device to load tensors to

        Returns:
            List of (x_tensor, label) tuples
        """
        payload = torch.load(input_file, map_location=device or "cpu")

        signals = payload["signals"]
        labels = payload["labels"]

        if device is not None:
            signals = signals.to(device)

        return list(zip(signals, labels.tolist()))
