"""Sample collection for synt benchmarking."""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


class SampleCollector:
    """Collect and cache high-confidence samples for benchmarking.

    For synt, samples are generated using SyntDataset.
    """

    def __init__(
        self,
        model_cfg: dict,
        config: dict,
        data_root: str = None,
    ):
        """Initialize SampleCollector.

        Args:
            model_cfg: Model configuration dict
            config: Benchmark configuration dict
            data_root: Override data root path (for cache storage)
        """
        self.model_cfg = model_cfg
        self.config = config
        self.setup = model_cfg["setup"]
        self.n_classes = model_cfg["n_classes"]
        self.fs = model_cfg["fs"]
        self.length = model_cfg["length"]
        self.n_times = model_cfg["n_times"]

        # Use data_root for cache storage (optional)
        self.data_root = Path(data_root or model_cfg.get("data_root", Path(__file__).resolve().parents[2] / "data" / "synt"))
        self.last_indices = None   # dataset rows of the last collect(), in sample order

        # Load dataset using SyntDataset
        self._load_dataset()

    def _load_dataset(self):
        """Load synt dataset using SyntDataset class."""
        from synt.data import SyntDataset

        # Set n_samples lower for benchmarking to avoid long generation times
        n_samples = 1000  # per class

        # Create dataset (will generate if cache doesn't exist)
        self.dataset = SyntDataset(
            setup=self.setup,
            n_samples=n_samples,
            fs=self.fs,
            length=self.length,
            bandwidth=5.0,
            return_mask=False,
            output_dir=str(self.data_root),
        )

        # Store data and labels
        self.data = self.dataset.data
        self.labels = self.dataset.labels

        print(f"[Info] Loaded {len(self.data)} samples for setup{self.setup}")

    def collect(
        self,
        score_fn: callable,
        max_samples: int = None,
        conf_thresh: float = None,
        seed: int = 42,
    ) -> List[Tuple[torch.Tensor, int]]:
        """Collect samples for benchmarking.

        For synt, we randomly sample from the dataset (no confidence filtering needed
        since the ground truth is synthetic and known).

        Args:
            score_fn: Model score function (not used for synt, but kept for interface)
            max_samples: Maximum number of samples to collect
            conf_thresh: Confidence threshold (not used for synt)
            seed: Random seed

        Returns:
            List of (x_tensor, label) tuples
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        max_samples = max_samples or self.config.get("max_samples", 200)

        # Sample evenly from each class
        samples_per_class = max(1, max_samples // self.n_classes)
        samples = []

        for cls_idx in range(self.n_classes):
            cls_mask = self.labels == cls_idx
            cls_indices = torch.where(cls_mask)[0]

            if len(cls_indices) == 0:
                continue

            # Randomly sample from this class
            n_samples = min(samples_per_class, len(cls_indices))
            chosen = cls_indices[torch.randperm(len(cls_indices))[:n_samples]]

            for idx in chosen:
                x = self.data[idx]
                y = self.labels[idx].item()
                samples.append((x, y, int(idx)))

        # Shuffle samples
        indices = torch.randperm(len(samples))
        samples = [samples[i] for i in indices]
        self.last_indices = [s[2] for s in samples]
        samples = [(x, y) for x, y, _ in samples]

        print(f"[Info] Collected {len(samples)} samples (setup{self.setup})")
        return samples

    def save_samples(self, samples: List[Tuple[torch.Tensor, int]], output_file: Path):
        """Save collected samples to disk.

        Args:
            samples: List of (x_tensor, label) tuples
            output_file: Path to save samples (.npz format)
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = torch.stack([s[0] for s in samples])
        labels = torch.tensor([s[1] for s in samples])

        extra = {}
        if self.last_indices is not None and len(self.last_indices) == len(samples):
            extra["indices"] = np.asarray(self.last_indices)   # dataset rows: compute_localization.py reads the masks from them
        np.savez(
            output_file,
            data=data.numpy(),
            labels=labels.numpy(),
            **extra,
        )

        print(f"[Info] Saved {len(samples)} samples to {output_file}")

    def load_samples(self, input_file: Path) -> List[Tuple[torch.Tensor, int]]:
        """Load samples from disk.

        Args:
            input_file: Path to load samples from (.npz format)

        Returns:
            List of (x_tensor, label) tuples
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Sample file not found: {input_file}")

        content = np.load(input_file)
        data = torch.tensor(content["data"], dtype=torch.float32)
        labels = torch.tensor(content["labels"], dtype=torch.long)

        samples = [(data[i], labels[i].item()) for i in range(len(data))]

        print(f"[Info] Loaded {len(samples)} samples from {input_file}")
        return samples
