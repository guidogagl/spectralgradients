"""Sample collection for arrhythmia benchmarking."""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Add parent directory to path for imports
from arrhythmia.data import (
    build_patients_data, flatten_segments_and_labels, rebalance_classes_n_and_f,
    resolve_data_dir, TrainConfig,
)


class SampleCollector:
    """Collect and cache high-confidence samples for benchmarking.

    For arrhythmia, samples are loaded from the MIT-BIH dataset.
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
            data_root: Override data root path
        """
        self.model_cfg = model_cfg
        self.config = config
        self.data_root = Path(data_root or model_cfg["data_root"])
        self.fs = model_cfg["fs"]
        self.n_times = model_cfg["n_times"]
        self.n_classes = model_cfg["n_classes"]

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load MIT-BIH arrhythmia dataset."""
        # Use same data pipeline as train.py
        data_dir = resolve_data_dir(str(self.data_root), self.data_root.parent)
        patients_data = build_patients_data(data_dir)
        all_segments, all_labels = flatten_segments_and_labels(patients_data)

        # Use rebalanced data for fairness
        balanced_segments, balanced_labels = rebalance_classes_n_and_f(
            all_segments, all_labels, target_n=20000
        )

        # Encode labels with the SAME class order the checkpoint was trained with
        # (sklearn LabelEncoder -> alphabetical: F, N, S, U, V). A fixed hand-written
        # map (N=0, S=1, V=2, F=3) silently explained the wrong class. Beats labelled
        # unclassifiable ("U") are dropped from the explanation pool.
        self.class_names = self._load_class_names()
        label_map = {c: i for i, c in enumerate(self.class_names)}
        keep = [i for i, l in enumerate(balanced_labels) if l in label_map and l != "U"]
        self.data = torch.tensor(
            np.array([balanced_segments[i] for i in keep]), dtype=torch.float32
        )
        self.labels = torch.tensor(
            [label_map[balanced_labels[i]] for i in keep], dtype=torch.long
        )

        print(f"[Info] Loaded {len(self.data)} samples for arrhythmia "
              f"(classes {self.class_names}, U excluded)")

    def _load_class_names(self) -> List[str]:
        """Read the class order from the trained checkpoint."""
        ckpt = torch.load(self.model_cfg["model_path"], map_location="cpu", weights_only=False)
        classes = ckpt.get("classes")
        if classes is None:
            raise RuntimeError("Checkpoint has no 'classes' field; cannot align label encoding")
        return [str(c) for c in classes]

    def collect(
        self,
        score_fn: callable,
        max_samples: int = None,
        conf_thresh: float = None,
        seed: int = 42,
        batch_size: int = 1024,
    ) -> List[Tuple[torch.Tensor, int]]:
        """Collect the benchmark sample set: correctly classified beats with confidence >= thresh.

        Protocol (same as the other case studies): a beat is eligible iff the model's argmax
        equals its label AND the predicted-class probability is >= ``conf_thresh``. The threshold
        is NEVER lowered. If fewer than ``max_samples`` beats are eligible, all eligible beats are
        used (this is the intended "every correctly classified test sample" regime when
        ``max_samples`` is large); otherwise a seeded random subset of size ``max_samples``.

        ``score_fn`` already returns softmax probabilities (see benchmark.build_score_fn); do not
        apply softmax again here (the former double softmax capped confidences near 1/n_classes
        and silently disabled the threshold).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        max_samples = max_samples or self.config.get("max_samples", 200)
        conf_thresh = conf_thresh or self.config.get("metrics", {}).get("conf_thresh", 0.6)

        n = len(self.data)
        keep = torch.zeros(n, dtype=torch.bool)
        confs = torch.zeros(n)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                xb = self.data[start:start + batch_size]
                yb = self.labels[start:start + batch_size]
                probs = score_fn(xb).detach().cpu()          # already softmax
                pred = probs.argmax(dim=-1)
                conf = probs[torch.arange(len(yb)), yb]
                keep[start:start + batch_size] = (pred == yb) & (conf >= conf_thresh)
                confs[start:start + batch_size] = conf

        eligible = torch.nonzero(keep).squeeze(1)
        per_class = torch.bincount(self.labels[eligible], minlength=len(self.class_names)).tolist()
        print(f"[Info] Eligible beats (pred==label & conf>={conf_thresh:.2f}): {len(eligible)}/{n} "
              f"per class {dict(zip(self.class_names, per_class))}")

        if len(eligible) > max_samples:
            perm = torch.randperm(len(eligible))[:max_samples]
            chosen = eligible[perm]
        else:
            chosen = eligible
        # deterministic order (dataset order) so chunking is reproducible
        chosen = torch.sort(chosen).values
        samples = [(self.data[i], int(self.labels[i])) for i in chosen.tolist()]
        print(f"[Info] Collected {len(samples)} samples (confidence >= {conf_thresh:.2f}, seed {seed})")
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

        np.savez(
            output_file,
            data=data.numpy(),
            labels=labels.numpy(),
        )

        print(f"[Info] Saved {len(samples)} samples to {output_file}")

    @staticmethod
    def load_samples(input_file: Path) -> List[Tuple[torch.Tensor, int]]:
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
