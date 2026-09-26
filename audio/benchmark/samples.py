"""Sample collection for audio benchmarking."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import soundfile as sf


class SampleCollector:
    """Collect and cache high-confidence samples for benchmarking.

    For audio, samples are loaded from the SpeechCommands test set.
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
        self.length = model_cfg["length"]

        # Set torchaudio backend to avoid torchcodec
        os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '0'

        # Load test dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load SpeechCommands test dataset file paths."""
        # Build the testing list path
        testing_list_path = self.data_root / "SpeechCommands" / "speech_commands_v0.02" / "testing_list.txt"

        if not testing_list_path.exists():
            # Try alternative path
            testing_list_path = self.data_root / "SpeechCommands" / "testing_list.txt"

        if testing_list_path.exists():
            with open(testing_list_path, "r") as f:
                self.test_files = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: scan the directory
            speech_dir = self.data_root / "SpeechCommands" / "speech_commands_v0.02"
            if not speech_dir.exists():
                speech_dir = self.data_root / "SpeechCommands"
            self.test_files = []
            for wav_file in speech_dir.glob("*/*.wav"):
                self.test_files.append(str(wav_file.relative_to(speech_dir)))

        # Build label map from the CHECKPOINT's own class order. The model was
        # trained on 35 words (no "menu"); the previous hardcoded 36-entry map
        # shifted every label after index 16 and broke label alignment.
        ckpt = torch.load(self.model_cfg["model_path"], map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("label_to_idx"):
            self.label_to_idx = dict(ckpt["label_to_idx"])
        elif isinstance(ckpt, dict) and ckpt.get("labels"):
            self.label_to_idx = {lab: i for i, lab in enumerate(ckpt["labels"])}
        else:
            raise RuntimeError("Audio checkpoint lacks label_to_idx/labels; cannot align labels.")
        self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}
        self.n_classes = len(self.label_to_idx)

        print(f"[Info] Found {len(self.test_files)} test files, {self.n_classes} classes")

    def _load_audio_file(self, file_path: Path) -> tuple[torch.Tensor, int, str]:
        """Load a single audio file using soundfile.

        Args:
            file_path: Path to audio file

        Returns:
            (waveform, sample_rate, label) tuple
        """
        # Load with soundfile
        audio, sr = sf.read(file_path, always_2d=False)

        # Convert to tensor
        waveform = torch.tensor(audio, dtype=torch.float32)

        # Add channel dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Get label from directory name
        label = file_path.parent.name

        return waveform, sr, label

    def collect(
        self,
        score_fn: callable,
        max_samples: int = None,
        conf_thresh: float = None,
        seed: int = 42,
    ) -> List[Tuple[torch.Tensor, int]]:
        """Collect samples for benchmarking.

        Filters samples by confidence threshold and randomly samples.

        Args:
            score_fn: Model score function
            max_samples: Maximum number of samples to collect
            conf_thresh: Minimum confidence threshold
            seed: Random seed

        Returns:
            List of (x_tensor, label) tuples
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        max_samples = max_samples or self.config.get("max_samples", 100)
        conf_thresh = conf_thresh or self.config.get("metrics", {}).get("conf_thresh", 0.6)

        # Find the speech commands directory
        speech_dir = self.data_root / "SpeechCommands" / "speech_commands_v0.02"
        if not speech_dir.exists():
            speech_dir = self.data_root / "SpeechCommands"

        # First, collect high-confidence samples
        candidates = []

        for file_rel in self.test_files:
            if len(candidates) >= max_samples * 3:  # Collect more then filter
                break

            file_path = speech_dir / file_rel
            if not file_path.exists():
                continue

            try:
                waveform, sr, label = self._load_audio_file(file_path)

                # Skip if label not in our map
                if label not in self.label_to_idx:
                    continue

                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Resample if needed (SpeechCommands is 16kHz)
                if sr != self.fs:
                    import torchaudio.transforms as T
                    resampler = T.Resample(sr, self.fs)
                    waveform = resampler(waveform)

                # Normalize to [-1, 1]
                waveform = waveform.clamp(-1, 1)

                # Trim/pad to 1 second
                target_len = int(self.fs * self.length)
                if waveform.shape[-1] > target_len:
                    # Trim from center
                    start = (waveform.shape[-1] - target_len) // 2
                    waveform = waveform[:, start:start + target_len]
                elif waveform.shape[-1] < target_len:
                    # Pad with zeros
                    pad_amount = target_len - waveform.shape[-1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

                true_label_idx = self.label_to_idx[label]

                # Get model confidence. NOTE: build_score_fn already applies
                # softmax, so use its output directly (previous code softmaxed
                # twice, flattening the confidence distribution).
                with torch.no_grad():
                    x = waveform.unsqueeze(0)  # Add batch dim
                    probs = score_fn(x)
                    true_conf = probs[0, true_label_idx].item()

                # Filter by confidence on true class
                if true_conf >= conf_thresh:
                    candidates.append((waveform.squeeze(0), true_label_idx))

            except Exception as e:
                # Skip problematic files
                continue

        # If not enough candidates, lower threshold (but stop recursion if threshold is tiny)
        if len(candidates) < max_samples:
            if conf_thresh > 1e-4:
                print(f"[Warning] Only {len(candidates)} samples above threshold {conf_thresh:.6f}, retrying...")
                return self.collect(score_fn, max_samples, conf_thresh * 0.5, seed)
            else:
                print(f"[Info] Using all {len(candidates)} available samples (threshold floor reached).")
                max_samples = len(candidates)

        # Randomly sample from candidates
        indices = torch.randperm(len(candidates))[:max_samples]
        samples = [candidates[i] for i in indices]

        print(f"[Info] Collected {len(samples)} samples (confidence >= {conf_thresh:.2f})")
        return samples

    def save_samples(self, samples: List[Tuple[torch.Tensor, int]], output_file: Path):
        """Save collected samples to disk.

        Args:
            samples: List of (x_tensor, label) tuples
            output_file: Path to save samples (.npz format)
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Find max length for padding
        max_len = max(s[0].shape[-1] for s in samples)

        data = torch.zeros(len(samples), max_len)
        labels = torch.tensor([s[1] for s in samples], dtype=torch.long)

        for i, (x, y) in enumerate(samples):
            data[i, :x.shape[-1]] = x

        np.savez(
            output_file,
            data=data.numpy(),
            labels=labels.numpy(),
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
