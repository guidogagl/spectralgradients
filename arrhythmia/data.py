import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import wfdb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]


VALID_BEAT_SYMBOLS = {"N", "L", "R", "A", "a", "J", "S", "V", "F", "!", "e", "j", "E", "f", "x", "Q"}
WINDOW_SIZE = 200


@dataclass
class TrainConfig:
    batch_size: int = 128
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    hidden_size: int = 160
    dropout_rate: float = 0.1
    patience: int = 10
    seed: int = 42
    num_workers: int = 0


class ECGDataset(Dataset):
    """Dataset per segmenti ECG con shape [N, T, C]."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y), "X e y devono avere la stessa lunghezza"
        assert len(x) > 0, "Dataset vuoto"
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def high_pass_filter(original_signal: np.ndarray, cutoff_frequency: float = 0.5, sampling_rate: float = 360, order: int = 3) -> np.ndarray:
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = sp.signal.butter(order, normal_cutoff, btype="highpass", analog=False)
    return sp.signal.filtfilt(b, a, original_signal)


def low_pass_filter(original_signal: np.ndarray, cutoff_freq: float = 150, sampling_rate: float = 360, order: int = 3) -> np.ndarray:
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = sp.signal.butter(order, normal_cutoff, btype="low", analog=False)
    return sp.signal.filtfilt(b, a, original_signal)


def notch_filter(signal: np.ndarray, notch_frequency: float = 60, quality_factor: float = 30, sampling_rate: float = 360) -> np.ndarray:
    b, a = sp.signal.iirnotch(notch_frequency / (sampling_rate / 2), quality_factor)
    return sp.signal.filtfilt(b, a, signal)


def signal_filtering(raw_signal: np.ndarray) -> np.ndarray:
    filtered_high_pass_signal = high_pass_filter(raw_signal)
    filtered_notch_signal = notch_filter(filtered_high_pass_signal)
    return low_pass_filter(filtered_notch_signal)


def filter_beat_annotations(annotations: wfdb.Annotation, valid_beat_symbols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    valid_indices = [i for i, symbol in enumerate(annotations.symbol) if symbol in valid_beat_symbols]
    filtered_samples = annotations.sample[valid_indices]
    filtered_symbols = np.array(annotations.symbol)[valid_indices]
    return filtered_symbols, filtered_samples


def simplify_label(label: str) -> str:
    if label in {"N", "L", "R", "j", "e"}:
        return "N"
    if label in {"A", "a", "S", "J"}:
        return "S"
    if label in {"V", "E"}:
        return "V"
    if label == "F":
        return "F"
    return "U"


def segmentation_process(signal: np.ndarray, filtered_samples: np.ndarray, filtered_symbols: np.ndarray, window_size: int = WINDOW_SIZE) -> Tuple[List[np.ndarray], List[str]]:
    segments: List[np.ndarray] = []
    labels: List[str] = []

    for r_peak, raw_label in zip(filtered_samples, filtered_symbols):
        label = simplify_label(raw_label)
        start = max(0, int(r_peak) - window_size // 2)
        end = min(len(signal), int(r_peak) + window_size // 2)
        segment = signal[start:end]
        if len(segment) == window_size:
            segments.append(segment)
            labels.append(label)

    return segments, labels


def normalization_process(raw_segments: Sequence[np.ndarray]) -> List[np.ndarray]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_segments = []
    for raw_segment in raw_segments:
        normalized_segment = scaler.fit_transform(np.array(raw_segment).reshape(-1, 1)).reshape(-1)
        normalized_segments.append(normalized_segment)
    return normalized_segments


def time_shift(signal: np.ndarray, shift: int = 5) -> np.ndarray:
    return np.roll(signal, shift)


def amplitude_scaling(signal: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    scale = np.random.uniform(*scale_range)
    return signal * scale


def rebalance_classes_n_and_f(all_segments: List[np.ndarray], all_labels: np.ndarray, target_n: int = 20000) -> Tuple[List[np.ndarray], np.ndarray]:
    labels_list = list(all_labels)

    indices_n = [i for i, label in enumerate(labels_list) if label == "N"]
    target_n = min(target_n, len(indices_n))
    selected_n_indices = np.random.choice(indices_n, target_n, replace=False)

    other_indices = [i for i, label in enumerate(labels_list) if label != "N"]
    final_indices = sorted(list(selected_n_indices) + other_indices)

    balanced_segments = [all_segments[i] for i in final_indices]
    balanced_labels = [labels_list[i] for i in final_indices]

    indices_f = [i for i, label in enumerate(balanced_labels) if label == "F"]
    if len(indices_f) >= 2:
        n_aug = min(401, len(indices_f))

        augmented_segments = []
        augmented_labels = []

        selected_shift = np.random.choice(indices_f, size=n_aug, replace=False)
        for i in selected_shift:
            shift_value = int(np.random.randint(-5, 6))
            augmented_segments.append(np.array(time_shift(balanced_segments[i], shift=shift_value)))
            augmented_labels.append("F")

        selected_amp = np.random.choice(indices_f, size=n_aug, replace=False)
        scaler = MinMaxScaler(feature_range=(0, 1))
        for i in selected_amp:
            scaled = amplitude_scaling(np.array(balanced_segments[i]))
            normalized_scaled = scaler.fit_transform(scaled.reshape(-1, 1)).reshape(-1)
            augmented_segments.append(normalized_scaled)
            augmented_labels.append("F")

        balanced_segments.extend(augmented_segments)
        balanced_labels.extend(augmented_labels)

    return balanced_segments, np.array(balanced_labels)


def discover_patient_ids(data_dir: Path) -> List[str]:
    record_ids = sorted({p.stem for p in data_dir.glob("*.dat") if p.stem.isdigit()})
    if not record_ids:
        record_ids = [str(i) for i in range(100, 235)]
    return record_ids


def build_patients_data(data_dir: Path) -> List[Dict]:
    patients_data = []
    patient_ids = discover_patient_ids(data_dir)

    print("Start loading and preprocessing all patients...")
    for patient_id in tqdm(patient_ids):
        record_path = data_dir / patient_id
        try:
            patient_record = wfdb.rdrecord(str(record_path))
            annotations = wfdb.rdann(str(record_path), "atr")

            filtered_symbols, filtered_samples = filter_beat_annotations(annotations, VALID_BEAT_SYMBOLS)

            first_lead_cleaned_signal = signal_filtering(patient_record.p_signal[:, 0])
            second_lead_cleaned_signal = signal_filtering(patient_record.p_signal[:, 1])

            first_raw_segments, first_labels = segmentation_process(first_lead_cleaned_signal, filtered_samples, filtered_symbols)
            second_raw_segments, second_labels = segmentation_process(second_lead_cleaned_signal, filtered_samples, filtered_symbols)

            first_segments = normalization_process(first_raw_segments)
            second_segments = normalization_process(second_raw_segments)

            segments = first_segments + second_segments
            labels = first_labels + second_labels

            patient_entry = {
                "patient_id": patient_id,
                "ecg_first_lead": first_lead_cleaned_signal,
                "ecg_second_lead": second_lead_cleaned_signal,
                "fs": patient_record.fs,
                "lead_names": patient_record.sig_name,
                "beats": segments,
                "beat_labels": labels,
            }
            patients_data.append(patient_entry)

        except FileNotFoundError:
            continue

    print(f"Loaded completed: {len(patients_data)} patients")
    return patients_data


def flatten_segments_and_labels(patients_data: Sequence[Dict]) -> Tuple[List[np.ndarray], np.ndarray]:
    all_segments: List[np.ndarray] = []
    all_labels: List[str] = []
    for patient in patients_data:
        for segment, label in zip(patient["beats"], patient["beat_labels"]):
            all_segments.append(np.array(segment))
            all_labels.append(label)
    return all_segments, np.array(all_labels)


def build_dataloaders(segments: List[np.ndarray], labels: np.ndarray, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    x = np.array(segments, dtype=np.float32)
    x = np.expand_dims(x, axis=-1)  # [N, 200, 1]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    dataset = ECGDataset(x, y)
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(indices, test_size=(1.0 - cfg.train_ratio), random_state=cfg.seed, stratify=y)
    val_relative = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    y_temp = y[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1.0 - val_relative), random_state=cfg.seed, stratify=y_temp)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_labels = torch.tensor([dataset.y[i].item() for i in train_idx], dtype=torch.long)
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float().clamp_min(1.0)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader, test_loader, label_encoder


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    class_names: Sequence[str],
    output_dir: Path,
    prefix: str,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []
    probabilities = []

    with torch.inference_mode():
        for batch_segments, batch_labels in test_loader:
            batch_segments = batch_segments.to(device)
            outputs = model(batch_segments)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            probabilities.extend(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    probabilities = np.array(probabilities)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(class_names)))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"Confusion Matrix ({prefix})")
    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
    }

    try:
        metrics["auc_macro_ovo"] = float(roc_auc_score(all_labels, probabilities, multi_class="ovo", average="macro"))
        metrics["auc_weighted_ovo"] = float(roc_auc_score(all_labels, probabilities, multi_class="ovo", average="weighted"))
    except Exception as exc:
        print(f"AUC calculation skipped: {exc}")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    if "auc_macro_ovo" in metrics:
        print(f"AUC-ROC (Macro OVO): {metrics['auc_macro_ovo']:.4f}")
        print(f"AUC-ROC (Weighted OVO): {metrics['auc_weighted_ovo']:.4f}")
    print(f"Saved confusion matrix to: {cm_path}")

    return metrics


def plot_training_curves(training_loss: List[float], validation_loss: List[float], accuracy_train: List[float], accuracy_val: List[float], output_dir: Path) -> None:
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(training_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 2)
    plt.plot(accuracy_train)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(2, 2, 3)
    plt.plot(validation_loss)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 4)
    plt.plot(accuracy_val)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved training curves to: {out_path}")


def resolve_data_dir(user_data_dir: str | None, script_dir: Path) -> Path:
    candidates = []
    if user_data_dir:
        candidates.append(Path(user_data_dir))
    candidates.extend(
        [
            REPO / "data" / "mitdb",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.dat")):
            return candidate.resolve()

    raise FileNotFoundError(
        "MIT-BIH dataset directory not found. Run `python arrhythmia/data.py --download data/mitdb` or place the records in one of: "
        f"{[str(c) for c in candidates]}"
    )


def download(dl_dir: Path) -> Path:
    """Fetch the 48 records of the MIT-BIH Arrhythmia Database (PhysioNet, ~107 MB) with wfdb."""
    dl_dir = Path(dl_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=str(dl_dir))
    return dl_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIT-BIH data pipeline (download, filtering, segmentation)")
    parser.add_argument("--download", metavar="DIR", help="download the MIT-BIH Arrhythmia Database into DIR")
    args = parser.parse_args()
    if args.download:
        print(f"Downloaded MIT-BIH to {download(args.download)}")
    else:
        parser.print_help()
