"""Train the SimpleConv1d ECG beat classifier on MIT-BIH (the paper's arrhythmia model).

Model: pure Conv1d -> ReLU -> MaxPool (no BN, no Dropout), which gives clean gradients for the
attribution benchmark. Reuses the data pipeline of ``arrhythmia/data.py`` (filtering, segmentation,
min-max normalisation, class rebalancing).

    python arrhythmia/train.py --data-dir data/mitdb --out models-retrained/arrhythmia-cnn.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from arrhythmia.data import (
    TrainConfig, build_patients_data, flatten_segments_and_labels,
    rebalance_classes_n_and_f, build_dataloaders, set_seed, resolve_data_dir,
    evaluate_model, plot_training_curves,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FS = 360.0
SIGNAL_LEN = 200
SEED = 42


class SimpleConv1d(nn.Module):
    """Minimal 1D CNN for ECG beat classification.

    Architecture:
        4 × (Conv1d → ReLU → MaxPool) → AdaptiveAvgPool → Linear

    No BatchNorm, no Dropout, no augmentation.
    Clean gradients for explainability.
    """

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 200 → 100
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2: 100 → 50
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3: 50 → 25
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 4: 25 → 1 (global pool)
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        """x: (B, T) raw waveform, (B, T, 1), or (B, 1, T)."""
        if x.dim() == 3 and x.shape[-1] == 1:
            # Input is (B, T, 1) from ECGDataset — convert to (B, T)
            x = x.squeeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (T,) → (1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, T) → (B, 1, T)
        # Z-score normalization (per-sample, differentiable)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std
        return self.classifier(self.features(x))


def train(args):
    set_seed(args.seed)
    cfg = TrainConfig(seed=args.seed, num_epochs=args.epochs, lr=1e-3, weight_decay=1e-4)
    out = Path(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data — same pipeline as LSTM
    data_dir = resolve_data_dir(args.data_dir, SCRIPT_DIR)
    patients_data = build_patients_data(data_dir)
    all_segments, all_labels = flatten_segments_and_labels(patients_data)
    balanced_segments, balanced_labels = rebalance_classes_n_and_f(all_segments, all_labels)
    train_loader, val_loader, test_loader, label_encoder = build_dataloaders(
        balanced_segments, balanced_labels, cfg,
    )
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    model = SimpleConv1d(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleConv1d parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-5)

    out.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(cfg.num_epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(-1) == yb).sum().item()
            train_total += xb.size(0)

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_correct += (logits.argmax(-1) == yb).sum().item()
                val_total += xb.size(0)

        scheduler.step()
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:>2d}/{cfg.num_epochs}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "classes": class_names,
            }
            ckpt_path = out
            torch.save(best_state, ckpt_path)
            print(f"  -> Saved (val_acc={val_acc:.4f})")

    # Test
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            test_correct += (logits.argmax(-1) == yb).sum().item()
            test_total += xb.size(0)

    test_acc = test_correct / test_total
    print(f"\nBest val_acc: {best_val_acc:.4f}")
    print(f"Test acc:     {test_acc:.4f}")

    # Save info
    info = {
        "model": "SimpleConv1d",
        "best_epoch": best_state["epoch"],
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "classes": class_names,
        "n_params": n_params,
        "fs": FS,
        "signal_length": SIGNAL_LEN,
    }
    with open(out.with_suffix(".training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved training info to {out.with_suffix('.training_info.json')}")

    # Full evaluation with existing evaluate_model function
    output_dir = out.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_model(model, test_loader, device, class_names, output_dir, prefix="cnn")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=None, help="MIT-BIH record directory (default: data/mitdb)")
    ap.add_argument("--out", default=str(SCRIPT_DIR.parent / "models-retrained" / "arrhythmia-cnn.pth"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    train(ap.parse_args())
