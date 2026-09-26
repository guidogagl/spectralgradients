import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

# ============================================================
# 1) Config
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO, "data", "speech")
NUM_WORKERS = 2
LABEL_SMOOTHING = 0.1
BEST_CHECKPOINT_PATH = os.path.join(REPO, "models-retrained", "audio-wavcnn.pt")

# ============================================================
# 3) Collate function: mono + padding variabile
# ============================================================
def make_collate_fn(label_to_idx):
    def collate_fn(batch):
        waveforms = []
        targets = []
        lengths = []

        max_len = 0
        for waveform, sr, label, speaker_id, utt_id in batch:
            # waveform shape: [channels, time]
            # Se più canali, media in mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # waveform shape finale: [1, time]
            waveforms.append(waveform)
            targets.append(label_to_idx[label])
            lengths.append(waveform.shape[1])
            if waveform.shape[1] > max_len:
                max_len = waveform.shape[1]

        # padding a destra fino a max_len del batch
        padded = []
        for w in waveforms:
            pad_amount = max_len - w.shape[1]
            if pad_amount > 0:
                w = nn.functional.pad(w, (0, pad_amount))
            padded.append(w)

        x = torch.stack(padded, dim=0)  # [B, 1, T]
        y = torch.tensor(targets, dtype=torch.long)
        seq_lengths = torch.tensor(lengths, dtype=torch.long)
        return x, y, seq_lengths

    return collate_fn

# ============================================================
# 4) Modello: CNN1d su waveform grezzo
# ============================================================
class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class WaveCNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            WaveBlock(1, 32, kernel_size=15, dropout=0.10),
            WaveBlock(32, 64, kernel_size=11, dropout=0.15),
            WaveBlock(64, 128, kernel_size=9, dropout=0.20),
            WaveBlock(128, 192, kernel_size=7, dropout=0.25),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def _wave_augment(self, x):
        # x: [B, 1, T]
        if torch.rand(1).item() < 0.7:
            gain = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.8, 1.2)
            x = x * gain
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * 0.003
            x = x + noise
        if torch.rand(1).item() < 0.5:
            max_shift = max(1, x.shape[-1] // 20)
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item())
            x = torch.roll(x, shifts=shift, dims=-1)
        return x.clamp(-1.0, 1.0)

    def forward(self, x, is_train=False):
        # x: [B, 1, T]
        if is_train:
            x = self._wave_augment(x)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        x = (x - mean) / std

        x = self.features(x)
        x = self.head(x)
        return x

# ============================================================
# 5) Train / Eval loops
# ============================================================
def run_epoch(loader, model, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for x, y, lengths in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x, is_train=is_train)
                loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Train the WaveCNN keyword classifier on Google Speech Commands v0.02")
    ap.add_argument("--data-root", default=DATA_ROOT, help="torchaudio root (the dataset is downloaded here if absent)")
    ap.add_argument("--out", default=BEST_CHECKPOINT_PATH, help="checkpoint path")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--download-only", action="store_true", help="download/verify the dataset and exit")
    return ap.parse_args()


def main():
    global DATA_ROOT, BEST_CHECKPOINT_PATH, EPOCHS, NUM_WORKERS
    args = parse_args()
    DATA_ROOT, BEST_CHECKPOINT_PATH, EPOCHS, NUM_WORKERS = args.data_root, args.out, args.epochs, args.workers
    torch.manual_seed(42)
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(BEST_CHECKPOINT_PATH)), exist_ok=True)

    # ============================================================
    # 2) Dataset: official SpeechCommands splits (downloaded on first use)
    # ============================================================
    train_set = SPEECHCOMMANDS(root=DATA_ROOT, download=True, subset="training")
    val_set = SPEECHCOMMANDS(root=DATA_ROOT, download=True, subset="validation")
    test_set = SPEECHCOMMANDS(root=DATA_ROOT, download=True, subset="testing")
    if args.download_only:
        print(f"Speech Commands ready under {DATA_ROOT} ({len(train_set)}/{len(val_set)}/{len(test_set)} clips)")
        return

    # Costruisco label map dal training set
    labels = sorted(list(set(item[2] for item in train_set)))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    print(f"Numero classi: {len(labels)}")
    print("Prime classi:", labels[:10])

    collate_fn = make_collate_fn(label_to_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = WaveCNNClassifier(num_classes=len(labels)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # ============================================================
    # 6) Training con checkpointing best validation
    # ============================================================
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            train_loader, model, criterion, device=DEVICE, optimizer=optimizer, scaler=scaler
        )
        val_loss, val_acc = run_epoch(val_loader, model, criterion, device=DEVICE, optimizer=None, scaler=None)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "labels": labels,
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "label_smoothing": LABEL_SMOOTHING,
                    },
                },
                BEST_CHECKPOINT_PATH,
            )
            print(f"Nuovo best checkpoint salvato: {BEST_CHECKPOINT_PATH} (val_acc={val_acc:.4f})")

    # Carico il miglior checkpoint prima del test finale
    if os.path.exists(BEST_CHECKPOINT_PATH):
        checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Caricato best checkpoint da epoca {checkpoint['epoch']} "
            f"con val_acc={checkpoint['val_acc']:.4f}"
        )

    # ============================================================
    # 7) Test finale
    # ============================================================
    test_loss, test_acc = run_epoch(test_loader, model, criterion, device=DEVICE, optimizer=None, scaler=None)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()