import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from synt.data import SyntDataset, SETUPS


class TimeConvNet(nn.Module):
    def __init__(
        self,
        input_shape,
        fs: int = 100,
        n_classes: int = 6,  # number of classes
    ):
        super().__init__()

        self.n_classes = n_classes
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=fs, stride=fs // 2),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Flatten(),
        )

        out_shape = self.conv(torch.rand(1, 1, *input_shape)).shape[-1]

        self.lin1 = nn.Linear(out_shape, self.n_classes)

    def forward(self, x):
        # x is batch_size, fs * length
        if len(x.shape) == 1:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        x = x.reshape(batch_size, 1, x.shape[-1])

        x = self.conv(x)
        x = self.lin1(x)

        if batch_size == 1:
            x = x.squeeze(0)
        return x


def _run_epoch(model, loader, criterion, device, train: bool, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_count += xb.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += xb.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def main():
    ap = argparse.ArgumentParser(description="Train the TimeConvNet classifier of each synthetic setup")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data-root", type=str, default=os.path.join(repo, "data", "synt-train"),
                    help="where the training datasets are generated/read (kept separate from the benchmark sets)")
    ap.add_argument("--n-samples", type=int, default=2000, help="samples per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=str, default=os.path.join(repo, "models-retrained"),
                    help="output directory; writes synt-setup{k}.pt and synt-setup{k}.metrics.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs, length, bandwidth = 100, 10, 5.0
    os.makedirs(args.out, exist_ok=True)

    for setup in range(len(SETUPS)):
        print(f"[Info] Training setup={setup} on {device}")
        dataset = SyntDataset(
            setup=setup,
            n_samples=args.n_samples,
            fs=fs,
            length=length,
            bandwidth=bandwidth,
            return_mask=False,
            output_dir=args.data_root,
        )

        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val
        train_set, val_set, test_set = random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        model = TimeConvNet(input_shape=(fs * length,), fs=fs, n_classes=dataset.n_class).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_state, best_val_acc = None, -1.0
        for epoch in range(args.epochs):
            train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, train=True, optimizer=optimizer)
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
            print(f"[Info] setup={setup} epoch={epoch+1:02d} train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {"model": model.state_dict(), "epoch": epoch + 1, "val_acc": val_acc}

        if best_state is None:
            print(f"[Warn] No best model captured for setup={setup}")
            continue
        model.load_state_dict(best_state["model"])
        train_loss, train_acc = _evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = _evaluate(model, test_loader, criterion, device)

        model_path = os.path.join(args.out, f"synt-setup{setup}.pt")
        torch.save(best_state, model_path)
        metrics = {"train": {"loss": train_loss, "acc": train_acc}, "val": {"loss": val_loss, "acc": val_acc},
                   "test": {"loss": test_loss, "acc": test_acc}}
        with open(os.path.join(args.out, f"synt-setup{setup}.metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Info] Saved {model_path} (val_acc={best_val_acc:.2f}, test_acc={test_acc:.2f})")


if __name__ == "__main__":
    main()
