"""Training loop with validation tracking and checkpoint saving."""

import time
from pathlib import Path

import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients, which are common in deep
        # networks and recurrent architectures. max_norm=1.0 is a conservative cap.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate weighted loss so the per-epoch average is sample-count-correct
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {"loss": total_loss / total, "accuracy": correct / total}


def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradient tracking during inference to save memory
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return {"loss": total_loss / total, "accuracy": correct / total}


def _step_scheduler(scheduler, val_loss: float) -> None:
    if scheduler is None:
        return
    # ReduceLROnPlateau monitors a metric and requires it to be passed explicitly.
    # All other schedulers (StepLR, CosineAnnealingLR, etc.) step unconditionally.
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs: int = 50,
    patience: int = 10,
    checkpoint_path: Path = None,
    scheduler=None,
) -> dict:
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_acc": 0.0,
        "best_val_loss": float("inf"),
    }

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Counter for epochs without improvement — triggers early stopping at `patience`
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        saved = ""
        if val_acc > history["best_val_acc"]:
            # New best validation accuracy — save the full checkpoint and reset patience.
            # We track val_acc (not val_loss) as the primary criterion because the
            # weighted CrossEntropyLoss can decrease while accuracy stagnates on
            # imbalanced classes.
            history["best_val_acc"] = val_acc
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            epochs_no_improve = 0

            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                        "val_loss": val_loss,
                        "history": history,  # saved with checkpoint for later plotting
                    },
                    checkpoint_path,
                )
            saved = " [SAVED]"
        else:
            epochs_no_improve += 1

        # Step the LR scheduler after validation so it responds to the latest metrics
        _step_scheduler(scheduler, val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"time={elapsed:.1f}s{saved}"
        )

        # Stop training early if validation accuracy has not improved for `patience` epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs (patience={patience}).")
            break

    return history


def load_checkpoint(
    model,
    checkpoint_path: Path,
    optimizer=None,
    device="cpu",
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        # Restore optimizer state so training can resume without a learning-rate reset
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_x = torch.randn(64, 10)
    dummy_y = torch.randint(0, 3, (64,))
    dataset = TensorDataset(dummy_x, dummy_y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = train(
        model, train_loader, val_loader,
        optimizer, criterion, device,
        num_epochs=3, patience=10, checkpoint_path=None,
    )
    print("\nHistory:")
    for k, v in history.items():
        print(f"  {k}: {v}")
