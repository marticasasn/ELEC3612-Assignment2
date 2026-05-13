"""Metrics computation and confusion matrix plotting for model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true, y_pred, class_names: list) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = list(range(len(class_names)))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            name: float(score) for name, score in zip(class_names, per_class_f1)
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
        ),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list,
    save_path: Path = None,
    title: str = "Confusion Matrix",
) -> None:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for classes with no true samples.
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_training_curves(
    history: dict,
    save_path: Path = None,
    title: str = "Training Curves",
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    if "best_epoch" in history:
        best_epoch = history["best_epoch"]
    else:
        best_epoch = int(np.argmax(history["val_acc"])) + 1

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.axvline(best_epoch, linestyle="--", color="gray", label=f"best epoch ({best_epoch})")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.axvline(best_epoch, linestyle="--", color="gray", label=f"best epoch ({best_epoch})")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def get_predictions(model, loader, device) -> tuple:
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(labels.cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_pred)


def evaluate_model(
    model,
    loader,
    device,
    class_names: list,
    checkpoint_path: Path = None,
    figures_dir: Path = None,
    model_name: str = "model",
) -> dict:
    history = None

    if checkpoint_path is not None:
        ckpt = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        history = ckpt.get("history", None)

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = get_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)

    print(metrics["classification_report"])

    cm_save = figures_dir / f"{model_name}_confusion_matrix.png" if figures_dir else None
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=cm_save,
        title=f"{model_name} — Confusion Matrix",
    )

    if history is not None:
        curve_save = figures_dir / f"{model_name}_training_curves.png" if figures_dir else None
        plot_training_curves(
            history,
            save_path=curve_save,
            title=f"{model_name} — Training Curves",
        )

    return metrics


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    class_names = ["Rock", "Pop", "Jazz"]
    y_true = rng.integers(0, 3, size=120)
    y_pred = rng.integers(0, 3, size=120)

    metrics = compute_metrics(y_true, y_pred, class_names)
    print(f"accuracy:     {metrics['accuracy']:.4f}")
    print(f"macro_f1:     {metrics['macro_f1']:.4f}")
    print(f"weighted_f1:  {metrics['weighted_f1']:.4f}")
    print(metrics["classification_report"])

    plot_confusion_matrix(y_true, y_pred, class_names)
