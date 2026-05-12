"""PyTorch Dataset classes for loading mel-spectrogram and MFCC tensors."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

VALID_SPLITS = {"train", "val", "test"}


def _validate_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _load_genre_mapping(path: Path) -> dict:
    """Load genre_mapping.json and return a dict of {int_index: genre_name}."""
    _require_file(path)
    with open(path) as f:
        raw = json.load(f)
    first_key = next(iter(raw))
    if isinstance(raw[first_key], int):
        # Format: {"Rock": 0, "Pop": 1, ...}
        return {v: k for k, v in raw.items()}
    else:
        # Format: {"0": "Rock", "1": "Pop", ...}
        return {int(k): v for k, v in raw.items()}


class SpectrogramDataset(Dataset):
    def __init__(self, tensors_dir: Path, split: str):
        _validate_split(split)
        tensors_dir = Path(tensors_dir)

        spec_path = tensors_dir / f"spectrograms_{split}.pt"
        labels_path = tensors_dir / f"labels_{split}.pt"
        mapping_path = tensors_dir / "genre_mapping.json"

        _require_file(spec_path)
        _require_file(labels_path)

        self.spectrograms = torch.load(spec_path, weights_only=True).to(torch.float32)
        self.labels = torch.load(labels_path, weights_only=True).to(torch.long)
        self.genre_mapping = _load_genre_mapping(mapping_path)
        self.split = split

        if self.spectrograms.ndim != 4 or self.spectrograms.shape[1:] != (1, 128, 128):
            raise ValueError(
                f"Expected spectrogram shape (N, 1, 128, 128), "
                f"got {tuple(self.spectrograms.shape)}"
            )

        if self.spectrograms.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Feature rows ({self.spectrograms.shape[0]}) != "
                f"label rows ({self.labels.shape[0]})"
            )

    def __len__(self) -> int:
        return self.spectrograms.shape[0]

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]

    @property
    def genres(self) -> list:
        return [self.genre_mapping[i] for i in sorted(self.genre_mapping)]

    @property
    def num_classes(self) -> int:
        return len(self.genre_mapping)

    def get_class_weights(self) -> torch.FloatTensor:
        n = len(self)
        c = self.num_classes
        counts = torch.bincount(self.labels, minlength=c).float()
        counts = torch.clamp(counts, min=1.0)
        return (n / (c * counts)).to(torch.float32)

    def __repr__(self) -> str:
        return (
            f"SpectrogramDataset | split={self.split} | n={len(self)} | "
            f"num_classes={self.num_classes} | shape={tuple(self.spectrograms.shape)}"
        )


class MFCCDataset(Dataset):
    def __init__(self, tensors_dir: Path, split: str):
        _validate_split(split)
        tensors_dir = Path(tensors_dir)

        mfcc_path = tensors_dir / f"mfccs_{split}.pt"
        labels_path = tensors_dir / f"labels_{split}.pt"
        mapping_path = tensors_dir / "genre_mapping.json"

        _require_file(mfcc_path)
        _require_file(labels_path)

        self.mfccs = torch.load(mfcc_path, weights_only=True).to(torch.float32)
        self.labels = torch.load(labels_path, weights_only=True).to(torch.long)
        self.genre_mapping = _load_genre_mapping(mapping_path)
        self.split = split

        if self.mfccs.ndim != 3 or self.mfccs.shape[1:] != (130, 40):
            raise ValueError(
                f"Expected MFCC shape (N, 130, 40), "
                f"got {tuple(self.mfccs.shape)}"
            )

        if self.mfccs.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Feature rows ({self.mfccs.shape[0]}) != "
                f"label rows ({self.labels.shape[0]})"
            )

    def __len__(self) -> int:
        return self.mfccs.shape[0]

    def __getitem__(self, idx):
        return self.mfccs[idx], self.labels[idx]

    @property
    def genres(self) -> list:
        return [self.genre_mapping[i] for i in sorted(self.genre_mapping)]

    @property
    def num_classes(self) -> int:
        return len(self.genre_mapping)

    def get_class_weights(self) -> torch.FloatTensor:
        n = len(self)
        c = self.num_classes
        counts = torch.bincount(self.labels, minlength=c).float()
        counts = torch.clamp(counts, min=1.0)
        return (n / (c * counts)).to(torch.float32)

    def __repr__(self) -> str:
        return (
            f"MFCCDataset | split={self.split} | n={len(self)} | "
            f"num_classes={self.num_classes} | shape={tuple(self.mfccs.shape)}"
        )


if __name__ == "__main__":
    data_dir = Path("data/processed")

    for cls, name in [(SpectrogramDataset, "SpectrogramDataset"), (MFCCDataset, "MFCCDataset")]:
        print(f"\n--- {name} ---")
        try:
            ds = cls(data_dir, split="train")
            print(repr(ds))
            print(f"len:         {len(ds)}")
            sample, label = ds[0]
            print(f"sample shape: {tuple(sample.shape)}, label: {label.item()}")
            print(f"num_classes: {ds.num_classes}")
            print(f"genres:      {ds.genres}")
            print(f"class_weights shape: {tuple(ds.get_class_weights().shape)}")
        except FileNotFoundError as e:
            print(f"Skipped — {e}")
