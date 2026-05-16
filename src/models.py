"""CNN 2D and BiLSTM architecture definitions for music genre classification."""

import torch
import torch.nn as nn

NUM_CLASSES = 12


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    # Standard Conv → BatchNorm → ReLU → MaxPool block.
    # BatchNorm stabilises training by normalising activations after each conv layer.
    # MaxPool(2,2) halves the spatial dimensions, doubling the receptive field.
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # Four convolutional blocks progressively double the channel depth
        # (1 → 32 → 64 → 128 → 256) while halving the spatial size at each stage.
        # Input: (batch, 1, 128, 128) → after 4x MaxPool: (batch, 256, 8, 8)
        self.features = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )

        # AdaptiveAvgPool squeezes spatial dims to a fixed 4×4 regardless of input size.
        # This decouples the classifier from the exact spectrogram resolution.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Two-stage FC head with Dropout for regularisation.
        # Dropout(0.5) after flatten is aggressive — reduces co-adaptation of features.
        # Dropout(0.3) before the final layer provides lighter regularisation.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),  # 256 channels × 4×4 spatial = 4096 inputs
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 128, 128):
            raise ValueError(
                f"Expected input shape (batch, 1, 128, 128), got {tuple(x.shape)}"
            )
        x = self.features(x)   # (batch, 256, 8, 8)
        x = self.pool(x)        # (batch, 256, 4, 4)
        return self.classifier(x)  # (batch, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (
            f"CNNClassifier | num_classes={self.num_classes} | "
            f"input=(batch, 1, 128, 128) | params={self.count_parameters():,}"
        )


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # Bidirectional LSTM processes 130 time frames of 40 MFCC coefficients.
        # bidirectional=True doubles the effective hidden size (256 × 2 = 512),
        # capturing both past and future temporal context at each frame.
        # inter-layer dropout=0.3 is applied between the two LSTM layers.
        self.lstm = nn.LSTM(
            input_size=40,      # 40 MFCC coefficients per frame
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,        # dropout between LSTM layers (not applied on last layer)
            bidirectional=True,
        )

        # Classifier head: input size is 512 = 256 (forward) + 256 (backward).
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (130, 40):
            raise ValueError(
                f"Expected input shape (batch, 130, 40), got {tuple(x.shape)}"
            )
        _, (hidden, _) = self.lstm(x)
        # hidden shape: (num_layers * 2, batch, hidden_size)
        # hidden[-2] = final forward hidden state of the last layer
        # hidden[-1] = final backward hidden state of the last layer
        # Concatenating them gives a single (batch, 512) summary of the full sequence.
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 512)
        return self.classifier(out)  # (batch, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (
            f"LSTMClassifier | num_classes={self.num_classes} | "
            f"input=(batch, 130, 40) | params={self.count_parameters():,}"
        )


if __name__ == "__main__":
    cnn = CNNClassifier()
    lstm = LSTMClassifier()

    for model in (cnn, lstm):
        print(model)
        print(f"Trainable parameters: {model.count_parameters():,}")
        print(model.summary())

    cnn_out = cnn(torch.randn(4, 1, 128, 128))
    print(f"\nCNN output shape:  {tuple(cnn_out.shape)}")

    lstm_out = lstm(torch.randn(4, 130, 40))
    print(f"LSTM output shape: {tuple(lstm_out.shape)}")
