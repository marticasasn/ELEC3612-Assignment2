"""CNN 2D and LSTM architecture definitions for music genre classification."""

import torch
import torch.nn as nn

NUM_CLASSES = 12


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
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

        self.features = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 128, 128):
            raise ValueError(
                f"Expected input shape (batch, 1, 128, 128), got {tuple(x.shape)}"
            )
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

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

        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
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
        # Take the final forward and backward hidden states from the last layer.
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, 512)
        return self.classifier(out)

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
