import torch
from torch import nn


class TetherNet(nn.Module):
    """64x64 grayscale -> absent/left/center/right line classification.

    Logits are perception outputs, never flight commands. A visible line is
    insufficient to infer physical tension or 3D cable angle.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(4),
            nn.Flatten(), nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.layers(x)
