import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    def __init__(self, num_classes=2, hidden_size=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.gru = nn.GRU(32, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        batch, time, channels, height, width = x.shape
        x = x.view(batch * time, channels, height, width)
        features = self.cnn(x).view(batch, time, -1)
        _, hidden = self.gru(features)
        return self.classifier(hidden[-1])


class Compact3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x).flatten(1)
        return self.classifier(x)


class MotionAwareCNNGRU(nn.Module):
    def __init__(self, num_classes=2, hidden_size=64):
        super().__init__()
        self.raw_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.motion_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.gru = nn.GRU(64, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, raw, motion):
        batch, time, channels, height, width = raw.shape

        raw = raw.view(batch * time, channels, height, width)
        motion = motion.view(batch * time, channels, height, width)

        raw_features = self.raw_cnn(raw).view(batch, time, -1)
        motion_features = self.motion_cnn(motion).view(batch, time, -1)
        features = torch.cat([raw_features, motion_features], dim=-1)

        _, hidden = self.gru(features)
        return self.classifier(hidden[-1])


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
