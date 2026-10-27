"""Configurable VGG6: four 3x3 convolutions and two max-pooling layers."""
import torch
from torch import nn


def activation_layer(name):
    layers = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
              'silu': nn.SiLU, 'gelu': nn.GELU}
    if name not in layers:
        raise ValueError(f'Unknown activation {name!r}')
    return layers[name]()


class VGG6(nn.Module):
    def __init__(self, activation='relu', batch_norm=True, channels=(64, 128), num_classes=10):
        super().__init__()
        if len(channels) != 2 or any(c <= 0 for c in channels):
            raise ValueError('channels must have two positive values')
        blocks = []
        in_ch = 3
        for out_ch in channels:
            for _ in range(2):
                blocks.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not batch_norm))
                if batch_norm:
                    blocks.append(nn.BatchNorm2d(out_ch))
                blocks.append(activation_layer(activation))
                in_ch = out_ch
            blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))
