from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate, utils


class FrameSNN(nn.Module):
    def __init__(self, num_classes: int, beta: float = 0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes, bias=False)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 5 or x.shape[2] != 2:
            raise ValueError(f"Expected input shape [T, B, 2, H, W], got {tuple(x.shape)}")

    def _forward_step(self, current: torch.Tensor) -> torch.Tensor:
        spk1 = self.lif1(self.conv1(current))
        spk2 = self.lif2(self.bn2(self.conv2(spk1)))
        spk3 = self.lif3(self.bn3(self.conv3(spk2)))
        pooled = self.pool(spk3).flatten(1)
        return self.lif_out(self.fc(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        utils.reset(self)
        self._validate_input(x)
        spikes = []
        for step in range(x.shape[0]):
            spikes.append(self._forward_step(x[step]))
        return torch.stack(spikes, dim=0)
