from __future__ import annotations

from dataclasses import dataclass

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import surrogate, utils


@dataclass(frozen=True)
class BackboneConfig:
    num_classes: int
    beta: float = 0.9
    low_weight_init: float = 1.0
    high_weight_init: float = 1.5


class EventSNN(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.w_low = nn.Parameter(torch.tensor(float(config.low_weight_init)))
        self.w_high = nn.Parameter(torch.tensor(float(config.high_weight_init)))

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False)
        self.lif1 = snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, config.num_classes, bias=False)
        self.lif_out = snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True)

    def collapse_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 4:
            raise ValueError(f"Expected input shape [T, B, 4, H, W], got {tuple(x.shape)}")
        on = self.w_low * x[:, :, 0] + self.w_high * x[:, :, 1]
        off = self.w_low * x[:, :, 2] + self.w_high * x[:, :, 3]
        return torch.stack([on, off], dim=2)

    @staticmethod
    def _winner_take_all(current: torch.Tensor) -> torch.Tensor:
        max_per_location = current.amax(dim=1, keepdim=True)
        winners = current >= max_per_location
        return current * winners.to(dtype=current.dtype)

    def _forward_step(self, current: torch.Tensor) -> torch.Tensor:
        coincidence = self._winner_take_all(self.conv1(current))
        spk1 = self.lif1(coincidence)
        spk2 = self.lif2(self.bn2(self.conv2(spk1)))
        spk3 = self.lif3(self.bn3(self.conv3(spk2)))
        pooled = self.pool(spk3).flatten(1)
        return self.lif_out(self.fc(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        utils.reset(self)
        collapsed = self.collapse_input(x)
        spikes = []
        for step in range(collapsed.shape[0]):
            spikes.append(self._forward_step(collapsed[step]))
        return torch.stack(spikes, dim=0)

    def confidence_ratio(self) -> torch.Tensor:
        return self.w_high / torch.clamp(self.w_low, min=1e-6)


def spike_rate_cross_entropy(spike_record: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(spike_record.sum(dim=0), targets)


def spike_accuracy(spike_record: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = spike_record.sum(dim=0).argmax(dim=1)
    return float((predictions == targets).float().mean().item())
