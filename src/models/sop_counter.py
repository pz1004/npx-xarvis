from __future__ import annotations

import torch
import torch.nn as nn


def parameter_memory_bytes(model: nn.Module) -> int:
    return int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _register_activation_hooks(model: nn.Module, accumulator: dict[str, int]) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        if isinstance(output, torch.Tensor):
            accumulator["peak"] = max(accumulator["peak"], int(output.numel() * output.element_size()))

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.AdaptiveAvgPool2d)):
            handles.append(module.register_forward_hook(hook))
    return handles


def _run_profile_forward(model: nn.Module, sample: torch.Tensor) -> None:
    device = next(model.parameters()).device
    with torch.no_grad():
        _ = model(sample.to(device))


def _remove_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def _register_compute_hooks(
    model: nn.Module,
    conv_hook,
    linear_hook,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    return handles


def measure_peak_activation_bytes(model: nn.Module, sample: torch.Tensor) -> int:
    accumulator = {"peak": 0}
    handles = _register_activation_hooks(model, accumulator)
    _run_profile_forward(model, sample)
    _remove_hooks(handles)
    return accumulator["peak"]


def count_event_sops(model: nn.Module, sample: torch.Tensor) -> int:
    total = {"ops": 0}

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor):
        input_tensor = inputs[0]
        nonzero = int(torch.count_nonzero(input_tensor).item())
        kernel_h, kernel_w = module.kernel_size
        ops = nonzero * kernel_h * kernel_w * (module.out_channels // module.groups)
        total["ops"] += int(ops)

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor):
        input_tensor = inputs[0]
        nonzero = int(torch.count_nonzero(input_tensor).item())
        total["ops"] += nonzero * module.out_features

    handles = _register_compute_hooks(model, conv_hook, linear_hook)
    _run_profile_forward(model, sample)
    _remove_hooks(handles)
    return total["ops"]


def count_dense_macs(model: nn.Module, sample: torch.Tensor) -> int:
    total = {"macs": 0}

    def conv_hook(module: nn.Conv2d, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        output_elements = int(output.numel())
        kernel_h, kernel_w = module.kernel_size
        macs = output_elements * (module.in_channels // module.groups) * kernel_h * kernel_w
        total["macs"] += macs

    def linear_hook(module: nn.Linear, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        total["macs"] += int(output.numel()) * module.in_features

    handles = _register_compute_hooks(model, conv_hook, linear_hook)
    _run_profile_forward(model, sample)
    _remove_hooks(handles)
    return total["macs"]
