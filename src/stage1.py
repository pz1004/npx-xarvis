from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Stage1Config:
    mode: str = "offline_noop"
    nominal_bias: dict[str, Any] = field(default_factory=dict)
    low_light_bias: dict[str, Any] = field(default_factory=dict)
    calibration_stats: dict[str, Any] = field(default_factory=dict)
    manifest_name: str = "default"


def run_stage1(config: Stage1Config) -> dict[str, Any]:
    if config.mode == "offline_noop":
        return {
            "mode": config.mode,
            "manifest_name": config.manifest_name,
            "bypassed": True,
            "nominal_bias": {},
            "low_light_bias": {},
            "calibration_stats": config.calibration_stats,
        }
    if config.mode == "sensor_manifest":
        return {
            "mode": config.mode,
            "manifest_name": config.manifest_name,
            "bypassed": False,
            "nominal_bias": dict(config.nominal_bias),
            "low_light_bias": dict(config.low_light_bias),
            "calibration_stats": dict(config.calibration_stats),
        }
    raise ValueError(f"Unsupported Stage 1 mode: {config.mode}")
