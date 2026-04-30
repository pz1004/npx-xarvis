from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common import FilterResult
from src.data.event_io import (
    EVENT_P,
    EVENT_T,
    EVENT_X,
    EVENT_Y,
    normalize_events,
)
from src.filters._shared import empty_filter_result, output_event_row, stack_output_rows


@dataclass(frozen=True)
class ProposedBalancedConfig:
    tau_ref_dig_us: int
    delta_t_us: int
    k0: int
    gamma: int
    tau_pair_us: int
    k_high: int
    alpha: float = 0.01
    r_max_hz: float = 5_000.0
    u_hot: int = 32
    t_recover_us: int = 1_000_000
    eps_us: int = 1
    stage_variant: str = "conf"
    stage1_mode: str = "offline_noop"


class ProposedBalancedFilter:
    def __init__(self, sensor_size: tuple[int, int, int], config: ProposedBalancedConfig):
        self.sensor_size = sensor_size
        self.config = config

    @staticmethod
    def estimate_rmax_hz(
        calibration_events: list[np.ndarray],
        percentile: float = 99.9,
        minimum_hz: float = 1_000.0,
        maximum_hz: float = 20_000.0,
    ) -> float:
        rates: list[float] = []
        for events in calibration_events:
            normalized = normalize_events(events)
            if len(normalized) < 2:
                continue
            dt = np.diff(normalized[:, EVENT_T])
            dt = dt[dt > 0]
            if len(dt) == 0:
                continue
            rates.extend((1_000_000.0 / dt).tolist())
        if not rates:
            return minimum_hz
        estimated = float(np.percentile(np.asarray(rates, dtype=np.float64), percentile))
        return float(np.clip(estimated, minimum_hz, maximum_hz))

    def apply(self, events: np.ndarray) -> FilterResult:
        events = normalize_events(events)
        if len(events) == 0:
            return empty_filter_result(stats={"accepted_count": 0, "mean_support": 0.0, "score_mode": "proposed"})

        width, height, _ = self.sensor_size
        config = self.config

        t_raw_last = np.full((height, width), -10**18, dtype=np.int64)
        p_raw_last = np.zeros((height, width), dtype=np.int8)
        rate_hz = np.zeros((height, width), dtype=np.float32)
        unsupported = np.zeros((height, width), dtype=np.uint16)
        hot = np.zeros((height, width), dtype=bool)

        t_acc_pos = np.full((height, width), -10**18, dtype=np.int64)
        t_acc_neg = np.full((height, width), -10**18, dtype=np.int64)

        accepted_mask = np.zeros(len(events), dtype=bool)
        confidence = np.zeros(len(events), dtype=np.int8)
        support = np.zeros(len(events), dtype=np.int16)
        pair_flag = np.zeros(len(events), dtype=np.int8)
        accepted_rows: list[np.ndarray] = []

        for idx, event in enumerate(events):
            x = int(event[EVENT_X])
            y = int(event[EVENT_Y])
            t = int(event[EVENT_T])
            p = int(event[EVENT_P])

            dt_raw = t - int(t_raw_last[y, x])
            pair = int((p != int(p_raw_last[y, x])) and (dt_raw < config.tau_pair_us))
            pair_flag[idx] = pair

            instant_rate_hz = 1_000_000.0 / max(dt_raw, config.eps_us)
            rate_hz[y, x] = float(config.alpha * instant_rate_hz + (1.0 - config.alpha) * rate_hz[y, x])

            if hot[y, x] and dt_raw >= config.t_recover_us:
                hot[y, x] = False
                unsupported[y, x] = 0
                rate_hz[y, x] = 0.0

            guarded = bool(hot[y, x] or dt_raw < config.tau_ref_dig_us)
            accepted = False
            conf = 0
            support_count = 0

            if not guarded:
                if config.stage_variant != "ref":
                    support_count = self._support_count(
                        x=x,
                        y=y,
                        t=t,
                        polarity=p,
                        t_acc_pos=t_acc_pos,
                        t_acc_neg=t_acc_neg,
                        delta_t_us=config.delta_t_us,
                    )
                    support[idx] = support_count

                    if config.stage_variant == "sup":
                        accepted = support_count >= config.k0
                    else:
                        accepted = support_count >= (config.k0 + config.gamma * pair)

                    if support_count == 0:
                        unsupported[y, x] = np.uint16(min(int(unsupported[y, x]) + 1, np.iinfo(np.uint16).max))
                    else:
                        unsupported[y, x] = 0

                    if rate_hz[y, x] > config.r_max_hz and unsupported[y, x] >= config.u_hot:
                        hot[y, x] = True
                else:
                    accepted = True

                if accepted:
                    accepted_mask[idx] = True
                    if config.stage_variant == "conf":
                        conf = 1 + int(support_count >= config.k_high)
                    else:
                        conf = 1
                    confidence[idx] = conf
                    accepted_rows.append(output_event_row(x, y, t, p, confidence=conf))

                    if config.stage_variant != "ref":
                        if p > 0:
                            t_acc_pos[y, x] = t
                        else:
                            t_acc_neg[y, x] = t
            else:
                confidence[idx] = 0

            t_raw_last[y, x] = t
            p_raw_last[y, x] = p

        output_events = stack_output_rows(accepted_rows)
        return FilterResult(
            accepted_mask=accepted_mask,
            confidence=confidence,
            support=support,
            pair_flag=pair_flag,
            output_events=output_events,
            stats={
                "accepted_count": int(np.sum(accepted_mask)),
                "mean_support": float(np.mean(support)) if len(support) else 0.0,
                "stage_variant": config.stage_variant,
                "score_mode": "proposed",
            },
        )

    @staticmethod
    def _support_count(
        x: int,
        y: int,
        t: int,
        polarity: int,
        t_acc_pos: np.ndarray,
        t_acc_neg: np.ndarray,
        delta_t_us: int,
    ) -> int:
        support_count = 0
        acc_map = t_acc_pos if polarity > 0 else t_acc_neg
        height, width = acc_map.shape
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if nx == x and ny == y:
                    continue
                lag = t - int(acc_map[ny, nx])
                if 0 < lag < delta_t_us:
                    support_count += 1
        return support_count
