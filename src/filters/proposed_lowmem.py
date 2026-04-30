from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common import FilterResult
from src.data.event_io import EVENT_P, EVENT_T, EVENT_X, EVENT_Y, normalize_events
from src.filters._shared import empty_filter_result, output_event_row, stack_output_rows


@dataclass(frozen=True)
class ProposedLowMemConfig:
    tau_ref_dig_us: int
    delta_t_us: int
    k0: int
    k_high: int
    alpha: float = 0.01
    r_max_hz: float = 5_000.0
    u_hot: int = 32
    t_recover_us: int = 1_000_000
    eps_us: int = 1


class ProposedLowMemFilter:
    def __init__(self, sensor_size: tuple[int, int, int], config: ProposedLowMemConfig):
        self.sensor_size = sensor_size
        self.config = config

    def apply(self, events: np.ndarray) -> FilterResult:
        events = normalize_events(events)
        if len(events) == 0:
            return empty_filter_result(stats={"score_mode": "proposed"})

        width, height, _ = self.sensor_size
        config = self.config
        row_raw = np.full(height, -10**18, dtype=np.int64)
        col_raw = np.full(width, -10**18, dtype=np.int64)
        row_acc = np.full(height, -10**18, dtype=np.int64)
        col_acc = np.full(width, -10**18, dtype=np.int64)
        row_rate = np.zeros(height, dtype=np.float32)
        col_rate = np.zeros(width, dtype=np.float32)
        row_unsupported = np.zeros(height, dtype=np.uint16)
        col_unsupported = np.zeros(width, dtype=np.uint16)
        row_hot = np.zeros(height, dtype=bool)
        col_hot = np.zeros(width, dtype=bool)

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
            dt_row = t - int(row_raw[y])
            dt_col = t - int(col_raw[x])
            row_rate[y] = float(config.alpha * (1_000_000.0 / max(dt_row, config.eps_us)) + (1.0 - config.alpha) * row_rate[y])
            col_rate[x] = float(config.alpha * (1_000_000.0 / max(dt_col, config.eps_us)) + (1.0 - config.alpha) * col_rate[x])

            if row_hot[y] and dt_row >= config.t_recover_us:
                row_hot[y] = False
                row_unsupported[y] = 0
                row_rate[y] = 0.0
            if col_hot[x] and dt_col >= config.t_recover_us:
                col_hot[x] = False
                col_unsupported[x] = 0
                col_rate[x] = 0.0

            dt_raw = min(dt_row, dt_col)
            guarded = bool(row_hot[y] or col_hot[x] or dt_raw < config.tau_ref_dig_us)
            if not guarded:
                support_count = 0
                for ny in (y - 1, y + 1):
                    if 0 <= ny < height and 0 < (t - int(row_acc[ny])) < config.delta_t_us:
                        support_count += 1
                for nx in (x - 1, x + 1):
                    if 0 <= nx < width and 0 < (t - int(col_acc[nx])) < config.delta_t_us:
                        support_count += 1
                support[idx] = support_count
                if support_count == 0:
                    row_unsupported[y] = np.uint16(min(int(row_unsupported[y]) + 1, np.iinfo(np.uint16).max))
                    col_unsupported[x] = np.uint16(min(int(col_unsupported[x]) + 1, np.iinfo(np.uint16).max))
                else:
                    row_unsupported[y] = 0
                    col_unsupported[x] = 0

                if row_rate[y] > config.r_max_hz and row_unsupported[y] >= config.u_hot:
                    row_hot[y] = True
                if col_rate[x] > config.r_max_hz and col_unsupported[x] >= config.u_hot:
                    col_hot[x] = True

                if support_count >= config.k0:
                    accepted_mask[idx] = True
                    confidence[idx] = 1 + int(support_count >= config.k_high)
                    accepted_rows.append(output_event_row(x, y, t, p, confidence=int(confidence[idx])))
                    row_acc[y] = t
                    col_acc[x] = t
            row_raw[y] = t
            col_raw[x] = t

        output_events = stack_output_rows(accepted_rows)
        return FilterResult(
            accepted_mask=accepted_mask,
            confidence=confidence,
            support=support,
            pair_flag=pair_flag,
            output_events=output_events,
            stats={
                "accepted_count": int(np.sum(accepted_mask)),
                "score_mode": "proposed",
            },
        )
