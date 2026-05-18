from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common import FilterResult
from src.data.event_io import EVENT_P, EVENT_T, EVENT_X, EVENT_Y, normalize_events
from src.filters._shared import empty_filter_result, output_event_row, stack_output_rows


@dataclass(frozen=True)
class STCFRCConfig:
    delta_t_us: int


class STCFRCFilter:
    def __init__(self, sensor_size: tuple[int, int, int], config: STCFRCConfig):
        self.sensor_size = sensor_size
        self.config = config

    def apply(self, events: np.ndarray) -> FilterResult:
        events = normalize_events(events)
        if len(events) == 0:
            return empty_filter_result(stats={"score_mode": "binary"})

        width, height, _ = self.sensor_size
        row_last = np.full(height, -10**18, dtype=np.int64)
        col_last = np.full(width, -10**18, dtype=np.int64)

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
            support_count = 0
            for ny in (y - 1, y + 1):
                if 0 <= ny < height and 0 < (t - int(row_last[ny])) < self.config.delta_t_us:
                    support_count += 1
            for nx in (x - 1, x + 1):
                if 0 <= nx < width and 0 < (t - int(col_last[nx])) < self.config.delta_t_us:
                    support_count += 1
            support[idx] = support_count
            if support_count >= 1:
                accepted_mask[idx] = True
                confidence[idx] = 1
                accepted_rows.append(output_event_row(x, y, t, p, confidence=1))
            row_last[y] = t
            col_last[x] = t

        output_events = stack_output_rows(accepted_rows)
        return FilterResult(
            accepted_mask=accepted_mask,
            confidence=confidence,
            support=support,
            pair_flag=pair_flag,
            output_events=output_events,
            stats={"accepted_count": int(np.sum(accepted_mask)), "score_mode": "binary"},
        )
