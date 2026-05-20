from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, types
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.common import FilterResult
from src.data.event_io import (
    EVENT_P,
    EVENT_T,
    EVENT_X,
    EVENT_Y,
    OUTPUT_CONFIDENCE,
    empty_output_events,
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
        self._init_state()

    def _init_state(self):
        width, height, _ = self.sensor_size
        self.t_raw_last = np.full((height, width), -10**18, dtype=np.int64)
        self.p_raw_last = np.zeros((height, width), dtype=np.int8)
        self.rate_hz = np.zeros((height, width), dtype=np.float32)
        self.unsupported = np.zeros((height, width), dtype=np.uint16)
        self.hot = np.zeros((height, width), dtype=bool)
        self.t_acc_pos = np.full((height, width), -10**18, dtype=np.int64)
        self.t_acc_neg = np.full((height, width), -10**18, dtype=np.int64)

    def _get_state(self):
        return (
            self.t_raw_last.copy(),
            self.p_raw_last.copy(),
            self.rate_hz.copy(),
            self.unsupported.copy(),
            self.hot.copy(),
            self.t_acc_pos.copy(),
            self.t_acc_neg.copy(),
        )

    def _set_state(self, state):
        (
            self.t_raw_last,
            self.p_raw_last,
            self.rate_hz,
            self.unsupported,
            self.hot,
            self.t_acc_pos,
            self.t_acc_neg,
        ) = state

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

        if HAS_NUMBA:
            # Call Numba JIT compiled function
            accepted_mask, confidence, support, pair_flag, new_t_raw_last, new_p_raw_last, new_rate_hz, new_unsupported, new_hot, new_t_acc_pos, new_t_acc_neg = _apply_filter_jit(
                events,
                width,
                height,
                config.tau_ref_dig_us,
                config.delta_t_us,
                config.k0,
                config.gamma,
                config.tau_pair_us,
                config.k_high,
                config.alpha,
                config.r_max_hz,
                config.u_hot,
                config.t_recover_us,
                config.stage_variant,
                self.t_raw_last,
                self.p_raw_last,
                self.rate_hz,
                self.unsupported,
                self.hot,
                self.t_acc_pos,
                self.t_acc_neg,
            )
            # Update internal state after JIT execution
            self.t_raw_last = new_t_raw_last
            self.p_raw_last = new_p_raw_last
            self.rate_hz = new_rate_hz
            self.unsupported = new_unsupported
            self.hot = new_hot
            self.t_acc_pos = new_t_acc_pos
            self.t_acc_neg = new_t_acc_neg

            accepted_indices = np.where(accepted_mask)[0]
            if len(accepted_indices) > 0:
                output_events_data = np.zeros((len(accepted_indices), 5), dtype=events.dtype)
                output_events_data[:, EVENT_X] = events[accepted_indices, EVENT_X]
                output_events_data[:, EVENT_Y] = events[accepted_indices, EVENT_Y]
                output_events_data[:, EVENT_T] = events[accepted_indices, EVENT_T]
                output_events_data[:, EVENT_P] = events[accepted_indices, EVENT_P]
                output_events_data[:, OUTPUT_CONFIDENCE] = confidence[accepted_indices]
                output_events = output_events_data
            else:
                output_events = empty_output_events()

        else:
            # Fallback to Python implementation if Numba is not available
            t_raw_last = self.t_raw_last
            p_raw_last = self.p_raw_last
            rate_hz = self.rate_hz
            unsupported = self.unsupported
            hot = self.hot
            t_acc_pos = self.t_acc_pos
            t_acc_neg = self.t_acc_neg

            accepted_mask = np.zeros(len(events), dtype=bool)
            confidence = np.zeros(len(events), dtype=np.int8)
            support = np.zeros(len(events), dtype=np.int16)
            pair_flag = np.zeros(len(events), dtype=np.int8)
            accepted_rows: list[np.ndarray] = []
            output_events = empty_output_events() # Initialize here to prevent UnboundLocalError

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
                        support_count = _support_count_jit(
                            x=x,
                            y=y,
                            t=t,
                            polarity=p,
                            t_acc_pos=t_acc_pos,
                            t_acc_neg=t_acc_neg,
                            delta_t_us=config.delta_t_us,
                            height=height,
                            width=width,
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

@njit(cache=True)
def _support_count_jit(
    x: int,
    y: int,
    t: int,
    polarity: int,
    t_acc_pos: np.ndarray,
    t_acc_neg: np.ndarray,
    delta_t_us: int,
    height: int,
    width: int,
) -> int:
    support_count = 0
    acc_map = t_acc_pos if polarity > 0 else t_acc_neg
    for ny in range(max(0, y - 1), min(height, y + 2)):
        for nx in range(max(0, x - 1), min(width, x + 2)):
            if nx == x and ny == y:
                continue
            lag = t - int(acc_map[ny, nx])
            if 0 < lag < delta_t_us:
                support_count += 1
    return support_count


@njit(cache=True)
def _apply_filter_jit(
    events: np.ndarray,
    width: int,
    height: int,
    tau_ref_dig_us: int,
    delta_t_us: int,
    k0: int,
    gamma: int,
    tau_pair_us: int,
    k_high: int,
    alpha: float,
    r_max_hz: float,
    u_hot: int,
    t_recover_us: int,
    stage_variant: str,
    t_raw_last_in: np.ndarray,
    p_raw_last_in: np.ndarray,
    rate_hz_in: np.ndarray,
    unsupported_in: np.ndarray,
    hot_in: np.ndarray,
    t_acc_pos_in: np.ndarray,
    t_acc_neg_in: np.ndarray,
):
    # Copy input state to local mutable variables
    t_raw_last = t_raw_last_in.copy()
    p_raw_last = p_raw_last_in.copy()
    rate_hz = rate_hz_in.copy()
    unsupported = unsupported_in.copy()
    hot = hot_in.copy()
    t_acc_pos = t_acc_pos_in.copy()
    t_acc_neg = t_acc_neg_in.copy()

    accepted_mask = np.zeros(len(events), dtype=types.boolean)
    confidence = np.zeros(len(events), dtype=types.int8)
    support_arr = np.zeros(len(events), dtype=types.int16)
    pair_flag_arr = np.zeros(len(events), dtype=types.int8)

    for idx in range(len(events)):
        x = events[idx, EVENT_X]
        y = events[idx, EVENT_Y]
        t = events[idx, EVENT_T]
        p = events[idx, EVENT_P]

        dt_raw = t - t_raw_last[y, x]
        pair = int((p != p_raw_last[y, x]) and (dt_raw < tau_pair_us))
        pair_flag_arr[idx] = pair

        instant_rate_hz = 1_000_000.0 / max(dt_raw, 1) # eps_us is 1
        rate_hz[y, x] = float(alpha * instant_rate_hz + (1.0 - alpha) * rate_hz[y, x])

        if hot[y, x] and dt_raw >= t_recover_us:
            hot[y, x] = False
            unsupported[y, x] = 0
            rate_hz[y, x] = 0.0

        guarded = bool(hot[y, x] or dt_raw < tau_ref_dig_us)
        accepted = False
        conf = 0
        support_count = 0

        if not guarded:
            if stage_variant != "ref":
                support_count = _support_count_jit(
                    x=x,
                    y=y,
                    t=t,
                    polarity=p,
                    t_acc_pos=t_acc_pos,
                    t_acc_neg=t_acc_neg,
                    delta_t_us=delta_t_us,
                    height=height,
                    width=width,
                )
                support_arr[idx] = support_count

                if stage_variant == "sup":
                    accepted = support_count >= k0
                else:
                    accepted = support_count >= (k0 + gamma * pair)

                if support_count == 0:
                    unsupported[y, x] = np.uint16(min(int(unsupported[y, x]) + 1, np.iinfo(np.uint16).max))
                else:
                    unsupported[y, x] = 0

                if rate_hz[y, x] > r_max_hz and unsupported[y, x] >= u_hot:
                    hot[y, x] = True
            else:
                accepted = True

            if accepted:
                accepted_mask[idx] = True
                if stage_variant == "conf":
                    conf = 1 + int(support_count >= k_high)
                else:
                    conf = 1
                confidence[idx] = conf

                if stage_variant != "ref":
                    if p > 0:
                        t_acc_pos[y, x] = t
                    else:
                        t_acc_neg[y, x] = t
        else:
            confidence[idx] = 0

        t_raw_last[y, x] = t
        p_raw_last[y, x] = p

    return accepted_mask, confidence, support_arr, pair_flag_arr, t_raw_last, p_raw_last, rate_hz, unsupported, hot, t_acc_pos, t_acc_neg
