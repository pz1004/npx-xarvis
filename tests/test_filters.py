from __future__ import annotations

import numpy as np

from src.filters.proposed_lowmem import ProposedLowMemConfig, ProposedLowMemFilter
from src.filters.proposed_balanced import ProposedBalancedConfig, ProposedBalancedFilter


def _filter_config(**overrides: int | float) -> ProposedBalancedConfig:
    base = {
        "tau_ref_dig_us": 0,
        "delta_t_us": 1000,
        "k0": 0,
        "gamma": 0,
        "tau_pair_us": 100,
        "k_high": 2,
        "r_max_hz": 1e9,
        "u_hot": 100,
        "warmup_us": 0,
    }
    base.update(overrides)
    return ProposedBalancedConfig(**base)


def test_raw_history_updates_on_reject_and_rejected_events_do_not_seed_support() -> None:
    config = _filter_config(tau_ref_dig_us=1, k0=1, gamma=1, u_hot=10)
    filter_impl = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config)
    events = np.asarray(
        [
            [1, 1, 0, 1],
            [2, 1, 10, 1],
            [1, 1, 50, -1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.accepted_mask.tolist() == [False, False, False]
    assert int(result.support[1]) == 0
    assert int(result.pair_flag[2]) == 1


def test_guarded_events_do_not_advance_unsupported_streak() -> None:
    config = _filter_config(tau_ref_dig_us=10, r_max_hz=0.0, u_hot=2)
    filter_impl = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config)
    events = np.asarray(
        [
            [1, 1, 100, 1],
            [1, 1, 105, 1],
            [1, 1, 200, 1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.accepted_mask.tolist() == [True, False, True]


def test_confidence_switches_at_k_high() -> None:
    config = _filter_config()
    filter_impl = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config)
    events = np.asarray(
        [
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 2, 1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.confidence.tolist() == [1, 1, 2]


def test_polarity_penalty_rejects_short_lag_opposite_polarity_event() -> None:
    base_events = np.asarray(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 2, 1],
            [1, 1, 3, -1],
        ],
        dtype=np.int64,
    )
    gamma_one = ProposedBalancedFilter(
        sensor_size=(3, 3, 2),
        config=_filter_config(gamma=1),
    )
    gamma_zero = ProposedBalancedFilter(
        sensor_size=(3, 3, 2),
        config=_filter_config(gamma=0),
    )
    penalized = gamma_one.apply(base_events)
    unpenalized = gamma_zero.apply(base_events)
    assert int(penalized.pair_flag[-1]) == 1
    assert bool(unpenalized.accepted_mask[-1]) is True
    assert bool(penalized.accepted_mask[-1]) is False


def test_hot_pixel_recovery_after_quiescent_interval() -> None:
    config = _filter_config(tau_ref_dig_us=10, r_max_hz=0.0, u_hot=2, t_recover_us=1000)
    filter_impl = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config)
    events = np.asarray(
        [
            [1, 1, 100, 1],
            [1, 1, 200, 1],
            [1, 1, 300, 1],
            [1, 1, 1400, 1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.accepted_mask.tolist() == [True, True, False, True]


def test_proposed_balanced_resets_state_between_sample_applications() -> None:
    config = _filter_config(tau_ref_dig_us=10, k0=0)
    reused_filter = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config)
    previous_sample = np.asarray(
        [
            [1, 1, 100, 1],
            [1, 1, 200, 1],
        ],
        dtype=np.int64,
    )
    target_sample = np.asarray([[1, 1, 100, 1]], dtype=np.int64)

    reused_filter.apply(previous_sample)
    reused_result = reused_filter.apply(target_sample)
    fresh_result = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config).apply(target_sample)

    assert reused_result.accepted_mask.tolist() == fresh_result.accepted_mask.tolist()
    assert reused_result.confidence.tolist() == fresh_result.confidence.tolist()


def test_balanced_warmup_accepts_initial_events_with_strict_k0() -> None:
    config = _filter_config(k0=1, warmup_us=5000)
    filter_impl = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config)
    events = np.asarray(
        [
            [1, 1, 0, 1],
            [0, 0, 1000, 1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.accepted_mask.tolist() == [True, True]
    assert result.support.tolist() == [0, 0]


def test_balanced_rejects_unsupported_event_after_warmup() -> None:
    config = _filter_config(k0=1, warmup_us=5000, delta_t_us=10_000)
    filter_impl = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config)
    events = np.asarray(
        [
            [1, 1, 0, 1],
            [1, 1, 6000, 1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.accepted_mask.tolist() == [True, False]
    assert int(result.support[1]) == 0


def test_balanced_warmup_seeds_accepted_history_for_strict_support() -> None:
    config = _filter_config(k0=1, warmup_us=5000, delta_t_us=10_000)
    filter_impl = ProposedBalancedFilter(sensor_size=(3, 3, 2), config=config)
    events = np.asarray(
        [
            [1, 1, 0, 1],
            [2, 1, 6000, 1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.accepted_mask.tolist() == [True, True]
    assert int(result.support[1]) == 1


def test_lowmem_hot_pixel_recovery_after_quiescent_interval() -> None:
    filter_impl = ProposedLowMemFilter(
        sensor_size=(3, 3, 2),
        config=ProposedLowMemConfig(
            tau_ref_dig_us=10,
            delta_t_us=1000,
            k0=0,
            k_high=2,
            r_max_hz=0.0,
            u_hot=2,
            t_recover_us=1000,
            warmup_us=0,
        ),
    )
    events = np.asarray(
        [
            [1, 1, 100, 1],
            [1, 1, 200, 1],
            [1, 1, 300, 1],
            [1, 1, 1400, 1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.accepted_mask.tolist() == [True, True, False, True]


def test_lowmem_warmup_seeds_row_column_support() -> None:
    filter_impl = ProposedLowMemFilter(
        sensor_size=(3, 3, 2),
        config=ProposedLowMemConfig(
            tau_ref_dig_us=0,
            delta_t_us=10_000,
            k0=1,
            k_high=2,
            warmup_us=5000,
        ),
    )
    events = np.asarray(
        [
            [1, 1, 0, 1],
            [1, 2, 6000, 1],
        ],
        dtype=np.int64,
    )
    result = filter_impl.apply(events)
    assert result.accepted_mask.tolist() == [True, True]
    assert int(result.support[1]) == 1
