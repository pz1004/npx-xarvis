"""Test Numba JIT optimization of proposed_balanced filter."""
import sys
sys.path.insert(0, '.')

import numpy as np
import time

from src.filters.proposed_balanced import ProposedBalancedFilter, ProposedBalancedConfig, HAS_NUMBA

print(f'Numba available: {HAS_NUMBA}')

# Create synthetic events for testing
np.random.seed(42)
n_events = 50000
sensor_size = (34, 34, 2)
events = np.zeros((n_events, 4), dtype=np.int64)
events[:, 0] = np.random.randint(0, 34, n_events)  # x
events[:, 1] = np.random.randint(0, 34, n_events)  # y
events[:, 2] = np.sort(np.random.randint(0, 5_000_000, n_events))  # t (5 seconds)
events[:, 3] = np.random.choice([-1, 1], n_events)  # p

config = ProposedBalancedConfig(
    tau_ref_dig_us=1000,
    delta_t_us=2000,
    k0=1,
    gamma=1,
    tau_pair_us=1000,
    k_high=2,
)

filt = ProposedBalancedFilter(sensor_size=sensor_size, config=config)

# Warm up (JIT compilation)
print("Warming up JIT...")
result = filt.apply(events[:100])
print(f'Warm-up done. Accepted: {result.stats["accepted_count"]}')

# Benchmark Numba path
start = time.time()
result = filt.apply(events)
elapsed_numba = time.time() - start
print(f'\n--- Numba JIT Path ---')
print(f'Events: {n_events}')
print(f'Time: {elapsed_numba:.4f}s')
print(f'Accepted: {result.stats["accepted_count"]} / {n_events} ({100*result.stats["accepted_count"]/n_events:.1f}%)')
print(f'Mean support: {result.stats["mean_support"]:.3f}')
print(f'Throughput: {n_events/elapsed_numba:.0f} events/s')

# Benchmark Python fallback path
filt_python = ProposedBalancedFilter(sensor_size=sensor_size, config=config)
start = time.time()
result_python = filt_python._apply_python(events[:5000], 34, 34, config)
elapsed_python = time.time() - start
print(f'\n--- Python Fallback Path (5000 events) ---')
print(f'Time: {elapsed_python:.4f}s')
print(f'Accepted: {result_python.stats["accepted_count"]} / 5000')
print(f'Throughput: {5000/elapsed_python:.0f} events/s')

# Estimate speedup
speedup = (elapsed_python / 5000) / (elapsed_numba / n_events)
print(f'\n--- Estimated Speedup ---')
print(f'Numba vs Python: ~{speedup:.1f}x faster')

# Verify correctness: compare results on small subset
print("\n--- Correctness Check ---")
small_events = events[:1000]
result_jit = filt.apply(small_events)
result_py = filt_python._apply_python(small_events, 34, 34, config)

match_accepted = np.array_equal(result_jit.accepted_mask, result_py.accepted_mask)
match_confidence = np.array_equal(result_jit.confidence, result_py.confidence)
match_support = np.array_equal(result_jit.support, result_py.support)
match_pair = np.array_equal(result_jit.pair_flag, result_py.pair_flag)

print(f'Accepted mask match: {match_accepted}')
print(f'Confidence match: {match_confidence}')
print(f'Support match: {match_support}')
print(f'Pair flag match: {match_pair}')

if all([match_accepted, match_confidence, match_support, match_pair]):
    print("\n✅ All outputs match! Numba JIT optimization is correct.")
else:
    print("\n❌ Mismatch detected! Investigating...")
    diffs = np.where(result_jit.accepted_mask != result_py.accepted_mask)[0]
    if len(diffs) > 0:
        print(f"  First mismatch at index {diffs[0]}")
        print(f"  JIT accepted: {result_jit.accepted_mask[diffs[0]]}")
        print(f"  Python accepted: {result_py.accepted_mask[diffs[0]]}")
