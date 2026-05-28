# Optimized full-step benchmark

Timestamp: 2026-05-28T16:23:02.340829+00:00

## Environment

- Python: 3.14.4 (main, May  1 2026, 05:51:55) [GCC 13.3.0]
- Platform: Linux-6.12.47-x86_64-with-glibc2.39
- NumPy: 2.4.6
- Numba available: True

## Comparison

| case | vehicles | steps | reference mean ms/step | optimized mean ms/step | speedup x | equivalence | top optimized component | notes |
|---|---:|---:|---:|---:|---:|---|---|---|
| medium_moderate | 600 | 50 | 491.1606 | 6.0169 | 81.630 | yes | longitudinal_random_draws | optimized backend active |
| medium_dense | 1500 | 50 | 2761.7621 | 20.0826 | 137.520 | yes | longitudinal_velocity_numba | optimized backend active |
| wide_moderate | 1000 | 50 | 1001.5073 | 10.0021 | 100.130 | yes | longitudinal_random_draws | optimized backend active |

## Recommendation

keep OptimizedBackend supported and merge indexing fast path
