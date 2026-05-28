# Optimized full-step benchmark

Timestamp: 2026-05-28T16:04:53.306111+00:00

## Environment

- Python: 3.14.4 (main, May  1 2026, 05:51:55) [GCC 13.3.0]
- Platform: Linux-6.12.47-x86_64-with-glibc2.39
- NumPy: 2.4.6
- Numba available: True

## Comparison

| case | vehicles | steps | reference mean ms/step | optimized mean ms/step | speedup x | equivalence | top optimized component | notes |
|---|---:|---:|---:|---:|---:|---|---|---|
| smoke_low_density | 20 | 50 | 3.8865 | 0.6471 | 6.006 | yes | longitudinal_random_draws | optimized backend active |
| smoke_dense | 150 | 50 | 33.5594 | 1.3742 | 24.421 | yes | longitudinal_random_draws | optimized backend active |

## Recommendation

keep OptimizedBackend supported but do not merge indexing fast path
