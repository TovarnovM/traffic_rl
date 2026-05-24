# Optimized full-step benchmark

Timestamp: 2026-05-24T09:47:01.048813+00:00

## Environment

- Python: 3.14.4 (main, May  1 2026, 05:51:55) [GCC 13.3.0]
- Platform: Linux-6.12.47-x86_64-with-glibc2.39
- NumPy: 2.4.6
- Numba available: True

## Comparison

| case | vehicles | steps | reference mean ms/step | optimized mean ms/step | speedup x | equivalence | top optimized component | notes |
|---|---:|---:|---:|---:|---:|---|---|---|
| smoke_low_density | 20 | 50 | 1.9888 | 0.1037 | 19.178 | yes | longitudinal_velocity | optimized backend active |
| smoke_dense | 150 | 50 | 8.0872 | 0.5909 | 13.686 | yes | longitudinal_velocity | optimized backend active |

## Recommendation

keep OptimizedBackend supported but do not merge indexing fast path
