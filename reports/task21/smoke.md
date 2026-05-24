# Optimized full-step benchmark

Timestamp: 2026-05-24T08:12:20.376212+00:00

## Environment

- Python: 3.14.4 (main, May  1 2026, 05:51:55) [GCC 13.3.0]
- Platform: Linux-6.12.47-x86_64-with-glibc2.39
- NumPy: 2.4.6
- Numba available: True

## Comparison

| case | vehicles | steps | reference mean ms/step | optimized mean ms/step | speedup x | equivalence | top optimized component | notes |
|---|---:|---:|---:|---:|---:|---|---|
| smoke_low_density | 20 | 50 | 1.9696 | 0.7498 | 2.627 | yes | lane_order_post_lane_change | optimized backend active |
| smoke_dense | 150 | 50 | 7.0357 | 2.4429 | 2.880 | yes | lane_order_post_lane_change | optimized backend active |

## Recommendation

merge only specific kernels
