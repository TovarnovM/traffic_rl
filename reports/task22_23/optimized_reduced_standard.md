# Optimized full-step benchmark

Timestamp: 2026-05-24T09:49:18.371765+00:00

## Environment

- Python: 3.14.4 (main, May  1 2026, 05:51:55) [GCC 13.3.0]
- Platform: Linux-6.12.47-x86_64-with-glibc2.39
- NumPy: 2.4.6
- Numba available: True

## Comparison

| case | vehicles | steps | reference mean ms/step | optimized mean ms/step | speedup x | equivalence | top optimized component | notes |
|---|---:|---:|---:|---:|---:|---|---|---|
| medium_moderate | 600 | 50 | 96.9242 | 4.1205 | 23.522 | yes | lane_change_proposals | optimized backend active |
| medium_dense | 1500 | 50 | 354.9731 | 13.6533 | 25.999 | yes | lane_change_proposals | optimized backend active |
| wide_moderate | 1000 | 50 | 191.7672 | 7.8746 | 24.353 | yes | lane_change_proposals | optimized backend active |

## Recommendation

keep OptimizedBackend supported and merge indexing fast path
