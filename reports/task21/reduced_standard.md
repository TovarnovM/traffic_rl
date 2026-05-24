# Optimized full-step benchmark

Timestamp: 2026-05-24T08:22:40.520273+00:00

## Environment

- Python: 3.14.4 (main, May  1 2026, 05:51:55) [GCC 13.3.0]
- Platform: Linux-6.12.47-x86_64-with-glibc2.39
- NumPy: 2.4.6
- Numba available: True

## Comparison

| case | vehicles | steps | reference mean ms/step | optimized mean ms/step | speedup x | equivalence | top optimized component | notes |
|---|---:|---:|---:|---:|---:|---|---|
| medium_moderate | 600 | 50 | 95.6518 | 10.8184 | 8.842 | yes | lane_order_post_lane_change | optimized backend active |
| medium_dense | 1500 | 50 | 356.4242 | 25.1126 | 14.193 | yes | lane_order_post_lane_change | optimized backend active |
| wide_moderate | 1000 | 50 | 190.0754 | 18.0323 | 10.541 | yes | lane_order_post_lane_change | optimized backend active |

## Recommendation

merge optimized backend
