# SNFS full-step phase benchmark

Timestamp: 2026-05-28T16:23:24.607611+00:00

## Environment

- Python: 3.14.4 (main, May  1 2026, 05:51:55) [GCC 13.3.0]
- Platform: Linux-6.12.47-x86_64-with-glibc2.39
- NumPy: 2.4.6

## Benchmark config

- Preset: smoke
- Seed: 0
- Warmups: 2
- Repeats: 3
- Default steps: 10

## Case summary

| case | lanes | road_length | density | vehicles | p_lane_change | steps | mean step ms | median step ms | top bottleneck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smoke_low_density | 2 | 100 | 0.10 | 20 | 0.50 | 50 | 3.3267 | 3.3216 | longitudinal_phase |
| smoke_dense | 3 | 100 | 0.50 | 150 | 0.50 | 50 | 31.1913 | 30.9730 | longitudinal_phase |

## Phase breakdown — smoke_low_density

| phase | mean ns/step | mean ms/step | percent of phased step |
|---|---:|---:|---:|
| validate_state_input | 143393.0 | 0.143393 | 4.31% |
| build_occupancy_pre_lane_change | 157284.5 | 0.157285 | 4.73% |
| build_lane_order_pre_lane_change | 87800.1 | 0.087800 | 2.64% |
| compute_neighbors_pre_lane_change | 87545.2 | 0.087545 | 2.63% |
| lane_change_collect_proposals | 351421.9 | 0.351422 | 10.56% |
| lane_change_resolve_conflicts | 1905.4 | 0.001905 | 0.06% |
| lane_change_apply | 8445.3 | 0.008445 | 0.25% |
| validation_and_final_checks | 597711.0 | 0.597711 | 17.97% |
| longitudinal_phase | 1861028.8 | 1.861029 | 55.94% |
| orchestration_and_copy_overhead | 30196.1 | 0.030196 | 0.91% |

## Phase breakdown — smoke_dense

| phase | mean ns/step | mean ms/step | percent of phased step |
|---|---:|---:|---:|
| validate_state_input | 169678.5 | 0.169678 | 0.54% |
| build_occupancy_pre_lane_change | 270307.8 | 0.270308 | 0.87% |
| build_lane_order_pre_lane_change | 170924.7 | 0.170925 | 0.55% |
| compute_neighbors_pre_lane_change | 544372.0 | 0.544372 | 1.75% |
| lane_change_collect_proposals | 3420586.5 | 3.420586 | 10.97% |
| lane_change_resolve_conflicts | 4881.9 | 0.004882 | 0.02% |
| lane_change_apply | 14907.2 | 0.014907 | 0.05% |
| validation_and_final_checks | 916014.9 | 0.916015 | 2.94% |
| longitudinal_phase | 25635401.8 | 25.635402 | 82.19% |
| orchestration_and_copy_overhead | 44274.4 | 0.044274 | 0.14% |

## Recommended next task

Run standard preset before final optimization choice; smoke data is local and limited.

_These benchmark results are local-environment measurements._
