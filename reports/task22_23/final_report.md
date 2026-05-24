# Task 22+23 final report

## Commands run

- `python -m pip install -e ".[numba]"`
- `pytest -q`
- `pytest -q tests/test_optimized_backend_equivalence.py`
- `pytest -q tests/test_lane_change_numba.py`
- `pytest -q tests/test_bench_optimized_full_step.py`
- `pytest -q tests/test_indexing_numba.py`
- `pytest -q tests/test_indexing_numba_fused.py`
- `pytest -q tests/test_backends_selection.py`
- `PYTHONPATH=src python benchmarks/benchmark_indexing.py --quick --repeat 1 --warmup 0`
- `python benchmarks/bench_optimized_full_step.py --preset smoke --backend both --repeats 3 --warmups 1 --steps 50 --out-json reports/task22_23/optimized_smoke.json --out-md reports/task22_23/optimized_smoke.md`
- `python benchmarks/bench_optimized_full_step.py --preset standard --backend both --repeats 3 --warmups 1 --steps 50 --cases medium_moderate,medium_dense,wide_moderate --out-json reports/task22_23/optimized_reduced_standard.json --out-md reports/task22_23/optimized_reduced_standard.md`

## Test results

- Full pytest: `265 passed`.
- `tests/test_optimized_backend_equivalence.py`: `2 passed`.
- `tests/test_lane_change_numba.py`: `38 passed`.
- `tests/test_bench_optimized_full_step.py`: `2 passed`.
- `tests/test_indexing_numba.py`: `8 passed`.
- `tests/test_indexing_numba_fused.py`: `4 passed`.
- `tests/test_backends_selection.py`: `5 passed`.

## Benchmark outputs

- `reports/task22_23/optimized_smoke.json`
- `reports/task22_23/optimized_smoke.md`
- `reports/task22_23/optimized_reduced_standard.json`
- `reports/task22_23/optimized_reduced_standard.md`

## Compact benchmark table (reduced standard)

| case | reference ms/step | optimized after (task22+23) ms/step | speedup vs reference | optimized before (task21) ms/step | speedup vs task21 optimized |
|---|---:|---:|---:|---:|---:|
| medium_moderate | 96.9242 | 4.1205 | 23.522x | 10.8184 | 2.625x |
| medium_dense | 354.9731 | 13.6533 | 25.999x | 25.1126 | 1.839x |
| wide_moderate | 191.7672 | 7.8746 | 24.353x | 18.0323 | 2.290x |

## Component breakdown after optimization

Top optimized component by case:

- medium_moderate: `lane_change_proposals`
- medium_dense: `lane_change_proposals`
- wide_moderate: `lane_change_proposals`

Indexing/neighbor path is reported via fused components:

- `pre_lane_change_index_neighbors_fused`
- `post_lane_change_index_neighbors_fused`

## Bottleneck movement from Task 21 to Task 22+23

- Task 21 bottleneck was primarily lane-order/indexing (`lane_order_post_lane_change`).
- After Task 22+23 fused indexing fast path, bottleneck moved away from indexing to lane-change proposal collection.

## Recommendation

keep OptimizedBackend supported and merge indexing fast path
