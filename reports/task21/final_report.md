# Task 21 Final Gate Report — OptimizedBackend Production Readiness

## Commands run

```bash
python -m pip install -e ".[numba]"
python -m pytest -q
python -m pytest -q tests/test_optimized_backend_equivalence.py
python -m pytest -q tests/test_lane_change_numba.py
python -m pytest -q tests/test_bench_optimized_full_step.py
python benchmarks/bench_optimized_full_step.py --preset smoke --backend both --repeats 1 --warmups 1 --steps 2 --out-json /tmp/task21_smoke.json --out-md /tmp/task21_smoke.md
```

## Test results

- Full pytest run: `210 passed, 46 skipped in 33.62s`
- Focused optimized backend test: `2 passed in 16.41s`
- Focused lane_change_numba test: `1 passed, 37 skipped in 0.65s`
- Focused benchmark test: `2 passed in 1.78s`

## Benchmark output paths

- Existing Task 21 artifact: `reports/task21/smoke.json`
- Existing Task 21 artifact: `reports/task21/smoke.md`
- Existing Task 21 artifact: `reports/task21/reduced_standard.json`
- Existing Task 21 artifact: `reports/task21/reduced_standard.md`
- Smoke sanity benchmark for markdown fix: `/tmp/task21_smoke.json`
- Smoke sanity benchmark for markdown fix: `/tmp/task21_smoke.md`

## Compact benchmark table (reduced-standard)

| case | vehicles | steps | reference ms/step | optimized ms/step | speedup x | equivalence | top optimized component | notes |
|---|---:|---:|---:|---:|---:|---|---|---|
| medium_moderate | 600 | 50 | 95.6518 | 10.8184 | 8.842 | yes | lane_order_post_lane_change | optimized backend active |
| medium_dense | 1500 | 50 | 356.4242 | 25.1126 | 14.193 | yes | lane_order_post_lane_change | optimized backend active |
| wide_moderate | 1000 | 50 | 190.0754 | 18.0323 | 10.541 | yes | lane_order_post_lane_change | optimized backend active |

## Optimized backend component breakdown (reduced-standard)

- Dominant component in all representative cases: `lane_order_post_lane_change`.
- Next-largest components are typically `lane_order_pre_lane_change`, followed by either `longitudinal_velocity` or `lane_change_proposals`.
- Small-cost components: `conflict_resolution`, `apply_lane_changes`, `position_advance`, and state-copy/orchestration overhead.

## Bottleneck movement

- With lane-change and longitudinal Numba kernels active, remaining time is concentrated in lane-order/neighbor-indexing rebuilds.
- The dominant residual cost is the duplicated indexing path before and after lane-change.

## Final recommendation

1. **Promote `OptimizedBackend` as a supported optional backend.**
2. **Keep Numba optional** (fallback to reference remains required/valid).
3. **Keep `ReferenceBackend` and `step_reference(...)` authoritative.**
4. **Next performance task:** optimize lane order / neighbor indexing rebuilds, especially duplicated pre/post lane-change indexing path.

