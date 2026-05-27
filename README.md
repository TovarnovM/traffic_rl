# traffic_rl / snfs_traffic

High-performance Revised S-NFS traffic simulator core for future reinforcement-learning experiments.

The repository currently provides a deterministic, testable, array-oriented simulator core with:

- authoritative reference semantics;
- a supported optional optimized Numba backend;
- a public backend selection API;
- strict reference-vs-optimized equivalence checks;
- RNG next-draw parity checks;
- runtime invariant checks;
- benchmark and report infrastructure.

Longitudinal model note: the current core implements a simplified S-NFS-style
longitudinal update (authoritative in `step_reference(...)`), not paper-exact
Revised S-NFS formulas. `SimulationParams.q` and `SimulationParams.P1` are
currently reserved for forward compatibility and intentionally unused by
longitudinal dynamics.

The project is **not yet an RL environment package**. RL actions, observations, rewards, Gymnasium wrappers, RLlib wrappers, and multi-agent wrappers are planned, but intentionally remain outside the current simulator-core layer.

---

## Current status

Repository state: after Tasks 1–23, including Task 22+23 follow-up.

Current conclusion:

- `step_reference(...)` is the authoritative semantic oracle.
- `ReferenceBackend` remains the simple reference implementation.
- `OptimizedBackend` is a supported optional backend when required Numba kernels are available.
- Numba remains optional. The base package must import and run without Numba installed.
- The project is ready to begin RL environment preparation, but not ready for RL training yet.

Current readiness estimate:

| Area | Status |
|---|---|
| Simulator core | Mature enough for next-stage use |
| Reference semantics | Stable oracle |
| Optional optimized backend | Supported, benchmarked, fallback-safe |
| RL action semantics | Not implemented |
| Observation/reward contracts | Not implemented |
| Simulator facade | Not implemented |
| Gymnasium/RLlib wrappers | Not implemented |

---

## Implemented

### Core simulator

- `SimulationParams` schema.
- `TrafficState` array schema.
- Periodic multi-lane `RingTopology`.
- Reproducible uniform-random scenario initializer.
- Reference head-cell occupancy.
- Reference lane ordering.
- Reference periodic-ring neighbor/gap indexing.
- Reference lane-change phase.
- Stochastic lane-change conflict resolution.
- Reference longitudinal same-lane update.
- Full reference step via `step_reference(...)`.
- Runtime invariant validation helper.

### Backend architecture

- Minimal backend protocol.
- `ReferenceBackend`.
- Supported optional `OptimizedBackend`.
- Public backend selector:
  - `"reference"`;
  - `"optimized"`;
  - `"auto"`.

### Optimized path

- Optional Numba indexing kernels.
- Optional Numba lane-change proposal helper.
- Optional Numba longitudinal helper.
- Fused Numba indexing/neighbor fast path used by `OptimizedBackend`.
- Graceful fallback to reference semantics when required Numba kernels are unavailable.

### Correctness validation

- Reference-vs-backend equivalence tests.
- Optimized-backend equivalence tests.
- Runtime invariant tests.
- RNG next-draw parity checks.
- Focused Numba indexing tests.
- Focused fused indexing/neighbor tests.
- Focused lane-change Numba tests.
- Focused benchmark script tests.

### Benchmarking/reporting

- Indexing benchmark.
- Full-step optimized-vs-reference benchmark.
- Component-level optimized backend profiling in benchmark code.
- Task 21 benchmark report.
- Task 22+23 benchmark report.

---

## Not implemented yet

### RL-facing layer

- Controlled RL action semantics.
- Action application contract.
- Observation schema.
- Reward schema.
- Episode semantics.
- Metrics/info schema for RL rollouts.
- Simulator facade.
- Gymnasium environments.
- RLlib wrappers.
- PettingZoo/multi-agent wrappers.

### Physics/model extensions

- Length-aware multi-cell occupancy.
- Length-aware bumper-to-bumper gaps.
- Body-cell bus collision geometry.
- Open-boundary topology.
- Open-boundary step semantics.
- New simulator physics beyond the current Revised S-NFS core.

### Performance extensions

- Further optimization of lane-change proposal collection.
- Further optimization of longitudinal velocity update.
- Cython kernels.

---

## Installation

Base install:

```bash
python -m pip install -e .
```

Install with optional Numba support:

```bash
python -m pip install -e ".[numba]"
```

No-install local development alternative:

```bash
PYTHONPATH=src python -c "import snfs_traffic; print(snfs_traffic.__version__)"
```

---

## Developer quick checks

Base checks:

```bash
python -m pip install -e .
pytest -q
python -c "import snfs_traffic; print(snfs_traffic.__version__)"
```

Numba-enabled checks:

```bash
python -m pip install -e ".[numba]"
pytest -q
```

Focused optimized backend checks:

```bash
pytest -q tests/test_optimized_backend_equivalence.py
pytest -q tests/test_lane_change_numba.py
pytest -q tests/test_indexing_numba.py
pytest -q tests/test_indexing_numba_fused.py
pytest -q tests/test_backends_selection.py
pytest -q tests/test_bench_optimized_full_step.py
```

Legacy indexing benchmark smoke check:

```bash
PYTHONPATH=src python benchmarks/benchmark_indexing.py --quick --repeat 1 --warmup 0
```

---

## Repository layout

```text
src/snfs_traffic/
  backends/       # ReferenceBackend, OptimizedBackend, backend selector
  core/           # state, params, indexing, lane-change, longitudinal, reference step
  topology/       # RingTopology and topology interfaces
  scenarios/      # initial state generation and vehicle mix metadata
  rules/          # reserved for rule/preset layer
  observations/   # reserved for future observation builders
  envs/           # reserved for future RL environment wrappers
  metrics/        # reserved for future rollout metrics
  io/             # reserved for future snapshots/export/import

benchmarks/       # benchmark CLIs and benchmark-local profiling
reports/          # generated benchmark reports
tests/            # correctness, equivalence, smoke, and benchmark-schema tests
```

The `core` package should remain framework-independent. It should not import Gymnasium, Ray, RLlib, Torch, Matplotlib, Pandas, or SciPy.

---

## Public core API

```python
from snfs_traffic.core import (
    SimulationParams,
    TrafficState,
    empty_state,
    max_supported_velocity,
    validate_state,
    INDEX_DTYPE,
    MISSING_GAP,
    MISSING_INDEX,
    build_occupancy,
    build_lane_order,
    compute_neighbors,
    step_longitudinal_reference,
    step_lane_change_reference,
    step_reference,
    validate_runtime_invariants,
    StepBackend,
    ReferenceBackend,
    get_reference_backend,
)
```

Topology API:

```python
from snfs_traffic.topology import RingTopology
```

Scenario API:

```python
from snfs_traffic.scenarios import (
    AV_BEHAVIOR_ID,
    AV_VEH_TYPE,
    BUS_BEHAVIOR_ID,
    BUS_VEH_TYPE,
    CONTROLLED_AV_BEHAVIOR_ID,
    HDV_BEHAVIOR_ID,
    HDV_VEH_TYPE,
    VehicleMix,
    make_uniform_random_state,
)
```

Backend API:

```python
from snfs_traffic.backends import (
    BACKEND_AUTO,
    BACKEND_OPTIMIZED,
    BACKEND_REFERENCE,
    OptimizedBackend,
    get_backend,
    get_optimized_backend,
    get_reference_backend,
    optimized_backend_available,
)
```

---

## Backend contract

A backend performs exactly one full simulation step.

Backends receive:

- `TrafficState`;
- `SimulationParams`;
- `RingTopology`;
- `np.random.Generator`.

Backends return:

- `TrafficState`.

All backends must preserve `step_reference(...)` semantics.

Required state-field equivalence:

- `lane`;
- `pos`;
- `vel`;
- `alive`;
- `controlled`;
- `changed_lane`;
- `last_lane_delta`.

Required global guarantees:

- runtime invariants pass;
- RNG draw order/parity is preserved;
- no hidden stochastic behavior is introduced;
- `ReferenceBackend` remains independent of Numba;
- Numba remains optional.

---

## Backend selection

Use:

```python
from snfs_traffic.backends import get_backend

backend = get_backend("auto")
```

Available backend names:

```python
get_backend("reference")
get_backend("optimized")
get_backend("auto")
```

Behavior:

- `"reference"` always returns `ReferenceBackend`.
- `"optimized"` returns `OptimizedBackend` when all required Numba kernels are available, otherwise falls back to `ReferenceBackend`.
- `"auto"` is the production-safe selector: prefer optimized when available, otherwise reference.

The reference backend remains the semantic oracle. The optimized backend is a performance implementation that must match the reference backend exactly.

---

## Minimal usage example

```python
import numpy as np

from snfs_traffic.backends import get_backend
from snfs_traffic.core import SimulationParams
from snfs_traffic.scenarios import make_uniform_random_state
from snfs_traffic.topology import RingTopology

params = SimulationParams(num_lanes=3, road_length=1000)
topology = RingTopology(num_lanes=3, length=1000)

state = make_uniform_random_state(
    num_lanes=3,
    road_length=1000,
    density=0.2,
    seed=1,
)

rng = np.random.default_rng(123)
backend = get_backend("auto")

for _ in range(100):
    state = backend.step(state, params, topology, rng)
```

---

## Reference step

The authoritative full-step implementation is:

```python
from snfs_traffic.core import step_reference
```

Use `step_reference(...)` when:

- validating new behavior;
- writing semantic tests;
- debugging optimized backend discrepancies;
- checking exact RNG behavior.

Optimized kernels and optimized backends must be compared against this reference path.

---

## Benchmarking

Indexing benchmark:

```bash
python benchmarks/benchmark_indexing.py \
  --quick \
  --out-json /tmp/snfs_indexing_bench.json \
  --out-md /tmp/snfs_indexing_bench.md
```

Optimized full-step smoke benchmark:

```bash
python benchmarks/bench_optimized_full_step.py \
  --preset smoke \
  --backend both \
  --repeats 3 \
  --warmups 1 \
  --steps 50 \
  --out-json reports/task22_23/optimized_smoke.json \
  --out-md reports/task22_23/optimized_smoke.md
```

Optimized reduced-standard benchmark:

```bash
python benchmarks/bench_optimized_full_step.py \
  --preset standard \
  --backend both \
  --repeats 3 \
  --warmups 1 \
  --steps 50 \
  --cases medium_moderate,medium_dense,wide_moderate \
  --out-json reports/task22_23/optimized_reduced_standard.json \
  --out-md reports/task22_23/optimized_reduced_standard.md
```

Benchmark numbers are environment-dependent. Use them for relative comparison inside the same environment, not as absolute production-performance claims.

---

## Latest representative benchmark status

Task 22+23 reduced-standard benchmark showed that the optimized backend remained equivalent to reference and significantly faster.

Representative reduced-standard results:

| case | reference ms/step | optimized ms/step | speedup vs reference | speedup vs Task 21 optimized |
|---|---:|---:|---:|---:|
| medium_moderate | 96.9242 | 4.1205 | 23.522x | 2.625x |
| medium_dense | 354.9731 | 13.6533 | 25.999x | 1.839x |
| wide_moderate | 191.7672 | 7.8746 | 24.353x | 2.290x |

Equivalence:

- `state_equal=true`;
- `rng_next_draw_equal=true`.

Final Task 22+23 recommendation:

```text
keep OptimizedBackend supported and merge indexing fast path
```

After Task 22+23, the main optimized-backend bottleneck moved from indexing/lane-ordering to lane-change proposal collection.

Current top representative component:

- `lane_change_proposals`.

---

## Current model limitations

The current simulator intentionally uses simplified head-cell semantics:

- occupancy is head-cell-only;
- gaps are head-cell-only;
- vehicle length is not used for occupancy/gap/collision geometry;
- bus body cells are not modeled as occupied cells;
- lane changes are lateral only and do not move longitudinal position;
- there are no same-step lateral swaps into previously occupied target cells;
- controlled vehicles are marked in state but do not yet receive external RL actions;
- open-boundary roads are not implemented;
- current runtime invariants validate current reference semantics, not future length-aware geometry.

These limitations are intentional for the current core stage. Do not change them casually while adding RL wrappers.

---

## Development principles

1. `step_reference(...)` is the semantic oracle.
2. Optimized code must match reference semantics exactly.
3. RNG draw order/parity is part of correctness.
4. Numba is optional.
5. Reference backend must remain simple and independent of Numba.
6. Do not add RL actions/observations/rewards directly into optimized kernels.
7. Do not add new physics while working on backend performance.
8. Prefer small, testable, surgical changes.
9. Benchmarks must not become correctness tests with hard performance thresholds.
10. Production hot paths should not contain profiling hooks.

---

## Roadmap summary

Immediate next stage:

1. Add simulator facade.
2. Define controlled action semantics.
3. Define observation schema.
4. Define reward schema.
5. Define episode/reset semantics.
6. Add Gymnasium wrapper after the simulator facade is stable.

The next architectural step should not be another low-level optimization pass unless performance becomes a blocker. The project is ready to begin RL environment preparation, but not ready for RL training yet.


## Visualization

Install optional visualization tooling:

```bash
python -m pip install -e ".[viz]"
```

CLI demo:

```bash
PYTHONPATH=src python -m snfs_traffic.visualization.demo \
  --output episode.mp4 \
  --steps 200 \
  --num-lanes 3 \
  --road-length 120 \
  --density 0.20 \
  --controlled-fraction 0.03
```

Visualization is a tooling side-feature: it observes `TrafficState` snapshots, does not step the simulator, does not consume RNG, and uses visual-only interpolation between discrete simulator steps. Default drawing uses head-cell-only bodies to match current simulator semantics. `body_mode="state_length"` only changes rendering and does not change collision/occupancy semantics.

```python
import numpy as np

from snfs_traffic.backends import get_backend
from snfs_traffic.core import SimulationParams, validate_runtime_invariants
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology
from snfs_traffic.visualization import RoadRenderConfig, RoadRenderer, VideoWriter

params = SimulationParams(num_lanes=3, road_length=120)
topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
state = make_uniform_random_state(num_lanes=3, road_length=120, density=0.20, seed=1, vehicle_mix=VehicleMix(controlled_fraction=0.03))
backend = get_backend("auto")
rng = np.random.default_rng(123)
renderer = RoadRenderer(params=params, topology=topology, config=RoadRenderConfig(interpolation_frames=8, camera="follow", follow="first-controlled"))
renderer.reset(state)
with VideoWriter("episode.mp4", fps=48) as video:
    for step in range(200):
        next_state = backend.step(state, params, topology, rng)
        validate_runtime_invariants(next_state, params, topology)
        video.write_many(renderer.render_step(next_state, step=step + 1))
        state = next_state
```

## Simulator facade
Use `TrafficSimulator` for reset/step/observe/rollout. `step(actions=None)` preserves selected backend semantics exactly. Choose backend via `backend="reference"|"optimized"|"auto"`. Enable runtime checks with `validate=True`.

## Controlled lateral actions
Actions are keyed by stable `vehicle_id` with lane deltas `-1` (left), `0` (stay), `+1` (right). IDs must be alive controlled vehicles. Under `require_all_controlled_actions=True`, every alive controlled vehicle must have an action; otherwise missing actions are treated as stay. Structural invalid inputs raise `ValueError`. Unsafe/out-of-bounds/occupied/conflict commands are rejected and reported via `ControlledActionResult`. Explicit zero/stay actions are not equivalent to `actions=None`.

## Local observations
Gym-free local observations are returned only for alive controlled vehicles. `obs.vehicle_id` aligns with rows. `obs.action_mask` columns are `[-1, 0, +1]`. Observation feature order is fixed and shape is `(n_controlled_alive, 21)` with default dtype `float32`.
