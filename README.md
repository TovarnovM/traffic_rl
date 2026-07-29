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

Longitudinal model note: the current core implements the full Revised S-NFS
longitudinal phase order in `step_reference(...)`: stochastic look-ahead via
`r/S`, slow-to-start via `q`, perspective capping, `P1..P4` keep-speed braking
branches, and leader-safe collision avoidance.

The project now includes a first end-to-end local RL training path. Alongside
the simulator and priority-vehicle Baseline 1/2 experiments, it provides a
single-agent PV coordinator, a fixed padded graph observation, and a compact
Graph-PPO implementation for RLlib. This is still a research MVP rather than a
general-purpose RL training package.

---

## Current status

Repository state: after full Revised S-NFS unit-length Numba longitudinal velocity acceleration.

Current conclusion:

- `step_reference(...)` is the authoritative semantic oracle.
- `ReferenceBackend` remains the simple reference implementation.
- `OptimizedBackend` is a supported optional backend when required Numba kernels are available.
- Numba remains optional. The base package must import and run without Numba installed.
- The project has an MVP centralized Graph-PPO path ready for local smoke runs
  and initial training experiments.

Current readiness estimate:

| Area | Status |
|---|---|
| Simulator core | Mature enough for next-stage use |
| Reference semantics | Stable oracle |
| Optional optimized backend | Supported, benchmarked, fallback-safe |
| RL action semantics | Implemented (lateral controlled facade/reference path) |
| Observation contract | Implemented (local controlled-only schema) |
| Reward contract | Implemented (MVP speed/lane-change/blocked/stopped components) |
| Episode lifecycle | Implemented (fixed-horizon truncation plus optional no-alive termination) |
| Simulator facade | Implemented |
| Gymnasium wrapper | Implemented (single controlled vehicle, lateral-only `Discrete(3)`) |
| Multi-agent/RLlib env | Implemented; subclasses RLlib `MultiAgentEnv` when Ray is installed |
| Priority Baseline 1/2 | Implemented with paired runner and metrics |
| Centralized PV Graph-PPO | Implemented as a local-training research MVP |
| PettingZoo wrapper | Not implemented |

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
- Optional Numba full Revised S-NFS longitudinal velocity kernel for unit-length vehicles.
- Optional Numba position-advance helper.
- Fused Numba indexing/neighbor fast path used by `OptimizedBackend`.
- Graceful fallback to reference semantics when required Numba kernels are unavailable.
- Explicit reference fallback when any alive vehicle has `length != 1`.

### RL-facing MVP

- Reward schema in `snfs_traffic.rl.rewards` with components: `speed_reward`, `lane_change_penalty`, `blocked_action_penalty`, `stopped_penalty`, and `total`.
- Episode lifecycle helpers in `snfs_traffic.rl.episode`; periodic ring-road episodes are fixed-horizon by default (`terminated=False` in normal operation, `truncated=True` at `max_steps`).
- Stable reset/step info schema with JSON-friendly scalar metrics.
- Optional Gymnasium wrapper `snfs_traffic.rl.env.SnfsTrafficEnv` for exactly one controlled vehicle.
- Synchronous `SnfsTrafficMultiAgentEnv` and speed-control extension; the former
  is a real RLlib `MultiAgentEnv` subclass when the optional Ray extra is installed.
- Action space is lateral-only `Discrete(3)`: `0=keep lane`, `1=request left`, `2=request right`.
- Observation space is a Gymnasium `Dict` using the existing local controlled-only observation vector plus lateral action mask.
- General-purpose RL policies and PettingZoo remain out of scope.

### Priority-vehicle baselines

- Explicit uncontrolled PV role with `vmax_controlled` and native S-NFS behavior.
- Reproducible one-PV and same-lane two-PV (`G`, `2G`) placement.
- Baseline 1 HDV flow and Baseline 2 yielding rule with safe deferral/cooldown.
- Fair 5%/10%/20% matched rule-based AV subsets.
- Paired fundamental diagrams, all/PV speeds, worst-PV metrics, lap metrics,
  background-flow constraint, bootstrap intervals, CSV/JSON, and plots.
- Protocol and commands: `src/snfs_traffic/baseline/PRIORITY_BASELINES.md`.

### Centralized PV Graph-PPO MVP

- One single-agent coordinator associated with the priority vehicle.
- A configurable local PV window; nearby AVs are graph vertices, while nearby
  HDVs are encoded in vertex features.
- Fixed-size padded observations and masks keep the RLlib tensor contract
  stable while physical vehicles enter and leave the PV neighborhood.
- Six physical neighbor relations and two message-passing layers with a shared
  per-node lateral-action head.
- Native Revised S-NFS behavior outside the controlled neighborhood.
- PV-speed reward with small applied/rejected intervention regularizers and no
  aggregate-flow reward term.
- Local RLlib runner with progress output and periodic checkpoints.

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
- Full Revised S-NFS unit-length Numba longitudinal benchmark report.

---

## Not implemented yet

### RL-facing layer

- Recurrent/temporal graph state beyond the current Markov observation.
- Multi-PV centralized coordination and additional RL algorithms.
- PettingZoo adapter.

### Physics/model extensions

- Road networks and junctions beyond the periodic ring.
- Calibrated heterogeneous human-driver parameter sets.
- Open-boundary topology.
- Open-boundary step semantics.
- New simulator physics beyond the current Revised S-NFS core.

### Performance extensions

- Further optimization of lane-change proposal collection.
- Further optimization of longitudinal collision-avoidance and RNG-draw costs.
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

Install with optional RL/Gymnasium support:

```bash
python -m pip install -e ".[rl]"
```

Install with RLlib support:

```bash
python -m pip install -e ".[rl,ray]"
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
  scenarios/      # initial states, vehicle mix, and priority placement
  rules/          # priority-yield baseline controller
  observations/   # local controlled-only observation schema
  rl/             # reward/episode helpers and single-/multi-agent envs
  envs/           # reserved for future RL environment wrappers
  metrics/        # priority-flow, speed, lap, and paired statistics
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

## Minimal Gymnasium MVP

Install the optional RL extra before importing the Gymnasium wrapper:

```bash
python -m pip install -e ".[rl]"
```

```python
from snfs_traffic.rl.env import SnfsTrafficEnv

env = SnfsTrafficEnv(backend="reference", seed=123)
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(0)
```

MVP environment contract:

- one controlled vehicle per environment;
- lateral-only action space `Discrete(3)`: `0=keep lane`, `1=request left`, `2=request right`;
- observation space is a Gymnasium `Dict` with the existing local controlled-only observation vector and action mask;
- reward is deterministic and includes `speed_reward`, `lane_change_penalty`, `blocked_action_penalty`, `stopped_penalty`, and `total`;
- current periodic ring-road episodes are fixed-horizon by default (`terminated=False` during normal operation, `truncated=True` at `max_steps`);
- RL training policies and PettingZoo are not implemented in this single-agent MVP.

#### Customization hooks

`SnfsTrafficEnv` exposes protected hooks for lightweight subclassing while the Gymnasium API is still MVP-level. Override `_compute_reward(...)` to customize reward calculation, override `_build_observation()` together with `observation_space` to customize observations, and override `_select_single_controlled(...)` to customize priority controlled-vehicle selection at reset. The default environment behavior remains the stable baseline.


### Lightweight multi-agent RL env

`SnfsTrafficMultiAgentEnv` exposes a synchronous multi-agent API. Ray remains
optional: with `.[ray]` installed, the class directly subclasses RLlib's
`MultiAgentEnv`; otherwise it retains the same lightweight Gymnasium API.

- agents are controlled vehicles;
- agent ids use `vehicle_<vehicle_id>`;
- `reset()` returns observation/info dicts keyed by agent id;
- `step(action_dict)` returns observation, reward, termination, truncation, and info dicts keyed by agent id;
- all active agents act simultaneously;
- `terminateds` and `truncateds` include `__all__`;
- all agents share the same lateral-only `Discrete(3)` action space;
- per-agent observations use the same `{"obs", "action_mask"}` schema as the single-agent MVP;
- protected hooks `_select_controlled(...)`, `_build_observations()`, and `_compute_agent_reward(...)` support lightweight subclass customization;
- constructor seeding produces a reproducible sequence of distinct episodes;
- lateral masks use the exact action order `stay, left, right`.


### Multi-agent speed-control env

`SnfsTrafficSpeedControlMultiAgentEnv` extends the lightweight multi-agent env with a per-agent `MultiDiscrete([3, 3])` action space.

- first component: lateral action, `0=stay`, `1=left`, `2=right`;
- second component: speed action, `0=brake`, `1=keep speed`, `2=accelerate`;
- speed actions map to `speed_delta ∈ {-1, 0, +1}`;
- requested speed is clipped by `vmax_controlled` and safety/gap constraints;
- uncontrolled HDV and priority vehicles still use normal Rev S-NFS dynamics;
- existing lateral-only envs remain unchanged.

### Centralized PV Graph-PPO

Install the local training dependencies:

```bash
python -m pip install -e ".[rl,ray,numba]"
```

Run the integration smoke first:

```bash
PYTHONPATH=src python research/train_pv_graph_ppo.py --smoke
```

Start a local training run, for example:

```bash
PYTHONPATH=src python research/train_pv_graph_ppo.py \
  --density 0.20 \
  --av-fraction 0.95 \
  --workers 8 \
  --iterations 1000 \
  --out-dir artifacts/rl/pv_graph_ppo
```

The environment exposes one fixed-size `MultiDiscrete` action vector. For each
active AV node, action indices are `0=left`, `1=stay`, and `2=right`; padded
nodes are masked to `stay`. AVs outside the local graph keep native Revised
S-NFS lane-changing behavior.

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
  --warmups 2 \
  --steps 50 \
  --out-json reports/full_rev_snfs_numba/optimized_smoke.json \
  --out-md reports/full_rev_snfs_numba/optimized_smoke.md
```

Optimized reduced-standard benchmark:

```bash
python benchmarks/bench_optimized_full_step.py \
  --preset standard \
  --backend both \
  --repeats 3 \
  --warmups 2 \
  --steps 50 \
  --cases medium_moderate,medium_dense,wide_moderate \
  --out-json reports/full_rev_snfs_numba/optimized_reduced_standard.json \
  --out-md reports/full_rev_snfs_numba/optimized_reduced_standard.md
```

Speed-control optimized rollout benchmark:

```bash
PYTHONPATH=src python benchmarks/bench_speed_control_rollout.py \
  --road-length 1000 \
  --num-lanes 4 \
  --density 0.30 \
  --num-controlled 1000 \
  --steps 500 \
  --warmup-steps 20 \
  --seed 123
```

This compares `backend="reference"` and `backend="optimized"` for `SnfsTrafficSpeedControlMultiAgentEnv` with `MultiDiscrete([3, 3])` speed-control actions.

Benchmark numbers are environment-dependent. Use them for relative comparison inside the same environment, not as absolute production-performance claims.

---

## Latest representative benchmark status

The current reference path implements the full Revised S-NFS longitudinal dynamics. The optimized backend accelerates the same full longitudinal velocity update with Numba for the common case where all alive vehicles have `length == 1`; if any alive vehicle has `length != 1`, `OptimizedBackend` intentionally falls back to reference semantics. Benchmark numbers are environment-dependent and should be compared only within the same machine/interpreter run. Older Task 22+23 speedups are historical pre-full-Rev-SNFS-longitudinal-Numba results and have been replaced here by the fresh report in `reports/full_rev_snfs_numba/optimized_reduced_standard.md`.

Representative reduced-standard results from this environment:

| case | reference mean ms/step | optimized mean ms/step | speedup | equivalence | top optimized component |
|---|---:|---:|---:|---|---|
| medium_moderate | 491.1606 | 6.0169 | 81.630x | yes | `longitudinal_random_draws` |
| medium_dense | 2761.7621 | 20.0826 | 137.520x | yes | `longitudinal_velocity_numba` |
| wide_moderate | 1001.5073 | 10.0021 | 100.130x | yes | `longitudinal_random_draws` |

Equivalence means both `state_equal=true` and `rng_next_draw_equal=true` in the benchmark output.

Fresh recommendation:

```text
keep OptimizedBackend supported and merge indexing fast path
```

Current optimized bottleneck after the Numba velocity kernel is longitudinal-related overall: medium-density cases are dominated by scalar longitudinal RNG draw generation, while the dense case is dominated by the Numba longitudinal velocity kernel, primarily its collision-avoidance propagation work. Lane-change proposal collection is now secondary in the representative reduced-standard profile.

---

## Current model limitations

The current simulator uses length-aware body validity with head-based lane ordering:

- body occupancy is length-aware for validity/collision checks while head occupancy still drives lane ordering;
- longitudinal and lane-change gaps use length-aware empty-cell conventions;
- body occupancy is length-aware and overlaps are invalid;
- head ordering remains based on head cells for indexing;
- lane changes are lateral only and do not move longitudinal position;
- there are no same-step lateral swaps into previously occupied target cells;
- controlled vehicles support facade-level lateral actions plus MVP
  reward/episode/Gymnasium wrapper semantics; the centralized PV Graph-PPO path
  is implemented, while broader RL training stacks are not;
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

Next roadmap:

1. freeze the Baseline 1/2 pilot density region from paired results;
2. run the Graph-PPO integration smoke in the target local environment;
3. train and evaluate the centralized PV coordinator against paired baselines;
4. add temporal state or multi-PV coordination only when experiments justify it;
5. add PettingZoo only if a downstream integration requires it.

The next architectural step should be driven by the first local Graph-PPO
learning curves and baseline comparisons, not by another low-level optimization
pass unless performance becomes a blocker.


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

Visualization is a tooling side-feature: it observes `TrafficState` snapshots, does not step the simulator, does not consume RNG, and uses visual-only interpolation between discrete simulator steps. Default drawing can use state lengths; visualization does not alter simulator collision semantics. `body_mode="state_length"` only changes rendering and does not change collision/occupancy semantics.

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

```python
from snfs_traffic.core import SimulationParams
from snfs_traffic.scenarios import VehicleMix
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator

sim = TrafficSimulator(
    params=SimulationParams(num_lanes=3, road_length=120),
    backend="auto",  # "reference" | "optimized" | "auto"
    rng_seed=123,
    scenario=ScenarioConfig(
        density=0.20,
        seed=1,
        vehicle_mix=VehicleMix(controlled_fraction=0.03),
    ),
    validate=True,
)
state = sim.reset()
obs = sim.observe()
state = sim.step(actions=None)
```

## Controlled lateral actions

- Action values are `-1`, `0`, `+1`.
  - `-1`: target lane `lane - 1`
  - `0`: stay
  - `+1`: target lane `lane + 1`
- Actions are keyed by stable `vehicle_id`.
- Action IDs must refer to alive controlled vehicles.
- `require_all_controlled_actions=True` requires an action for every alive controlled vehicle.
- `require_all_controlled_actions=False` fills missing controlled actions with stay (`0`).
- Structural invalid inputs raise `ValueError`.
- Unsafe / out-of-bounds / target-occupied / controlled-conflict commands are reported in `ControlledActionResult`.
- Explicit stay actions are not equivalent to `actions=None`.
- `actions=None` preserves selected backend behavior exactly.

## Local observations

- Gym-free observation API.
- Rows are alive controlled vehicles only and align with `obs.vehicle_id`.
- Default dtype is `float32`; shape is `(n_alive_controlled, 21)`.
- `action_mask` shape is `(n_alive_controlled, 3)` with columns `[-1, 0, +1]`.
- Exact 21 feature names in order:
  1. `ego_lane_norm`
  2. `ego_pos_norm`
  3. `ego_vel_norm`
  4. `front_gap_norm`
  5. `front_rel_speed_norm`
  6. `back_gap_norm`
  7. `back_rel_speed_norm`
  8. `left_exists`
  9. `left_cell_free`
  10. `left_front_gap_norm`
  11. `left_front_rel_speed_norm`
  12. `left_back_gap_norm`
  13. `left_back_rel_speed_norm`
  14. `left_safe`
  15. `right_exists`
  16. `right_cell_free`
  17. `right_front_gap_norm`
  18. `right_front_rel_speed_norm`
  19. `right_back_gap_norm`
  20. `right_back_rel_speed_norm`
  21. `right_safe`

## Visualization with simulator rollout

```python
from snfs_traffic.visualization import RoadRenderConfig, RoadRenderer, VideoWriter

state = sim.reset()
renderer = RoadRenderer(
    params=sim.params,
    topology=sim.topology,
    config=RoadRenderConfig(interpolation_frames=2),
)
renderer.reset(state)

with VideoWriter("episode.mp4", fps=24) as video:
    for snap in sim.iter_rollout(steps=50):
        video.write_many(renderer.render_step(snap.state, step=snap.step))
```
