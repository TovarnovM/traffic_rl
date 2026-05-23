# PLAN.md — Revised S-NFS Traffic Simulator Roadmap

Актуальное состояние: после завершения **Tasks 1–11** в загруженном репозитории.

Проект — чистая новая реализация Revised S-NFS traffic simulator для будущих multi-agent reinforcement learning экспериментов. Главная архитектурная линия остаётся прежней: массивное NumPy-состояние, маленькое и тестируемое core-ядро, отсутствие Python object graph в hot loop, постепенный переход от reference NumPy/Python реализации к оптимизированному backend.

Важно: `step_reference(...)` уже реализован как композиция reference lane-change phase и reference longitudinal phase с сохранением lane-change флагов после longitudinal шага. Tasks 1–11 are complete. Task 11 завершил выделение backend-neutral pure-array indexing kernels и их equivalence coverage без изменения physics. Следующий непосредственный task — Task 12: optional Numba implementation of the indexing kernels with reference equivalence tests.

---

# 0. Текущий статус реализации

## Уже реализовано

```text
Task 1 — project bootstrap and import smoke tests.
Task 2 — core SimulationParams and TrafficState schemas.
Task 3 — reference periodic RingTopology.
Task 4 — minimal reproducible uniform-random scenario initializer.
Task 5 — reference head-cell occupancy / lane order / neighbor indexing.
Task 6 — reference longitudinal same-lane step without lane changes.
Task 7 — reference lane-change phase using Eq. (8)/(9), P_CL = p_lane_change = 0.5 by default, and stochastic conflict resolution.
Task 8 — full reference step composing lane-change phase and longitudinal phase.
Task 9 — runtime invariant suite / random rollout invariant tests.
Task 10 — minimal backend contract and reference-vs-backend equivalence scaffolding.
Task 11 — backend-neutral pure-array indexing kernels and reference equivalence tests.
```

Текущее ядро содержит:

```text
src/snfs_traffic/
  core/
    __init__.py
    params.py
    state.py
    types.py
    indexing.py
    step_reference.py              # longitudinal + full reference step composition
    lane_change_reference.py       # reference lane-change phase
  topology/
    base.py
    ring.py
  scenarios/
    init.py
  rules/
  observations/
  envs/
  metrics/
  io/
```

Текущие тесты покрывают:

```text
tests/test_imports.py
tests/test_state_schema.py
tests/test_ring_topology.py
tests/test_init_scenarios.py
tests/test_indexing.py
tests/test_indexing_kernels.py
tests/test_snfs_longitudinal.py
tests/test_snfs_lane_change.py
tests/test_snfs_full_step.py
tests/test_runtime_invariants.py
```

## Текущий публичный core API

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
    step_reference,
    step_lane_change_reference,
    validate_runtime_invariants,
    StepBackend,
    ReferenceBackend,
    get_reference_backend,
)
```

## Текущий публичный topology API

```python
from snfs_traffic.topology import RingTopology
```

## Текущий публичный scenario API

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

## Следующий непосредственный task

```text
Task 12 — optional Numba implementation of the indexing kernels with reference equivalence tests.
```

Tasks 1–11 are complete. Следующий приоритет — optional Numba implementation of indexing kernels behind the established pure-array kernel contract.

---

# 1. Главная архитектурная декомпозиция

Проект разделяется на независимые слои:

```text
snfs_traffic/
  core/           # быстрый state transition engine и reference physics
  topology/       # ring/open/segment graph/intersections later
  scenarios/      # initial state generation, spawn policies later
  rules/          # HDV/AV/RL/infrastructure rule ids and presets
  observations/   # full-state, local grid, graph observation builders
  envs/           # Gymnasium/RLlib/PettingZoo adapters
  metrics/        # validation metrics, fundamental diagram, rollout stats
  io/             # snapshots, restore, export/import
```

Критическое правило:

```text
core не импортирует gymnasium, ray, torch, rllib, matplotlib, pandas, scipy.
```

`core` должен оставаться маленьким, быстрым, framework-independent и проверяемым.

Запрещено строить hot loop вокруг объектов `Vehicle`, `RoadLane`, `TrafficModel` или любого object graph. Машины — это строки в массивах.

---

# 2. Dependency policy

Текущий базовый runtime dependency:

```text
numpy
```

Текущий dev/test dependency:

```text
pytest
```

Добавлять позже только отдельными task-ами:

```text
numba       # только начиная с Numba backend / kernel tasks
gymnasium   # только начиная с Gym env task
ray/rllib   # только начиная с RLlib adapter task, не в core
```

Не добавлять в `core`, `scenarios`, `topology` на текущих этапах:

```text
gymnasium
ray
rllib
torch
pandas
scipy
matplotlib
networkx
```

---

# 3. Зафиксированный runtime-контракт текущего reference core

## 3.1. TrafficState schema

`TrafficState` хранит только массивы:

```python
vehicle_id:      int32[N]
lane:            int16[N]
pos:             int32[N]
vel:             int16[N]
length:          int16[N]
veh_type:        int16[N]
behavior_id:     int16[N]
alive:           bool[N]
last_lane_delta: int8[N]
changed_lane:    bool[N]
controlled:      bool[N]
```

`pos` на текущем этапе означает **head cell**. Для `length > 1` тело машины пока не размечается. `length` является metadata и сейчас не участвует в occupancy/gaps/collision checks.

## 3.2. Reference indexing state

Текущий indexing слой реализует:

```python
occupancy:       int32[num_lanes, road_length]  # -1 или vehicle array index
lane_order:      int32[num_lanes, road_length]
lane_counts:     int32[num_lanes]
lane_rank:       int32[N]
front_id:        int32[N]
back_id:         int32[N]
front_gap:       int32[N]
back_gap:        int32[N]
```

Текущий public API:

```python
build_occupancy(state, params) -> occupancy
build_lane_order(occupancy, *, n_vehicles) -> lane_order, lane_counts, lane_rank
compute_neighbors(state, lane_order, lane_counts, lane_rank, topology) -> front_id, back_id, front_gap, back_gap
```

Обязательный контракт indexing:

```text
- occupancy marks head cells only;
- occupancy[lane, head_pos] = vehicle array index;
- vehicle_id не используется как индекс occupancy;
- inactive vehicles не попадают в occupancy/lane_order/neighbors;
- duplicate alive head cells запрещены;
- lane_order хранит vehicle array indices в порядке возрастающего pos;
- lane_rank[i] = rank машины i внутри своей полосы;
- front/back neighbors считаются только внутри одной lane;
- one-vehicle lane policy: front_id/back_id/front_gap/back_gap = -1;
- gaps are head-cell empty gaps;
- length сейчас игнорируется в occupancy и gaps.
```

Это намеренно. Полная length-aware occupancy и bumper-to-bumper gaps остаются отдельным поздним milestone.

## 3.3. Longitudinal reference phase

Текущий API:

```python
step_longitudinal_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
) -> TrafficState
```

Текущие semantics:

```text
- validates state;
- validates periodic RingTopology compatibility;
- validates rng;
- computes same-lane neighbor indexing from input state;
- updates vel and pos for alive vehicles;
- does not change lane;
- resets changed_lane[:] = False and last_lane_delta[:] = 0 because this phase performs no lane changes;
- keeps inactive vehicles unchanged;
- validates result and rebuilds occupancy;
- remains head-cell-only and length-ignored.
```

Важное ограничение: это текущая explicit reference semantics, а не гарантированная финальная paper-exact Revised S-NFS longitudinal equation mapping. Если будет отдельно предоставлена формальная версия уравнений, её нужно интегрировать отдельным task-ом.

## 3.4. Lane-change reference phase

Текущий API:

```python
step_lane_change_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
) -> TrafficState
```

Текущие semantics:

```text
- validates state;
- validates periodic RingTopology compatibility;
- validates rng;
- computes all lane-change decisions from old state / old indexing;
- uses Eq. (8) incentive criterion;
- uses Eq. (9) safety criterion;
- uses params.p_lane_change, default 0.5;
- uses stochastic side tie-break when both adjacent lanes are eligible;
- uses stochastic conflict resolution for simultaneous target-cell conflicts;
- changes only lane and lane-change flags;
- does not change pos or vel;
- sets changed_lane[i] = True and last_lane_delta[i] = target_lane - old_lane for accepted movers;
- resets changed_lane/last_lane_delta for non-movers;
- disallows same-step lateral swaps into previously occupied target cells;
- keeps head-cell-only, length-ignored semantics.
```

## 3.5. Full-step target semantics

Целевой public API после следующего task:

```python
step_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
) -> TrafficState
```

Фиксируем порядок полного reference шага:

```text
1. lane-change phase;
2. recompute indexing implicitly inside longitudinal phase;
3. longitudinal movement phase.
```

Критически важно: longitudinal movement должен считаться после перестроений, по новым lane assignment. То есть машина, перестроившаяся в более свободную полосу, должна ускоряться/двигаться относительно новой полосы, а не старой.

`step_longitudinal_reference(...)` сейчас сбрасывает `changed_lane` и `last_lane_delta`, поэтому будущий `step_reference(...)` обязан сохранить флаги lane-change phase и восстановить их после longitudinal phase.

Правильная композиция:

```python
after_lc = step_lane_change_reference(state, params, topology, rng)
lane_delta = after_lc.last_lane_delta.copy()
changed_lane = after_lc.changed_lane.copy()

after_long = step_longitudinal_reference(after_lc, params, topology, rng)
out = after_long.copy()
out.last_lane_delta = lane_delta
out.changed_lane = changed_lane

validate_state(out, params)
build_occupancy(out, params)
return out
```

После полного шага `state.changed_lane` / `state.last_lane_delta` должны описывать lateral movement, произошедший именно в этом полном шаге. Будущие observation builders смогут использовать эти поля как recent lane-changing behavior.

---

# 4. Ближайшая очередь task-ов

## Task 8 — full reference step composition

Цель: добавить маленький, явный, тестируемый `step_reference(...)`, который только композирует уже реализованные фазы.

Files:

```text
Modify:
  src/snfs_traffic/core/step_reference.py
  src/snfs_traffic/core/__init__.py
  README.md

Add:
  tests/test_snfs_full_step.py
tests/test_runtime_invariants.py
```

Не добавлять новую физику. Не менять lane-change или longitudinal rules без необходимости.

Обязательные проверки:

```text
- public import of step_reference;
- no input state mutation;
- full step equals manual phase composition with same rng;
- lane-change phase happens before longitudinal movement;
- lane-change flags survive longitudinal phase;
- one-lane full step equals longitudinal step;
- p_lane_change=0 blocks lateral movement but still allows longitudinal update;
- conflict resolution participates in full step;
- deterministic with same seed;
- different seeds can produce different outcomes;
- invalid rng/topology rejected;
- random rollout remains valid;
- density=1.0 remains valid;
- bus length ignored intentionally.
```

Expected after Task 8:

```python
from snfs_traffic.core import step_reference
```

## Task 9 — runtime invariant suite / random rollout tests

Цель: вынести повторяющиеся runtime-проверки состояния в явный reusable слой, чтобы зафиксировать correctness contract перед оптимизацией.

Suggested files:

```text
Add:
  src/snfs_traffic/core/invariants.py
  tests/test_invariants_random_rollouts.py
```

Возможный API:

```python
validate_runtime_invariants(state, params, topology) -> None
```

Проверки:

```text
- validate_state passes;
- build_occupancy succeeds;
- occupied head-cell count equals alive vehicle count;
- no duplicate alive head cells;
- alive lanes are in [0, num_lanes);
- alive positions are in [0, road_length);
- alive velocities are non-negative;
- uncontrolled alive velocities <= params.vmax_default;
- controlled alive velocities <= params.vmax_controlled;
- changed_lane == (last_lane_delta != 0) for alive vehicles;
- last_lane_delta values are only -1, 0, +1;
- alive count is stable for closed periodic ring rollouts;
- length remains ignored by occupancy/gaps under current reference semantics.
```

Rollout coverage:

```text
num_lanes: 1, 2, 3, 4
road_length: small and medium values
density: 0.05, 0.2, 0.5, 0.8, 1.0
steps: 50–200 depending on test cost
seeds: multiple fixed seeds
```

Этот task должен не менять physics. Его задача — зафиксировать invariant contract перед Numba/backend work.

## Completed Task 10 — minimal backend contract and reference-vs-backend equivalence scaffolding

Статус: выполнено.

Что реализовано:

```text
- добавлен минимальный backend contract (StepBackend protocol);
- добавлен ReferenceBackend, делегирующий в step_reference(...);
- добавлен get_reference_backend() singleton accessor;
- добавлены reference-vs-backend equivalence tests на фиксированных seeds/scenarios;
- зафиксированы ограничения RNG/head-cell semantics/documentation без изменения physics.
```

## Completed Task 11 — backend-neutral pure-array indexing kernels and reference equivalence tests

Статус: выполнено.

Что реализовано:

```text
- добавлен `src/snfs_traffic/core/indexing_kernels.py`;
- добавлены pure-array kernels: `build_occupancy_kernel`, `build_lane_order_kernel`, `compute_neighbors_kernel`;
- публичные wrappers в `src/snfs_traffic/core/indexing.py` сохранили валидацию входов и делегируют вычисления kernels;
- добавлены equivalence tests (`tests/test_indexing_kernels.py`), доказывающие совпадение outputs kernels с публичными reference wrappers;
- Numba/Cython/optimized backend в рамках Task 11 не добавлялись;
- physics/RNG/public API semantics не менялись.
```

## Task 12 — optional Numba implementation of the indexing kernels with reference equivalence tests

Цель: опционально ускорить occupancy/lane_order/neighbors поверх уже зафиксированного pure-array kernel contract, не меняя semantics.

Обязательные требования:

```text
- reference Python/NumPy implementation remains available;
- Numba implementation is optional and must preserve outputs;
- tests compare Numba kernel outputs to reference kernel/wrapper outputs over many random states;
- no change in head-cell-only semantics;
- no length-aware logic;
- no RL actions;
- no simulator facade yet unless needed only for backend selection.
```

## Task 13 — Numba full-step equivalence

Цель: accelerated full-step backend, эквивалентный `step_reference(...)` на зафиксированных scenarios/seeds.

Обязательные требования:

```text
- same RNG semantics must be explicitly handled or documented;
- if exact RNG equivalence is hard, split deterministic kernels and stochastic draw preparation;
- reference step remains source of truth;
- tests compare output arrays and invariants;
- performance benchmark can be tiny and optional, not a hard correctness dependency.
```

## Task 14 — thin simulator facade

Цель: добавить минимальный удобный facade поверх `TrafficState`, `SimulationParams`, `RingTopology`, `step_reference` / backend step.

Пример целевого API:

```python
sim = SnfsSimulator(params, topology, initial_state, seed=123)
state = sim.state
state = sim.step()
```

Ограничения:

```text
- facade must stay thin;
- no Gymnasium yet;
- no RL actions yet unless separate task explicitly defines them;
- no object graph of Vehicle/RoadLane;
- state remains array-oriented.
```

## Task 15 — controlled action semantics

Цель: определить, как external RL actions влияют на controlled vehicles.

Это нельзя делать неявно. Нужно отдельное специфицированное решение:

```text
action_accel: int8[N]  # e.g. -1, 0, +1
action_lane:  int8[N]  # e.g. -1, 0, +1
```

Надо решить:

```text
- action validity and clipping;
- interaction with safety constraints;
- whether controlled actions override or bias reference lane-change probabilities;
- how controlled vehicles interact with p_lane_change;
- whether HDV/AV internal rules remain unchanged;
- deterministic behavior under fixed seed.
```

До этого момента controlled vehicles отличаются только `vmax_controlled` и флагом `controlled`; внешние RL actions не реализованы.

## Task 15 — observation builders

Цель: добавить первые observation builders без Gym/RLlib.

Возможные builders:

```text
- full-state observation;
- ego/local lane-window observation;
- occupancy grid observation;
- later graph/GNN-compatible observation.
```

Ограничения:

```text
- observations не должны менять simulation state;
- observations не должны тянуть torch/ray/rllib;
- output должен быть NumPy arrays / plain dicts;
- graph-specific framework integration позже отдельным task-ом.
```

## Task 16 — metrics and rollout diagnostics

Цель: добавить метрики без тяжёлых dependencies.

Примеры:

```text
- mean speed;
- flow / throughput on ring;
- density by lane;
- lane-change count/rate;
- stopped vehicle count;
- collision/duplicate occupancy assertions;
- fundamental-diagram-friendly summaries.
```

No matplotlib/pandas in core. Export plotting или dataframe conversion — только отдельно и вне hot core.

## Task 17 — Gymnasium environment wrapper

Цель: добавить Gymnasium-compatible env поверх уже стабильного simulator facade and observation/action semantics.

Только здесь можно добавлять `gymnasium`.

Ограничения:

```text
- Gym wrapper не должен загрязнять core;
- core остаётся importable without gymnasium;
- tests должны проверять импорт core без gymnasium при необходимости;
- env action/observation spaces должны соответствовать уже утверждённым semantics.
```

## Task 18 — multi-agent / RLlib adapter

Цель: добавить multi-agent adapter, когда уже есть:

```text
- stable simulator facade;
- controlled action semantics;
- observation builders;
- reward/metric basics;
- Gymnasium wrapper.
```

Ray/RLlib не должны попадать в core imports.

---

# 5. Later milestones

## 5.1. Length-aware geometry

Текущая реализация намеренно head-cell-only. Отдельный будущий milestone:

```text
- multi-cell occupancy for vehicle bodies;
- bumper-to-bumper gaps;
- bus/body collision geometry;
- lane-change safety based on body extents;
- validation of mixed lengths.
```

Это нельзя добавлять маленькими незаметными правками в текущие reference tasks, потому что это меняет фундаментальную semantics indexing/collisions.

## 5.2. Open-boundary topology

Сейчас поддержан periodic ring. Отдельный future milestone:

```text
- open segment topology;
- spawn/despawn policies;
- boundary inflow/outflow;
- route/lane availability constraints;
- validation for non-periodic indexing and neighbor semantics.
```

## 5.3. More exact Revised S-NFS paper equations

Если будут предоставлены формальные уравнения/таблицы/параметры для полного paper-exact Revised S-NFS longitudinal update, интегрировать их отдельным task-ом.

Важно не смешивать:

```text
- current operational reference semantics;
- paper-exact equation mapping;
- performance backend;
- RL action semantics.
```

Каждая из этих тем должна быть отдельной проверяемой задачей.

## 5.4. IO / snapshots / reproducibility

Позже:

```text
- snapshot save/load;
- deterministic rollout replay;
- compact binary or npz export;
- scenario config serialization;
- benchmark fixtures.
```

Не нужно до stabilization of core step and invariants.

---

# 6. Global non-goals for current phase

До завершения reference full step + invariant suite не делать:

```text
- Gymnasium env;
- RLlib/PettingZoo adapter;
- graph observations;
- controlled external action semantics;
- simulator facade with broad feature surface;
- Numba/Cython kernels;
- pandas/matplotlib reporting;
- length-aware occupancy;
- open-boundary traffic;
- object-oriented vehicle/lane model.
```

---

# 7. Current correctness contract summary

На текущем этапе проект гарантирует только это:

```text
- state is array-oriented and validated;
- topology is periodic RingTopology;
- scenario initializer can generate reproducible uniform-random states;
- occupancy/lane_order/neighbors are head-cell-only and same-lane;
- longitudinal phase updates velocity/position without lane changes;
- lane-change phase updates lane and lane-change flags without longitudinal movement;
- lane-change conflicts for same target head cell are resolved stochastically;
- inactive vehicles are ignored by occupancy/lane_order/neighbors and are not moved by phases;
- vehicle length is metadata only for current reference core.
```

Не гарантируется пока:

```text
- full composed step in public API;
- external controlled RL actions;
- observations;
- metrics;
- simulator facade;
- RL environments;
- length-aware body occupancy;
- open boundary behavior;
- Numba/Cython acceleration;
- final paper-exact longitudinal equation mapping.
```

---

# 8. Recommended immediate validation commands

После каждого task-а запускать:

```bash
python -m pip install -e .
pytest -q
```

Для текущего состояния особенно важны:

```bash
pytest -q tests/test_snfs_longitudinal.py
pytest -q tests/test_snfs_lane_change.py
pytest -q tests/test_indexing.py
pytest -q
```

После Task 8 добавить:

```bash
pytest -q tests/test_snfs_full_step.py
pytest -q
```

После Task 9 добавить:

```bash
pytest -q tests/test_invariants_random_rollouts.py
pytest -q
```