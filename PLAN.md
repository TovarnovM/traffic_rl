# PLAN.md — Revised S-NFS Traffic Simulator Roadmap

Актуальное состояние: после завершения Tasks 1–5.

Проект — чистая новая реализация Revised S-NFS traffic simulator для будущих multi-agent reinforcement learning экспериментов. Главная архитектурная линия: массивное NumPy-состояние, маленькое и тестируемое core-ядро, отсутствие Python object graph в hot loop, постепенный переход от reference NumPy/Python к Numba/Cython backend.

---

# 0. Текущий статус реализации

## Уже реализовано

```text
Task 1 — project bootstrap and import smoke tests.
Task 2 — core SimulationParams and TrafficState schemas.
Task 3 — reference periodic RingTopology.
Task 4 — minimal reproducible uniform-random scenario initializer.
Task 5 — reference head-cell occupancy / lane order / neighbor indexing.
```

Текущее ядро содержит:

```text
src/snfs_traffic/
  core/
    params.py
    state.py
    types.py
    indexing.py
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

Текущий публичный core API:

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
)
```

Текущий публичный topology API:

```python
from snfs_traffic.topology import RingTopology
```

Текущий публичный scenario API:

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

## Текущий следующий task

```text
Task 6 — reference longitudinal Revised S-NFS without lane changes.
```

Именно с него надо продолжать. Переходить к lane-change, envs, observations или Numba раньше нельзя: сначала нужен корректный reference longitudinal step.

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
numba       # только начиная с Numba kernel tasks
gymnasium   # только начиная с Gym env task
ray/rllib   # только начиная с RLlib adapter task, не в core
```

Не добавлять в core/scenarios/topology на текущих этапах:

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

# 3. Целевой runtime-контракт

## 3.1. TrafficState schema

Текущий `TrafficState` хранит только массивы:

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

`pos` на текущем этапе означает head cell. Для `length > 1` тело машины пока не размечается.

## 3.2. Reference indexing state

Текущий Task 5 уже реализовал reference indexing:

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

Обязательный контракт Task 5:

```text
- occupancy marks head cells only;
- occupancy[lane, head_pos] = vehicle_index;
- vehicle_id не используется как индекс occupancy;
- inactive vehicles не попадают в occupancy/lane_order;
- duplicate alive head cells запрещены;
- lane_order хранит vehicle array indices в порядке возрастающего pos;
- lane_rank[i] = rank машины i внутри своей полосы;
- front/back neighbors считаются только внутри одной lane;
- one-vehicle lane policy: front_id/back_id/front_gap/back_gap = -1;
- gaps are head-cell empty gaps;
- length сейчас игнорируется в occupancy и gaps.
```

Это намеренно. Полная length-aware occupancy и bumper-to-bumper gaps остаются Task 26.

## 3.3. Step API target

Целевой simulator API:

```python
sim.reset(seed, scenario_config)
state_view = sim.state_view()
sim.step(actions)
```

Где `actions` — необязательные намерения controlled vehicles:

```python
action_accel: int8[N]  # -1, 0, +1, or 0 for uncontrolled
action_lane:  int8[N]  # -1, 0, +1, or 0 for uncontrolled
```

Не все машины обязаны быть RL-controlled. Для HDV/AV actions генерируются внутренними rules.

## 3.4. Целевой порядок фаз полного шага

Фиксируем один порядок:

```text
1. build occupancy / lane order
2. compute front/back neighbors
3. decide lane-change intents
4. resolve lane-change conflicts
5. commit lane changes
6. rebuild occupancy / lane order
7. compute front/back neighbors
8. compute longitudinal velocity update
9. apply collision avoidance
10. commit positions
11. update metrics/events
```

Это соответствует главному требованию модели: сначала перестроения на соседнюю полосу, потом независимое обновление состояний, скоростей и позиций. Межшаговых interaction после commit positions нет.

---

# 4. Topology strategy

MVP topology уже реализована как periodic multi-lane ring:

```python
RingTopology(
    num_lanes=4,
    length=1500,
)
```

Текущий API:

```python
boundary == "periodic"
normalize_pos(pos)
forward_distance(from_pos, to_pos)
signed_delta(from_pos, to_pos)
```

Критическое правило: расстояния через boundary считаются через topology, а не вручную через `% road_length` по всему коду.

Дальнейшая стратегия:

```text
MVP = RingTopology.
Open boundary = отдельный Task 27.
Segment graph / ramps / intersections = отдельный Task 28.
```

Не надо сейчас преждевременно строить универсальный road graph. Это сломает простоту hot loop до того, как будет проверено S-NFS ядро.

---

# 5. План этапов / Codex tasks

## Task 1. Bootstrap нового проекта и базовая структура

**Status:** completed.

**Цель:** создать чистый проект без переноса старой объектной архитектуры.

**Реализовано:**

```text
- pyproject.toml
- README.md
- src/snfs_traffic/
- tests/
- package import smoke tests
- .devcontainer / docker-related files
```

**Важный результат:** проект стартовал как array-oriented simulator, без переноса старых `Vehicle`, `RoadLane`, `TrafficModel` как основы.

---

## Task 2. Configs и типизированные структуры состояния

**Status:** completed.

**Цель:** зафиксировать state schema и параметры модели.

**Реализовано:**

```text
src/snfs_traffic/core/state.py
src/snfs_traffic/core/params.py
src/snfs_traffic/core/types.py
tests/test_state_schema.py
```

Текущие основные структуры:

```python
SimulationParams
TrafficState
empty_state
validate_state
max_supported_velocity
```

Текущий `SimulationParams` включает:

```python
num_lanes
road_length
vmax_default
vmax_controlled
G
q
r
S
P1
P2
P3
P4
p_lane_change
```

Проверяется:

```text
- dtype массивов;
- shape compatibility;
- lane/pos/vel ranges;
- copy semantics;
- params validation.
```

---

## Task 3. Reference topology: periodic ring segment

**Status:** completed.

**Цель:** сделать минимальную topology без overengineering.

**Реализовано:**

```text
src/snfs_traffic/topology/base.py
src/snfs_traffic/topology/ring.py
tests/test_ring_topology.py
```

Текущий `RingTopology`:

```python
class RingTopology:
    num_lanes: int
    length: int
    boundary == "periodic"

    def normalize_pos(pos): ...
    def forward_distance(from_pos, to_pos): ...
    def signed_delta(from_pos, to_pos): ...
```

---

## Task 4. Scenario initializer: uniform random initial state

**Status:** completed.

**Цель:** уметь создавать корректное начальное состояние.

**Реализовано:**

```text
src/snfs_traffic/scenarios/init.py
tests/test_init_scenarios.py
```

Текущий API:

```python
make_uniform_random_state(
    *,
    num_lanes: int,
    road_length: int,
    density: float,
    seed: int | np.integer,
    vehicle_mix: VehicleMix | None = None,
) -> TrafficState
```

Поддерживается metadata для:

```text
- HDV
- AV
- controlled AV
- bus
```

Важное ограничение:

```text
bus length хранится только в state.length;
body cells автобусов пока не занимают отдельные occupancy cells;
уникальность гарантируется только для head cells.
```

---

## Task 5. Occupancy и lane order reference implementation

**Status:** completed.

**Цель:** построить корректный reference indexing layer для будущих S-NFS dynamics, lane-change, observations, metrics и invariant checks.

**Реализовано:**

```text
src/snfs_traffic/core/indexing.py
tests/test_indexing.py
exports from src/snfs_traffic/core/__init__.py
README status update
```

Текущий API:

```python
INDEX_DTYPE = np.int32
MISSING_INDEX = -1
MISSING_GAP = -1

build_occupancy(state, params) -> np.ndarray
build_lane_order(occupancy, *, n_vehicles) -> tuple[np.ndarray, np.ndarray, np.ndarray]
compute_neighbors(state, lane_order, lane_counts, lane_rank, topology) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Контракт:

```text
- HEAD-CELL indexing only;
- occupancy shape = (params.num_lanes, params.road_length);
- occupancy dtype = int32;
- occupancy stores vehicle array index, not vehicle_id;
- inactive vehicles ignored;
- duplicate alive head cells rejected;
- lane_order ordered by increasing pos;
- lane_rank missing/inactive = -1;
- front/back neighbors are same-lane only;
- lanes with 0 or 1 vehicle return missing neighbors/gaps;
- front_gap/back_gap are head-cell empty gaps;
- state.length ignored for now.
```

Это важный foundation. Не менять семантику Task 5 при реализации Task 6–8 без отдельного task.

---

## Task 6. Reference longitudinal Revised S-NFS без lane changes

**Status:** next.

**Цель:** реализовать продольную динамику Revised S-NFS для одной/нескольких полос без перестроений.

Этот task должен использовать уже готовый indexing layer:

```text
build_occupancy
build_lane_order
compute_neighbors
```

**Файлы:**

```text
src/snfs_traffic/core/step_reference.py
tests/test_snfs_longitudinal.py
```

**Что сделать:**

```text
- acceleration;
- slow-to-start;
- perspective / anticipation;
- random braking;
- collision avoidance;
- movement / position commit.
```

Параметры default preset уже есть в `SimulationParams`:

```text
G = 15
q = 0.99
r = 0.99
S = 2
P1 = 0.999
P2 = 0.99
P3 = 0.98
P4 = 0.01
vmax_default = 5
vmax_controlled = 6
```

**Scope для Task 6:**

```text
- no lane-change decisions;
- no lane-change conflict resolution;
- no action API beyond optional future shape if needed;
- no simulator facade;
- no observations;
- no metrics;
- no Numba;
- no length-aware occupancy;
- no multi-cell bus collision geometry.
```

**Ожидаемый API:**

Минимально и без overengineering:

```python
def step_longitudinal_reference(
    state: TrafficState,
    params: SimulationParams,
    topology: RingTopology,
    rng: np.random.Generator,
) -> TrafficState:
    ...
```

Допустим in-place вариант только если он явно документирован и покрыт тестами:

```python
def step_longitudinal_reference_inplace(...): ...
```

Лучше начать с copy-return или explicit in-place policy, но не смешивать оба поведения неявно.

**Критические тесты:**

```text
- deterministic при одинаковом seed;
- velocity всегда в [0, vmax(vehicle)];
- positions normalized on ring;
- нет duplicate occupied head cells after movement;
- stopped vehicle remains valid;
- single vehicle accelerates up to vmax;
- close front vehicle limits speed by gap;
- adjacent front vehicle forces speed 0;
- wraparound front vehicle handled correctly;
- multi-lane vehicles interact only within same lane;
- inactive vehicles ignored;
- density=1.0 full occupancy remains valid;
- controlled vehicles use vmax_controlled, uncontrolled use vmax_default;
- length ignored explicitly for now.
```

**Success criteria:**

```bash
pytest -q tests/test_snfs_longitudinal.py
pytest -q
```

---

## Task 7. Reference lane-change Kukida/S-NFS

**Status:** planned.

**Цель:** реализовать перестроение на соседнюю полосу до longitudinal update.

**Файлы:**

```text
src/snfs_traffic/core/lane_change_reference.py
tests/test_lane_change_reference.py
```

**Сделать:**

```text
- candidate target lanes: lane - 1, lane + 1;
- incentive criterion;
- safety criterion;
- stochastic p_lane_change;
- conflict resolution;
- commit lane changes;
- update changed_lane and last_lane_delta.
```

**MVP conflict resolution:**

```text
Если несколько машин хотят в одну target head cell, все конфликтующие остаются на месте.
Разрешить перестроение только если целевая head cell свободна и нет конкурента.
```

Не добавлять сложный приоритет раньше времени.

**Success criteria:**

```text
- только соседние полосы;
- нельзя выйти за пределы lane;
- нельзя попасть в занятую head cell;
- stochastic lane-change воспроизводим по seed;
- changed_lane и last_lane_delta обновляются;
- no duplicate head cells after lane-change commit.
```

---

## Task 8. Полный reference step

**Status:** planned.

**Цель:** собрать полный шаг симуляции в Python/NumPy без Numba.

**Файлы:**

```text
src/snfs_traffic/core/step_reference.py
tests/test_step_reference.py
```

**Целевой порядок:**

```text
1. build occupancy / lane order
2. compute front/back neighbors
3. decide lane-change intents
4. resolve lane-change conflicts
5. commit lane changes
6. rebuild occupancy / lane order
7. compute front/back neighbors
8. compute longitudinal velocity update
9. apply collision avoidance
10. commit positions
11. return events
```

**API sketch:**

```python
@dataclass(frozen=True, slots=True)
class StepEvents:
    changed_lane_count: int
    collision_count: int
    inserted_count: int
    removed_count: int


def step_reference(state, params, topology, rng, actions=None) -> tuple[TrafficState, StepEvents]:
    ...
```

`collision_count` должен быть 0 в норме, но полезен для assert/debug.

**Success criteria:**

```text
- 1000 шагов на random state без нарушения инвариантов;
- deterministic rollout с одинаковым seed;
- actions=None работает для HDV/AV internal rules;
- no duplicate head cells after every step.
```

---

## Task 9. Invariant suite и property-like rollout tests

**Status:** planned.

**Цель:** перед Numba жёстко зафиксировать корректность reference implementation.

**Файлы:**

```text
src/snfs_traffic/core/invariants.py
tests/test_invariants_random_rollouts.py
```

**Инварианты:**

```text
- no duplicate occupied head cells;
- lane in [0, num_lanes);
- pos in [0, road_length);
- vel in [0, vmax(vehicle)];
- lane changes only -1/0/+1;
- no teleport except periodic boundary movement;
- alive vehicles count stable for ring topology;
- indexing consistency after each step.
```

**Rollout matrix:**

```text
lanes: 1, 2, 4
road_length: 50, 100, 1500
density: 0.05, 0.2, 0.4, 0.7
steps: 100-1000
```

---

## Task 10. Numba indexing kernels

**Status:** planned.

**Цель:** начать ускорение с уже проверенного indexing layer, а не с полного step.

**Файлы:**

```text
src/snfs_traffic/core/kernels/indexing_numba.py
tests/test_indexing_numba_equivalence.py
benchmarks/bench_indexing.py
```

**Сделать Numba-версии:**

```python
build_occupancy_numba(...)
build_lane_order_numba(...)
compute_neighbors_numba(...)
```

**Требование:** результаты byte-for-byte совпадают с reference implementation на тестовых состояниях.

**Success criteria:**

```text
- equivalence tests;
- benchmark показывает ускорение или хотя бы отсутствие регресса;
- Numba не попадает в core dependency до этого task.
```

---

## Task 11. Numba full step kernel

**Status:** planned.

**Цель:** получить быстрый production step.

**Файлы:**

```text
src/snfs_traffic/core/kernels/step_numba.py
tests/test_step_numba_equivalence.py
benchmarks/bench_step.py
```

**Подход к RNG:**

На первом этапе не делать сложный internal Numba RNG. Лучше использовать pre-sampled random arrays:

```python
rand_slow_start[N]
rand_perspective[N]
rand_brake[N]
rand_lane_change[N]
```

Python/Env генерирует random arrays, Numba kernel их потребляет. Это проще, тестируемее и достаточно быстро.

**Success criteria:**

```text
- step_numba совпадает с step_reference при одинаковых random arrays;
- benchmark печатает steps/sec и vehicle_steps/sec;
- target case 4 lanes / 1500 cells / 2000 vehicles не является bottleneck.
```

---

## Task 12. SnfsSimulator facade

**Status:** planned.

**Цель:** закрыть сырые массивы удобным, но тонким API.

**Файлы:**

```text
src/snfs_traffic/core/simulator.py
tests/test_simulator_api.py
```

**API sketch:**

```python
class SnfsSimulator:
    def reset(...): ...
    def step(actions=None): ...
    def state_view(): ...
    def copy_state(): ...
    def restore_state(snapshot): ...
```

Параметр:

```python
backend: Literal["reference", "numba"]
```

**Важно:** `SnfsSimulator.step()` не строит observations и не считает rewards. Только physics/state transition.

---

## Task 13. Metrics и fundamental diagram

**Status:** planned.

**Цель:** иметь научную проверку ядра, а не только unit tests.

**Файлы:**

```text
src/snfs_traffic/metrics/basic.py
src/snfs_traffic/metrics/fundamental.py
examples/run_fundamental_diagram.py
tests/test_basic_metrics.py
```

**Метрики:**

```text
- density;
- mean speed;
- flow;
- lane-change rate;
- local density around vehicle;
- local flow around vehicle.
```

**Success criteria:**

```bash
python examples/run_fundamental_diagram.py --quick
```

Создаёт `.csv`/`.npz` с кривыми.

---

## Task 14. Snapshot `.npz` / restore

**Status:** planned.

**Цель:** заменить pickle object graph на компактный массивный snapshot.

**Файлы:**

```text
src/snfs_traffic/io/snapshot.py
tests/test_snapshot.py
```

**API sketch:**

```python
save_snapshot(path, state, params, rng_state, metadata)
load_snapshot(path) -> Snapshot
```

**Формат:**

```text
snapshot.npz:
  vehicle_id
  lane
  pos
  vel
  length
  veh_type
  behavior_id
  alive
  last_lane_delta
  changed_lane
  controlled
  params_json
  rng_state_json
  metadata_json
```

**Success criteria:**

```text
- save/load сохраняет rollout exactly;
- snapshot не зависит от Python class graph;
- можно использовать warm-start после stabilization.
```

---

## Task 15. Rule IDs и минимальная гибкость поведения

**Status:** planned.

**Цель:** сделать расширяемые правила без ООП в hot loop.

**Файлы:**

```text
src/snfs_traffic/rules/ids.py
src/snfs_traffic/rules/presets.py
tests/test_behavior_ids.py
```

Текущие scenario IDs уже есть, но task должен привести их к устойчивому публичному месту в `rules`:

```python
HDV_REVISED_SNFS = 1
AV_REVISED_SNFS = 2
RL_CONTROLLED = 3
BUS = 4
AGGRESSIVE_INTERCEPTOR = 5
```

**Важно:** не делать plugin system, исполняющую Python callbacks в hot loop.

Правильная модель:

```text
prototype mode: Python reference rules;
production mode: behavior_id + numba branch.
```

---

## Task 16. Controlled actions semantics

**Status:** planned.

**Цель:** правильно реализовать RL-действия как намерения, а не как прямую телепортацию состояния.

**Файлы:**

```text
src/snfs_traffic/core/actions.py
tests/test_controlled_actions.py
```

**API sketch:**

```python
@dataclass(frozen=True, slots=True)
class ActionBatch:
    vehicle_indices: np.ndarray
    accel: np.ndarray  # -1, 0, +1
    lane: np.ndarray   # -1, 0, +1
```

Dense mapping:

```python
action_accel[N]
action_lane[N]
```

**Семантика:**

```text
- accel modifies desired velocity by -1/0/+1;
- lane is attempted lane-change intent;
- collision/safety rules still apply;
- invalid lane action at boundary becomes no-op;
- uncontrolled vehicles ignore external actions.
```

---

## Task 17. Full-state observation API

**Status:** planned.

**Цель:** дать observation builders исчерпывающую информацию без прямого доступа к simulator internals.

**Файлы:**

```text
src/snfs_traffic/observations/state_view.py
tests/test_state_view.py
```

**API sketch:**

```python
@dataclass(frozen=True, slots=True)
class StateView:
    lane: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    veh_type: np.ndarray
    behavior_id: np.ndarray
    front_id: np.ndarray
    back_id: np.ndarray
    front_gap: np.ndarray
    back_gap: np.ndarray
    topology: RingTopology
```

`StateView` должен быть read-only или явно documented immutable during step.

---

## Task 18. Local grid observation builder

**Status:** planned.

**Цель:** воспроизвести local grid observation, но отдельно от env.

**Файлы:**

```text
src/snfs_traffic/observations/local_grid.py
tests/test_local_grid_observation.py
```

**API sketch:**

```python
LocalGridObservationBuilder(
    cells_back: int,
    cells_front: int,
    include_velocity: bool,
    include_lane_change_sign: bool,
)
```

Возврат:

```python
obs: np.ndarray  # shape=(num_lanes, width) for one ego
```

или batch для нескольких агентов.

**Success criteria:**

```text
- корректный wraparound на кольце;
- target/interceptor encoding optional;
- batch для нескольких agents;
- тесты на ручных конфигурациях.
```

---

## Task 19. AgentManager

**Status:** planned.

**Цель:** отделить физическую машину от RL-агента.

**Файлы:**

```text
src/snfs_traffic/envs/agent_manager.py
tests/test_agent_manager.py
```

**API sketch:**

```python
class AgentManager:
    def reset(sim) -> dict[str, int]: ...
    def active_agents(sim) -> dict[str, int]: ...
    def map_actions(action_dict) -> ActionBatch: ...
```

Режимы:

```text
- FixedControlledVehicles;
- AllControlledVehicles;
- RadiusAroundTargetAgents;
- GroupControllerAgent.
```

Ключевой контракт:

```text
agent_id -> vehicle_idx
```

а не subclass машины.

---

## Task 20. RewardFunction и TaskSpec

**Status:** planned.

**Цель:** вынести reward/termination из env.

**Файлы:**

```text
src/snfs_traffic/tasks/base.py
src/snfs_traffic/tasks/interception.py
tests/test_interception_task.py
```

**API sketch:**

```python
class TaskSpec:
    def reset(sim): ...
    def compute_rewards(prev_state, state, events, agent_manager): ...
    def compute_terminated(...): ...
    def compute_truncated(...): ...
```

Первый task:

```text
InterceptionConvergenceTask
```

Минимальные reward components:

```text
- reward for reducing distance to target;
- proximity bonus;
- bonus for occupying cell in front of target;
- penalty for overshooting target;
- lane-change penalty.
```

---

## Task 21. Gymnasium single-agent adapter

**Status:** planned.

**Цель:** сделать простой RL smoke-test.

**Файлы:**

```text
src/snfs_traffic/envs/gym_single.py
tests/test_gym_single_env.py
```

**API sketch:**

```python
class SingleAgentTrafficEnv(gym.Env):
    def reset(...): ...
    def step(action): ...
```

**Важно:** на этом task можно добавить `gymnasium` dependency. Раньше — не надо.

**Не делать:** RLlib, APPO, GNN.

---

## Task 22. RLlib MultiAgentEnv adapter

**Status:** planned.

**Цель:** тонкая RLlib-обвязка поверх готового simulator/task/observation stack.

**Файлы:**

```text
src/snfs_traffic/envs/rllib_multi.py
tests/test_rllib_multi_env.py
```

**API sketch:**

```python
class RllibTrafficMultiAgentEnv(MultiAgentEnv):
    def reset(...): ...
    def step(action_dict): ...
```

Возврат:

```python
obs: dict[agent_id, obs]
rewards: dict[agent_id, float]
terminateds: dict[agent_id, bool] + "__all__"
truncateds: dict[agent_id, bool] + "__all__"
infos: dict[agent_id, dict]
```

RLlib не должен влиять на архитектуру core.

---

## Task 23. Sync vectorized env внутри одного процесса

**Status:** planned.

**Цель:** простая поддержка vectorized env без преждевременной батчевой Numba-магии.

**Файлы:**

```text
src/snfs_traffic/envs/vector.py
tests/test_vector_env.py
benchmarks/bench_vector_env.py
```

**API sketch:**

```python
class SyncVectorTrafficEnv:
    def reset(): ...
    def step(list_of_action_dicts): ...
```

Первый вариант — честный loop по envs.

Не делать пока один огромный batched kernel для разных env: это усложнит layout из-за переменного числа машин/агентов.

---

## Task 24. Graph observation interface, без финальной GNN-логики

**Status:** planned.

**Цель:** заложить безопасный контракт для будущего PyTorch GNN encoder.

**Файлы:**

```text
src/snfs_traffic/observations/graph.py
tests/test_graph_observation_contract.py
```

**API sketch:**

```python
@dataclass(frozen=True, slots=True)
class GraphObservation:
    node_features: np.ndarray      # float32 [num_nodes, node_dim]
    edge_index: np.ndarray         # int64 [2, num_edges]
    edge_features: np.ndarray      # float32 [num_edges, edge_dim]
    node_vehicle_indices: np.ndarray
    controlled_node_mask: np.ndarray
```

Первый builder:

```text
- nodes = all vehicles within radius R from ego;
- edges = front/back same lane + adjacent lane nearest front/back.
```

Не пытаться на этом task выбрать “идеальный” GNN-граф.

---

## Task 25. Benchmark suite и performance gates

**Status:** planned.

**Цель:** не потерять производительность при дальнейшей разработке.

**Файлы:**

```text
benchmarks/bench_core.py
benchmarks/bench_observations.py
benchmarks/bench_rllib_env.py
```

Сценарии:

```text
small: 3 lanes, 500 cells, 350 vehicles
target: 4 lanes, 1500 cells, 2000 vehicles
dense: 4 lanes, 1500 cells, 3500 vehicles
```

Метрики:

```text
- core steps/sec;
- vehicle-steps/sec;
- local grid obs/sec;
- graph obs/sec;
- full env steps/sec.
```

Benchmark должен печатать JSON/CSV для сравнения commits.

---

## Task 26. Bus / length > 1 support

**Status:** planned, intentionally delayed.

**Цель:** аккуратно добавить длинные ТС, не ломая ядро.

Почему не раньше: `length > 1` усложняет occupancy, gaps, collision checks и lane-change safety. Если впихнуть это в первые tasks, MVP станет сильно сложнее и медленнее.

**Файлы:**

```text
src/snfs_traffic/core/lengths.py
tests/test_vehicle_lengths.py
```

**Сделать:**

```text
- зафиксировать, что pos означает: front cell или rear cell;
- occupancy validation для intervals;
- gap calculation с учетом length;
- lane-change safety с учетом length;
- migration path from head-cell-only tests.
```

**Success criteria:**

```text
- автобусы не пересекаются с легковыми;
- gap перед автобусом/за автобусом корректный;
- старые тесты length=1 не ломаются;
- head-cell-only assumptions заменены явно, а не неявно.
```

---

## Task 27. Open boundary

**Status:** planned, after stable ring core.

**Цель:** добавить inflow/outflow без fake vehicle hacks.

**Файлы:**

```text
src/snfs_traffic/topology/open_segment.py
src/snfs_traffic/scenarios/spawn.py
tests/test_open_boundary.py
```

**Сделать:**

```python
SpawnPolicy
RemovePolicy
```

Примеры:

```text
- stochastic inflow;
- fixed inflow interval;
- density-controlled inflow.
```

**Success criteria:**

```text
- машины появляются только если entry cells свободны;
- машины удаляются при выходе;
- density/flow metrics работают;
- open boundary не является особым типом машины.
```

---

## Task 28. Segment graph topology draft

**Status:** planned, after stable ring/open core.

**Цель:** подготовить основу для съездов/перекрестков, но не делать полноценную городскую симуляцию.

**Файлы:**

```text
src/snfs_traffic/topology/segment_graph.py
tests/test_segment_graph_topology.py
```

**API sketch:**

```python
RoadSegment
LaneConnector
TransferPolicy
```

MVP:

```text
- два последовательных сегмента;
- merge/split metadata;
- transfer vehicle from segment A to B.
```

Этот этап делать только после стабильного ring/open ядра.

---

# 6. Рекомендуемый порядок выполнения

Текущий актуальный порядок:

```text
DONE  1. Task 1  — bootstrap
DONE  2. Task 2  — state/params
DONE  3. Task 3  — ring topology
DONE  4. Task 4  — initializer
DONE  5. Task 5  — indexing
NEXT  6. Task 6  — longitudinal S-NFS reference
TODO  7. Task 7  — lane change reference
TODO  8. Task 8  — full reference step
TODO  9. Task 9  — invariants
TODO 10. Task 10 — numba indexing
TODO 11. Task 11 — numba full step
TODO 12. Task 12 — simulator facade
TODO 13. Task 13 — metrics/fundamental diagram
TODO 14. Task 14 — snapshots
TODO 15. Task 15 — behavior/rule ids
TODO 16. Task 16 — controlled actions
TODO 17. Task 17 — full state observation
TODO 18. Task 18 — local grid observation
TODO 19. Task 19 — agent manager
TODO 20. Task 20 — task/reward
TODO 21. Task 21 — gym single-agent
TODO 22. Task 22 — RLlib multi-agent
TODO 23. Task 23 — vector env
TODO 24. Task 24 — graph observation contract
TODO 25. Task 25 — benchmark suite
TODO 26. Task 26 — buses / length > 1
TODO 27. Task 27 — open boundary
TODO 28. Task 28 — segment graph / ramps / intersections
```

Нельзя сейчас перескакивать сразу к env/RLlib/GNN. Без reference S-NFS step все эти слои будут строиться на пустоте.

---

# 7. MVP v0.1 definition

MVP v0.1 готов, когда есть:

```text
- ring road;
- 4 lanes / 1500 cells / 2000 vehicles target scenario;
- HDV/AV/RL-controlled vehicles;
- Revised S-NFS longitudinal update;
- stochastic lane change;
- full reference step;
- invariant rollout suite;
- Numba backend;
- deterministic seeds;
- local grid observation;
- single-agent Gym env;
- multi-agent RLlib env;
- fundamental diagram example;
- benchmark core/env throughput.
```

Не входит в MVP v0.1:

```text
- перекрестки;
- open boundary;
- автобусы как полноценные length>1 occupying bodies;
- сложный graph observation;
- custom APPO;
- красивая визуализация.
```

Это не недостаток. Это правильная отсечка.

---

# 8. Главные риски

## Риск 1. Слишком рано делать универсальную topology

Если сразу делать перекрестки, съезды, traffic lights, multi-segment kernels — ядро станет сложным до того, как будет проверено. Поэтому:

```text
interface discipline — да;
сложная topology implementation — позже.
```

## Риск 2. Слишком красивая rule plugin system

Python callbacks в hot loop убьют производительность. Гибкость должна быть двухуровневой:

```text
prototype mode: Python reference rules;
production mode: behavior_id + Numba branch.
```

## Риск 3. Observation станет bottleneck

Даже если S-NFS step быстрый, graph/local observations могут съесть всё. Observation builders надо benchmark-ить отдельно.

## Риск 4. RLlib начнет диктовать архитектуру

Нельзя. RLlib — адаптер. Simulator, task logic, observations и rewards должны жить отдельно.

## Риск 5. Раннее length-aware моделирование сломает MVP

`length > 1` кажется маленькой фичей, но она меняет occupancy, gaps, collision avoidance и lane-change safety. До Task 26 buses должны оставаться metadata-only через `state.length`.

## Риск 6. Непроверенная reference physics перед Numba

Numba надо писать только после того, как reference behavior закрыт тестами. Иначе Numba kernel законсервирует ошибки.

---

# 9. Ближайший практический шаг

Следующий Codex task должен быть:

```text
Task 6 — implement reference longitudinal Revised S-NFS without lane changes.
```

Он должен получить весь контекст:

```text
- current TrafficState/SimulationParams schema;
- current RingTopology API;
- current scenario initializer;
- current indexing API and head-cell-only semantics;
- no lane change yet;
- no length-aware occupancy yet;
- no Numba yet;
- tests must prove deterministic, safe, same-lane longitudinal motion.
```