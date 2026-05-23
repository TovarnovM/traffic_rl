
# 0. Главная декомпозиция нового проекта

Я бы разделил проект на 6 независимых слоев:

```text
snfs_traffic/
  core/           # быстрый S-NFS state transition engine
  rules/          # HDV/AV/RL/infrastructure rule specs
  topology/       # ring/open/segment graph/intersections later
  observations/   # full-state, local grid, graph observation builders
  envs/           # Gymnasium/RLlib/PettingZoo adapters
  experiments/    # конкретные задачи: interception, traffic light, flow control
```

Важное правило:

```text
core не импортирует gymnasium, ray, torch, rllib, matplotlib, pandas.
```

`core` должен быть маленьким, быстрым и проверяемым.

---

# 1. Целевой runtime-контракт

## 1.1. State

Базово:

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

Scratch/state для индексации:

```python
occupancy:       int32[num_lanes, road_length]  # -1 или vehicle index
lane_counts:     int32[num_lanes]
lane_order:      int32[num_lanes, max_vehicles_per_lane]
lane_rank:       int32[N]

front_id:        int32[N]
back_id:         int32[N]
front_gap:       int32[N]
back_gap:        int32[N]
```

## 1.2. Step API

Минимальный API:

```python
sim.reset(seed, scenario_config)
state_view = sim.state_view()
sim.step(actions)
```

Где `actions` — необязательный массив намерений:

```python
action_accel: int8[N]  # -1, 0, +1, or 0 for uncontrolled
action_lane:  int8[N]  # -1, 0, +1, or 0 for uncontrolled
```

Не все машины обязаны быть RL-controlled. Для HDV/AV действия генерируются внутренним rule kernel.

## 1.3. Порядок фаз

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

Это прямо соответствует твоему требованию: сначала перестроение на соседнюю полосу, потом независимое обновление состояний, скоростей и позиций.

---

# 2. Важное ограничение по топологии

Ты хочешь теоретическую возможность перекрестков и съездов. Это правильно, но если сразу делать полноценный road graph, проект утонет.

Мой компромисс:

```text
MVP topology = linear lane segment + periodic ring boundary.
Архитектурно topology = набор LaneSegment + boundary/transfer interface.
Intersections/ramps = будущие SegmentTransferPolicy, но не в первом ядре.
```

То есть в коде сразу не надо пришивать `road_length` как единственный возможный мир. Но первые kernels должны работать на одном сегменте:

```python
SegmentTopology(
    num_lanes=4,
    length=1500,
    boundary="periodic",
)
```

Позже можно добавить:

```text
RoadGraphTopology:
  segments
  lane connectors
  merge/split transfer rules
  traffic light controllers
```

Но это отдельный этап после стабильного S-NFS ядра.

---

# 3. План этапов / Codex tasks

## Task 1. Bootstrap нового проекта и базовая структура

**Цель:** создать чистый проект без переноса старой объектной архитектуры.

**Что сделать:**

```text
- pyproject.toml
- src/snfs_traffic/
- tests/
- benchmarks/
- examples/
- README.md
```

Минимальные зависимости:

```text
numpy
numba
pytest
gymnasium      # можно не использовать сразу, но допустимо
```

Torch/RLlib пока не добавлять в core-зависимости.

**Ожидаемый результат:**

```bash
pytest -q
python -c "import snfs_traffic"
```

проходят.

**Важно:** не переносить старые классы `Vehicle`, `RoadLane`, `TrafficModel` как основу.

---

## Task 2. Конфиги и типизированные структуры состояния

**Цель:** зафиксировать state schema и параметры модели.

**Файлы:**

```text
src/snfs_traffic/core/state.py
src/snfs_traffic/core/params.py
src/snfs_traffic/core/types.py
tests/test_state_schema.py
```

**Сделать:**

```python
@dataclass
class SimulationParams:
    num_lanes: int
    road_length: int
    vmax_default: int
    vmax_controlled: int
    G: int
    q: float
    r: float
    S: int
    P1: float
    P2: float
    P3: float
    P4: float
    p_lane_change: float
```

```python
@dataclass
class TrafficState:
    lane: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    length: np.ndarray
    veh_type: np.ndarray
    behavior_id: np.ndarray
    alive: np.ndarray
    last_lane_delta: np.ndarray
    changed_lane: np.ndarray
```

**Проверки:**

* dtype корректный;
* shape одинаковый;
* значения lane/pos/vel в допустимых диапазонах;
* state можно скопировать без shared mutable багов.

**Success criteria:**

```bash
pytest -q tests/test_state_schema.py
```

---

## Task 3. Reference topology: ring segment

**Цель:** сделать минимальную топологию без overengineering.

**Файлы:**

```text
src/snfs_traffic/topology/base.py
src/snfs_traffic/topology/ring.py
tests/test_ring_topology.py
```

**Сделать:**

```python
class RingTopology:
    num_lanes: int
    length: int

    def normalize_pos(pos): ...
    def forward_distance(from_pos, to_pos): ...
    def signed_delta(from_pos, to_pos): ...
```

**Почему это важно:** все расстояния, especially для target/interceptor и graph edges, должны считаться через topology, а не вручную через `% road_length` по всему коду.

**Success criteria:**

* distance across boundary работает;
* signed distance работает;
* нет прямой логики кольца в env/observations.

---

## Task 4. Scenario initializer: равномерная и random плотность

**Цель:** уметь создавать корректное начальное состояние.

**Файлы:**

```text
src/snfs_traffic/scenarios/init.py
tests/test_init_scenarios.py
```

**Сделать:**

```python
make_uniform_random_state(
    num_lanes: int,
    road_length: int,
    density: float,
    seed: int,
    vehicle_mix: VehicleMix,
)
```

Поддержать:

```text
- HDV
- AV
- controlled AV
- buses пока можно только length field, без сложной логики
```

**Инварианты:**

* нет двух машин в одной клетке;
* `N <= num_lanes * road_length`;
* позиции валидны;
* seed дает воспроизводимый state.

**Важно:** автобусы пока лучше не вводить в симуляцию движения как полноценные length>1, но поле `length` предусмотреть. Полная поддержка длинных ТС — отдельный task.

---

## Task 5. Occupancy и lane order reference implementation

**Цель:** построить быстрый и надежный способ находить соседей.

**Файлы:**

```text
src/snfs_traffic/core/indexing.py
tests/test_indexing.py
```

**Сделать:**

```python
build_occupancy(state, params) -> occupancy
build_lane_order(occupancy) -> lane_order, lane_counts, lane_rank
compute_neighbors(state, lane_order, lane_counts, lane_rank, topology) -> front/back/gaps
```

**Решение:** для масштаба `4 lanes * 1500 cells = 6000 cells` сканирование occupancy по клеткам дешевое. Это лучше, чем сортировка `N log N` на каждом шаге.

**Инварианты:**

* для каждой машины корректный front/back на кольце;
* gaps корректны через boundary;
* при одной машине на lane front/back может быть `-1` или self-handled policy — нужно явно зафиксировать.

**Success criteria:**

* тесты на простые ручные конфигурации;
* тесты boundary wraparound;
* тесты нескольких lanes.

---

## Task 6. Reference longitudinal Revised S-NFS без перестроений

**Цель:** реализовать движение по S-NFS для одной полосы/нескольких полос без lane change.

**Файлы:**

```text
src/snfs_traffic/core/step_reference.py
tests/test_snfs_longitudinal.py
```

**Сделать фазу velocity update:**

```text
- acceleration
- slow-to-start
- perspective/anticipation
- random braking
- collision avoidance
- movement
```

Параметры взять из статьи как default preset:

```text
G = 15
q = 0.99
r = 0.99
S = 2
P1 = 0.999
P2 = 0.99
P3 = 0.98
P4 = 0.01
```

В статье эти параметры использовались для Revised S-NFS экспериментов, а обычные машины имели `vmax=5`, interceptor/RL agents — `vmax=6`. 

**Success criteria:**

* deterministic при seed;
* velocity в `[0, vmax]`;
* нет пересечений;
* простые ручные cases проходят.

---

## Task 7. Reference lane-change Kukida/S-NFS

**Цель:** реализовать перестроение на соседнюю полосу до longitudinal update.

**Файлы:**

```text
src/snfs_traffic/core/lane_change_reference.py
tests/test_lane_change_reference.py
```

**Сделать:**

```text
- candidate target lanes: lane - 1, lane + 1
- incentive criterion
- safety criterion
- stochastic p_lane_change
- conflict resolution
- commit lane changes
```

По статье lane change использует incentive criterion, safety criterion и вероятность `P_CL = 0.5`; схема на странице 5 показывает сравнение forward/back gaps в текущей и новой полосе. 

**Критически важно зафиксировать conflict resolution:**

Когда две машины хотят в одну клетку target lane:

```text
вариант MVP: разрешить только если целевая клетка свободна и нет конкурента;
иначе все конфликтующие остаются в своих полосах.
```

Не надо сейчас делать сложный приоритет. Это можно добавить позже.

**Success criteria:**

* только соседние полосы;
* нельзя выйти за пределы lane;
* нельзя попасть в занятую клетку;
* stochastic lane-change воспроизводим по seed;
* `changed_lane` и `last_lane_delta` обновляются.

---

## Task 8. Полный reference `step_reference`

**Цель:** собрать полный шаг симуляции в Python/NumPy без Numba.

**Файлы:**

```text
src/snfs_traffic/core/step_reference.py
tests/test_step_reference.py
```

**Сделать:**

```python
def step_reference(state, params, rng, actions=None, scratch=None) -> StepEvents:
    ...
```

**StepEvents:**

```python
@dataclass
class StepEvents:
    changed_lane_count: int
    collision_count: int  # должен быть 0, но удобно для assert/debug
    inserted_count: int
    removed_count: int
```

**Success criteria:**

* 1000 шагов на случайном state без нарушения инвариантов;
* deterministic rollout с одинаковым seed;
* при `actions=None` все машины HDV/AV используют internal rules.

---

## Task 9. Invariant suite и property-like tests

**Цель:** перед Numba жестко зафиксировать правильность.

**Файлы:**

```text
src/snfs_traffic/core/invariants.py
tests/test_invariants_random_rollouts.py
```

**Инварианты:**

```text
- no duplicate occupied cells
- lane in [0, num_lanes)
- pos in [0, road_length)
- vel in [0, vmax(vehicle)]
- lane changes only -1/0/+1
- no teleport except periodic boundary movement
- alive vehicles count stable for ring topology
```

**Тесты:**

```text
- densities: 0.05, 0.2, 0.4, 0.7
- lanes: 1, 2, 4
- road_length: 50, 100, 1500
- rollout: 100-1000 steps
```

**Success criteria:**

```bash
pytest -q tests/test_invariants_random_rollouts.py
```

---

## Task 10. Numba indexing kernels

**Цель:** начать ускорение не с полного step, а с индексации.

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

**Требование:** результат byte-for-byte совпадает с reference на тестовых состояниях.

**Success criteria:**

* equivalence tests;
* benchmark показывает ускорение или хотя бы отсутствие регресса;
* Numba cache включен.

---

## Task 11. Numba full step kernel

**Цель:** получить быстрый production step.

**Файлы:**

```text
src/snfs_traffic/core/kernels/step_numba.py
tests/test_step_numba_equivalence.py
benchmarks/bench_step.py
```

**Сделать:**

```python
step_numba(
    lane, pos, vel, length, veh_type, behavior_id,
    alive, last_lane_delta, changed_lane,
    params_arrays,
    action_accel,
    action_lane,
    rng_state,
    scratch,
)
```

**Важно:** не пытаться сразу сделать идеальный RNG. На первом этапе допустим deterministic pre-sampled random arrays:

```python
rand_slow_start[N]
rand_perspective[N]
rand_brake[N]
rand_lane_change[N]
```

То есть Python/Env генерирует случайные массивы, Numba kernel их потребляет. Это проще, тестируемее и часто быстрее.

**Success criteria:**

* `step_numba` совпадает с `step_reference` при одинаковых random arrays;
* 4 lanes / 1500 cells / 2000 vehicles не является bottleneck;
* benchmark печатает `steps/sec` и `vehicle_steps/sec`.

---

## Task 12. `SnfsSimulator` facade

**Цель:** закрыть сырые массивы удобным, но тонким API.

**Файлы:**

```text
src/snfs_traffic/core/simulator.py
tests/test_simulator_api.py
```

**Сделать:**

```python
class SnfsSimulator:
    def reset(...)
    def step(actions=None)
    def state_view()
    def copy_state()
    def restore_state(snapshot)
```

Параметр:

```python
backend: Literal["reference", "numba"]
```

**Важно:** `SnfsSimulator.step()` не должен строить observation. Только физика.

**Success criteria:**

* reference и numba backend взаимозаменяемы;
* одинаковый seed дает одинаковый rollout;
* можно прогнать stabilization phase 1500 steps.

---

## Task 13. Metrics и fundamental diagram

**Цель:** сразу иметь научную проверку ядра.

**Файлы:**

```text
src/snfs_traffic/metrics/basic.py
src/snfs_traffic/metrics/fundamental.py
examples/run_fundamental_diagram.py
tests/test_basic_metrics.py
```

**Метрики:**

```text
- density
- mean speed
- flow
- lane change rate
- local density around vehicle
- local flow around vehicle
```

В статье для анализа использовались local density, local flux и lane-changing activity по observation window; эти метрики стоит поддержать как first-class validation tools, а не как код внутри эксперимента. 

**Success criteria:**

```bash
python examples/run_fundamental_diagram.py --quick
```

создает `.csv`/`.npz` с кривыми.

---

## Task 14. Snapshot `.npz` / restore

**Цель:** заменить pickle объектного графа на компактный массивный snapshot.

**Файлы:**

```text
src/snfs_traffic/io/snapshot.py
tests/test_snapshot.py
```

**Сделать:**

```python
save_snapshot(path, state, params, rng_state, metadata)
load_snapshot(path) -> Snapshot
```

Формат:

```text
snapshot.npz:
  lane
  pos
  vel
  length
  veh_type
  behavior_id
  alive
  last_lane_delta
  changed_lane
  metadata_json
```

**Success criteria:**

* save/load сохраняет rollout exactly;
* snapshot не зависит от Python class graph;
* можно использовать warm-start после stabilization.

---

## Task 15. Rule IDs и минимальная гибкость поведения

**Цель:** сделать расширяемые правила без ООП в hot loop.

**Файлы:**

```text
src/snfs_traffic/rules/ids.py
src/snfs_traffic/rules/presets.py
tests/test_behavior_ids.py
```

**Rule IDs:**

```python
HDV_REVISED_SNFS = 1
AV_REVISED_SNFS = 2
RL_CONTROLLED = 3
AGGRESSIVE_INTERCEPTOR = 4
BUS = 5
```

**Важно:** на этом этапе не делать plugin-систему, которая исполняет Python callback в hot loop.

Правильная модель:

```text
behavior_id выбирает ветку внутри numba kernel.
новое production-правило = новый behavior_id + тесты + kernel branch.
экспериментальное Python-правило = только reference/prototype backend.
```

**Success criteria:**

* HDV и AV могут иметь разные `vmax`, `p_lane_change`, safety behavior;
* RL_CONTROLLED использует action arrays;
* uncontrolled машины игнорируют external actions.

---

## Task 16. Controlled actions semantics

**Цель:** правильно реализовать RL-действия как намерения.

**Файлы:**

```text
src/snfs_traffic/core/actions.py
tests/test_controlled_actions.py
```

**Сделать:**

```python
@dataclass
class ActionBatch:
    vehicle_indices: np.ndarray
    accel: np.ndarray  # -1,0,+1
    lane: np.ndarray   # -1,0,+1
```

Маппинг в dense arrays:

```python
action_accel[N]
action_lane[N]
```

Семантика:

```text
- accel modifies desired velocity by -1/0/+1
- lane is attempted lane-change intent
- collision/safety rules still apply
- invalid lane action at boundary becomes no-op
```

Это соответствует статье: действие `[a1, a2]`, где оба компонента дискретны `{-1,0,1}`, является намерением, но после него всё равно применяется collision avoidance и CA movement. 

**Success criteria:**

* controlled vehicle can accelerate/decelerate;
* unsafe lane action не приводит к collision;
* no-op работает.

---

## Task 17. Full-state observation API

**Цель:** дать среде «исчерпывающую информацию», но не заставлять каждого агента получать огромный объект.

**Файлы:**

```text
src/snfs_traffic/observations/state_view.py
tests/test_state_view.py
```

**Сделать:**

```python
class StateView:
    lane
    pos
    vel
    veh_type
    behavior_id
    front_id
    back_id
    front_gap
    back_gap
    topology
```

**Важно:** `StateView` должен быть read-only или documented immutable during step.

**Решение:** env предоставляет полный `StateView`, а конкретный `ObservationBuilder` сам выбирает фильтр.

**Success criteria:**

* StateView строится дешево;
* не копирует большие массивы без необходимости;
* observation builders используют StateView, а не simulator internals.

---

## Task 18. Local grid observation builder

**Цель:** воспроизвести старое observation-поведение, но правильно отделить его от env.

**Файлы:**

```text
src/snfs_traffic/observations/local_grid.py
tests/test_local_grid_observation.py
```

**Сделать:**

```python
LocalGridObservationBuilder(
    cells_back: int,
    cells_front: int,
    include_velocity: bool,
    include_lane_change_sign: bool,
)
```

Возвращать:

```python
obs[agent, lanes, width]
```

Или для одного агента:

```python
obs: np.ndarray[int16, shape=(num_lanes, width)]
```

В статье использовалось окно вокруг interceptor, матрица `l x w`, где `w = wf + wb + 1`, а значения кодировали пустую клетку, скорость и факт перестроения. 

**Success criteria:**

* корректный wraparound на кольце;
* target/interceptor encoding optional;
* batch для нескольких агентов;
* тесты на ручных конфигурациях.

---

## Task 19. AgentManager

**Цель:** отделить физическую машину от RL-агента.

**Файлы:**

```text
src/snfs_traffic/envs/agent_manager.py
tests/test_agent_manager.py
```

**Сделать:**

```python
class AgentManager:
    def reset(sim) -> dict[agent_id, vehicle_idx]
    def active_agents(sim) -> dict[agent_id, vehicle_idx]
    def map_actions(action_dict) -> ActionBatch
```

Режимы:

```text
- FixedControlledVehicles
- AllControlledVehicles
- RadiusAroundTargetAgents
- GroupControllerAgent
```

**Важно:** `agent_id -> vehicle_idx`, а не subclass машины.

**Success criteria:**

* dynamic active agents;
* stable agent ids within episode;
* terminated agents корректно исчезают из action_dict expectation;
* поддержка parameter sharing естественная.

---

## Task 20. RewardFunction и TaskSpec

**Цель:** вынести reward/termination из env.

**Файлы:**

```text
src/snfs_traffic/tasks/base.py
src/snfs_traffic/tasks/interception.py
tests/test_interception_task.py
```

**Сделать:**

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

Минимальные reward components из статьи:

```text
- reward for reducing distance to target
- proximity bonus
- bonus for occupying cell in front of target
- penalty for overshooting target
- lane-change penalty
```

В статье reward был суммой таких компонентов для interception/convergence задачи. 

**Success criteria:**

* reward воспроизводим;
* task не зависит от RLlib;
* можно заменить reward без изменения simulator.

---

## Task 21. Gymnasium single-agent adapter

**Цель:** сделать простой RL smoke-test.

**Файлы:**

```text
src/snfs_traffic/envs/gym_single.py
tests/test_gym_single_env.py
```

**Сделать:**

```python
class SingleAgentTrafficEnv(gym.Env):
    reset()
    step(action)
```

**Не делать:** RLlib, APPO, GNN.

**Success criteria:**

* env проходит `gymnasium.utils.env_checker`;
* random policy rollout 100 episodes без падений;
* simulator backend можно выбрать `reference/numba`.

---

## Task 22. RLlib MultiAgentEnv adapter

**Цель:** тонкая RLlib-обвязка поверх уже готовых компонентов.

**Файлы:**

```text
src/snfs_traffic/envs/rllib_multi.py
tests/test_rllib_multi_env.py
```

**Сделать:**

```python
class RllibTrafficMultiAgentEnv(MultiAgentEnv):
    def reset(...)
    def step(action_dict)
```

Возвращать:

```python
obs: dict[agent_id, obs]
rewards: dict[agent_id, float]
terminateds: dict[agent_id, bool] + "__all__"
truncateds: dict[agent_id, bool] + "__all__"
infos: dict[agent_id, dict]
```

**Success criteria:**

* random multi-agent rollout;
* dynamic agents работают;
* parameter sharing не требует особого кода;
* нет импорта RLlib в core.

---

## Task 23. Sync vectorized env внутри одного процесса

**Цель:** простая поддержка vectorized env без преждевременной батчевой Numba-магии.

**Файлы:**

```text
src/snfs_traffic/envs/vector.py
tests/test_vector_env.py
benchmarks/bench_vector_env.py
```

**Сделать:**

```python
class SyncVectorTrafficEnv:
    def reset()
    def step(list_of_action_dicts)
```

Первый вариант просто loop по envs. Это честно и полезно.

**Не делать пока:** один огромный batched kernel для разных env. Это усложнит state layout из-за переменного числа машин/агентов.

**Success criteria:**

* `num_envs=8/16/32`;
* reproducible seeds;
* benchmark throughput.

---

## Task 24. Graph observation interface, без финальной GNN-логики

**Цель:** заложить безопасный контракт для будущего PyTorch GNN encoder.

**Файлы:**

```text
src/snfs_traffic/observations/graph.py
tests/test_graph_observation_contract.py
```

**Сделать dataclass:**

```python
@dataclass
class GraphObservation:
    node_features: np.ndarray      # float32 [num_nodes, node_dim]
    edge_index: np.ndarray         # int64 [2, num_edges]
    edge_features: np.ndarray      # float32 [num_edges, edge_dim]
    node_vehicle_indices: np.ndarray
    controlled_node_mask: np.ndarray
```

Первый builder может быть простым:

```text
- nodes = all vehicles within radius R from ego
- edges = front/back same lane + adjacent lane nearest front/back
```

**Важно:** этот task не должен пытаться подобрать «идеальный» GNN-граф. Только надежный контракт и тестируемая геометрия.

**Success criteria:**

* edge distances корректны на кольце;
* нет self-edge, если явно не включен;
* node_vehicle_indices правильно маппятся назад в simulator state;
* output можно напрямую конвертировать в torch tensors.

---

## Task 25. Benchmark suite и performance gates

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
- core steps/sec
- vehicle-steps/sec
- local grid obs/sec
- graph obs/sec
- full env steps/sec
```

**Success criteria:** benchmark печатает JSON/CSV, чтобы можно было сравнивать commits.

---

## Task 26. Bus / length > 1 support

**Цель:** аккуратно добавить длинные ТС, не ломая ядро.

**Почему не раньше:** length>1 усложняет occupancy, gaps, collision checks и lane change safety. Если впихнуть это в первые tasks, можно сильно замедлить MVP.

**Файлы:**

```text
src/snfs_traffic/core/lengths.py
tests/test_vehicle_lengths.py
```

**Сделать:**

```text
- определить, что pos означает: front cell или rear cell
- occupancy validation для intervals
- gap calculation с учетом length
- lane-change safety с учетом length
```

**Success criteria:**

* автобусы не пересекаются с легковыми;
* gap перед автобусом/за автобусом корректный;
* старые тесты length=1 не ломаются.

---

## Task 27. Open boundary

**Цель:** добавить inflow/outflow без FakeHeadVehicle.

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

Например:

```text
- stochastic inflow
- fixed inflow interval
- density-controlled inflow
```

**Важно:** open boundary не должна быть особым типом машины.

**Success criteria:**

* машины появляются только если entry cells свободны;
* машины удаляются при выходе;
* density/flow metrics работают.

---

## Task 28. Segment graph topology draft

**Цель:** подготовить основу для съездов/перекрестков, но не делать полноценную городскую симуляцию.

**Файлы:**

```text
src/snfs_traffic/topology/segment_graph.py
tests/test_segment_graph_topology.py
```

**Сделать:**

```python
RoadSegment
LaneConnector
TransferPolicy
```

MVP:

```text
- два последовательных сегмента
- merge/split metadata
- transfer vehicle from segment A to B
```

**Важно:** этот этап делать только после стабильного ring/open ядра. Иначе он преждевременно усложнит hot loop.

---

# 4. Рекомендуемый порядок выполнения

Я бы шел строго так:

```text
1. Task 1  — bootstrap
2. Task 2  — state/params
3. Task 3  — ring topology
4. Task 4  — initializer
5. Task 5  — indexing
6. Task 6  — longitudinal S-NFS reference
7. Task 7  — lane change reference
8. Task 8  — full reference step
9. Task 9  — invariants
10. Task 10 — numba indexing
11. Task 11 — numba full step
12. Task 12 — simulator facade
13. Task 13 — metrics/fundamental diagram
14. Task 14 — snapshots
15. Task 15 — behavior/rule ids
16. Task 16 — controlled actions
17. Task 17 — full state observation
18. Task 18 — local grid observation
19. Task 19 — agent manager
20. Task 20 — task/reward
21. Task 21 — gym single-agent
22. Task 22 — RLlib multi-agent
23. Task 23 — vector env
24. Task 24 — graph observation contract
25. Task 25 — benchmark suite
```

А уже потом:

```text
26. buses / length > 1
27. open boundary
28. segment graph / ramps / intersections
```

---

# 5. Что считаю MVP v0.1

MVP v0.1 готов, когда есть:

```text
- ring road
- 4 lanes / 1500 cells / 2000 vehicles
- HDV/AV/RL-controlled vehicles
- revised S-NFS longitudinal update
- stochastic lane change
- numba backend
- deterministic seed
- local grid observation
- single-agent Gym env
- multi-agent RLlib env
- fundamental diagram example
- benchmark core/env throughput
```

Без:

```text
- перекрестков
- open boundary
- автобусов length>1
- сложного graph observation
- custom APPO
- красивой визуализации
```

Это не недостаток. Это правильная отсечка.

---

# 6. Главные риски

## Риск 1. Слишком рано делать универсальную topology

Если сразу делать перекрестки, съезды, traffic lights, multi-segment kernels — ядро станет сложным до того, как будет проверено. Поэтому: **интерфейс под topology — да, реализация сложной topology — позже**.

## Риск 2. Слишком красивая rule plugin system

Python callbacks в hot loop убьют производительность. Гибкость должна быть двухуровневой:

```text
prototype mode: Python reference rules
production mode: behavior_id + numba branch
```

## Риск 3. Observation станет bottleneck

Даже если S-NFS step быстрый, graph/local observations могут съесть всё. Поэтому observation builders надо benchmark-ить отдельно.

## Риск 4. RLlib начнет диктовать архитектуру

Нельзя. RLlib — адаптер. Simulator и task logic должны жить отдельно.

---
