# PLAN.md — Revised S-NFS Traffic Simulator Roadmap

Current state: after Tasks 1–23, including Task 22+23 follow-up.

The project is a clean, array-based Revised S-NFS-style traffic simulator core intended for future reinforcement-learning experiments. It currently has a reliable reference simulator, a supported optional optimized backend, backend selection, correctness tests, RNG-parity checks, runtime invariant checks, and benchmark/report infrastructure.

Important semantic note: the current longitudinal phase is intentionally a
simplified S-NFS-style variant (reference-defined behavior), not paper-exact
Revised S-NFS. Parameters `q` and `P1` are retained in `SimulationParams` for
compatibility but are currently unused by longitudinal dynamics.

The project is ready to begin RL environment preparation, but not ready for RL training yet.

---

## 1. Current status snapshot

### 1.1 What is stable now


- Standalone rollout visualization tooling (`snfs_traffic.visualization`) is implemented as an optional side-feature and intentionally decoupled from simulator stepping semantics.
- Core simulator state transition is implemented.
- `step_reference(...)` is the authoritative semantic oracle.
- `ReferenceBackend` is available and remains simple.
- `OptimizedBackend` is a supported optional backend when required Numba kernels are available.
- Backend selection supports `"reference"`, `"optimized"`, and `"auto"`.
- Numba remains optional.
- Optimized backend preserves reference state equivalence and RNG next-draw parity in focused tests and benchmark checks.
- Runtime invariant validation is available.
- Benchmark scripts and reports exist for indexing and optimized full-step performance.

### 1.2 What is not implemented yet

- Simulator facade.
- Controlled RL action semantics.
- Action application contract.
- Observation schema.
- Reward schema.
- Episode lifecycle.
- RL metrics/info schema.
- Gymnasium wrapper.
- RLlib wrapper.
- PettingZoo/multi-agent wrapper.

### 1.3 Current readiness estimate

| Target | Readiness | Notes |
|---|---:|---|
| Continue simulator-core development | High | Reference/optimized architecture is established. |
| Begin RL environment preparation | 75–80% | Correct next step is facade + contracts. |
| Start actual RL training experiments | 40–50% | RL actions/observations/rewards/env wrappers are missing. |

The blocker for RL training is no longer low-level simulator correctness. The blocker is the missing RL-facing API design.

---

## 2. Core project principles

### 2.1 Reference-first correctness

`step_reference(...)` is the authoritative semantic oracle.

All optimized implementations must match it exactly unless a future task explicitly changes model semantics.

Correctness includes:

- state field equivalence;
- runtime invariants;
- RNG next-draw parity;
- deterministic behavior under fixed seed;
- consistent behavior across supported backends.

### 2.2 Optional optimization

Optimized backends and kernels are optional.

Numba must remain an optional dependency.

The base package must import and run without Numba.

`ReferenceBackend` must never depend on Numba.

### 2.3 No hidden RL semantics inside core kernels

The current simulator core should remain independent from RL-specific concerns.

Do not add directly to low-level kernels:

- rewards;
- observations;
- Gymnasium API;
- RLlib API;
- policy objects;
- learning logic.

RL-facing behavior should be layered above the simulator core through a clean simulator facade and environment wrappers.

### 2.4 Surgical development

Prefer small, verifiable tasks.

Do not refactor unrelated areas.

Do not rewrite physics while working on wrappers.

Do not rewrite wrappers while working on backend performance.

Keep reference/debug helpers intact unless removal is explicitly justified and covered by tests.

---

## 3. Completed work by layer

### 3.1 Project bootstrap

Completed:

- package skeleton;
- source layout;
- tests layout;
- basic import checks;
- project metadata;
- optional extras for Numba;
- development quick checks.

### 3.2 Core state and parameters

Completed:

- `TrafficState`;
- `SimulationParams`;
- vehicle arrays;
- controlled-vehicle flag;
- lane-change flags;
- runtime invariant helper;
- periodic `RingTopology`.

Current limitation:

- controlled vehicles are marked but do not yet receive external RL actions.

### 3.3 Scenario initialization

Completed:

- reproducible uniform-random scenario generation;
- density-controlled initialization;
- support for vehicle mix metadata;
- tests for deterministic initialization.

Current limitation:

- no scenario curriculum/config facade yet;
- no RL episode reset API yet.

### 3.4 Reference indexing

Completed:

- head-cell occupancy;
- lane order;
- lane counts;
- lane rank;
- front/back neighbor IDs;
- front/back gaps;
- periodic wraparound semantics;
- public validated wrappers;
- pure-array indexing kernels;
- equivalence tests.

Current limitation:

- head-cell-only occupancy;
- no length-aware geometry;
- no body-cell collision geometry.

### 3.5 Lane-change phase

Completed:

- reference lane-change proposal phase;
- stochastic lane-change attempt logic;
- target-cell conflict resolution;
- accepted lane-change application;
- lane-change flags update;
- optional Numba proposal helper;
- focused lane-change Numba tests.

Current limitation:

- no external controlled action override;
- lane-change proposal collection is now the primary optimized-backend bottleneck.

### 3.6 Longitudinal phase

Completed:

- reference longitudinal update;
- optional Numba longitudinal helper;
- position advance;
- tests against reference behavior.

Current limitation:

- no external controlled acceleration/speed action yet;
- no future model-family variants yet.

### 3.7 Full reference step

Completed:

- full `step_reference(...)`;
- lane-change phase followed by longitudinal phase;
- deterministic RNG usage;
- reference rollout tests;
- runtime invariant validation after rollouts.

### 3.8 Backend abstraction

Completed:

- backend protocol;
- `ReferenceBackend`;
- `OptimizedBackend`;
- public backend selector:
  - `"reference"`;
  - `"optimized"`;
  - `"auto"`.

Current behavior:

- `"reference"` always uses `ReferenceBackend`;
- `"optimized"` uses `OptimizedBackend` when required Numba kernels are available, otherwise falls back to reference;
- `"auto"` prefers optimized when available, otherwise reference.

### 3.9 Optimized backend

Completed:

- supported optional optimized full-step backend;
- optional Numba indexing/lane-change/longitudinal kernels;
- fused indexing/neighbor fast path;
- optimized backend equivalence tests;
- RNG next-draw parity checks;
- fallback behavior when Numba is unavailable.

Current representative benchmark status:

| case | reference ms/step | optimized ms/step | speedup vs reference | speedup vs Task 21 optimized |
|---|---:|---:|---:|---:|
| medium_moderate | 96.9242 | 4.1205 | 23.522x | 2.625x |
| medium_dense | 354.9731 | 13.6533 | 25.999x | 1.839x |
| wide_moderate | 191.7672 | 7.8746 | 24.353x | 2.290x |

Final Task 22+23 recommendation:

```text
keep OptimizedBackend supported and merge indexing fast path
```

Current optimized bottleneck:

- `lane_change_proposals`.

### 3.10 Benchmark/report infrastructure

Completed:

- indexing benchmark;
- optimized full-step benchmark;
- smoke benchmark;
- reduced-standard benchmark;
- component breakdown;
- JSON/Markdown benchmark output;
- Task 21 report;
- Task 22+23 report.

Current policy:

- no hard performance thresholds in pytest;
- benchmarks are non-CI performance tools;
- component profiling remains in benchmark code, not production hot path.

---

## 4. Current architecture

Target layering:

```text
low-level core
  |
  |-- step_reference(...)
  |-- ReferenceBackend
  |-- OptimizedBackend
  |
simulator facade                  # next stage
  |
  |-- reset(...)
  |-- step(actions)
  |-- scenario config
  |-- backend config
  |-- controlled vehicle selection
  |-- optional invariant checks
  |
RL contracts                      # after facade
  |
  |-- action schema
  |-- observation schema
  |-- reward schema
  |-- episode semantics
  |-- metrics/info
  |
environment wrappers              # after contracts
  |
  |-- Gymnasium single-agent
  |-- Gymnasium vector-compatible option
  |-- multi-agent wrapper if needed
  |-- RLlib/PettingZoo only after core env is stable
```

Critical dependency rule:

```text
core must not import gymnasium, ray, rllib, torch, matplotlib, pandas, scipy, or networkx.
```

---

## 5. What should not be done next

Do not start directly with Gymnasium/RLlib wrappers.

Reason:

A wrapper built directly on low-level `TrafficState` and backend APIs will couple RL code to simulator internals too early.

Before wrappers, the project needs:

1. simulator facade;
2. controlled action contract;
3. observation contract;
4. reward contract;
5. episode/reset contract;
6. metrics/info contract.

Do not introduce new physics during the first RL-env preparation tasks.

Do not optimize `lane_change_proposals` before defining the simulator facade unless benchmark evidence shows it blocks development.

Do not add action semantics inside Numba kernels first. Define reference/control semantics first, then optimize later if necessary.

---

## 6. Immediate next stage: RL environment preparation

The next stage should build a clean layer between the simulator core and RL wrappers.

Primary goal:

```text
Create a stable simulator facade and formalize the boundary where external RL control enters the simulation.
```

The correct immediate next task is:

```text
Task 24 — simulator facade and deterministic episode skeleton
```

Task 24 should not add Gymnasium yet.

---

## 7. Proposed upcoming tasks

### Task 24 — Simulator facade and deterministic episode skeleton

Goal:

Create a stable high-level simulator facade without adding Gymnasium yet.

Expected additions:

- `TrafficSimulator` or `Simulator`;
- `ScenarioConfig`;
- `BackendConfig`;
- deterministic `reset(seed=...)`;
- deterministic `step(actions=None)`;
- current state access;
- controlled vehicle selection placeholder;
- optional runtime invariant validation;
- backend selection integration;
- rollout helper for tests/debug.

Important constraints:

- do not add Gymnasium;
- do not add RLlib;
- do not add PettingZoo;
- do not add rewards yet unless minimal placeholders are strictly needed;
- do not change simulator physics;
- do not break direct backend usage;
- do not add hidden stochastic behavior.

Success criteria:

- facade reset is deterministic under fixed seed;
- facade step matches backend step when no external actions are supplied;
- `backend="reference"`, `backend="optimized"`, and `backend="auto"` work through the facade;
- runtime invariants can be enabled/disabled;
- tests cover reset/step/determinism/backend selection;
- `pytest -q` passes;
- README includes a small facade usage snippet.

Recommended files to consider:

```text
src/snfs_traffic/simulator.py
src/snfs_traffic/config.py
# or
src/snfs_traffic/sim/
  __init__.py
  simulator.py
  config.py
```

Do not over-abstract. The first facade should be small and explicit.

---

### Task 25 — Controlled action contract

Goal:

Define how external control enters the simulator.

Suggested first action mode:

```text
lane_delta ∈ {-1, 0, +1}
```

Possible later action extensions:

- desired speed;
- acceleration;
- combined lateral + longitudinal action;
- continuous control variants.

For the first RL MVP, prefer the smallest useful action space.

Required design decisions:

- how controlled vehicles are selected;
- whether all controlled vehicles receive actions every step;
- what happens when an action is missing;
- what happens when an action is invalid;
- how action masks are represented;
- whether invalid actions are clipped, ignored, rejected, or penalized;
- whether controlled actions override stochastic lane-change attempts;
- how RNG parity is defined when external actions replace stochastic attempts.

Important warning:

This is the highest-risk design point before RL. Do not bury it inside optimized kernels.

Success criteria:

- action schema is explicit;
- invalid action behavior is explicit;
- reference path supports controlled actions;
- optimized path either supports the same behavior or falls back clearly;
- tests compare controlled and uncontrolled behavior.

---

### Task 26 — Observation contract

Goal:

Define what the agent sees.

Recommended first observation type:

```text
ego-centric local observation for each controlled vehicle
```

Candidate fields:

- ego lane;
- ego velocity;
- normalized position or omitted position;
- front gap current lane;
- front relative speed current lane;
- back gap current lane;
- left-lane front/back gaps if lane exists;
- right-lane front/back gaps if lane exists;
- action mask;
- optional global density/lane count metadata.

Avoid full global occupancy as the first default observation unless there is a specific experiment requiring it.

Success criteria:

- observation builder is independent from Gymnasium;
- observation shape is deterministic;
- observation dtype is documented;
- observation tests cover road wraparound and lane boundaries;
- action mask is consistent with lane boundaries and occupancy/safety logic.

---

### Task 27 — Reward contract

Goal:

Define reward components independently from Gymnasium.

Candidate first reward components:

- speed/progress reward;
- lane-change cost;
- invalid-action penalty;
- unsafe-gap penalty if applicable;
- optional collision/termination penalty if future collision semantics are added.

Keep reward decomposed:

```python
reward_total = reward_speed + reward_lane_change + reward_invalid + reward_safety
```

The `info` dict should expose reward components.

Success criteria:

- reward function is pure/testable;
- reward components are documented;
- reward scale is reasonable;
- tests cover simple scenarios.

---

### Task 28 — Episode semantics and metrics

Goal:

Define episode lifecycle.

Required decisions:

- max episode steps;
- `terminated` vs `truncated`;
- reset randomization;
- controlled vehicle lifecycle;
- whether controlled vehicles can disappear/die;
- what happens if no controlled vehicle is available;
- per-step info schema;
- per-episode summary metrics.

Candidate metrics:

- mean speed;
- controlled mean speed;
- flow proxy;
- lane changes;
- invalid actions;
- safety violations;
- reward components;
- backend used;
- seed;
- scenario config.

Success criteria:

- deterministic episode rollout under fixed seed;
- metrics are stable and tested;
- episode end behavior is explicit.

---

### Task 29 — Gymnasium single-agent environment

Goal:

Add the first real RL wrapper after facade/contracts are stable.

Recommended initial scope:

- one controlled ego vehicle;
- discrete lateral action;
- fixed scenario config;
- fixed observation schema;
- fixed reward schema;
- Gymnasium API:
  - `reset(seed=None, options=None)`;
  - `step(action) -> obs, reward, terminated, truncated, info`.

Do not add RLlib yet.

Success criteria:

- `gymnasium.Env` compliance;
- deterministic reset with seed;
- smoke random-agent rollout;
- tests for observation/action spaces;
- tests for terminated/truncated behavior;
- no direct dependency of core kernels on Gymnasium.

---

### Task 30 — Multi-agent environment design

Goal:

Only after single-agent env works, design multi-agent control.

Possible options:

- custom multi-agent facade;
- PettingZoo ParallelEnv;
- RLlib MultiAgentEnv.

Do not implement all wrappers at once.

Required decisions:

- agent IDs;
- controlled vehicle assignment;
- per-agent observations;
- shared/global rewards vs individual rewards;
- agent appearance/disappearance;
- action dict validation.

Success criteria:

- deterministic multi-agent rollout;
- clear mapping between vehicle IDs and agent IDs;
- tests for missing/extra actions;
- tests for done/truncation semantics.

---

## 8. Performance roadmap

Current bottleneck after Task 22+23:

- `lane_change_proposals`.

Recommended future performance task:

```text
Optional Task P1 — optimize lane-change proposal collection
```

Goal:

Reduce current optimized-backend bottleneck after fused indexing.

Constraints:

- preserve exact semantics;
- preserve RNG parity;
- do not change action semantics;
- do not add RL behavior;
- do not alter conflict resolution semantics.

Potential strategies:

- reduce per-vehicle branching;
- specialize dense/sparse paths;
- precompute lane availability masks;
- fuse more read-only data access;
- reduce temporary allocations;
- split controlled/uncontrolled paths only after controlled action contract is finalized.

Do not start this before Task 24 unless performance blocks facade work.

---

## 9. Model-extension roadmap

These are not prerequisites for first RL environment preparation.

Potential future tasks:

- length-aware multi-cell occupancy;
- length-aware gaps;
- bus body-cell collision geometry;
- open-boundary topology;
- inflow/outflow;
- richer behavior profiles;
- paper-exact longitudinal variants if needed.

Warning:

Any physics change must be treated as a semantic change and must update:

- reference implementation;
- tests;
- optimized backend;
- benchmarks;
- docs.

Do not mix these changes with RL wrapper tasks.

---

## 10. Documentation roadmap

Current docs should stay synchronized with project stage.

Immediate documentation needs:

- keep README status current;
- keep backend support status clear;
- document simulator facade once added;
- document action/observation/reward contracts before Gymnasium wrapper;
- keep benchmark reports in `reports/`;
- avoid claiming RL readiness before wrappers exist.

Recommended docs after Task 24:

- facade usage example;
- scenario config example;
- backend config example;
- deterministic rollout example.

Recommended docs after Tasks 25–27:

- action schema;
- observation schema;
- reward components;
- invalid action semantics;
- episode semantics.

Recommended docs after Task 29:

- Gymnasium environment usage;
- random-agent rollout example;
- minimal training smoke example only if actually tested.

---

## 11. Suggested Task 24 prompt

```text
Task 24 — simulator facade and deterministic episode skeleton

Repository context
==================

The repository is after Tasks 1–23, including Task 22+23 follow-up.

The simulator core has:
- step_reference(...) as the semantic oracle;
- ReferenceBackend;
- supported optional OptimizedBackend;
- backend selector get_backend("reference" | "optimized" | "auto");
- optional Numba kernels;
- runtime invariant validation;
- deterministic scenario initialization;
- benchmark/report infrastructure.

Goal
====

Add a small high-level simulator facade that future RL wrappers can use.

Do not add Gymnasium, RLlib, PettingZoo, rewards, or observations yet.

Expected implementation
=======================

Add a simple facade, for example:

- TrafficSimulator or Simulator;
- ScenarioConfig;
- BackendConfig or backend name field;
- reset(seed=...) method;
- step(actions=None) method;
- current state access;
- controlled vehicle selection placeholder;
- optional runtime invariant validation;
- backend selection integration.

Requirements
============

- reset is deterministic under fixed seed;
- step(actions=None) matches selected backend behavior exactly;
- backend="reference", backend="optimized", and backend="auto" work;
- no simulator physics changes;
- no hidden RNG changes;
- no Gymnasium dependency;
- no RLlib dependency;
- no rewards/observations unless minimal placeholders are unavoidable;
- existing direct backend API remains valid.

Tests
=====

Add focused tests for:

- deterministic reset;
- deterministic rollout under fixed seed;
- reference backend through facade;
- optimized/auto backend through facade, including fallback-safe behavior;
- optional runtime invariant checks;
- state returned/accessed by facade.

Validation
==========

Run:

python -m pip install -e ".[numba]"
pytest -q
pytest -q tests/test_optimized_backend_equivalence.py
pytest -q tests/test_backends_selection.py

Documentation
=============

Update README with a short facade usage example and clarify that Gymnasium wrappers are still planned, not implemented.
```

---

## 12. Readiness summary

Current project level:

- core simulator: mature enough for next-stage use;
- optimized backend: supported optional backend;
- benchmark status: good;
- semantic safety: good;
- RL API: missing;
- Gymnasium readiness: not yet;
- RL training readiness: not yet.

The correct next move is to build the simulator facade and formalize control boundaries before implementing environment wrappers.
