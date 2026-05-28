# PLAN.md — Revised S-NFS Traffic Simulator Roadmap

Current state: after Tasks 1–23, including Task 22+23 follow-up.

The project is a clean, array-based Revised S-NFS-style traffic simulator core intended for future reinforcement-learning experiments. It currently has a reliable reference simulator, a supported optional optimized backend, backend selection, correctness tests, RNG-parity checks, runtime invariant checks, and benchmark/report infrastructure.

Important semantic note: the current longitudinal phase implements the full
Revised S-NFS phase order in the reference backend. `q` controls slow-to-start,
`r/S` controls stochastic look-ahead, and `P1..P4` are keep-speed probabilities
for random braking (`1 - Pk` braking probability).

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

- Reward schema.
- Episode lifecycle.
- Gymnasium wrapper.
- RLlib wrapper.
- PettingZoo/multi-agent wrapper.

### 1.3 Current readiness estimate

| Target | Readiness | Notes |
|---|---:|---|
| Continue simulator-core development | High | Reference/optimized architecture is established. |
| Begin RL environment preparation | 85–90% | Facade/actions/local observations are in place; next is reward + episode/env contracts. |
| Start actual RL training experiments | 40–50% | Facade-level controlled lateral actions and local observations are implemented; rewards, episode semantics, metrics/info contract, and env wrappers are still missing. |

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

- controlled vehicles now support facade-level lateral action control via the reference action path.

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

- head ordering remains head-based;
- body occupancy/gap validation is length-aware;
- optimized backend falls back to reference for non-unit lengths.

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
simulator facade                  # implemented MVP
  |
  |-- reset(...)
  |-- step(actions)
  |-- scenario config
  |-- backend config
  |-- controlled vehicle selection
  |-- optional invariant checks
  |
remaining RL contracts            # next: reward, episode, metrics/info
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

Before wrappers, the project status is:

1. simulator facade: done;
2. controlled action contract: done for lateral facade/reference path;
3. observation contract: done for local controlled-only schema;
4. reward contract: pending;
5. episode/reset contract: pending;
6. metrics/info contract: pending.

Do not introduce new physics during the first RL-env preparation tasks.

Do not optimize controlled-action paths or `lane_change_proposals` further unless profiling shows it blocks reward/episode/env work.

Do not add action semantics inside Numba kernels first. Define reference/control semantics first, then optimize later if necessary.

---

## 6. Completed facade milestone (historical)

The simulator facade milestone has been completed:
- facade-level reset/step/observe/rollout APIs are implemented;
- controlled lateral actions are implemented via reference action path;
- local controlled-only observations are implemented.

Current roadmap remains:
1. reward schema;
2. episode reset/termination/truncation contract;
3. Gymnasium wrapper;
4. optional optimized controlled-action path after profiling.

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

Do not start this before reward/episode/env contracts unless performance blocks development priorities.

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

Recommended docs after facade milestone:

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

