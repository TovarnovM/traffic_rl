# snfs-traffic

`snfs-traffic` is a new clean implementation of a Revised S-NFS traffic simulator.

## Design goal

The long-term design target is a high-performance, array-oriented simulation core.

## Planned future capabilities

- Multi-agent reinforcement learning environments.
- Graph/GNN-compatible observation builders.
- Flexible vehicle behavior rule presets.
- Numba/Cython-optimized simulation step.

## Current status

Implemented:
- Project skeleton.
- Core `SimulationParams` schema.
- Core `TrafficState` array schema.
- Reference periodic `RingTopology` for a multi-lane ring segment.
- Reproducible uniform-random scenario initializer.
- Reference head-cell occupancy / lane ordering / neighbor indexing.
- Reference longitudinal same-lane step.
- Reference lane-change phase using the paper's incentive/safety criteria and stochastic P_CL attempt.
- Stochastic conflict resolution for simultaneous lane-change target-cell conflicts.
- Full reference step composing lane-change phase and longitudinal phase.
- Reusable runtime invariant validation helper.
- Random rollout invariant tests for the current reference core.
- Minimal backend step protocol.
- `ReferenceBackend` wrapper around `step_reference`.
- Reference-vs-backend equivalence tests.
- Backend-neutral pure-array indexing kernels for occupancy, lane order, and periodic-ring neighbor/gap computation.
- Public validated indexing wrappers delegating to pure-array indexing kernels.
- Equivalence tests proving pure-array indexing kernels match the public reference indexing API.
- Validation tests for params/state/topology/scenario/indexing/longitudinal/lane-change/full-step behavior.

Not implemented yet:
- Optimized backend.
- Numba/Cython kernels.
- Controlled RL action semantics.
- Observations.
- Metrics.
- Simulator facade.
- RL environments.
- Length-aware multi-cell occupancy.
- Length-aware bumper-to-bumper gaps.
- Body-cell bus collision geometry.
- Open-boundary topology/step semantics.


## Backend contract

A backend performs exactly one full simulation step and must match `step_reference` semantics.

Backends receive:
- `TrafficState`
- `SimulationParams`
- `RingTopology`
- `np.random.Generator`

Backends return:
- `TrafficState`

Current backend limitations:
- only `ReferenceBackend` exists;
- no optimized backend exists yet;
- backend equivalence tests currently compare `ReferenceBackend` against `step_reference`;
- future optimized backends must pass the same equivalence tests;
- RNG must come from the provided `np.random.Generator`;
- occupancy/gaps/collision checks remain head-cell-only;
- vehicle length is ignored by occupancy and gap logic;
- controlled vehicles still do not receive external RL actions.

## Developer quick checks

```bash
python -m pip install -e .
pytest -q
python -c "import snfs_traffic; print(snfs_traffic.__version__)"
```

No-install alternative:

```bash
PYTHONPATH=src python -c "import snfs_traffic; print(snfs_traffic.__version__)"
```

## Current model limitations

- Current full step is head-cell-only.
- Vehicle length is ignored by occupancy, gaps, and collision checks.
- Lane changes are lateral only and do not move position.
- No same-step lateral swaps into previously occupied target cells.
- Controlled vehicles do not yet receive external actions.
- Runtime invariant helper validates current reference semantics, not future length-aware geometry.
- Full paper-exact longitudinal equations are not implemented unless separately added.
- Lane-change flags in returned full-step state describe lateral motion during that full step.
