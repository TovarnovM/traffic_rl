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
- Reference head-cell occupancy / lane ordering / neighbor indexing (head-cell only; vehicle body cells are not marked).
- Reference longitudinal same-lane step without lane changes.
- Reference lane-change phase using the paper's incentive/safety criteria and stochastic P_CL attempt.
- Stochastic conflict resolution for simultaneous lane-change target-cell conflicts.
- Full reference step composing lane-change phase and longitudinal phase.
- Validation tests for params/state/topology/scenario/indexing/longitudinal/lane-change/full-step behavior.

Not implemented yet:
- Controlled RL action semantics.
- Observations.
- Metrics.
- Simulator facade.
- RL environments.
- Length-aware multi-cell occupancy.
- Length-aware bumper-to-bumper gaps.
- Body-cell bus collision geometry.
- Numba/Cython kernels.
- Open-boundary topology/step semantics.

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
- Full paper-exact longitudinal equations are not implemented unless separately added.
- Lane-change flags in returned full-step state describe lateral motion during that full step.
