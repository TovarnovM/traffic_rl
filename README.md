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
- Validation tests for params/state/topology/scenario/indexing/longitudinal behavior.

Not implemented yet:
- Lane changing.
- Full reference step with lane-change phase.
- Length-aware multi-cell occupancy.
- Length-aware bumper-to-bumper gaps.
- Simulator facade.
- RL environments.
- Graph/local observations.
- Metrics.
- Numba/Cython kernels.

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

- Current longitudinal step uses head-cell-only gaps.
- Vehicle length is currently ignored by occupancy and longitudinal gaps.
- Exact paper-equation mapping of Revised S-NFS probabilities is deferred unless formal equations are supplied.
