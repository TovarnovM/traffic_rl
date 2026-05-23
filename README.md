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

- Project skeleton only.
- No S-NFS traffic dynamics implemented yet.

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
