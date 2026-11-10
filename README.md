# traffic_rl — Physica A v1.1

Source, trained weights and compact reproduction materials for the **current Physica A study of Graph/NodeMLP/GridCNN priority-vehicle control** in Revised S-NFS traffic. This release does not include the separate/joint RL study, manuscript text or reviewer correspondence.

## Contents

- [Training simulator and models](<src/snfs_traffic/>) and [research entrypoints](<research/>).
- [Frozen MC/statistics core](<mc_toolkit/README.md>) and [training/MC reproduction guide](<reproduction/README.md>), including the 15 training configurations and frozen selection/seeds/profiles.
- [Weights](<weights/README.md>): 17 selected and 15 terminal checkpoints, **28 unique numerical bundles** from 15 independent trainings; no optimizer checkpoints.
- [Final results](<results/physica_a_v1.1/README.md>): **38 figures**, five main and five supplementary tables, and the 42-domain results.
- [Compact inputs](<results/physica_a_v1.1/inputs/README.md>) and [publication generators](<tools/publication/README.md>). Stored estimates, intervals, support and missing values are retained; no new inference is needed for redraw.

## Redraw figures and tables

Python 3.11; run from the repository root. The destination must not exist.

```sh
python3.11 -m venv .venv-figures
. .venv-figures/bin/activate
python -m pip install -r requirements-figures.txt
python tools/publication/generate.py --output ../physica-a-v1.1-redraw
```

This redraws 37 numerical figures and copies the original vector schematic master. It needs no model inference, raw episode archives or original worktree. Table-only export is also available in the [generator guide](<tools/publication/README.md>).

## Repeat experiments

Follow the [reproduction guide](<reproduction/README.md>): training, selected-policy MC and final statistical extraction are separate, expensive, **explicitly user-run** steps. Use separate training and MC environments: both source trees contain a deliberately different `snfs_traffic` package. Recorded dependencies are in [training requirements](<requirements-training.txt>) and [MC requirements](<requirements-mc.txt>); select the appropriate Torch CPU/CUDA wheel explicitly.

Training used 64 actors, batches of 16,384 steps and 1,221 updates (20,004,864 steps per line). Final evaluation contains 30,690 physical episodes. Bootstrap intervals use the frozen crossed training/environment design; compact aggregate inputs alone cannot reproduce bootstrap inference. Rerunning MC supplies new episode data. Bitwise retraining/resume and identical PDF bytes are not promised.

## Release boundary

Raw archives, runtime logs, intermediate checkpoints, local campaign scripts and article files are excluded. Historical paths inside immutable scientific metadata identify origins, not required runtime locations. Test source is retained, but **no tests, experiments, rendering, builds or installation were run during v1.1 preparation**; checks were limited to source inspection and file/metadata integrity.

See the [local release and GitHub upload guide](<RELEASE_v1.1.md>). Exclusions apply to the v1.1 tree; older Git history is not rewritten.
