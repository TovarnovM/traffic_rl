# LOCAL Physica A v1.1 reproduction

This directory covers the current Physica A paper only. Commands below are
**optional user-run operations**. None were executed during release preparation.
No empirical reproduction or bitwise continuation is claimed.
Run commands from the release root using the appropriate Python environment.

The dependency chain is frozen training lines → candidate checkpoints → frozen
selection → selected native bundles → independent MC episodes → final statistics
→ figures. This release reuses the published selection for MC; retraining does
not automatically replace that selection. One training line is a sequential
1221-update trajectory with 64 CPU sampling actors and one CPU/GPU learner.
MC parallelism is across independent complete episodes, with one numeric thread
per process. No mass launcher or cluster service is needed. Full MC comprises
30,690 physical episodes (9,930 with mechanism observations); archive volume is
large (the source preparation estimated about 400 GB uncompressed). Runtime is
hardware-dependent and has not been measured here. Choose resources and output
storage before running. MC uses native retry/budget/publication semantics;
training publishes an immutable functional checkpoint after each update.

## Environments and ownership

Use **separate environments/PYTHONPATHs** for training and MC/statistics. Both trees
contain `snfs_traffic`; never merge them or put both source roots on PYTHONPATH.

- Training: `src/`, `research/`; Ray 2.57, Torch 2.5.1+cu118,
  Gymnasium 1.2.2, NumPy 2.1.2. The nongraph trainer imports the graph trainer.
- MC: `mc_toolkit/src/`; numeric inference requires no Ray; Numba 0.67.
  Full package-tree source digest:
  `3ff57f9e84d10390a6a184358332ea12bdf400011f2b6008d6d767030678492e`.
- Statistics: the frozen MC tree plus `operations/paper_outputs/{cli,data,inference}.py`
  and the `operations/fresh_test` dataset/plan/store closure. NumPy **2.1.2**
  is required for final bootstrap; rendering uses Matplotlib 3.10.

The parent release supplies these trees, dependencies, `weights/`, and curated
`results/physica_a_v1.1/`. This directory neither installs dependencies nor copies
weights. Repository release v1.1 does not change the frozen scientific package's
0.1 version/digest. An old wheel hash is not a claim about a rebuilt wheel.

## Redraw existing compact outputs

The supplied compact final inputs support the current article's 37 numerical
figures and vector master generators under `tools/publication`:

```bash
python tools/publication/generate.py --output local/publication-redraw
```

Use a new output directory. Compact inputs support redraw only, **not bootstrap
recomputation**. Original raw archives are not shipped.

## Recompute final statistics from new MC

Follow the preparation, native run/collect and ingestion sequence below. The
bridge creates a new complete dataset (`index.json`, `ingestion.json`, `results/`)
from the new native publications; no old accepted dataset or private queue receipt
is required. Then run:

```bash
PYTHONPATH=mc_toolkit/src python reproduction/paper_outputs.py extract \
  --index local/new-paper-stage/index.json --freeze reproduction/configs/frozen \
  --output local/new-statistics --final
```

The index resolves its absolute `shared_dir` to this new dataset. Keep that
location stable. For relocation, the existing `paper_outputs.py stage --index
INDEX --freeze reproduction/configs/frozen --output NEW_STAGE --max-bytes N`
creates a verified mirror, consumed with `extract --stage NEW_STAGE` instead of
`--index`; add `--include-archives` for diagnostics/illustrations. The default
mirror byte cap is 1 GiB. Never substitute native collect JSON for the dataset index.

`extract --final` calls `data.comparison_points(..., final=True)` and
`inference.bootstrap`, preserving the W7 scale-keyed and L2000 length-keyed
streams: 49,999 replicates, master seed 2026102401, 30 environments, whole-training
and environment crossed blocks, and the frozen comparison families. No new FDR
procedure is added. `--diagnostics` is cell-batched and cannot be combined with
a full multi-cell `--final` extraction; use separate descriptive cell scopes.
`--illustrations` needs archives. Neither is mandatory for final scalar inference.

## New MC using the selected published weights

Preparation is metadata-only. It checks the frozen input hashes and full MC source
digest, scans **only the explicitly supplied weights directory** for native
`manifest.json` files, matches selected manifest hashes, and verifies both native
bundle files against the original seal. It ignores the catalog layout and the
11 archive-only terminal bundles. Only the 17 selected bundles enter test plans.
The 28 unique release bundles comprise selected 17 plus terminal 15, overlap 4.

```bash
PYTHONPATH=mc_toolkit/src python reproduction/prepare_mc.py \
  --weights weights --output local/mc-requests --results local/mc-results \
  --spool local/mc-spool --deadline-utc YOUR_FUTURE_UTC_ISO_TIMESTAMP \
  --workers 4 --max-attempts 30690
```

Choose a real future UTC timestamp and resources; the numbers above are an
example, not a measured or authorized compute budget. Preparation does not run MC.
It writes disjoint `full12` (20,760 tasks) and `mechanism` (9,930 tasks) plans,
run requests, collect requests, and `transport.json`. Run them explicitly (keep
these environment settings for ingestion/statistics too):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PYTHONPATH=mc_toolkit/src python -m physica_mc run --request local/mc-requests/full12-run.json
PYTHONPATH=mc_toolkit/src python -m physica_mc run --request local/mc-requests/mechanism-run.json
PYTHONPATH=mc_toolkit/src python -m physica_mc collect --request local/mc-requests/full12-collect.json
PYTHONPATH=mc_toolkit/src python -m physica_mc collect --request local/mc-requests/mechanism-collect.json
```

Each run request contains `plan`, `output_root`, `spool_root`, `workers`, `threads`,
`device`, `observer`, and `budget` (`deadline_utc`, `max_attempts`, `max_workers`,
`cpu_limit`, `max_gpu_workers`). Attempt caps apply separately to the two plans.
Native resume/collection uses those stable plan/output/spool locations.

Transport follows `physica_test_plan.model_transport` semantics: preserve bundle
checksums and scientific identities. Because native `planning.make_plan` hashes
transport paths and task membership, the two portable plans have **new plan IDs**.
The original plan, task IDs, candidate identities, freeze packet, selection,
profiles and seeds remain unchanged. `transport.json` records this distinction.
Logical panel aliases remain in `b1-logical-references.json` and `panels.json`;
they are not additional physical replicates.

Native per-episode analysis/export remain available, for example request shapes:

```json
{"input_path":"local/episode/completion.json","profile":"reproduction/configs/frozen/analysis-full12.json","output_root":"local/analysis","require_scientific":true}
```

```json
{"analysis_paths":["local/analysis/ANALYSIS_FILE.json"],"output_dir":"local/episode-tables","require_scientific":true}
```

Pass these to `python -m physica_mc analyze --request REQUEST.json` and `export`
respectively, with `PYTHONPATH=mc_toolkit/src`. Use actual native publication and
analysis paths; use the mechanism profile for mechanism observations. These are
per-episode derivatives, not the paper's crossed bootstrap. The native CLI also
supports task, plan, episode, verify, publish, select and freeze; legacy conversion
requires `convert --trusted-legacy`. There are no `stats` or `plot` subcommands.

After both collections report complete, stop modifying the native requests and
result roots, and ingest into a **new** directory:

```bash
PYTHONPATH=mc_toolkit/src python reproduction/ingest_mc.py \
  --requests local/mc-requests --output local/new-paper-stage
PYTHONPATH=mc_toolkit/src python reproduction/paper_outputs.py extract \
  --index local/new-paper-stage/index.json --freeze reproduction/configs/frozen \
  --output local/new-statistics --final
```

The bridge reuses `runner.collect_plan` acceptance, including native ledger-bound
executions/attempts and exited lifecycle checks. Missing, invalid, conflicting or
incomplete collections fail. It checks the exact disjoint frozen task/alias
partitions and unchanged model identities, then uses
`physica_test_store.publish_episode` to verify and archive every accepted episode.
The store enforces exact native observer profiles, frozen source, CPU/one-thread
execution and canonical task/result identities. Native UUID attempt IDs remain
unchanged. `dataset.coverage` supplies the original physical denominators and
panel overlaps; logical aliases never become extra replicates.

`ingestion.json` binds exact execution plans, run/collect requests, native ledger
hashes, accepted marker hashes and the new output location. Its canonical digest
is the **new local input binding** in every envelope and index. Original logical
plan/freeze IDs identify the frozen scientific design; separate execution plan
IDs identify the transported runs. This is not an original-run snapshot, and no
historical attempt sequence or private receipt is fabricated. Inputs are hashed
again before completion. The exclusively and atomically published `index.json`
is the dataset completion marker. A failure leaves no usable index; retain or
remove that incomplete output manually and retry with a different new directory.
Existing output directories are always refused.

**Runtime storage/cost:** ingestion is not metadata-only. Native collection and
store publication read/validate NPZ arrays; the store packages deterministic
`episode.tar.gz` archives and scalar `result.json` envelopes, without simulation
or bootstrap. Provision substantial additional archive space alongside the native
roughly 400 GB uncompressed data, plus temporary space for one episode archive.
Compression size and runtime have not been measured. The store requires native
marker directories on a recognized local filesystem (e.g. ext4/xfs); packaging
uses temporary files there. Keep the native ledgers/results unchanged throughout
ingestion. No packaging or scientific execution was performed during this change.

## New training

`configs/training-lines.json` contains exactly 15 lines, derived from frozen
`policies.json`: Graph seeds 1510000001–1530000001, Node
1540000001–1560000001, Grid 1570000001–1590000001; Graph half-penalty
1610000001–1630000001 and double-penalty 1640000001–1660000001.
Both coefficients scale jointly by 0.5/1/2, from applied 0.001 and rejected
0.002. No independent-coefficient or reward active-set normalization experiment
is added. The existing PPO node-mask averaging remains intact.

All lines use 64 actors × 1 environment × 256 fragment = 16,384 batch,
minibatch 1,024, 6 epochs, learning rate 0.0002; gamma .995, GAE lambda .95,
clip .2, entropy .01, value coefficient 1, value clip 10, KL coefficient .2 and
target .01. Graph is 128 × 3 message layers (355,460 parameters); matched Node
and Grid have 354,308 and 353,124. Training samples density .1/.2/.3/.4 and
AV fraction .9/.95/1, on length 1000 with 4 lanes, 1000 warmup + 1000 measured
steps and the original warmup cache. 1221 iterations = 20,004,864 steps.
FP32, AMP off, TF32 off. Generic research CLI defaults do not encode this protocol.

```bash
PYTHONPATH=src python reproduction/train.py --line E3graph_s1 \
  --output local/training/E3graph_s1 --ray-cpus 65 --worker-cpus 1 --device cuda
```

Choose CPU reservations and device explicitly. GPU selection can use the user's
normal `CUDA_VISIBLE_DEVICES`; no private UUID is assumed. All actors stay in a
new local Ray runtime. OMP, MKL, OPENBLAS, NUMEXPR and NUMBA thread
environment variables are set to one for the driver and workers; the worker hook
also checks Torch intra/inter-op pools at one. CPU learner execution is supported but changes hardware;
changed hardware/placement can change trajectories and performance.

For continuation, add `--restore PATH_TO_ITER_DIRECTORY` and choose a **new output
directory**. Only trusted checkpoints from this wrapper with matching source,
architecture, seed and scientific config identity are accepted. Native Algorithm
save/restore retains weights, optimizer moments, LR/KL and counters; source-derived
digests verify restoration. The immutable `COMPLETE.json` and atomic `LATEST`
pointer identify committed checkpoints. Failed updates can resume from a prior
complete checkpoint. Sampler/environment/warmup-cache/RNG state is not restored
bitwise. Published numeric weight bundles are inference assets, not optimizer
checkpoints. All updates are checkpointed; provision storage accordingly.

## Source attribution and static verification

`training/checkpoints.py` extracts scientific/checkpoint functions from
`physica_a_revision_r1/analysis/e3_gpu_pilot.py`; `training/configuration.py`
adapts `e3_distributed_pilot.py`; joint penalty scaling follows `w7_train.py`.
Only the local resource boundary is new. Loss, action masking, optimizer behavior,
model encoders and environment transitions remain owned by `src/snfs_traffic` and
the two existing research trainers. `source-closure.json` records source hashes
and imports for the original top-level package and the 11 requested research scripts.
These attribution hashes cover original source, not the new wrappers, and do not
validate runtime behavior. Training checkpoint identity separately hashes local
source and wrapper files; wrapper changes intentionally invalidate old identities.

Frozen JSONs are byte-identical scientific/consumer inputs. `frozen-files.json`
checks the **included subset**. `FREEZE_MANIFEST.json` is the unchanged historical
seal, whose full artifact set includes material intentionally not shipped here;
this is not a new full-seal validation claim. No protocol/TZ/reviewer documents,
private launch handoffs or campaign logs are copied.

Preparation verification was limited to stdlib AST parsing, JSON metadata,
source attribution and hash comparisons. No tests, builds, installs, training,
inference, statistics, plotting or container execution were performed.
