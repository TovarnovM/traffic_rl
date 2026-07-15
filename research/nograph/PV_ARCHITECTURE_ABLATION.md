# Edge Graph-PPO architecture ablation

This experiment keeps the centralized PV environment, reward, action masks,
factorized nodewise PPO loss, mixed reset grid, training budget, and paired
evaluation protocol fixed.  Only the policy encoder changes.

## Frozen models

| Preset | Inter-node interaction | Parameters |
|---|---|---:|
| Existing Edge Graph-PPO `h=128, layers=3` | Dynamic nearest-AV graph with edge features | 355,460 |
| `node-mlp-matched` | None; shared self-residual node MLP | 354,308 |
| `grid-cnn-matched` | Fixed PV-centred `4 × 41` road grid | 353,124 |
| `grid-cnn-large` | Wider/deeper fixed road grid | 1,390,596 |

The two matched baselines differ from Edge Graph-PPO by 0.32% and 0.66%,
respectively.  Parameter matching controls trainable capacity, not forward
compute; report both sampled environment steps and wall-clock throughput.

`node-mlp-matched` consumes `node_features`, `node_mask`, `action_mask`, and
`global_features`.  Every active AV is encoded independently; its actor cannot
consume another AV embedding.  The critic still uses the masked mean of all AV
embeddings, exactly as in the graph policy.

The CNN models deterministically scatter the same packed AV node features into
a PV-centred raster with four lanes and relative positions `[-10, +30]`.
Dynamic `neighbor_index`, `neighbor_mask`, and `edge_features` are ignored.
Five longitudinal dilations `1,2,4,8,16` cover the complete 41-cell window.
The large model adds three residual blocks and uses 128 instead of 80 channels.

In strict terminology this is a fixed-grid convolutional encoder without a
dynamic vehicle graph.  A convolution can itself be interpreted as message
passing on a regular lattice, so the paper should not call it an architecture
without any relational inductive bias.

## Installation and smoke checks

```bash
python -m pip install -e ".[rl,ray,numba,viz]"

for MODEL in node-mlp-matched grid-cnn-matched grid-cnn-large; do
  PYTHONPATH=src python -u research/train_pv_nongraph_ppo.py \
    --model-preset "$MODEL" \
    --smoke \
    --num-gpus 1 \
    --out-dir "/workspace/artifacts/rl/pv_nongraph_ppo/smoke_${MODEL}"
done
```

Smoke checkpoints are architecture-specific.  A Graph-PPO checkpoint cannot
be restored into any non-graph model, and checkpoints cannot be exchanged
between the three presets.

## Matched 20M training protocol

Run the following command separately for each value of `MODEL` and `TAG`:

```bash
MODEL=node-mlp-matched
TAG=node_mlp_matched

PYTHONPATH=src python -u research/train_pv_nongraph_ppo.py \
  --model-preset "$MODEL" \
  --road-length 1000 \
  --num-lanes 4 \
  --train-densities 0.10,0.20,0.30,0.40 \
  --train-av-fractions 0.9,0.95,1.00 \
  --front-distance 30 \
  --back-distance 10 \
  --sensor-distance 60 \
  --cooldown-steps 1 \
  --warmup-steps 1000 \
  --warmup-cache \
  --episode-steps 1000 \
  --backend optimized \
  --applied-penalty 0.001 \
  --rejected-penalty 0.002 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --clip-param 0.20 \
  --lr 0.0002 \
  --entropy-coeff 0.01 \
  --vf-loss-coeff 1.0 \
  --vf-clip-param 10.0 \
  --workers 32 \
  --envs-per-worker 1 \
  --num-gpus 1 \
  --rollout-fragment-length 256 \
  --train-batch-size 16384 \
  --minibatch-size 1024 \
  --epochs 6 \
  --iterations 2000 \
  --stop-timesteps 20000000 \
  --checkpoint-freq 25 \
  --seed 20260715 \
  --out-dir "/workspace/artifacts/rl/pv_nongraph_ppo/${TAG}_20m"
```

Use these pairs:

```text
MODEL=node-mlp-matched  TAG=node_mlp_matched
MODEL=grid-cnn-matched TAG=grid_cnn_matched
MODEL=grid-cnn-large   TAG=grid_cnn_large
```

The primary comparison keeps the same `minibatch-size=1024`.  If and only if
`grid-cnn-large` produces CUDA OOM, record the deviation and retry that model
with `--minibatch-size 512`; do not silently change its PPO settings.

The first seed is useful for screening.  An architecture claim in the paper
requires at least three independent training seeds for Edge Graph-PPO and each
reported non-graph policy, with the same seed set and checkpoint-selection
budget.

## Restore and checkpoint selection

A restore smoke is available independently:

```bash
RUN=/workspace/artifacts/rl/pv_nongraph_ppo/node_mlp_matched_20m

PYTHONPATH=src python -u research/evaluate_pv_nongraph_ppo.py \
  --checkpoint "$RUN/checkpoints" \
  --densities 0.30 \
  --av-fraction 1.00 \
  --smoke \
  --out-dir "$RUN/evaluation_restore_smoke"
```

The selector evaluates requested periodic checkpoints plus final on identical
selection seeds, ranks deterministic PV speed at `rho=0.30, AV=1.00`, and then
runs the full matrix on a different held-out seed set:

```bash
PYTHONPATH=src python -u research/select_pv_policy_checkpoint.py \
  --policy-kind nongraph \
  --checkpoint-root "$RUN/checkpoints" \
  --checkpoint-iterations 825,875,900 \
  --include-final \
  --selection-density 0.30 \
  --selection-av-fraction 1.00 \
  --selection-runs 20 \
  --selection-seed 300001 \
  --full-densities 0.10,0.20,0.30,0.40 \
  --full-av-fractions 0.90,0.95,1.00 \
  --full-runs 20 \
  --full-seed 400001 \
  --workers 20 \
  --measure-steps 2000 \
  --backend optimized \
  --out-dir "$RUN/evaluation_selected"
```

The script refuses to reuse the selection seed for the final matrix.  It
creates `checkpoint_leaderboard.csv`, `checkpoint_selection.json`, and three
full evaluation directories.  Each evaluator invocation first verifies that
RLlib and the extracted multiprocessing CPU model choose exactly the same
deterministic action.

The currently training Edge Graph-PPO uses the identical selector by changing
only the policy kind and run directory:

```bash
GRAPH_RUN=/workspace/artifacts/rl/pv_graph_ppo/mixed_big_h128_l3_20m

PYTHONPATH=src python -u research/select_pv_policy_checkpoint.py \
  --policy-kind graph \
  --checkpoint-root "$GRAPH_RUN/checkpoints" \
  --checkpoint-iterations 825,875,900 \
  --include-final \
  --selection-density 0.30 \
  --selection-av-fraction 1.00 \
  --selection-runs 20 \
  --selection-seed 300001 \
  --full-densities 0.10,0.20,0.30,0.40 \
  --full-av-fractions 0.90,0.95,1.00 \
  --full-runs 20 \
  --full-seed 400001 \
  --workers 20 \
  --measure-steps 2000 \
  --backend optimized \
  --out-dir "$GRAPH_RUN/evaluation_selected"
```

## Direct architecture comparison

Pass the learned-policy raw `*_runs.csv` files from the same held-out seed set.
Repeat a label for its three AV-fraction files.  Put Edge Graph-PPO first so a
positive pairwise delta means an advantage for the graph model.

```bash
PYTHONPATH=src python -u research/compare_pv_architectures.py \
  --input graph=/path/graph_av_0p90_runs.csv \
  --input graph=/path/graph_av_0p95_runs.csv \
  --input graph=/path/graph_av_1p00_runs.csv \
  --input node_mlp=/path/node_mlp_av_0p90_runs.csv \
  --input node_mlp=/path/node_mlp_av_0p95_runs.csv \
  --input node_mlp=/path/node_mlp_av_1p00_runs.csv \
  --input cnn_matched=/path/cnn_matched_av_0p90_runs.csv \
  --input cnn_matched=/path/cnn_matched_av_0p95_runs.csv \
  --input cnn_matched=/path/cnn_matched_av_1p00_runs.csv \
  --input cnn_large=/path/cnn_large_av_0p90_runs.csv \
  --input cnn_large=/path/cnn_large_av_0p95_runs.csv \
  --input cnn_large=/path/cnn_large_av_1p00_runs.csv \
  --equivalence-margin 0.10 \
  --bootstrap-samples 10000 \
  --seed 900001 \
  --out-dir /workspace/artifacts/rl/pv_architecture_comparison
```

The comparator requires every architecture to contain exactly the same
scenario, warmup, assignment, AV-selection, and measurement seeds.  It aborts
instead of silently comparing unpaired trajectories.  The equivalence margin
must be chosen before examining results; `0.10 cells/step` above is an example,
not a universally justified threshold.  It also writes a comparison PNG with
PV-speed curves and paired deltas relative to the first input label.

## Interpretation

- Edge Graph-PPO above both matched models supports a graph inductive-bias and
  sample-efficiency claim.
- A large CNN that catches the graph model weakens a necessity claim but still
  supports graph parameter efficiency.
- A large CNN that remains worse strengthens the case for dynamic vehicle
  relations, provided training-seed uncertainty and learning curves agree.
- A CNN that wins means the paper must not claim that the graph architecture is
  the source of the performance gain.

Checkpoint selection must use deterministic validation, never training return.
The final paper should report held-out PV speed, background-flow effects,
lane-change burden, parameter count, environment steps, and wall-clock cost.
