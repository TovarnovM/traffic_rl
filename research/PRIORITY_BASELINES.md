# Priority-vehicle Baseline 1/2 protocol

This document fixes the Phase 0 experiment contract implemented by
`research/priority_baselines.py`.

## Fixed model defaults

- periodic four-lane ring, `road_length=1000` cells;
- ordinary HDV maximum speed: 5 cells/step;
- priority-vehicle (PV) maximum speed: 6 cells/step;
- a PV is uncontrolled and retains native Revised S-NFS lane changes;
- all vehicles have unit length in these baselines;
- priority cases: one PV, two PVs in one lane near gap `G`, and two PVs near
  gap `2G`;
- matched rule penetration rates: 5%, 10%, and 20% of non-PV traffic.

Matched penetration sets use one seeded ordering per paired case, so they are
nested: `5% ⊂ 10% ⊂ 20%`.

PV roles are tagged only after the HDV-only warm-up. Therefore density,
positions, velocities, and the stabilized snapshot are identical at the start
of every paired measurement.

## Controllers

`baseline1_hdv` has no yielding rule. Apart from the explicitly tagged PVs,
the flow is HDV-only.

`baseline2_all_hdv_yield` enables the yielding rule for every non-PV vehicle.
When a PV is within `detection_distance` cells behind a vehicle in the same
lane, the vehicle requests the safe adjacent lane with the largest minimum
front/back clearance. An unsuccessful or unsafe request is not emitted, so the
vehicle defers to native S-NFS behavior. Successfully moved vehicles enter a
per-vehicle cooldown.

`baseline2_matched_*` uses the identical rule but only for a seeded subset of
non-PV vehicles. Those vehicles are tagged as rule-based AVs while remaining
uncontrolled. This is the fair comparison for later MARL penetration rates.

## Pairing and metrics

For each `(density, replication, PV case)`, all modes receive a copy of the
same warmed state and the same measurement RNG seed. Output includes:

- fundamental-diagram flow for all and background traffic;
- mean speeds for all traffic, background traffic, and PVs;
- the worst per-PV mean speed;
- stopped fractions and lane-change rates;
- PV normalized time loss, distance, completed laps, and lap time;
- rule detections, requests, applied/rejected requests, unsafe deferrals, and
  cooldown suppressions.

The paired summary marks an `effective_region` only when there are at least
`min_effective_pairs`, the paired bootstrap 95% lower bound for PV speed gain
is positive, and mean background-flow loss is at most 5%.

## Commands

Fast one-PV pilot:

```bash
PYTHONPATH=src python research/priority_baselines.py \
  --priority-scenarios one \
  --densities 0.05,0.10,0.15,0.20,0.25,0.30 \
  --runs 5 --warmup-steps 1000 --measure-steps 2000 \
  --backend auto --workers 8 \
  --out-dir priority_baseline_results/pilot_one_pv
```

Full configured sweep:

```bash
PYTHONPATH=src python research/priority_baselines.py \
  --road-length 1000 --num-lanes 4 \
  --priority-scenarios one,two-g,two-2g \
  --matched-fractions 0.05,0.10,0.20 \
  --density-min 0.02 --density-max 0.90 --density-count 30 \
  --runs 10 --warmup-steps 1000 --measure-steps 2000 \
  --backend auto --workers 24 --no-validate
```

The runner writes raw runs, grouped summaries, raw paired deltas, aggregated
paired inference, a JSON manifest, and a four-panel plot.

`--no-validate` disables the expensive post-step runtime-invariant pass, but
not the simulator's normal state/occupancy safety checks. Use it for the full
sweep only after the test suite and a validated smoke run pass for the commit.
