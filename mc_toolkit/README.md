# Frozen Monte Carlo and statistics core

This package provides numerical Graph/NodeMLP/GridCNN inference, paired traffic episodes, deterministic task plans, collection, selection, local diagnostics and statistical analysis for the Physica A experiment.

The two source packages are intentionally preserved separately from the root training package. Their scientific source digest is `3ff57f9e84d10390a6a184358332ea12bdf400011f2b6008d6d767030678492e`. Do not merge the vendored simulator with the larger training simulator: package-relative filenames and bytes are part of the scientific identity.

The repository release is **v1.1**; the frozen MC package retains its original **0.1.0** version and source bytes. Historical wheel hashes do not certify a new wheel build.

Use a separate MC environment and the [reproduction guide](<../reproduction/README.md>). From the repository root, the native entrypoint is:

```sh
PYTHONPATH=mc_toolkit/src python -m physica_mc COMMAND --request REQUEST.json --output RESPONSE.json
```

Commands include `plan`, `episode`, `run`, `collect`, `analyze`, `export`, `select` and `freeze`. Requests must supply explicit plans, paths and execution budgets; there is no implicit campaign launch. Numerical bundles do not require Ray. Legacy checkpoint conversion requires Ray and explicit trust because pickle can execute code.

The retained [publication operations](<operations/paper_outputs/cli.py>) support extraction, statistical summaries, rendering and tables. Private cluster campaign scheduling and local runtime histories are not included. The supplied test sources were **not run** during release preparation.
