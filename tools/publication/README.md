# CURRENT Physica A — LOCAL v1.1

These consumers use only the included derived publication inputs and the release's
`mc_toolkit/operations/paper_outputs/{style,render}.py`. Paths are resolved from
`__file__`, independent of the working directory. No source worktree, raw archive,
cache, episode replay, policy inference, training, or journal source is required.
The frozen MC packages and shared renderer remain owned by the release maintainer.

From the repository root, with NumPy and Matplotlib installed using the
[figure requirements](<../../requirements-figures.txt>):

```sh
python tools/publication/generate.py --output ../physica-a-v1.1-redraw
```

The output path must
be **new** and its parent must exist. The command refuses existing destinations.
It writes `global/`, `local/`, `validation/`, manuscript-named `figures/`, and
`tables/`. `COMPLETE.json` is written last; a failure leaves its incomplete
output directory for inspection. Retry with a different new output path.
Nothing is written into the release's preserved figures or tables by default.

The 38 manuscript/SI figure slots comprise 12 global, 24 local (including the
12 frozen paired traces), one main validation figure, and one schematic.
**37 are numerical redraws; `method_schematic.pdf` is copied from the included
original vector PDF master.** No exact editable source generator for that master
was found among the inspected publication scripts. The PDF can be edited with a
vector editor. The generic `render_conceptual()` has a conceptual layout, but its
identity with the original master has not been established; it is deliberately
not substituted. No PDF byte-match claim is made for regenerated figures.
`figure_mapping.json` records the original numerical figure names and hashes.

Tables alone (stdlib; no plotter imported or rendering dependencies needed):

```sh
python tools/publication/export_tables.py --output ../physica-a-v1.1-tables
```

Exports cover five main tables, five SI tables and the unnumbered 42-domain
blocks. CSV-backed effect, background-audit, recorded-pair and historical E1
counts are formatted from included rows. Other table data/captions/bodies are
exact table-only extracts; `extraction_manifest.json` records source hashes and
line coordinates. Table `\input` fragments are expanded in standalone outputs.
No article prose outside tables, journal response, class or bibliography is
included. TeX fragments retain native macros and equation references: a host
TeX document supplies its normal `booktabs`, `P` column, `\PV`, `\eqref`, etc.
No TeX compilation, dependency installation or PDF table build is performed.

The original plotting functions, policy rosters, figure arithmetic, display
precision, support and missing-value rules are preserved. Loader adaptations
validate compact provenance instead of opening excluded native scopes. Existing
`mean54` and window point-difference display expressions remain in the original
plotting definitions; no new statistical estimator or uncertainty is introduced.
The main validation loader reverses only the sidecar's documented fixed-control
x-coordinate display projection before reusing `render._curve_pages`.

See the [input provenance README](../../results/physica_a_v1.1/inputs/README.md)
for coverage, exact-string preservation and the limits of provenance validation.
Implementation review used source inspection, stdlib AST parsing, and metadata /
copy / projection consistency checks only. **No runtime tests, builds, plots,
scientific calculations, inference, training or scientific module imports were
run.** Rendering and visual equivalence remain unverified at runtime.
