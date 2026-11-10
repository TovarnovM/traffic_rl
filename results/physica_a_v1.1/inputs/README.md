# Compact current Physica A publication inputs

Only CURRENT Physica A main/SI publication data are included. These inputs do not
serve the separate/joint RL articles or a review. The original read-only source
was `traffic_rl_physica_a_revision_r1`; historical path strings below identify
sources, and are never opened by release consumers.

- `global-data/points.csv.gz`: all 136,626 aggregated points, compressed without
  changing the original plaintext bytes. `manifest.json` is the original global
  manifest. No global episodes file is included. The recorded 30,690 episode
  count is source provenance, not a claim that raw episodes were revalidated.
- `local/points.csv.gz`: 109,536 unique exact rows from seven final sidecars:
  primary_summary, events_summary, context_counts, spatial_per_lane,
  temporal_summary, onset_summary and supp_window_sensitivity. Deduplication key
  is `(source_file, integer source_row)`; duplicate row dictionaries must agree.
  Rows are ordered by that key. All 42 columns are retained, including original
  native coordinates/hash and every value, SD, NA reason, status, unit, phase,
  support and identity string. No rounding, NA filling, pooling or reduction.
- `local/projection.csv.gz`: 109,584 sidecar-to-native coordinate mappings (48
  duplicate appearances removed from the compact points). CSV record numbers are
  one-based, header = 1. Each sidecar source hash/header/count is in provenance.
- `local/episode_identity.csv.gz`: 8,640 exact scope-level identity records,
  projected to scope ID, source row, task ID, group ID, replicate and native code
  digest. Their union is 7,770 distinct physical episodes across 36 scopes;
  aliases across scopes do not add independent observations. No measured episode
  metrics or operational logs are shipped in this projection.
- `local/provenance.json`: shared frozen snapshot/freeze/binding/native-code
  identity; original file hashes; per-scope scientific manifest fields, coverage
  rows, group roster and original counts. Coverage row position + 2 is its native
  CSV coordinate. Profiles are copied exactly as `analysis-full12.json` and
  `analysis-mechanism.json`; both original-file and canonical JSON hashes remain
  bound. Progress was projected to scientific identity/completion fields only;
  jobs, budget, execution, timing, resource and log metadata are omitted.
- `recorded/`: 12 frozen Graph/H2 pairs, 12,006 rows each, 144,072 total; six
  series per pair (two methods × speed/lane/distance), every native step 0..2000.
  Native scientific columns and row/episode mapping remain strings. Six repeated
  provenance fields are stored once in the pair JSON and restored by the loader.
  Original native CSV/JSON hashes and final sidecar CSV hash bind each projection.
  `global-data/illustrations.json` retains all twelve frozen selection proofs;
  the n=90 selection pool is not two-episode population evidence. No pair is
  regenerated or selected anew.
- `validation/`: exact native candidate CSV/JSON for scales 0.5, 1.0 and 2.0
  (39, 121 and 39 plotted rows). The main sidecar stores fixed controls with
  categorical x and preserves the original source_x/source_x_label.
- `tables/`: only table bodies/captions/data from the canonical main and SI,
  numerical CSV/TeX fragments and extraction coordinates/hashes. The recorded
  figure wrapper is intentionally excluded. `historical/` contains exact compact
  E1 audit CSVs and report_facts JSON, with no editorial Markdown.
- `masters/method_schematic.pdf`: original editable vector PDF master, copied
  as an explicit exception to numerical redraw, without a generator-match claim.

`packing.json` describes projections, row counts and compressed/plaintext SHA256.
Gzip uses level 9, mtime=0 and an empty filename. Projected CSVs use UTF-8 and
stdlib DictWriter CRLF, original column order and unmodified strings; their
payload hashes describe the projection, not the original source file bytes.
Global points decompress to the original source file hash exactly. Included
uncompressed files and compressed payloads are bound by `inventory.json`, whose
hash is pinned in `tools/publication/inputs.py`.

At regeneration, consumers validate included hashes, 36-scope coverage, original
source coordinates, complete branch/grid contracts, profile/event definitions,
physical identities and 30-environment support, the 7,770-episode union, frozen
pair identity/selection/units and all 2,001 trace steps. Original source hashes
identify excluded native files; they do **not** require or claim runtime access
to those files, raw archives or private caches. Provenance verifies a frozen
projection, not a fresh replay of its underlying experiment.

Native empty onset remains value='', na_reason='', n_defined=0, support=0.0 and
status=missing when that exception occurs. Temporal raw covariance zero remains
separate from undefined normalized covariance. No new CI or family pooling is
introduced. Original scientific validation branches remain enabled.

No generator, scientific module, test, build or plot was run during preparation.
Only stdlib packaging, AST inspection and metadata consistency were performed.
