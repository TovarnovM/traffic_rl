# Final Physica A results

This is the current main-paper and supplementary result set, not the older full diagnostic gallery or the separate/joint RL study.

- [Figures](<figures/>): **38 original final PDFs** — eight main and 30 supplementary. [The figure index](<FIGURES.json>) records hashes and generator mappings. Reference filenames omit the publication's `revision_` prefix; regenerated outputs retain the publication names.
- [Tables](<tables/>): five main and five supplementary tables, plus the complete 42-domain blocks; [the table index](<tables/TABLES.json>) identifies them. Native CSV/TeX fragments are included. TeX table sources use the original macros/equation references, as explained in the [export guide](<../../tools/publication/README.md>); they are not a full article document.
- [Compact scientific inputs](<inputs/README.md>): 136,626 global aggregate points; 109,536 unique local diagnostic rows; all 12 frozen paired trace illustrations; validation curves; exact table data and separate historical E1 summaries. Large repeated trace provenance is stored once, and CSV payloads are compressed without changing numerical strings.

## Regeneration

From the repository root, after installing [figure requirements](<../../requirements-figures.txt>):

```sh
python tools/publication/generate.py --output ../physica-a-v1.1-redraw
```

The destination must be new. This redraws 37 numerical figures, copies the original vector schematic master and exports the tables; it does not overwrite these reference outputs. The master is an editable PDF, not a claimed reconstruction from an unavailable original design script.

## Interpretation

The 30,690-episode count belongs to the original completed experiment; raw episodes are not included. Local diagnostics cover 36 scopes / 7,770 distinct physical episodes, without counting logical aliases as new replicates. `R` is the mean of paired background **flow** quotients, not a speed ratio or ratio of means. Stored NA values, support and conditional confidence intervals retain their original meaning. Historical E1 multiplicity counts are a separate older cohort, not new-test FDR results.

File identity, compact projections and scientific source hashes were checked during packaging. Generators, plots, tests and calculations were **not executed** for this release.
