# Model weights — Physica A v1.1

[The catalogue](<catalog.json>) maps scientific model identities to repository-relative NPZ/JSON files, training lines, checkpoint iterations, seeds, joint penalty scales and selection rules.

- **17 selected bundles** reproduce the policies used in the reported comparisons.
- **15 terminal bundles** retain every training line at iteration 1221 / 20,004,864 sampled steps.
- Four bundles belong to both sets: **28 unique bundles from 15 independent trainings**, about 40.7 MB including manifests.

Terminal weights are not replacements for the selected policies. In particular, the missing Graph S₀.₉₅ line remains infeasible; extra terminal files do not change its support from 2/3 or create additional independent trainings.

Bundles are the original numerical inference arrays and immutable scientific manifests, copied with SHA-256 verification. They are not full RLlib training checkpoints: optimizer state, unfinished episodes, sampler/RNG state and runtime logs are intentionally absent. Repeating training uses the supplied source/configuration; exact training continuation cannot be reconstructed from an inference NPZ alone.

Historical source paths inside immutable manifests are provenance identifiers, not required filesystem locations. Runtime paths are supplied through the catalogue/transport mapping. No models were evaluated or tested during release preparation.
