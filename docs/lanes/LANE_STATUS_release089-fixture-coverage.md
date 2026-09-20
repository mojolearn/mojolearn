# CPU/GPU fixture evidence for release 0.8.9

The scoped admission of 75 previously incomplete GPU lanes leaves zero missing CPU-applicable numeric cell parts among non-parallel lanes. The audit covers all nine fixtures and every recorded numeric part, including current batch, gradient, ragged, replay and full-step protocols; it does not treat N/A declarations as numerical results.

The table contains 7,452 exact CPU/GPU numeric pairs across 214 non-parallel numeric lanes. Eighteen GPU model-byte parts have an explicit CPU role exception: categorical and tensor CTR tables each load the GPU-saved model on CPU instead of writing a CPU model (nine fixtures each). These existing roles are enumerated in `release089-cpu-gpu-fixture-audit.json`. All 273 harness lanes remain in the table, and the table reports zero conflicts.

Fresh GPU collection added 504 stable Metal cells and 171 stable AMD cells. Their 2,601 numeric parts with CPU references matched exactly. Source/build provenance is preserved in the respective `2026-09-20_apple-gap-full-parts` and `2026-09-20_amd-neural-full-parts` record directories. The scoped regeneration retained the separately admitted CPU linalg fixtures and saved-host model evidence.

The machine-readable audit decodes each reference column (an integer points to the entry reference; a pair carries an explicit value), requires an equal GPU hash for each numeric CPU part, and separately lists recorded CPU N/A roles. It also reports parallel-lane counts without claiming that GPU-only parallel APIs have CPU implementations. This evidence does not assert every GPU vendor ran every cell, nor does it replace exact-wheel release qualification.
