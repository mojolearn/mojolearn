# Next-release parallel reference admission

All 25 selected Apple lanes completed all nine fixtures with full parts and one repeat. The 24-lane source445 column contributed 819 numeric comparisons and the corrected sourcebcdf causal-model column contributed 18; all 837 matched CPU reference values. Both used the declared existing source819 native binaries. Full provenance is preserved with the raw records. Fresh CPU source445 records provide current ARIMA and RBF model bytes.

The historical one-device cross-vendor baseline was **810 incomplete part-cells / 1,539 vendor values**. The current harness contract before this admission had **846 / 1,647**: nine newly required RBF model parts plus 27 previously agreeing ARIMA train/infer/batch parts invalidated by the new lane revision. The fresh ARIMA column carries `trend-metadata-1`; no old AMD or NVIDIA ARIMA record is relabeled as current.

After scoped admission there are **441 incomplete numeric part-cells / 504 vendor values**: Apple **0**, AMD **378** (351 missing and 27 stale N/A), NVIDIA **126** (90 missing and 36 stale N/A). Across all 59 parallel lanes and nine fixtures, 1,494 numeric part-cells agree on all three GPU vendors and 2,844 parts are declared inapplicable. Current Apple numeric requirements are complete; this does not mean AMD/NVIDIA or physical two-device execution is complete.

Regeneration also admitted 387 numeric and 18 N/A AMD values from the already committed `2026-09-19_par-lane-amd-class/...par-one.json` column. This is recovery of admissible existing evidence, not new AMD execution. The reused record names source `331cdfaa32df4927dc3c5adc9a4c014820200289` and package version 0.8.8. Its current admission predicates, all nine input/held-out hashes, source commit object, and every contributed lane revision were explicitly checked. It supplies no current ARIMA evidence. `reused-amd-validation.json` retains that verification and its historical native fingerprints; those binaries are not claimed to qualify the next release wheel. The before/after machine-readable reports preserve the remaining collection plans and reasons.

The standard scoped `verify --all --batch-checks --emit-reference --reference-table` command read the full `bench/results/identity_break` tree plus only the completed `bench/results/parallel-next/apple-4459379a4/all-nine.json`. It selected the union of the 24-lane column and `par-causal-lm`. The failed initial Apple base run is retained for diagnosis and was not used as an admission input. All unselected cells, including their values and record provenance, were checked unchanged after record-index normalization. The resulting table has zero conflicts.

Table SHA256: `c8bd574dcecc6bded270659bcaeeabd0bdff28bfd56b5425beaae2cf574480c5`.

This update is for the next release. It changes no published 0.8.9 wheel, historical report, or DOI. No two-GPU qualification is inferred from this one-device reference evidence.

Actionable per-vendor lane/fixture/part requirements are saved in `amd-collection-plan.json` and `nvidia-collection-plan.json`; the full audit also preserves the reasons and CPU targets for each gap.

The reused AMD source0.8.8/`331cdfaa32df4927dc3c5adc9a4c014820200289` binaries and Apple source819 native binaries provide reference evidence, not qualification of a new wheel. The exact next-release wheel still requires installed-wheel validation before publication.
