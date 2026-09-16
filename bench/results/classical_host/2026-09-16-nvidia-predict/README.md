# DBSCAN, AgglomerativeClustering and SpectralClustering predict, from a saved model (2026-09-16)

Branch `lane/saved-model-reference-gaps`. Four lanes, nine fixtures each, 36
fixture directories. DEVIATION 2740 (`dbscan`, `agglomerative`) and
DEVIATION 2860 (`spectral`, `spectral-precomputed`) are new capability;
neither cuML nor scikit-learn labels new rows under these estimators.

Until this recording existed, `mojolearn.host_model()` dispatched all four
formats and no gate covered any of them, so a user could call the predict and
had nothing to check it against.

## Why it did not exist before

`tools/classical_host_gate.py record` had been dead on main since
`--lane-rule-only` was added (lane/ties-sabotage, 2026-09-15). That flag is
declared on the `check` subparser only and `main()` read it unguarded, so every
`record` run raised `AttributeError: 'Namespace' object has no attribute
'lane_rule_only'` before `do_record` ran a line. It was found by a rented GPU
box refusing, and fixed with `getattr`.

## Where each column ran

* **The recording.** One RunPod NVIDIA A100-SXM4-80GB, `sm_80`, commit
  `9af299bc7`. `record` fits exactly what the `identity_break` lane fits,
  requires its own probe hash to equal that lane's `infer` cell, saves the
  model, reloads it through the class's own `load`, and requires the reload to
  predict the same bits before writing `expected.json`.
* **The CPU host route, twice, on two architectures.**
  * x86-64 on the recording box itself: `check.json`, `check.log`.
  * arm64 on the M4, one core, through the slot helper:
    `check.arm64.json`, `check.arm64.log`.
  Both read `gate verdict IDENTICAL (36 fixtures, exit 0)`.
* **The sabotage arm, on both architectures, on EVERY cell.** A second host set
  built with `-D MOJOLEARN_TRANSDUCTIVE_PREDICT_SABOTAGE=1` (estimators) and
  `-D MOJOLEARN_SPECTRAL_PREDICT_SABOTAGE=1` (metrics), checked with
  `--expect-mismatch --every-fixture`: `EXPECTED MISMATCH SEEN`, and the
  `unmoved` list is EMPTY in `sabotage.every-fixture.json`,
  `sabotage.every-lane.json` and `sabotage.arm64.json`. Every one of the 36
  cells was watched to FAIL before its PASS was believed.

## The identity columns behind these cells

`bench/results/identity_break/2026-09-16_predict-nvidia/`: two NVIDIA columns
(A100 `sm_80`, RTX 2000 Ada `sm_89`) over six lanes and nine fixtures, plus an
Apple M4 Metal column for `spectral` and `spectral-precomputed` at the
published 512-row size. `spectral`'s 2026-09-15 Apple and CPU columns were
taken at 2000 rows and the diff tool's `LANE_REVISIONS` already refuses them.

## What is owed

The AMD recording, at the next release record. AMD was left alone entirely for
this lane by Andrew's standing instruction.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
