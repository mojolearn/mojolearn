# Installed CV verifier checkpoint

2026-09-18, branch lane/cv-wheel-verifier, based on 4bb628f8e.

The CV capture implementation and picklable worker scorer now ship as
`mojolearn._verify_parallel_cv` and `mojolearn._parallel_cv_witness`.
The two tools entry points are thin compatibility wrappers. Root integrates
`python -m mojolearn verify-cross-validation` with `main(argv=None)`.

Capture flags: `--devices 0,1 --out NEW_DIRECTORY --require-installed`;
optional `--require-backend cuda|hip`. Offline comparison: `--compare LEFT RIGHT`
where each path names a report.json. The output directory is never overwritten.
Each completed run is atomically checkpointed with embedded fold receipts and
saved-model witnesses; partial failures retain the completed evidence.

Installed capture requires a noneditable matching distribution and verifies
actual Python/native SHA256 against wheel RECORD. Source profiles hash the
shipped CV runner/scorer, model-selection, pool/worker, GPU witness and reused
RECORD verifier. Checkout, guarded archive, and installed wheel identities
remain distinct; an installed wheel does not manufacture a source commit.

Numerics remain classifier/regressor, five uneven folds, one/two/reversed GPU
layouts, two repeats, model/reload/prediction/loss/score bytes and four comparator
negative controls per estimator. Offline receipts require all twelve runs,
eight controls, exact source/native/fixture profiles, distinct GPU placement,
and matching model replay. Cross-vendor comparison ignores native binary hashes
but requires matching source and all numerical witnesses.

Validation: 38 focused source, scorer, negative-control and complete-receipt
unit tests passed. These tests are synthetic and do not supply hardware proof.
Actual installed-wheel two-GPU capture, physical execution tracing and native
arithmetic sabotage remain owed. The prior NVIDIA pod disappeared before CV
capture; no new rental is authorized under the latest user constraint.
