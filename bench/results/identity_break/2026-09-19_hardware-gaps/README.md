# Hardware verification gap closure

The tracked missing-part ledger is closed: **NVIDIA 648/648**, **AMD 630/630**,
with **zero numerical differences**. `closure-ledger.json` maps every one of
those 1,278 fixture-parts to a retained raw capture, its SHA256, and the expected
reference hash. Previously compatible captures were reused; the full matrix
was not rerun.

Admission checks require IDENTICAL mode, at least two repeats, matching fixture
and heldout digests, current lane revision, the existing reference hashes, and
the prescribed batch protocol for batch witnesses. NVIDIA includes the 18
previously missing single-device resampling parts in addition to its two-device
parts. AMD covers the corresponding 630 two-device parts. This is the finite
requested fixture set, not a claim about every possible model, input or device.

The last AMD job captured only the three remaining resampling fixtures. Its raw
result and job summary are `amd-resample-last-two.json` and
`amd-resample-last-summary.json`. Both final remaining ledgers report zero.
All rented pods were deleted and verified absent; the last pod's lifecycle
receipt is `final-resource-cleanup.json`.

## Other closed configurations

- Eleven newer parallel lanes: nine fixtures and two repeats, 378 numerical
  parts on each vendor. Older lanes' missing model/batch witnesses were filled
  without repeating their already compatible captures.
- Distributed execution: 30 numerical cases and 10 transport controls per vendor.
- Cross-validation: 12 one/two/reversed-device runs and eight comparator controls
  per vendor. Canonical NVIDIA/AMD receipt comparisons match.
- Loaded language models: 24 existing tiny checkpoint cases, both layer layouts
  `(0,1)` and `(1,0)`, 144 baseline parts per layout on each vendor. Distinct
  worker devices and completed layer-run RPCs were observed.

These numerical/placement receipts do not claim external kernel execution
traces, large external checkpoints, long contexts, or additional CPU training
ports. Those are separate scopes.

## Fixes and candidate provenance

`c3b5783c` fixed JSON serialization of verifier environment receipts; all 123
existing Linux native/runtime files were unchanged. `1aa581748` fixed NumPy
conversion of empty Arrays; all six previously failed AMD radius fixtures now
match all 18 train/infer/model reference parts. Source transport stopped creating
untracked bytecode during manifest validation. Distinct private wheel hashes,
raw captures, comparisons and scoped validation are retained here.

The separate libm-removal candidate is documented in
`../../portable_math/2026-09-19/README.md`. Its small Apple/AMD smoke does not
relabel these prior full-lane captures as new runtime qualification. Published
0.8.8 remains unchanged; no release tag or PyPI artifact was replaced.

`HISTORY.md` preserves earlier progress reports and their then-current counts.
