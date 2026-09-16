# lane/saved-model-reference-gaps

The audit in `docs/lanes/LANE_STATUS_lane-expose-inference-surface.md` and the
registry note at `python/mojolearn/host_surface.py` (`SAVED_MODEL_INFERENCE_OWED`)
say DBSCAN, agglomerative and spectral PREDICT are shipped with their GPU
reference recordings still owed, and that spectral's precomputed-affinity
variant is pending. This file establishes what is actually missing, from the
registry read BY IMPORT and from the records on main, before anything is
recorded.

## How the gap was measured

* The lane registry was read by import, never by grep:
  `python3 -c "import identity_break; sorted(identity_break.LANES)"` reads
  **212** lanes at `06295a5da`. Twelve of them match dbscan, agglomerative or
  spectral.
* Every `bench/results/identity_break/**/*.json` on main was walked and each
  `cells[lane/fixture]` part was classified by whether the recorded value is a
  real digest.

### The first scan was wrong, and the way it was wrong is the point

The first pass accepted a part as recorded when its value was a list of
16-character strings. `n/a:transductive` is **exactly sixteen characters**, so
that scan reported NVIDIA, AMD and Apple columns for every one of these lanes.
It was a check that could not fail. The kept copy is
`~/mojolearn-evidence/saved-model-reference-gaps/gap-scan-LOOSE-WRONG.txt`, and
the corrected scan, which requires `^[0-9a-f]{16}$`, is
`gap-scan-strict.txt` beside it. The corrected scan's first act was to print
the matches rather than a count, which is how the placeholder was seen.

The same false positive is in the record itself: at commit `1eea14f80`
(`bench/results/identity_break/2026-09-14_166-lanes/`) every one of these lanes
carries `"infer": ["n/a:transductive", "n/a:transductive"]` and
`"model": ["n/a:no-save", "n/a:no-save"]` on all three GPU columns, because
`predict` did not exist until 2026-09-15.

## The true gap

No GPU or CPU column holds a real `infer`, `model` or `batch` digest for these
lanes anywhere except the two 2026-09-15 lane records. What exists:

| lane | part | Apple/Metal | NVIDIA | AMD | CPU |
|---|---|---|---|---|---|
| `dbscan` | infer, model, batch | 4 fixtures | **none** | none | 4 fixtures |
| `dbscan-brute-l1` | infer, model, batch | 4 fixtures | **none** | none | 4 fixtures |
| `dbscan-weighted` | infer, model, batch | 4 fixtures | **none** | none | 4 fixtures |
| `agglomerative` | infer, model, batch | 4 fixtures | **none** | none | 4 fixtures |
| `spectral` | infer, model, batch | 2 fixtures | **none** | none | 9 fixtures |
| `spectral-precomputed` | infer, model, batch | 2 fixtures | **none** | none | 9 fixtures |

* Apple/Metal, transductive lanes: `bench/results/identity_break/2026-09-15_transductive-predict/apple-m4.json`
  at `ef1647619`, fixtures base, ties, dupes, denormal. Its `vendor` field reads
  `arm64` rather than `apple-m4`, which is why a vendor-keyed search misses it;
  that record's README names it the Metal column.
* Apple/Metal, spectral lanes: `bench/results/identity_break/2026-09-15_spectral-predict/metal/apple-m4.json`
  at `b886dbc97`, fixtures base and ties. It sits one directory deeper than the
  other columns, so a non-recursive glob misses it.
* CPU: `2026-09-15_transductive-predict/cpu-x86.json` (`bd9ef2eee`) and
  `2026-09-15_spectral-predict/cpu-x86.json` (`0a6957015`).

**So the owed column is NVIDIA, and only NVIDIA.** AMD is owed too but is left
alone entirely by Andrew's standing instruction.

## Two things the audit got wrong

1. **Spectral's precomputed-affinity variant is NOT pending.** It is
   implemented and measured. `identity_break.LANES` registers
   `spectral-precomputed`; `python/mojolearn/_spectral_impl.py` accepts
   `affinity="precomputed"` in `fit`, in `predict` (an `(n_new, n_train)`
   matrix) and in `save`/`load`, which write and read the `affinity` scalar;
   `python/mojolearn/tests/test_spectral_predict.py` covers it. Both its CPU
   column (9 fixtures) and its Metal column (2 fixtures) carry real digests.
   Nothing is owed there but the NVIDIA column.
2. **The `owed.json` files in those two records are stale.** Both were written
   before their own Metal columns were taken and still list `apple-m4` as
   missing.

## What IS missing besides the column

`host_surface.inference_lanes()` and `tools/classical_host_gate.py`'s `LANES`
agree exactly, at 79 lanes, and none of the four predict lanes is in either.
So there is no `bench/results/classical_host/` recording for them, and there
cannot be one until the gate declares them. That is a code gap, not just a
recording gap, and it is this lane's first change.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
