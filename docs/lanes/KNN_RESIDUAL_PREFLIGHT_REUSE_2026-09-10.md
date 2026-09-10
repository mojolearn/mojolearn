# Next kNN experiment: reuse Apple exponent admission

Current NVIDIA defaults already include eight query rows per distance thread
and query batches of512 (within the measured index range). The earlier Apple
+39% request cost compares exact repair against disabled repair before accepted
whole-chain preflight; it is historical, not the current residual overhead.

A concrete untried candidate is to compute the minimum nonzero exponent once
per query/index vector, alongside the existing norm preparation. Current
`neighbors/checks/pinned_distance_tile.mojo::_rt_accumulate_tile` rereads the
complete feature chain to reconstruct those minima for every register tile.
Each query vector repeats across index-column tiles, and every index vector
repeats across query-row tiles. Metadata would remove this duplicate feature
scan while leaving every distance FMA and final selector unchanged.

The exact admission rule stays `q_min + y_min >= 151`. Metadata must ignore
exponent-zero inputs (the numerical load flushes them), initialize empty minima
to255 and preserve the existing treatment of nonfinite exponent255. A thread
reduces minima across its clamped row/column indices. Unsafe tiles retain the
integer repair. Compute metadata from each call's actual buffers; pointer-only
caching is invalid because callers can mutate inputs in place. Keep metadata
alive through all index/query tiles and all asynchronous launches. No production
candidate is implemented yet: extra preparation launches/buffers might outweigh
the saved scans on small requests.

First run the authored phase diagnostic under root's device/build lock:

```sh
bash -n tools/knn_residual_phase_probe.sh
bash tools/knn_residual_phase_probe.sh apple /fresh/absolute/apple-knn-phases
bash tools/knn_residual_phase_probe.sh nvidia /fresh/absolute/nvidia-knn-phases
```

Run only the host-appropriate device command. It compares current safe default
against safe NO_PREFLIGHT on Apple, and legacy256-query batches on NVIDIA.
Neither arm disables arithmetic repair. Three rounds follow two warmups, with
reversed arm order in a second pass; four shapes include the public k10/k15
requests, low feature count and ragged tails. Every paired complete output dump
must compare equal. Existing phase hooks report distance, selection, merge,
transpose and norm work. Synchronization perturbs totals: these measurements
are diagnostics, not new qualified opponent prices or end-to-end speed claims.
The script has not been run by its author.

Only if distance/preflight remains material should the metadata candidate be
implemented and gated against the strict396584-triple oracle, actual production
tile oracle and layout fixtures, including all-zero/subnormal vectors, exponent
threshold150/151, ragged dimensions, and an in-place mutation between calls.
The admitted tile decision must match the old per-tile preflight for every
fixture, in addition to output identity.

Do not repeat rejected partitioned selectors, deeper16/32 scans, two-word warp
redux, composite-key value decode or four-feature cached preflight without new
evidence. These are documented in the retained selector/residual results.

Root continuation: the small Apple phase smoke passes two reversed-order full-output comparisons. Large drift and ranking reversal prevent a speed conclusion. The script now enforces build serialization/two jobs; MOJOLEARN_KNN_PHASE_SMOKE=1 selects only the small fixture. Evidence: bench/results/knn_phase_2026-09-10/.

## Large-target continuation

`MOJOLEARN_KNN_PROBE_MODE=price` now disables phase instrumentation and runs
five timed rounds after two warmups, in both arm orders, on the complete
400k/4k/d32 k10/k15 grid plus low-feature and ragged controls. This mode
refuses `MOJOLEARN_KNN_PHASE_SMOKE=1`. Summarize retained output with:

```sh
MOJOLEARN_KNN_PROBE_MODE=price bash tools/knn_residual_phase_probe.sh apple /fresh/absolute/apple-knn-prices
python tools/knn_probe_summary.py /fresh/absolute/apple-knn-prices > /fresh/absolute/apple-knn-prices/summary.json
```

The summary requires all large shapes and both orders, verifies complete output
hashes and sample counts, reports within-run drift, and never automatically
approves a default. Historical Apple preflight acceptance used 400k index rows
and 1000 queries; it did not establish the full 4000-query target. The audit
records that scope explicitly in `PERFORMANCE_GATE_AUDIT_2026-09-10.md`.
