# Catalog matrix geometry screen

Tools-only descendant of compiled source
`9f1a1657c6e054573a495425e223fd73e0c174db`. No kernel changes, builds,
local timing, or queue edits. Manager review precedes any M3 execution.

The hard-pinned original quality report SHA256 is
`f2bcde131ffea5ff54cea85920c3a49ec7888b5d4b2824e2077189b50aa973a0`.
It is an unrestricted **FAIL**: all ten candidates fail `nt-vector`.
All eleven other fixtures PASS and are byte-identical to incumbent.
The harness validates all twelve rows, exact metadata/counters, recomputes
status from metrics, checks all eleven matrix output hashes against G0,
and requires precisely that sole failure. This is matrix-only eligibility
(N >= 2), not a retroactive unrestricted PASS. Actual shared-route G1/G5
quality with its predeclared vector fallback is separate evidence.

Predeclared cases, dimensions explicitly **M, N, K**:

| Case | M | N | K | Operation |
|---|---:|---:|---:|---|
| dense-nn | 2048 | 512 | 512 | A B |
| square-nn | 1024 | 1024 | 1024 | A B |
| tall-projection-nn | 32768 | 64 | 220 | A B |
| lowwidth-kmeans-nt | 32768 | 8 | 220 | A B^T |
| odd-nt | 4097 | 71 | 221 | A B^T |
| gram-nt | 1024 | 1024 | 220 | A A^T, aliased input |

Each shape runs G0 through G10 exactly once in fresh sequential processes:
66 scored calls, zero warmups, zero opponents, no opportunistic repeats.
Each process verifies counters start at zero and only its selected variant
increments to one. Imports/input preparation are outside the score. The
scored binding call includes fresh device allocation, uploads, GEMM,
download, synchronization and cleanup. Full host output sum immediately
following is the first read; both intervals and their sum are reported.
This measures cold standalone calls, not steady-state resident GEMM or
estimator throughput. Fresh processes avoid giving later variants a shared
initialized context advantage, but cannot remove temporal machine drift.
At n=1 per arm small differences remain inconclusive.

The compiled-source ancestry, unchanged compiled files, manifest A/B hashes,
actual imported B.so and quality report are checked in every child process.
Variant 0 is incumbent inside the probe B binary; variants 1..10 select
candidate geometry. Manifest A.so is verified but is not the enabled probe.
No package installation is changed. Partial run logs remain evidence and
tags cannot be reused. There is no automatic promotion or timing retry.

After manager review, use the M3 serial queue with REPORT_PATH pointing to
the original report (not a modified report):

```
CMD lane/apple-fast-catalog-matrix-timing catalog-matrix-t-v1 MOJOLEARN_NUMERIC_MODE=fast ~/board-0834/cache/venv/bin/python tools/catalog_gemm_matrix_timing.py 9f1a1657c6e054573a495425e223fd73e0c174db catalog-matrix-t-v1 REPORT_PATH f2bcde131ffea5ff54cea85920c3a49ec7888b5d4b2824e2077189b50aa973a0
```

Result: `~/mq/out/catalog-matrix-t-v1-timing/report.json`. Geometry selection
only; any production candidate still needs actual estimator quality and
call-plus-first-read timing. Shape-scale accuracy is not established by the
small standalone quality fixtures alone. No result clears opponent quality.
