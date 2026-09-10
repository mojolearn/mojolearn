# IDENTICAL kNN selector, September 9 continuation

Final safe source: `729ffa55` on `lane/knn-selector`. No FAST, DETERMINISTIC,
or tree code changes. Final NVIDIA code retains the software round-then-FTZ
FMA seam. Hardware FTZ FMA timings are rejected evidence, never final prices.

## Accepted changes and measured result

Butterfly composite-key reduction, eight independent candidate loads followed
by the original ordered insertions, and constant-k specialization for k=10/15.
The constant-k specialization defaults on NVIDIA only; the generic selector
remains available with `MOJOLEARN_KNN_IDENTICAL_GENERIC_K` and other k values
use it. Forced Apple specialization was checked for bit equality.

H100 80GB HBM3, driver 580.126.09, same pod, 32 features, dyadic-v1 fixture,
seven timed rounds after two warmups per request/device arm. Old arm is
`f2cff18e` with `-D MOJOLEARN_KNN_IDENTICAL_TREE_SELECT=1` and
`-D MOJOLEARN_KNN_IDENTICAL_SOFTWARE_FTZ=1`, reconstructing main's arithmetic
and reduction before this lane. Final arm is `729ffa55`, default flags.
The harness changes only timing and validation; the old source's harness was
updated to accept `MOJOLEARN_KNN_ROOT` for an isolated `/root/knn-base` tree.

| n / queries / k | Before request ms | Final request ms | Speedup |
|---|---:|---:|---:|
| 100,000 / 4,000 / 10 | 16.803003 | 10.322649 | 1.628x |
| 400,000 / 4,000 / 10 | 66.536189 | 40.665239 | 1.636x |
| 400,000 / 4,000 / 15 | 67.105424 | 45.006222 | 1.491x |

All 16 cases improve; full table in `h100-safe-before-after.csv`.
Every request/device comparison reports zero mismatched cells; before/after
index and distance fingerprints agree, including no-index-tile k=10/15 arms.
The H100 model and driver match the pinned cuML tuple in
`bench/OPPONENT_REFERENCE.md`. At 400k/4000/k10 the pinned cuML request is
10.225 ms, so the ratio decreases from 6.51x to 3.98x. At 100k/4000/k10,
10.322649 ms versus pinned 3.480 ms is 2.97x. The opponent was not rerun.

L40S driver 570.124.06, UMAP 1M rows x32 features, k15, 200 epochs,
dyadic-v1, one timed round (no separate warmup), same pod and fixture:

| phase | Before ms | Safe final ms |
|---|---:|---:|
| kNN | 33193.850253 | 23851.685916 |
| graph | 10282.132337 | 10474.614343 |
| spectral | 8927.926714 | 9154.631789 |
| optimize | 4629.638169 | 4896.412252 |
| total | 57033.548365 | 48377.345090 |

Total elapsed improves 15.18%, kNN phase 28.14%. Both runs emit exactly
20,914,722 edges and embedding FNV1a64 `160286130194205729`. This is an
own-before/after measurement, not a refreshed cuML ratio. This L40S driver
does not match the pinned opponent tuple. CUDA 13 bundled ptxas failed on
the older driver; system `/usr/local/cuda/bin/ptxas` was used at BOTH build
and runtime. Runtime-only override is insufficient for already-built images.

## Numeric rejection and Apple limitation

The proposed NVIDIA hardware `fma.rn.ftz.f32` optimization was rejected after
an adversarial Mamba gate found `0x3f7fffff * 0x00800000 + 0` gives zero,
where exact rounding followed by FTZ must give `0x00800000`. It was removed
from kNN production and its matrix row in `729ffa55`; all final prices above
were rebuilt without it. Other usual fixtures passing is not evidence that
this arithmetic substitution was sound.

The new independent eight-triple distance boundary gate also exposes a
PREEXISTING Apple Metal FMA issue: the old software seam returns zero on
that same triple. The ordinary Apple fixtures pass, but the new boundary
gate does not. An exact kNN-only integer correction is under investigation;
there is no claim of universal Apple/NVIDIA FMA equality. No global numerics
or tree arithmetic was changed to mask this finding.

## Evidence directories

- `knn_h100_before`, `knn_h100_final`: final safe same-H100 16-shape prices.
- `knn_before_umap`, `knn_corrected`: safe L40S 1M before/after and corrected
  NVIDIA gates. `gates.done=1` records an orchestration refusal: an interrupted
  layout build left an evidence directory, and the retry correctly refused
  overwrite. The separate fresh `layout-safe` rerun passed all four143628-cell
  hashes; see `safe-layout-cell-check.txt`. All remaining gate statuses are0.
- `l40s-f2`: initial old-source profiles and full ordinary-fixture gate.
- `knn_unroll`, `knn_specialize`: intermediate hardware-FTZ profiles, useful
  only for isolating selector gains. Their absolute timings are REJECTED.
- `knn_partition`, `knn_probes`: rejected partition and deeper-load probes.
- `knn_final`: full ordinary-fixture gates on the intermediate hardware-FTZ
  arm; retained to document that these fixtures missed the numeric flaw.

Large textual logs are individually gzip-compressed without changing their
uncompressed bytes. No compiled binaries or credentials belong in evidence.
SHA256SUMS records the retained files.
