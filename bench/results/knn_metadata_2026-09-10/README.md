# Apple kNN request-local metadata, September 10

Forced candidate built from aeddd83e, Apple M4 IDENTICAL, Mojo 1.0.0
(ed45d567), dyadic-v1. Current `default` uses complete-chain exponent scans
inside every register tile; candidate `control` sets
`MOJOLEARN_EXPERIMENTAL_KNN_PREFLIGHT_METADATA=1` and computes each input
vector's minimum nonzero biased exponent once per request. It reduces those
minima across the same tile rows/columns, so the admission decision is exactly
the existing complete-chain minimum. Unsafe tiles retain exact zero-FMA repair.
FMA order and selection are unchanged. Metadata occupies the tail of existing
partial-result scratch and is rebuilt after every input mutation; its lifetime
ends after the existing final synchronization. The default build adds no scratch
allocation or metadata launch.

## Ordinary request evidence

Five rounds after two warmups for each arm, both execution orders. Includes
metadata preparation and allocation in request/device timing, with phase timers
disabled. Fixed four-shape experiment under the shared build lock, nice 19,
two compiler workers and OMP/OPENBLAS workers. Ordinary desktop activity remained;
there is no continuous clock/thermal telemetry or certified uncontended-host claim.

| Index / queries / features / k | Order | Current request ms | Metadata request ms | Request saving |
|---|---|---:|---:|---:|
| 400000 / 4000 / 32 / 10 | current first | 849.940 | 654.728 | 23.0% |
| 400000 / 4000 / 32 / 10 | metadata first | 880.852 | 654.823 | 25.7% |
| 400000 / 4000 / 32 / 15 | current first | 871.992 | 670.155 | 23.1% |
| 400000 / 4000 / 32 / 15 | metadata first | 914.559 | 695.273 | 24.0% |
| 400000 / 1000 / 8 / 15 | current first | 147.527 | 138.569 | 6.1% |
| 400000 / 1000 / 8 / 15 | metadata first | 153.781 | 143.182 | 6.9% |
| 65537 / 129 / 17 / 10 | current first | 7.096 | 6.668 | 6.0% |
| 65537 / 129 / 17 / 10 | metadata first | 7.482 | 6.772 | 9.5% |

The large target supports 23.0–25.7% lower request time (1.30–1.35× speedup),
with all four large within-arm request/device last/first ratios 0.983–1.022.
The reverse-order baseline is slower than the first order; both paired orders
independently favor metadata. The small ragged reverse-order baseline device
sequence has 1.710 last/first drift and is not promotion evidence.

All eight complete index/distance output pairs match, including request/device
comparison within each call and cross-order output hashes. This result is an
own-arm improvement, not a cuML price; no new opponent was run. Absolute times
must not be compared with the previous audit window's historical baseline.
The forced comparison supports the scoped default decision and final validation
below; small-case improvements did not expand the default scope.

## Numerical gates

- Independent integer oracle: 396,584 cases, four arms (per-step repair,
  simulated underflow repair, existing complete-chain tile, metadata tile), all
  exactly match; no tolerance. The GPU metadata preprocessing kernel is exercised.
- Reusing the same allocation after in-place mutation to zeros, subnormals,
  infinities, NaNs and normals refreshes the expected exponent minima.
- 24 scalar-versus-register layout cases check every distance and selected pair,
  including cancellation, FTZ operands and ragged feature/row/column dimensions.
  Large pricing additionally checks all selected index/distance output bytes.

## Reproduction and provenance

```sh
python3 tools/knn_zero_fma_oracle.py
# Build/run neighbors/checks/zero_fma_candidate_check.mojo with IDENTICAL.
# Build/run bench/knn_index_layout_main.mojo with IDENTICAL and the metadata flag.
MOJOLEARN_KNN_PROBE_MODE=price MOJOLEARN_KNN_PROBE_COMPARISON=metadata pixi run --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml bash tools/knn_residual_phase_probe.sh apple /tmp/knn-metadata-prices-20260910
python3 tools/knn_probe_summary.py /tmp/knn-metadata-prices-20260910
```

Retained files include individual samples, all complete output files, summaries,
compiler/host identity, compressed build logs, source commit/empty working patch,
local binary hashes and validation fixture/source hashes. Executables and the
regenerable oracle text are local; their hashes are in the manifests. `SHA256SUMS`
covers the retained files. Trees were untouched.

## Scoped default adopted and measured

The final policy enables metadata only on Apple IDENTICAL, the transposed
register path without vendor top-k, Euclidean `return_sqrt=True`, exactly
400000 index rows / 4000 queries / 32 features / k10 or k15. The explicit
experimental flag can still force metadata outside this measured scope.
`MOJOLEARN_KNN_IDENTICAL_NO_METADATA=1` disables the new default while retaining
complete-chain preflight and exact repair. Other dimensions, metrics and
columns retain their existing dispatch. The two below-scope controls therefore
compare the same existing algorithm; their timing differences are noise.

The final source is aeddd83e plus `default-price/working-tree.patch`; separate
binary hashes identify this actual default build. Five rounds after two warmups,
both orders, ordinary phase-disabled requests:

| Index / queries / features / k | Order | Scoped default request ms | Disabled request ms | Request saving |
|---|---|---:|---:|---:|
| 400000 / 4000 / 32 / 10 | default first | 642.981 | 856.831 | 25.0% |
| 400000 / 4000 / 32 / 10 | disabled first | 748.700 | 890.776 | 15.9% |
| 400000 / 4000 / 32 / 15 | default first | 668.901 | 875.495 | 23.6% |
| 400000 / 4000 / 32 / 15 | disabled first | 903.554 | 1071.877 | 15.7% |
| 400000 / 1000 / 8 / 15 | default first | 147.171 | 154.170 | 4.5% |
| 400000 / 1000 / 8 / 15 | disabled first | 209.214 | 206.404 | -1.4% |
| 65537 / 129 / 17 / 10 | default first | 7.120 | 7.164 | 0.6% |
| 65537 / 129 / 17 / 10 | disabled first | 8.536 | 9.169 | 6.9% |

The actual scoped default saves 15.7–25.0% request time on the two large targets
in both orders. The reverse pass entered a slower regime: large device last/first
ratios span 0.876–1.116, and disabled k15 request drift reaches 1.068. Retain this
limitation; the earlier 23–26% forced-window saving is not a universal claim.
The decision rests on the stable original large paired experiment plus positive
ordinary request results for both actual-default orders. No extra runs were
selected to obtain a more favorable window.

All eight final full-output pairs match, including both below-scope fallback
controls and cross-order hashes. All four fixtures' output hashes also match the
original forced comparison. The default policy is deliberately restricted to the
measured targets rather than generalized from small controls.

```sh
MOJOLEARN_KNN_PROBE_MODE=price MOJOLEARN_KNN_PROBE_COMPARISON=metadata-default pixi run --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml bash tools/knn_residual_phase_probe.sh apple /tmp/knn-metadata-default-scoped-20260910
python3 tools/knn_probe_summary.py /tmp/knn-metadata-default-scoped-20260910
```

An initial default/control rebuild was stopped before a completed timing pair
when the runtime policy was narrowed to the measured metric/layout as well as
dimensions. Those partial files are local and excluded from price evidence.
Phase class timers exclude upfront metadata preparation; the ordinary prices
above include it. The harness requires price mode for metadata comparisons.

Root integration: final merged layout harness also cross-compiled for Linux
x86-64-v3/sm_90 with IDENTICAL enabled. This checks NVIDIA code generation
only; no NVIDIA execution or new timing claim. Log and sidecar hashes are
retained in merged-nvidia-compile files.
