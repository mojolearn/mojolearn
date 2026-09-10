# NVIDIA kNN aligned index loads: complete requests, September 10

Base 5d97c28c. H100 80GB HBM3, driver 580.126.09; exact GPU UUID in
`raw/knn-request/gpu.csv`. No opponent was rerun. This is the public
`knn_search` request, including allocation, transfers, norms, transpose,
distance, selection, merging, synchronization and host ordering.

## Large-data promotion evidence

400,000 index rows, 4,000 queries, 32 features, k=10/15, IDENTICAL,
L2SqrtExpanded, existing dyadic-v1 cuML fixture. Query batch 512, index
partition 65,536. Full index stride is 400,000, including the last partition
of 6,784 columns and last query batch of 416 rows.

| Window / k | Scalar request median ms | Vector request median ms | Reduction |
|---|---:|---:|---:|
| first / 10 | 27.689049 | 26.687143 | 3.62% |
| first / 15 | 32.153455 | 31.149189 | 3.12% |
| reversed shape order / 10 | 27.719551 | 26.735258 | 3.55% |
| reversed shape order / 15 | 32.154780 | 31.147359 | 3.13% |

Each process alternates arm order, with two warmups and 31 timed samples
per arm. Environment mutation, poisoning and output comparisons occur
outside timing. Every returned distance bit and index matches the first
scalar call. All samples, including the one slower vector pair in the
second k10 window, remain in the logs. `summary.json` includes paired ratios.
This is a request gain; the earlier 6.33% result was a distance-tile gain.

## Scope and arithmetic

Only NVIDIA's existing transposed register distance route, these measured
shapes and metric, acquire the default. Alignment admission checks the
full index stride, partition offset, and partition width, each divisible
by four; DeviceBuffer provides the aligned base. Other partitions use
scalar loads. Ragged query rows retain the existing clamping and masked
stores. Feature order, FMA/FTZ, epilogue, selection and tie ordering stay
identical. Apple metadata dispatch is unaffected.

The upstream transport reference is RAFT
`cpp/include/raft/linalg/detail/contractions.cuh:193-219`. cuVS's ordinary
small-k default dispatch remains fusedL2Knn at
`cpp/src/neighbors/detail/knn_brute_force.cuh:443-460`; this is an optimization
of our declared IDENTICAL departure, not a claim to have ported that fusion.
The existing DEVIATION 505 applies.

Large correctness controls use 400,003 and 400,004 index rows, 4,001 queries,
d32/k15, seven samples per arm. The first refuses vector transport because
of its stride; the second exercises aligned transport with ragged query
rows and index partitions. Both compare all selected output cells. These
controls do not widen default scope. The previous distance-tile arithmetic
fixtures and full 33,554,432-cell check remain in `knn_loads_2026-09-10`;
this continuation does not label selected-output comparisons as another
full all-pairs arithmetic check.

The candidate-only corruption replaces one vector-loaded input by 123.
At the actual 400k/4k/k10 target, the first scalar request passes, then the
vector request fails with `request output bytes moved between rounds`.
The nonzero exit and source are retained. This establishes public-path
reach and a reacting oracle.

## Reproduction and final default check

`reproduction/initial.sh` and `controls.sh` describe the initial two-arm
trials against the exact modified source snapshots under `raw/knn-request`.
`reproduction/sabotage-kernel.mojo` is the deliberate corruption source.

The maintained `bench/knn_vector_request_main.mojo`, built IDENTICAL with
`MOJOLEARN_KNN_VECTOR_REQUEST_CHECK=1`, checks three arms: 0 scalar,
1 explicitly requested vector, 2 actual scoped default. Without that build
flag the production dispatcher ignores the environment override.
`reproduction/final.sh` builds that check and the ordinary reference price
binary without the override flag, runs both large k values, and compares
all interleaved index/distance bytes using `cmp`. Raw logs retain warmups
and all request/device timings; only ordinary default request prices belong
in the cached-opponent comparison.

No GEMM, attention, Mamba or Transformer timings were taken here. Trees
were not edited. Selection remains the next major kNN target, with nearly
as much synchronized phase time as distance at k15 in the preceding capture.

Final ordinary default request medians: **26.661871 ms k10**, **31.126293 ms
k15**, 15 samples after two warmups; device medians 25.538397 / 29.877126 ms.
Cached cuML request references remain 10.225 / 10.817 ms, giving 2.61x / 2.88x.
They come from another physical rental with the recorded matching model and
driver, not a fresh paired opponent run.

Final three-arm medians (scalar / explicit vector / default) are
27.678031 / 26.679653 / 26.679805 ms at k10 and
32.111093 / 31.107047 / 31.117942 ms at k15. All output comparisons passed.
Summary fields named vector refer to requested arm 1; ragged stride 400003
actually uses scalar fallback, as required by alignment admission.

Binaries and complete output dumps are stored with deterministic gzip.
`packed-artifacts.json` retains their uncompressed hashes. `SHA256SUMS`
covers retained files. The measured source snapshots are included; the
maintained harness subsequently changes only its fallback environment
sizes to the large target, adds a printed arm legend, and updates documentation. Every measured shape was
set explicitly. The dispatcher comment about Apple scope was moved beside
the Apple decision without changing executable code.

Rental j3bjovngd06qtc: DELETE 204 at 09:48:48 EDT, verified GET 404 at
09:48:53 EDT. No GPU rental remains from this continuation.
