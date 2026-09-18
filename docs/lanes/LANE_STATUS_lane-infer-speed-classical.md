# LANE STATUS: lane/infer-speed-classical (2026-09-17)

Faster INFERENCE in IDENTICAL mode for the classical estimators, on the CPU
host path and on NVIDIA, with no output bit moved. Two changes, DEVIATIONS
2920 and 2921. Written for a session with no memory of this lane; every
number below is in `bench/results/infer_speed_classical_2026-09-17/` and the
raw JSONs, logs and pod records are outside the repo under
`~/mojolearn-evidence/infer-speed-classical/`.

## What changed

### DEVIATION 2920: the CPU host inference entries read X in place and split rows across threads

Files: `core/host_predict_threads.mojo` (new), `core/classical_host_predict.mojo`,
`core/knn_host_predict.mojo`, `kde/host/kde_oracle.mojo`,
`bindings/_mojolearn_estimators_host.mojo`.

Before, `ols_predict`, `tsvd_transform`, `pca_transform`,
`qn_decision_function` and `kde_score_samples` on the host copied X into a
List (`read_f32`: 440 MB per predict for a 500,000 x 220 X), walked every
output row on the calling thread (`host_gemm_nt` over `m * n` cells,
`host_knn_search` over every query, `oracle_score_samples` over every query
with two `n_query x n_train` stage matrices, 800 MB each at the KDE block),
and copied the result out element by element.

Now each entry has an `_into` twin that reads X through the caller's pointer
and writes the caller's output directly, and the output ROWS are split into
contiguous tasks: `host_predict_task_count(rows)` is MOJOLEARN_CPU_THREADS
when set to a positive integer (the one-core tooling exports 1, which makes
every path serial), else one task per physical core, never more than the
rows. One task runs on the calling thread without the pool.

WHY NO BIT MOVES, per path (the invariant is the same one, stated once):
every output row is a function of its own input row and the fitted state
alone, and a task keeps every statement of a row in its order.

- gemm cell (`host_pinned_cell_ptr`, the ONE spelling; the List door and the
  `MOJOLEARN_HOST_SABOTAGE` descending arm route through it): the ascending
  feature FMA/FTZ chain and the final `0.0 + acc` are per cell; a task writes
  its rows' cells ascending. The intercept and bias epilogues (one statement
  per output) run on the calling thread after the join, as before.
- PCA transform: a task centers each row into its own row buffer with
  `host_center_cell`, the statement `shift_columns_kernel` spells per cell,
  then the cells read the same flushed values the whole centered copy held.
- k-NN: a query row's distances (`host_l2_expanded_cell_ptr`,
  `host_metric_cell_ptr`), its composite-key selection and the estimator's
  insertion sort read the index, the norms and its own row, and write its own
  `k` slots; one `dist_row` scratch per task. The norms are computed once,
  before the split, as before.
- KDE: the norms, the log weights, `log_sw` and the kernel norm first, in
  `oracle_score_samples`'s order; a task then spells that function's row
  statements per query (every training row's distance, `oracle_log_kernel`,
  the weight's log, `oracle_logsumexp_row` over the row, the two
  subtractions). The stage matrices the checks read are not materialized;
  `oracle_score_samples` itself is unchanged for the checks.

This is not a kernel-matrix row: it changes which thread computes a row,
never what the row computes. The CPU columns below prove it at the default
task count, at 1, on x86 and on the M4.

### DEVIATION 2921: the fitted k-NN index stays on the device across `kneighbors` calls

Files: `neighbors/resident_index.mojo` (new), `neighbors/estimator.mojo`,
`bindings/_mojolearn.mojo`, `python/mojolearn/neighbors.py`.

Before, `NearestNeighbors.kneighbors` uploaded the whole index on every
call. Measured with the per-call upload on the RTX 4090 (the floor probe):
one Istella-S query against the 400,000 x 220 index cost 48 ms and 4,000
queries 140 ms; one taxi query 4.2 ms and 4,000 queries 42 ms. The upload
and its staging were the floor of the call.

Now the FIRST `kneighbors` after `fit` uploads the index once
(`knn_index_prepare`, a `_Global` registry in the shape of the forest's
FOREST-RESIDENT-1, an entry owning its DeviceContext and buffer) and keeps
the handle on the instance; later calls search through
`knn_search_resident`. `fit` is untouched. The handle is keyed on the index
array's address and shape: a refit replaces the array and the next call
uploads again; the old copy is released then and when the instance is
collected (`__del__`). An in-place write into the array `fit` was given,
after the first call, is not seen by later calls, as it is not by cuML,
whose `fit` copies the index to the device. A binding without the entry
(the CPU host binding, a CPU-only install, a `host_model` subclass) takes
the per-call path as before.

WHY NO BIT MOVES: `_knn_search_traced_retaining` was split into
`_knn_search_plan` (the refusals and the plan, before any allocation) and
`_knn_search_on_device_index` (everything after the index is on the device:
the query upload, the norms, the transposed layout, the distance chain, the
selection, the sort, the outputs). `knn_search` calls both with its own
upload between them; `knn_search_resident` calls both with the resident
buffer. The statements after the upload are the same statements over the
same bytes. The cuda column of the rebuilt binding below is the proof, and
the race worker records that the resident door was actually taken.

## Identity evidence

Harness: `tools/identity_break.py`, fixtures base,ties,odd,dupes,wide, two
repeats, IDENTICAL mode. Columns taken 2026-09-17:

| column | box | build | commit |
|---|---|---|---|
| cuda (x86_64) | RunPod RTX 4090, driver 580.159.04, Mojo 1.0.0 | GPU core + estimators at base | e3213a59a |
| cpu-before | pod CPU, AMD EPYC 7282 (64 vCPU) | host core + estimators at base | e3213a59a |
| cpu-after | pod CPU | host core + estimators, this lane, default task count | 6e08dc2fe |
| cpu-after-1thread | pod CPU | same set, MOJOLEARN_CPU_THREADS=1 | 6e08dc2fe |
| cpu-sabotage | pod CPU | same source, `-D MOJOLEARN_HOST_SABOTAGE=1` | 6e08dc2fe |
| cpu-after1-m4 | Apple M4, one core | host core + estimators, this lane, MOJOLEARN_CPU_THREADS=1 | 6e08dc2fe |
| cuda-after-core | pod RTX 4090 | GPU core binding rebuilt with DEVIATION 2921 | 9c3510897 |

The touched lanes (32; the estimators host family's inference lanes and the
core host family's k-NN lanes): ols, ols-no-intercept, ols-weighted, ridge,
ridge-no-intercept, logistic, logistic-l1, logistic-elasticnet,
logistic-multiclass, logistic-unpenalized-no-intercept, pca, pca-whiten,
pca-full-whiten, tsvd, kde, kde-cosine-minkowski, kde-epanechnikov-l1,
kde-exponential-chebyshev, kde-linear-cosine, kde-tophat-sqeuclidean,
kde-weighted, knn, knn-chebyshev, knn-clf, knn-clf-distance, knn-cosine,
knn-manhattan, knn-minkowski-p3, knn-rbc, knn-reg, knn-reg-distance,
knn-sqeuclidean.

| diff | train | infer/model | batch | exit |
|---|---|---|---|---|
| cuda, cpu-before, cpu-after, cpu-after-1thread | IDENTICAL x4 = 160 | IDENTICAL x4 = 320 | IDENTICAL x4 = 160 | 0 |
| cpu-before vs cpu-after | IDENTICAL = 160 | IDENTICAL = 320 | IDENTICAL = 160 | 0 |
| cuda, cpu-after, cpu-after1-m4 | IDENTICAL x3 = 160 | IDENTICAL x3 = 320 | IDENTICAL x3 = 160 | 0 |
| cuda vs cpu-sabotage (the control) | DIVERGENT = 160 | DIVERGENT = 226, IDENTICAL = 94 | DIVERGENT = 160 | 1 |
| cpu-after vs cpu-sabotage | DIVERGENT = 160 | DIVERGENT = 226, IDENTICAL = 94 | DIVERGENT = 160 | 1 |

No cell REFUSED in any touched-lane diff. The 94 sabotage-IDENTICAL
infer/model cells are the model column of lanes whose saved bytes the
sabotage does not reach; every lane has DIVERGENT infer cells on every
fixture (the per-lane table is in the diff file). The control was seen to
fail before the pass was read.

Spot check, base fixture, the other classical lanes (65 requested): the pod
carried GPU builds of the core and estimators families only, so 51 lanes
whose family was not built read REFUSED on every column alike (agglomerative,
arima*, cholesky, gp*, svc*, svr*, kernel methods, scalers, mixture, hdbscan,
umap, ivf, embedding, iforest, spectral, holtwinters, kpss, metrics,
gemm*); the 14 served (dbscan, dbscan-brute-l1, dbscan-weighted, kmeans,
kmeans-array, kmeans-random, kmeans-weighted, kmeans-sqrt, kmeans-classic-pp,
kmeans-cosine, radius, radius-chebyshev, radius-manhattan,
radius-minkowski-p3) read IDENTICAL x3 (cuda, cpu-before, cpu-after) on
every compared cell. A fuller spot check needs those families built; none
of their source changed.

Core binding rebuilt (DEVIATION 2921), cuda-before-core vs cuda-after-core
over the 22 lanes the `_mojolearn` binding serves (knn x11, radius x4,
kmeans x7), five fixtures: train IDENTICAL = 110, infer/model IDENTICAL = 210 (N/A = 10, the kmeans lanes' model cells), batch IDENTICAL = 105, exit 0. cuda-after-core against cpu-after and cpu-after-1thread on the 11 k-NN lanes both columns carry: IDENTICAL x3 = 160 train, 320 infer/model, 160 batch (the 55 radius and kmeans cells the touched CPU columns did not run read ONE-COLUMN, not DIVERGENT). cuda-after-core against cpu-sabotage: DIVERGENT on every k-NN train, infer and batch cell (55 each), exit 1. The race worker on the rebuilt binding recorded `resident_index: true`; on the base binding `false`.

Files: `~/mojolearn-evidence/infer-speed-classical/pod-pull/isc_ident/*.json`
and `diff.*.txt`, `.../ident-mac/cpu-after1-m4-touched.json` and
`diff.touched.cuda-podcpu-m4.txt`, `.../pod-pull/isc_post/`.

## Speed evidence

Box: RunPod secure cloud, one NVIDIA GeForce RTX 4090 (driver 580.159.04,
CUDA 13.0 image runpod/pytorch:2.4.0), AMD EPYC 7282 host, 64 vCPU, 251 GB.
Data: the `tools/classical_two_datasets.py prep` blocks staged from R2
(taxi and Istella-S; ols and pca on the big block's 500,000 eval rows; the
knn block's 400,000-row index; the kde block's 100,000-row fit set).
Harness: `bench/speed/classical_ladder_infer.py`, models fit ONCE on the
GPU and saved, every arm predicting from the same saved bytes; arms in
alternating processes, order rotated per outer round, 5 outer rounds x (1
warmup + 3 timed) calls, host array in and host array out with the upload
inside the clock; the output digest required equal across arms, rounds and
calls; spread = max/min over the 15 samples, admission 1.10.

Arms: cpu-before (host set at base), cpu-after (this lane, default task
count), cpu-after-1thread (MOJOLEARN_CPU_THREADS=1, the in-place read alone),
gpu-base (the GPU estimators binding at base, context for the same rows).
The CPU k-NN and KDE cells use the first 100 and 200 queries of the block
(the index and the training set whole); a serial 4,000-query Istella call
takes minutes. Ratios are paired medians over the first arm.

CPU host path, the clean rerun (nothing else on the box; `cpu-after-16` is
MOJOLEARN_CPU_THREADS=16, the socket's physical cores, against the default
count, which `num_physical_cores()` reports as 64 on this 2 x 16-core, 64
vCPU pod, so the default oversubscribes here):

| lane | dataset | arm | rows | rounds | median ms | min ms | max ms | spread | paired ratio over first arm | output digest |
|---|---|---|---|---|---|---|---|---|---|---|
| kde | istella | cpu-after | 200 | 5x3 | 1723.499 | 1548.535 | 1874.908 | 1.211 | 0.0677 | equal |
| kde | istella | cpu-after-16 | 200 | 5x3 | 1387.191 | 1367.059 | 1480.327 | 1.083 | 0.0541 | equal |
| kde | istella | cpu-before | 200 | 5x3 | 25582.880 | 25502.668 | 25958.710 | 1.018 | 1 (base) | equal |
| kde | taxi | cpu-after | 200 | 5x3 | 146.381 | 90.752 | 182.630 | 2.012 | 0.0587 | equal |
| kde | taxi | cpu-after-16 | 200 | 5x3 | 137.305 | 109.066 | 171.916 | 1.576 | 0.0673 | equal |
| kde | taxi | cpu-before | 200 | 5x3 | 2220.424 | 2192.823 | 2302.975 | 1.050 | 1 (base) | equal |
| knn | istella | cpu-after | 100 | 5x3 | 3743.100 | 3368.427 | 4255.378 | 1.263 | 0.0873 | equal |
| knn | istella | cpu-after-16 | 100 | 5x3 | 3431.944 | 3185.016 | 3651.253 | 1.146 | 0.0805 | equal |
| knn | istella | cpu-before | 100 | 5x3 | 42974.073 | 42685.126 | 43464.482 | 1.018 | 1 (base) | equal |
| knn | taxi | cpu-after | 100 | 5x3 | 578.501 | 545.392 | 673.520 | 1.235 | 0.1602 | equal |
| knn | taxi | cpu-after-16 | 100 | 5x3 | 566.365 | 543.762 | 613.085 | 1.127 | 0.1577 | equal |
| knn | taxi | cpu-before | 100 | 5x3 | 3601.861 | 3576.720 | 3642.471 | 1.018 | 1 (base) | equal |
| ols | istella | cpu-after | 500000 | 5x3 | 50.825 | 36.419 | 75.677 | 2.078 | 0.0874 | equal |
| ols | istella | cpu-after-16 | 500000 | 5x3 | 45.317 | 42.413 | 60.557 | 1.428 | 0.0735 | equal |
| ols | istella | cpu-before | 500000 | 5x3 | 598.175 | 573.835 | 946.223 | 1.649 | 1 (base) | equal |
| ols | taxi | cpu-after | 500000 | 5x3 | 6.086 | 4.332 | 10.121 | 2.336 | 0.1697 | equal |
| ols | taxi | cpu-after-16 | 500000 | 5x3 | 6.445 | 5.049 | 7.594 | 1.504 | 0.1720 | equal |
| ols | taxi | cpu-before | 500000 | 5x3 | 35.869 | 34.436 | 51.666 | 1.500 | 1 (base) | equal |
| pca | istella | cpu-after | 500000 | 5x3 | 279.084 | 250.690 | 338.377 | 1.350 | 0.0551 | equal |
| pca | istella | cpu-after-16 | 500000 | 5x3 | 188.631 | 171.615 | 256.876 | 1.497 | 0.0375 | equal |
| pca | istella | cpu-before | 500000 | 5x3 | 4918.256 | 4769.113 | 5401.935 | 1.133 | 1 (base) | equal |
| pca | taxi | cpu-after | 500000 | 5x3 | 23.798 | 18.824 | 30.159 | 1.602 | 0.0821 | equal |
| pca | taxi | cpu-after-16 | 500000 | 5x3 | 32.833 | 28.543 | 46.524 | 1.630 | 0.1149 | equal |
| pca | taxi | cpu-before | 500000 | 5x3 | 290.260 | 271.873 | 320.536 | 1.179 | 1 (base) | equal |

The serial baselines are tight (spread at most 1.05). The threaded arms are
noisier (1.08 to 2.34 across 15 samples; the min and max are in the table)
because a 64-task split over a shared, hyperthreaded pod is at the mercy of
the scheduler; every ratio above is 6x to 18x, far outside that noise, and
`cpu-after-16` is the steadier arm. A cell whose spread exceeds 1.10 is not
quoted as a precise number, only as a bound: the 500,000-row OLS on
Istella-S went from about 600 ms to under 76 ms on every one of 15 samples.

The first race (taken while the pod also ran the identity columns, so its
threaded arms are noisier still) carried a fourth arm,
`cpu-after-1thread` (MOJOLEARN_CPU_THREADS=1): the in-place read and the
copy-free write ALONE, on one thread, with tight spreads (at most 1.104):

| lane | dataset | arm | rows | rounds | median ms | min ms | max ms | spread | paired ratio over first arm | output digest |
|---|---|---|---|---|---|---|---|---|---|---|
| kde | istella | cpu-after | 200 | 5x3 | 1677.886 | 1537.847 | 1822.393 | 1.185 | 0.0645 | equal |
| kde | istella | cpu-after-1thread | 200 | 5x3 | 18272.326 | 18201.588 | 18821.795 | 1.034 | 0.7105 | equal |
| kde | istella | cpu-before | 200 | 5x3 | 25668.034 | 25422.492 | 26329.257 | 1.036 | 1 (base) | equal |
| kde | istella | gpu-base | 2000 | 5x3 | 77.540 | 74.571 | 82.349 | 1.104 | n/a rows differ | equal |
| kde | taxi | cpu-after | 200 | 5x3 | 161.817 | 101.026 | 181.037 | 1.792 | 0.0736 | equal |
| kde | taxi | cpu-after-1thread | 200 | 5x3 | 1675.794 | 1671.326 | 1682.988 | 1.007 | 0.7611 | equal |
| kde | taxi | cpu-before | 200 | 5x3 | 2201.171 | 2188.379 | 2239.934 | 1.024 | 1 (base) | equal |
| kde | taxi | gpu-base | 2000 | 5x3 | 50.242 | 47.659 | 51.334 | 1.077 | n/a rows differ | equal |
| knn | istella | cpu-after | 100 | 5x3 | 3662.810 | 3331.615 | 4083.755 | 1.226 | 0.0844 | equal |
| knn | istella | cpu-after-1thread | 100 | 5x3 | 33262.574 | 32314.814 | 34202.716 | 1.058 | 0.7645 | equal |
| knn | istella | cpu-before | 100 | 5x3 | 43157.700 | 42838.596 | 44557.923 | 1.040 | 1 (base) | equal |
| knn | istella | gpu-base | 4000 | 5x3 | 130.587 | 109.532 | 144.951 | 1.323 | n/a rows differ | equal |
| knn | taxi | cpu-after | 100 | 5x3 | 588.778 | 517.254 | 653.194 | 1.263 | 0.1604 | equal |
| knn | taxi | cpu-after-1thread | 100 | 5x3 | 2663.979 | 2656.177 | 2720.325 | 1.024 | 0.7432 | equal |
| knn | taxi | cpu-before | 100 | 5x3 | 3584.145 | 3574.023 | 3617.706 | 1.012 | 1 (base) | equal |
| knn | taxi | gpu-base | 4000 | 5x3 | 43.868 | 37.425 | 44.818 | 1.198 | n/a rows differ | equal |
| ols | istella | cpu-after | 500000 | 5x3 | 39.398 | 34.004 | 83.245 | 2.448 | 0.0615 | equal |
| ols | istella | cpu-after-1thread | 500000 | 5x3 | 428.419 | 419.530 | 444.768 | 1.060 | 0.6609 | equal |
| ols | istella | cpu-before | 500000 | 5x3 | 788.424 | 578.225 | 937.120 | 1.621 | 1 (base) | equal |
| ols | istella | gpu-base | 500000 | 5x3 | 57.542 | 40.796 | 82.561 | 2.024 | 0.0974 | equal |
| ols | taxi | cpu-after | 500000 | 5x3 | 5.708 | 4.164 | 7.748 | 1.861 | 0.1526 | equal |
| ols | taxi | cpu-after-1thread | 500000 | 5x3 | 19.953 | 18.804 | 20.767 | 1.104 | 0.5539 | equal |
| ols | taxi | cpu-before | 500000 | 5x3 | 37.415 | 34.819 | 54.316 | 1.560 | 1 (base) | equal |
| ols | taxi | gpu-base | 500000 | 5x3 | 6.090 | 4.241 | 8.378 | 1.975 | 0.1628 | equal |
| pca | istella | cpu-after | 500000 | 5x3 | 295.331 | 280.000 | 372.244 | 1.329 | 0.0601 | equal |
| pca | istella | cpu-after-1thread | 500000 | 5x3 | 2476.553 | 2453.985 | 2515.708 | 1.025 | 0.4882 | equal |
| pca | istella | cpu-before | 500000 | 5x3 | 5091.846 | 4733.778 | 5413.584 | 1.144 | 1 (base) | equal |
| pca | istella | gpu-base | 500000 | 5x3 | 67.238 | 55.355 | 95.308 | 1.722 | 0.0126 | equal |
| pca | taxi | cpu-after | 500000 | 5x3 | 25.396 | 19.740 | 33.609 | 1.703 | 0.0851 | equal |
| pca | taxi | cpu-after-1thread | 500000 | 5x3 | 134.299 | 129.338 | 140.498 | 1.086 | 0.4523 | equal |
| pca | taxi | cpu-before | 500000 | 5x3 | 298.412 | 276.396 | 332.270 | 1.202 | 1 (base) | equal |
| pca | taxi | gpu-base | 500000 | 5x3 | 10.964 | 8.654 | 20.142 | 2.327 | 0.0338 | equal |

Per path, one thread: OLS 0.55 (taxi) and 0.66 (Istella-S), PCA 0.45 and
0.49, k-NN 0.74 and 0.76, KDE 0.76 and 0.71 of the base time; the rest of
the gain is the row split. The GPU arm (`gpu-base`, the estimators binding
at base) is context for the same 500,000 rows: after this lane the CPU host
path on the pod's 32 cores is within about 1x of the RTX 4090 on OLS (5.7 vs
6.1 ms taxi, 39 vs 58 ms Istella-S, the upload inside the GPU clock) and 2x
to 4x of it on PCA.


k-NN GPU, resident index, gpu-before (base core binding) vs gpu-after
(rebuilt), the whole 4,000-query block:

| lane | dataset | arm | rows | rounds | median ms | min ms | max ms | spread | paired ratio over first arm | output digest |
|---|---|---|---|---|---|---|---|---|---|---|
| knn | istella | gpu-after | 4000 | 5x3 | 84.181 | 74.413 | 88.615 | 1.191 | 0.6521 | equal |
| knn | istella | gpu-before | 4000 | 5x3 | 129.908 | 109.051 | 137.981 | 1.265 | 1 (base) | equal |
| knn | taxi | gpu-after | 4000 | 5x3 | 34.128 | 28.075 | 35.218 | 1.254 | 0.7753 | equal |
| knn | taxi | gpu-before | 4000 | 5x3 | 44.070 | 37.664 | 44.397 | 1.179 | 1 (base) | equal |

The race worker recorded `resident_index: true` on gpu-after and `false` on
gpu-before. Spreads 1.18 to 1.27 (a query batch of 4,000 on a shared pod);
the paired ratio is the median over the five rounds of the per-round
medians' ratio, and every one of the five per-round ratios was below 0.85
on both datasets (in the JSON).


GPU floor probe (per-call cost, 5 calls after a warmup, base binding):

| lane | dataset | rows | base binding median ms | rebuilt core binding (DEVIATION 2921) median ms |
|---|---|---|---|---|
| ols | taxi | 1 | 0.403 | 0.401 |
| ols | taxi | 100 | 0.389 | 0.381 |
| ols | taxi | 10000 | 0.449 | 0.443 |
| ols | taxi | 500000 | 4.886 | 3.891 |
| ols | istella | 1 | 0.554 | 0.397 |
| ols | istella | 100 | 0.558 | 0.410 |
| ols | istella | 10000 | 1.413 | 1.381 |
| ols | istella | 500000 | 53.316 | 51.871 |
| pca | istella | 1 | 0.422 | 0.450 |
| pca | istella | 100 | 0.434 | 0.441 |
| pca | istella | 10000 | 1.620 | 1.468 |
| pca | istella | 500000 | 54.192 | 50.506 |
| knn | taxi | 1 | 4.165 | 1.404 |
| knn | taxi | 100 | 5.013 | 2.454 |
| knn | taxi | 10000 | 41.912 | 33.346 |
| knn | taxi | 4000 | 42.102 | 33.458 |
| knn | istella | 1 | 48.245 | 6.511 |
| knn | istella | 100 | 60.658 | 10.263 |
| knn | istella | 10000 | 140.302 | 88.133 |
| knn | istella | 4000 | 140.259 | 87.373 |
| kde | istella | 1 | 27.263 | 27.504 |
| kde | istella | 100 | 32.497 | 31.602 |
| kde | istella | 10000 | 76.023 | 75.772 |
| kde | istella | 2000 | 77.930 | 74.809 |

Base column: the console capture of the base run (its JSON was overwritten
by the after run of the same script; the capture is
`pod-pull/isc_out/gpu_floor_probe_base.console.txt`). OLS, PCA and KDE ran
the same estimators binding both times and read within noise; the k-NN rows
are the resident index: one Istella-S query 48.2 to 6.5 ms, one taxi query
4.2 to 1.4 ms. What the probe says about the remaining floors: a one-row OLS
or PCA call is 0.4 ms (context, allocation, two syncs and the readback), a
one-row KDE call 27.5 ms (the 100,000 x 220 training set uploaded and
validated per call), so the KDE training set is the next resident candidate.


## Commands

```
# pod (RunPod RTX 4090), from the shipped source tree at /root/mojolearn
sh tools/trees_leg.sh rent --gpu "NVIDIA GeForce RTX 4090" --minutes 150   # TREES_LEG_STATE=~/mojolearn-evidence/infer-speed-classical/pod
pixi run python3 tools/classical_two_datasets.py prep --data /root/ctd-data --lanes ols,pca,knn,kde --datasets taxi,istella
MOJOLEARN_HOST_OUTDIR=/root/hostbins/before sh bindings/build_estimators_host.sh; ... build_core_host.sh   # base, then after (this lane's tree), then
MOJOLEARN_HOST_OUTDIR=/root/hostbins/sabotage MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1" sh bindings/build_estimators_host.sh
MOJOLEARN_GPU_ARCHS=sm_89 bash bindings/build.sh; bash bindings/build_estimators.sh
pixi run python3 bench/speed/classical_ladder_infer.py fit --data /root/ctd-data --models /root/isc_models
# identity, one column per arm (CPU columns from a source tree with no GPU .so, MOJOLEARN_HOST_DIR naming the set)
python3 tools/identity_break.py --require-backend cuda --lanes <touched> --fixtures base,ties,odd,dupes,wide --repeats 2 --json cuda-touched.json
MOJOLEARN_HOST_DIR=/root/hostbins/after python3 tools/identity_break.py --require-backend cpu --lanes <touched> --fixtures base,ties,odd,dupes,wide --repeats 2 --json cpu-after-touched.json
python3 tools/identity_break.py --diff cuda-touched.json cpu-before-touched.json cpu-after-touched.json cpu-after1-touched.json
# speed
python3 bench/speed/classical_ladder_infer.py race --data /root/ctd-data --models /root/isc_models --arms arms.json --out /root/isc_race --lanes ols,pca,knn,kde --datasets taxi,istella --outer 5 --rounds 3 --warmup 1 --rows-cpu-knn 100 --rows-cpu-kde 200
```
The exact scripts run are `~/mojolearn-evidence/infer-speed-classical/pod/{setup_base,after_setup,identity_runs,post_race}.sh`.

## Owed

- Apple and AMD columns of the touched lanes and of the rebuilt core binding
  at the next release record (this lane takes neither; the CPU columns and
  the cuda column are here).
- The spot check of the classical lanes whose GPU families were not built on
  this pod (list above); their source did not change.
- A resident training set for SVC predict, the same door as DEVIATION 2921;
  not built here. (The KDE `score_samples` half of this item was closed the
  same day by lane/knn-tiled-distance as DEVIATION 3003: Istella-S 72.6 to
  54.0 ms, taxi 47.5 to 40.9.)
- (closed 2026-09-17 by lane/knn-tiled-distance, DEVIATION 3002)
  KNeighborsClassifier and KNeighborsRegressor now predict through the
  resident index; the label and target columns are still uploaded per call.
