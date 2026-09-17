# LANE STATUS: lane/knn-tiled-distance (2026-09-17)

Faster brute-force k-nearest-neighbors search and KDE scoring on NVIDIA in
IDENTICAL mode, no output bit moved, measured against cuML and cuVS on the
same box with the two real blocks. Four DEVIATIONs, 3000 to 3003. Written
for a session with no memory of this lane. Every number below is in
`bench/results/knn_tiled_2026-09-17/` (round1, round2, final); the raw JSON,
consoles, logs and pod records are outside the repo under
`~/mojolearn-evidence/knn-tiled/` (pull1, pull2, pull3, pod).

Box for everything here. RunPod secure cloud pod fc3i4usbkd8ahz, one NVIDIA
GeForce RTX 4090 (driver 580.126.09, CUDA 13.0 host, image
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04), 48 vCPU, 251 GB,
$0.74 per hour. Mojo 1.0.0 from the repository's pixi lock. Main at
c85657041 is the base of every "before" arm; the branch's commits are the
"after" arms. Data staged from R2 (taxi_speed.npz, istella_speed.npz) and
cut by `tools/classical_two_datasets.py prep` as before. The kNN blocks are
the 400,000-row index and 4,000 queries of each dataset (220 features on
Istella-S, 11 on taxi); the KDE blocks are 100,000 fit rows and 2,000
queries.

## What changed

### DEVIATION 3000: the distance tile through shared memory

`neighbors/checks/smem_distance_tile.mojo`, kernel-matrix row
`knn_smem_distance_tile_for` (`checks/kernel_matrix.mojo`), wired in
`neighbors/impl/detail/knn_brute_force.mojo::_tiled_brute_force_knn_impl`.

Before, the transposed IDENTICAL arm computed a `query_tile x index_tile`
distance matrix with `pinned_distance_register_tile_kernel`, each thread
owning an 8 x 4 register tile and issuing, per feature step, eight query
loads and one four-wide index load from global memory, each through `ftz`
(a bit test and a select under IDENTICAL). The 65,536-column index tile is
57.7 MB at d = 220 and was re-read from L2 once per eight query rows, 512
times per 4,096-query tile.

Now one block of 256 threads owns 64 query rows x 128 index columns. For
each slice of 16 features it stages the slice of the query rows and of the
transposed index columns into shared memory once, flushed through `ftz` at
the store, and each thread advances its 8 x 4 accumulators from three
16-byte shared loads per feature step. The index slice is read from L2 once
per 64 rows and no operand is flushed more than once.

WHY NO BIT MOVES. Every cell is still ONE ascending chain over the feature
axis from +0.0, `acc = _rt_step(ftz(q[row, f]), ftz(yt[f, col]), acc)`, the
register tile's own step function (`pinned_distance_tile.mojo::_rt_step`,
on NVIDIA the rounded FMA then the hardware flush), with the register
tile's epilogue, clamp and root, statement for statement. Staging changes
where an operand is read from and how many cells share its load; `ftz` is
idempotent, so the value read from shared memory is the value `_rt_load`
produced per step. No chain is split, folded or reordered, and no feature
past `d` is ever stepped (the slice's inner trip count is its real length;
a zero-padded step would turn a -0.0 accumulator into +0.0).

The tile also honors DEVIATION 2629's exact-chain admission (row
`knn_distance_exact_chain_for`, now ON on NVIDIA where the smem tile runs):
the block reduces the request-local per-row exponent metadata over its 64
rows and 128 columns and, when the three clauses hold for the whole block,
takes `_rt_step_exact` (the same rounded FMA without the flush, which is
the identity on every value it would see under the proof in
`pinned_distance_tile.mojo`); a block that fails keeps the flushed chain.
On the register tile this admission was measured neutral on 2026-09-11 and
stayed off; on the smem tile the flush is the second ALU op of every step
and admission is worth 6.5 ms of the Istella-S distance class (table below).

### DEVIATION 3001: the block top-k, no distance matrix

Same file, kernel-matrix row `knn_block_topk_select_for`, k <= 16 by
default (`KNN_BLOCK_TOPK_MAX_K`; `-D MOJOLEARN_KNN_BLOCK_TOPK_ALL_K=1`
lifts it to the selector's 64 for an A/B).

With the row on, the smem tile writes no matrix. After the chains, each
thread row (32 lanes x 4 columns, one warp on NVIDIA) holds 8 complete rows
of the block's 128 columns and pops each row's k smallest composite keys
(`select_radix_identical.mojo::composite_key(distance, tile-local column,
select_min=True)`, the small-k selector's key): per rank, every lane offers
its smallest remaining key, a 32-bit warp minimum over the distance halves
and one ballot name the lowest lane holding that minimum (lane order is
column order, so that lane holds the smallest column among equal distances,
which is the composite order), and that lane writes the key and retires
it. The eight rows are interleaved so the shuffle chains overlap. The keys
go to `part[(row * n_col_blocks + col_block) * k + rank]`, and
`partial_keys_select_kernel` (one block per row) selects the row's k
smallest from the union of the per-block lists with the same UInt64
minima, writing the distance as `twiddle_out` of the key's high half (the
exact inverse of the key's `twiddle_in`) and the tile-local column as the
index, which is exactly what the small-k selector wrote, so
`partial_topk_merge_kernel` merges column tiles as before.

WHY NO BIT MOVES. Keys are unique (they carry the column), so the k smallest
keys of the row's column tile are a subset of the union of the per-block
lists whatever the partition; the rank phases pop the same minima
ascending; the distance bits are the epilogue's (a bijection out of the
key), as the unfused selector's gather of the matrix cell was. This is the
argument the small-k selector and DEVIATION 2667 already rest on.

Why 2667 (the fused distance-and-select launch) lost and this wins: 2667 had
one block own one row and read 1.125 operands per cell per feature; here
the block owns 64 rows and the rank loop runs on values already in
registers.

### DEVIATION 3002: resident index for KNeighborsClassifier and KNeighborsRegressor

`neighbors/estimator.mojo` (`knn_classifier_predict_resident`,
`knn_regressor_predict_resident`), `neighbors/resident_index.mojo`
(`knn_index_classify`, `knn_index_regress`), `bindings/_mojolearn.mojo`
(`knn_classify_resident`, `knn_regress_resident`),
`python/mojolearn/neighbors.py`. The same registry, handle, refit-release,
`__del__` and no-handle-in-pickle rules as DEVIATION 2921, through
`NearestNeighbors._resident_index_handle`. The classifier binding carries
the handle as `params[0]` because a Python binding takes at most eight
arguments and the classifier already used all eight. The search is
`_knn_search_on_device_index` and the vote and fold are
`_knn_classifier_vote` and `_knn_regressor_vote` unchanged, so every
statement after the upload is the per-call entry's. The label and target
columns are still uploaded per call (1.6 MB at 400,000 rows).

### DEVIATION 3003: resident KDE fit set

`kde/resident_fit.mojo` (new; `kde_fit_prepare`, `kde_fit_release`,
`kde_score_samples_resident`), `bindings/_mojolearn_estimators.mojo`,
`python/mojolearn/density.py`. The first `score_samples` validates
(`kde_fit_validate`, DEVIATION 604's finiteness scan through
`kde_validate_data_ptr`, the cosine zero-row rule) and uploads the training
rows and the weights once; every later call validates and uploads the
queries only, runs `kde/impl/kde.mojo::score_samples` over the same device
bytes with the same `sum_weights`, and downloads through the same pinned
buffer. The handle is keyed on the training array, the weights array, the
kernel, the metric and the bandwidth; a refit, `__del__` and a pickle
follow DEVIATION 2921's rules. The refusals are the one-shot entry's, by
name and in its order, with one difference named in the file: a bad
training set and a bad query count in the same call now name the training
set first.

### Phase timers

`-D MOJOLEARN_KNN_PHASE_TIMERS=1` now also prints one `KNN_PHASE_TIMERS
request ...` line per search from `neighbors/estimator.mojo` (query upload,
norms, the tiled arm, readback, host sort), beside the arm's own transpose,
distance, select and merge classes. Trial builds only, never shipped.

## Phase attribution (deliverable 1)

Timer builds serialize the queue after every launch class, so a timed call
is slower than an untimed one; the split is the measurement. 4,000 queries
against the 400,000-row index, 7 column tiles of 65,536, query tile 4,000,
medians over three timed calls after a warmup; `phase0` is the branch with
every new row off (the register tile, the small-k selector), `smem` the
tile alone, `smemx` the tile with exact-chain admission, `topk` the block
top-k, `topkx` the shipped NVIDIA default (tile, admission, block top-k for
k <= 16). Milliseconds.

| arm | dataset | k | call (timed) | upload | norms | transpose | distance | select | merge | readback | sort |
|---|---|---|---|---|---|---|---|---|---|---|---|
| phase0 | istella | 1 | 77.8 | 1.28 | 0.42 | 0.93 | 58.60 | 7.08 | 0.07 | 0.02 | 0.00 |
| phase0 | istella | 10 | 83.2 | 1.30 | 0.43 | 0.93 | 59.35 | 7.04 | 0.07 | 1.05 | 0.16 |
| phase0 | istella | 64 | 149.1 | 0.34 | 0.39 | 0.87 | 56.48 | 67.92 | 0.12 | 0.17 | 1.08 |
| phase0 | taxi | 1 | 31.0 | 1.03 | 0.22 | 0.08 | 14.57 | 8.61 | 0.07 | 0.02 | 0.00 |
| phase0 | taxi | 10 | 33.1 | 1.04 | 0.22 | 0.08 | 14.55 | 7.58 | 0.07 | 1.09 | 0.08 |
| phase0 | taxi | 64 | 100.6 | 0.03 | 0.22 | 0.08 | 14.38 | 68.21 | 0.11 | 0.17 | 0.61 |
| smem | istella | 10 | 60.8 | 1.29 | 0.42 | 0.93 | 37.86 | 7.14 | 0.07 | 1.05 | 0.16 |
| smem | istella | 64 | 128.7 | 0.36 | 0.39 | 0.87 | 36.54 | 67.68 | 0.14 | 0.17 | 1.11 |
| smem | taxi | 10 | 32.2 | 1.04 | 0.22 | 0.08 | 13.62 | 7.57 | 0.07 | 1.09 | 0.08 |
| smemx | istella | 10 | 56.9 | 1.29 | 0.43 | 0.92 | 31.35 | 7.01 | 0.07 | 1.05 | 0.16 |
| smemx | istella | 64 | 127.4 | 0.34 | 0.39 | 0.87 | 31.54 | 67.67 | 0.12 | 0.17 | 1.11 |
| smemx | taxi | 10 | 33.3 | 1.03 | 0.22 | 0.08 | 13.65 | 7.57 | 0.07 | 1.09 | 0.09 |
| topk | istella | 1 | 47.9 | 0.37 | 0.40 | 0.88 | 38.93 | 0.18 | 0.07 | 0.02 | 0.00 |
| topk | istella | 10 | 58.2 | 0.36 | 0.40 | 0.87 | 45.43 | 1.44 | 0.07 | 1.00 | 0.16 |
| topk | istella | 64 | 151.6 | 0.34 | 0.39 | 0.87 | 98.19 | 26.97 | 0.12 | 0.17 | 1.11 |
| topk | taxi | 1 | 11.3 | 0.03 | 0.22 | 0.08 | 9.40 | 0.18 | 0.06 | 0.02 | 0.00 |
| topk | taxi | 10 | 25.3 | 0.03 | 0.22 | 0.08 | 18.64 | 1.52 | 0.07 | 1.02 | 0.08 |
| topk | taxi | 64 | 115.0 | 0.04 | 0.20 | 0.08 | 68.99 | 26.92 | 0.11 | 0.17 | 0.61 |
| topkx | istella | 10 | 56.9 | 0.34 | 0.40 | 0.89 | 42.22 | 1.47 | 0.07 | 1.01 | 0.16 |
| topkx | taxi | 10 | 26.0 | 0.03 | 0.22 | 0.08 | 19.06 | 1.52 | 0.07 | 1.03 | 0.08 |

Under `topk` the distance class holds the tile and its in-block rank loop,
and the select class is the partial-key selector. What the table says:

- Before, the distance class was 59 ms of an 83 ms Istella-S call at k 10
  and the selector 7 ms; on taxi 14.6 and 7.6 of 33. The transfers (upload
  1 ms, readback 1 ms), the norms and the host sort are small at every
  shape; the one-query floor (5 to 7 ms on Istella-S) is the seven
  column-tile launch sequence and the transposition, not transfers.
- The shared-memory tile takes the Istella-S distance class from 59.4 to
  37.9 ms; admission takes it to 31.4. At d = 11 (taxi) the tile is nearly
  neutral (14.6 to 13.6) because the chain is 11 steps and the per-cell
  epilogue and the matrix write dominate.
- The block top-k removes the selector (7 ms) and the matrix; its rank
  loop costs about 8 ms inside the tile at k 10 on Istella-S (37.9 to 45.4)
  and 27 ms at k 64, where the partial-key selector adds another 27 ms, so
  k 64 stays on the matrix and the small-k selector. On taxi the matrix
  write was most of the distance class, so the block top-k halves the call
  at k 10 and takes k 1 from 31 to 11 ms.
- At k 64 the small-k selector is 68 ms on both datasets, more than the
  distances; that is the next target and not this lane's (the selector's
  chain is its own file's).

## Speed (deliverable 2), interleaved arms

`bench/speed/classical_ladder_infer.py race`, models fit once and saved,
one process per arm per round, order rotated per round, 5 outer rounds x
(1 warmup + 3 timed) calls, host array in and host array out with the query
upload inside the clock, every digest required equal across arms, rounds
and calls. `base` is main's binding, `after0` the branch with the rows off
(the resident doors only). Paired ratio is the median over rounds of the
per-round medians' ratio to `base`.

Round 2 (`bench/results/knn_tiled_2026-09-17/round2/race_knn_summary.tsv`),
4,000 queries, k 10:

| dataset | arm | median ms | min ms | max ms | spread | paired ratio | digests |
|---|---|---|---|---|---|---|---|
| istella | base | 82.46 | 73.77 | 84.58 | 1.147 | 1 | equal |
| istella | after0 | 82.04 | 72.34 | 83.74 | 1.158 | 0.996 | equal |
| istella | smem | 60.89 | 52.26 | 63.21 | 1.209 | 0.754 | equal |
| istella | smemx | 56.52 | 48.65 | 58.28 | 1.198 | 0.695 | equal |
| istella | topk | 58.71 | 55.42 | 61.32 | 1.106 | 0.729 | equal |
| istella | topkx | 53.49 | 50.73 | 56.32 | 1.110 | 0.649 | equal |
| taxi | base | 33.66 | 26.66 | 34.80 | 1.305 | 1 | equal |
| taxi | after0 | 33.33 | 25.41 | 34.94 | 1.375 | 0.999 | equal |
| taxi | smem | 31.54 | 24.59 | 32.92 | 1.339 | 0.938 | equal |
| taxi | smemx | 33.01 | 24.77 | 34.12 | 1.378 | 0.982 | equal |
| taxi | topk | 25.01 | 23.75 | 25.43 | 1.071 | 0.708 | equal |
| taxi | topkx | 25.69 | 25.45 | 30.34 | 1.192 | 0.764 | equal |

Spreads above 1.10 are the query batch on a shared pod (the first rounds of
a cell run slower; the per-round ratios are in the JSON and every one of
the five is below 0.80 for `topkx` on Istella-S and below 0.85 on taxi).
The floor probe below (5 calls in one process after a warmup, spreads at
most 1.05 at 4,000 queries) is the tighter number.

Floor probe (`round2/floor_probe.txt`, `round1/floor_probe.txt`; medians of
5 calls, ms):

| dataset | k | queries | base | after0 | smem | smemx | topk | topkx |
|---|---|---|---|---|---|---|---|---|
| istella | 1 | 4000 | 78.98 | 77.83 | 57.28 | 55.07 | 45.98 | 42.50 |
| istella | 10 | 4000 | 80.76 | 81.18 | 60.27 | 57.05 | 57.71 | 53.39 |
| istella | 64 | 4000 | 150.59 | 150.90 | 131.05 | 127.48 | 151.65 | 147.17 |
| istella | 1 | 1 | 6.63 | 5.28 | 5.27 | 7.93 | 6.60 | 6.84 |
| istella | 10 | 1 | 5.27 | 6.31 | 6.25 | 7.80 | 5.26 | 7.96 |
| taxi | 1 | 4000 | 29.53 | 27.50 | 26.68 | 27.78 | 10.13 | 10.71 |
| taxi | 10 | 4000 | 31.79 | 29.75 | 28.99 | 30.05 | 22.91 | 23.42 |
| taxi | 64 | 4000 | 97.60 | 97.42 | 96.60 | 96.84 | 116.16 | not run |
| taxi | 1 | 1 | 1.27 | 0.97 | 0.99 | 1.13 | 0.75 | 0.88 |
| taxi | 10 | 1 | 1.29 | 2.11 | 2.11 | 2.22 | 0.92 | 1.08 |

The one-query Istella-S floor moves by the admission kernels (two launches
over 400,000 rows, about 1.5 ms) and by which arm's launch sequence runs;
it is a launch-count floor and the shipped default (`topkx`, k 10) reads
7.96 ms against base's 5.27. A one-query call has no work to tile; the
lane accepted that trade for the 4,000-query numbers. The exact-chain
admission could be skipped below a query count if the one-query floor
matters; not done here.

Classifier, regressor and KDE (round 1 floor probe, the resident doors;
the classifier's labels are `row % 7`, the regressor's target is column
0; k 10 for both; `resident` read from the instance, never assumed):

| lane | dataset | queries | base ms | after0 ms | smem ms | topk ms |
|---|---|---|---|---|---|---|
| knn-clf | istella | 1 | 41.10 | 9.38 | 9.36 | 9.55 |
| knn-clf | istella | 4000 | 123.98 | 88.79 | 68.70 | 66.01 |
| knn-clf | taxi | 1 | 7.97 | 6.83 | 5.64 | 5.82 |
| knn-clf | taxi | 4000 | 41.87 | 37.88 | 35.12 | 32.14 |
| knn-reg | istella | 1 | 41.16 | 7.99 | 8.00 | 8.20 |
| knn-reg | istella | 4000 | 124.18 | 83.32 | 63.45 | 63.25 |
| knn-reg | taxi | 1 | 3.54 | 1.40 | 1.30 | 1.43 |
| knn-reg | taxi | 4000 | 41.22 | 29.92 | 29.10 | 28.54 |
| kde | istella | 1 | 21.91 | 3.35 | 3.36 | 3.35 |
| kde | istella | 2000 | 68.63 | 51.05 | 51.10 | 51.09 |
| kde | taxi | 1 | 3.87 | 2.83 | 2.81 | 2.81 |
| kde | taxi | 2000 | 49.59 | 41.03 | 41.03 | 41.07 |

KDE race (round 1, `round1/race_kde_summary.tsv`, 2,000 queries, 5 x 3,
spreads at most 1.08): Istella-S 72.56 ms base to 54.03 ms after0 (paired
0.745), taxi 47.45 to 40.93 (0.863), digests equal. The one-query KDE call
went 21.9 to 3.4 ms on Istella-S; what remains at 2,000 queries is the
device work of the tiled pass (DEVIATION 2625), which this lane did not
touch.

## Opponents on the same box (deliverable 4)

cuML 26.8.0 (`cuml-cu12`), cuVS 26.8.1 (`cuvs-cu12`; 26.8.0 has no wheel),
cupy 14.2.0, numpy 2.4.6, the pod's system Python 3.11, index and queries
uploaded before the clock, fit or build before the clock, `kneighbors` or
`search` timed with a device synchronize, 5 calls after a warmup
(`round2/opponents_probe.txt`, `round2/opponents_probe.json`). Our numbers
in the same row are the floor probe's for the shipped default (`topkx`)
and, at k 64, the matrix arm (`smemx`, which the default takes above
k 16); ours include the query upload and the readback, theirs do not.

| dataset | k | queries | ours ms | cuML brute ms | cuVS brute_force ms |
|---|---|---|---|---|---|
| istella | 1 | 4000 | 42.50 | 63.71 | 63.53 |
| istella | 10 | 4000 | 53.39 | 63.57 | 63.07 |
| istella | 64 | 4000 | 127.48 | 65.45 | 64.92 |
| istella | 1 | 1 | 6.84 | 1.98 | 1.32 |
| istella | 10 | 1 | 7.96 | 1.95 | 1.33 |
| taxi | 1 | 4000 | 10.71 | 6.62 | 6.35 |
| taxi | 10 | 4000 | 23.42 | 7.18 | 6.92 |
| taxi | 64 | 4000 | 96.84 | 27.73 | 27.50 |
| taxi | 1 | 1 | 0.88 | 1.05 | 0.83 |
| taxi | 10 | 1 | 1.08 | 1.04 | 0.80 |

Their distance GEMM runs cuBLAS at TF32 by default and their selection is
not the composite-key order, so these are their fast numbers against our
identical ones and nothing more (NO "price of determinism" reading). At
the Istella-S shape (220 features) the shipped default is at their number
at k 1 and 10 and 2x theirs at k 64; at the taxi shape (11 features) it is
1.6x theirs at k 1 and 3.3x at k 10, where the per-cell epilogue, the key
work and the seven column tiles cost more than an 11-step chain; and the
one-query Istella-S floor is 4x to 6x theirs (their search is one launch,
ours is seven column tiles plus the transposition and the admission). The
two-datasets race (`tools/classical_two_datasets.py race --arms
ours,cuml-gpu`, `round1/cuml_race.txt`, the `topk` arm at k 10, 5 rounds)
read Istella-S 64.9 ms ours against 68.7 cuML and taxi 33.0 against 8.6
(that arm's kNN call includes the first-call index upload on round 0 only
and the host validation every round).

KDE, cuML `KernelDensity` (gaussian, euclidean, Scott bandwidth), 2,000
queries against 100,000 fit rows: Istella-S 4.8 ms cuML against our 51.1
(after DEVIATION 3003; 68.6 before), taxi 0.93 against 41.0. One query:
0.83 against 3.35 and 0.29 against 2.81. The remaining gap is the device
pass, not the residency. CORRECTED 2026-09-17: this paragraph first named
the `n_query x n_train` matrix as the cause; that was measured on
2026-09-12 and is not it (`bench/OPPONENT_REFERENCE.md`, the September 12
KDE section, and `kde/impl/neighbors/kernel_density.mojo`, DEVIATION 2690):
the matrix WRITE is about 1.5 ms of a 36.9 ms device entry, the per-cell
distance arithmetic is 25.0 ms, and the fused pass that never writes the
matrix was built, gated bit for bit and read 109.8 ms. What binds is the
contract's serial ascending sum over all `n_train` terms of a query, which
one thread must own. The two-datasets race read the same
(`round1/cuml_race.txt`).

## Identity (deliverable 5)

`tools/identity_break.py`, fixtures base,ties,odd,dupes,wide, two repeats,
IDENTICAL. Lanes, 22: knn, knn-chebyshev, knn-clf, knn-clf-distance,
knn-cosine, knn-manhattan, knn-minkowski-p3, knn-rbc, knn-reg,
knn-reg-distance, knn-sqeuclidean, radius, radius-chebyshev,
radius-manhattan, radius-minkowski-p3, kde, kde-cosine-minkowski,
kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine,
kde-tophat-sqeuclidean, kde-weighted. Columns:

| column | build | rows |
|---|---|---|
| cuda-base | main c85657041, GPU core and estimators | none of this lane |
| cpu-base | main, host core and estimators (`MOJOLEARN_HOST_DIR`) | the CPU reference before |
| cpu-after | the branch, host core and estimators | the CPU reference after (host search untouched) |
| cuda-after0 | the branch, rows off | resident doors only |
| cuda-smem, cuda-smemx | `-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1` (+ `_EXACT_CHAIN=1`) | DEVIATION 3000 (+ 2629 per block) |
| cuda-topk, cuda-topkx | + `-D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1` | DEVIATION 3001 |
| cuda-sabo | topk + `-D MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1` | the tile's reach control |
| cuda-sabox | topkx + `-D MOJOLEARN_KNN_EXACT_CHAIN_SABOTAGE=1` | the admission's reach control |
| cuda-final | the branch, no defines (the shipped defaults) | see the final section |

Diffs (`round1/identity_diff_summaries.txt`, `round2/...`):

| diff | train | infer/model | batch | exit |
|---|---|---|---|---|
| cuda-base, cuda-after0, cuda-smem, cuda-topk (22 lanes) | IDENTICAL 110 | IDENTICAL 220 | IDENTICAL 110 | 0 |
| cuda-base vs cpu-base (before) | IDENTICAL 110 | IDENTICAL 220 | IDENTICAL 110 | 0 |
| cuda-after0 vs cpu-after | IDENTICAL 110 | IDENTICAL 220 | IDENTICAL 110 | 0 |
| cuda-topk vs cpu-after (after) | IDENTICAL 110 | IDENTICAL 220 | IDENTICAL 110 | 0 |
| cpu-base vs cpu-after | IDENTICAL 110 | IDENTICAL 220 | IDENTICAL 110 | 0 |
| cuda-base, cuda-smem, cuda-topk, cuda-smemx, cuda-topkx (15 knn and radius lanes) | IDENTICAL 75 | IDENTICAL 150 | IDENTICAL 75 | 0 |
| cuda-topkx vs cpu-after (15 lanes) | IDENTICAL 75 | IDENTICAL 150 | IDENTICAL 75 | 0 |
| cuda-smemx vs cpu-after (15 lanes) | IDENTICAL 75 | IDENTICAL 150 | IDENTICAL 75 | 0 |
| cuda-topk vs cuda-sabo (the control) | DIVERGENT 20, IDENTICAL 55 | DIVERGENT 20, IDENTICAL 130 | DIVERGENT 20, IDENTICAL 55 | 1 |
| cuda-topkx vs cuda-sabox (the control) | DIVERGENT 22, IDENTICAL 53 | DIVERGENT 22, IDENTICAL 128 | DIVERGENT 22, IDENTICAL 53 | 1 |
| cpu-after vs cuda-sabox | DIVERGENT 22, IDENTICAL 53 | DIVERGENT 22, IDENTICAL 128 | DIVERGENT 22, IDENTICAL 53 | 1 |

No cell REFUSED or MOVED. The DIVERGENT cells are the lanes whose search
runs the transposed IDENTICAL arm on the L2 expanded metrics (knn,
knn-sqeuclidean, knn-clf, knn-clf-distance, knn-reg, knn-reg-distance),
every fixture; the metric lanes (manhattan, chebyshev, cosine, minkowski)
never enter the tile and read IDENTICAL against the sabotage, which is the
dispatch witness. The controls were seen to fail before the pass was read.
The 22-lane and 15-lane counts differ because round 2 ran the kde lanes
only on the arms whose estimators binding changed (after0; the kde lanes
are in the 22-lane rows above and in the final section).

## The shipped default (final section)

`cuda-final` is the branch built with NO defines at the merge candidate
(`bench/results/knn_tiled_2026-09-17/final/`), so it is the NVIDIA IDENTICAL
column a user gets: the smem tile, the per-block exact-chain admission and
the block top-k for k <= 16, plus the resident doors.

Identity, 22 lanes, five fixtures, two repeats: cuda-final vs cpu-after vs
cpu-base IDENTICAL 110 train, 220 infer/model, 110 batch, exit 0;
cuda-base vs cuda-final the same; cuda-final vs cuda-sabox DIVERGENT 22 on
every part (the 15 knn and radius lanes; the 7 kde lanes read ONE-COLUMN
because the sabotage arm did not run them), exit 1.

Race (base vs final, 5 x 3, digests equal):

| lane | dataset | rows | base median ms | final median ms | final min ms | final spread | paired ratio |
|---|---|---|---|---|---|---|---|
| knn | istella | 4000 | 82.27 | 53.11 | 50.42 | 1.082 | 0.647 |
| knn | taxi | 4000 | 32.78 | 24.40 | 24.33 | 1.105 | 0.754 |
| kde | istella | 2000 | 72.25 | 53.73 | 50.85 | 1.059 | 0.744 |
| kde | taxi | 2000 | 47.47 | 40.95 | 40.88 | 1.004 | 0.863 |

Floor probe on the final binding (medians of 5 calls, ms): knn istella
4,000 queries k 1 42.47, k 10 53.36, k 64 126.98; taxi 10.67, 23.45, 96.61;
knn-clf istella 4,000 queries 56.26 (base 123.98), taxi 27.38 (41.87);
knn-reg istella 53.24 (124.18), taxi 23.69 (41.22); one query knn-clf
istella 10.91 (41.10), knn-reg 9.37 (41.16). Against cuML on the same box
through `tools/classical_two_datasets.py race` (k 10, 5 rounds): Istella-S
54.4 ms ours, 68.4 cuML; taxi 26.3 ours, 8.8 cuML.

## Pod

RunPod fc3i4usbkd8ahz, NVIDIA GeForce RTX 4090, $0.74 per hour, created
17:37 UTC, reaped 18:32 UTC and verified gone (HTTP 404), 55 minutes,
about $0.68; leases 150 + 120 minutes, never reached.
`~/mojolearn-evidence/knn-tiled/pod/` holds the create response, the arm
log and the reaped pod id; `pull3/ktd_out/` is the complete pod output.

## Commands

```
# Mac, from the lane worktree (commit first)
export MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key; export RUNPOD_API_KEY=$(cat $MOJOLEARN_RUNPOD_KEY_FILE)
TREES_LEG_NAME=mojolearn-knn-tiled TREES_LEG_STATE=$HOME/mojolearn-evidence/knn-tiled/pod TREES_LEG_CUDA_VERSIONS=13.0 \
  MOJOLEARN_STAGE_KEYS="gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz" \
  sh tools/trees_leg.sh rent --gpu "NVIDIA GeForce RTX 4090" --minutes 150
git archive --format=tar c85657041 -- . ':!bench/results' ':!mamba/corpus' ':!bench/oracle*' ':!archive' ':!upstream' ':!docs' ':!paper' \
  | gzip | sh tools/trees_leg.sh ssh 'rm -rf /root/mojolearn-base && mkdir -p /root/mojolearn-base && cd /root/mojolearn-base && tar xzf -'
sh tools/trees_leg.sh ssh 'apt-get update && apt-get install -y rsync'   # the image lacks rsync; push and pull need it
# pod, from /root/mojolearn (tools/knn_tiled_body.sh, stage by stage; each writes /root/ktd_out/<stage>)
sh tools/knn_tiled_body.sh setup; sh tools/knn_tiled_body.sh rebuild; sh tools/knn_tiled_body.sh phase
sh tools/knn_tiled_body.sh race; sh tools/knn_tiled_body.sh identity; sh tools/knn_tiled_body.sh pip; sh tools/knn_tiled_body.sh opponents
sh tools/knn_tiled_body.sh rebuild2; ... phase2; race2; identity2; cuvs; final-build; final-run
sh tools/trees_leg.sh pull /root/ktd_out ~/mojolearn-evidence/knn-tiled/pull3/
```

## Owed and not done

- Apple and AMD columns of DEVIATIONs 3000 to 3003 at the next release
  record. The three rows are NVIDIA only; the kernels compile on every
  column (the ballot takes the 64-lane half on CDNA) but are untimed and
  unverified there, so the defaults stay off.
- k 64 on the block top-k loses and stays on the matrix; the small-k
  selector at k 64 is 68 ms on both datasets, the largest remaining class.
- The one-query Istella-S floor (about 8 ms shipped, 5 before) is the
  seven-tile launch sequence plus the admission launches; skipping the
  admission below a query count, or a wider index tile now that no matrix
  is written, would recover it. Not done.
- KDE's device pass (51 ms Istella-S against cuML's 4.8) is untouched; the
  residency took the 20 ms of per-call staging that was this lane's.
- The label and target columns of the classifier and regressor are still
  uploaded per call.
- A double-buffered slice loop and a 8 x 8 register tile were not tried.
