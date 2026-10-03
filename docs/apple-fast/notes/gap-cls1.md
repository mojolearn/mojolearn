# gap-cls1: FAST Apple rows behind the best opponent (lane/apple-fast-gap-cls1)

Rows (docs/apple-fast/BOARD_M3_FAST.md): bayesian-ridge taxi 404 vs 73.2 ms, knn istella (classical) 1,805 vs 566,
ard taxi 41.9 vs 17.5, ridge-clf taxi 121 vs 77.9, nearest-centroid taxi 129 vs 96.9. Code read only (no local runs);
line numbers are on the lane head. Every candidate is a build-time define, FAST + Apple only, default OFF.

## Time-breakdown hypotheses

### bayesian-ridge taxi (algos: fit 1M x 16 + predict 100k; tools/bench_board_algos.py:229)
Route: BayesianRidge.fit (python/mojolearn/_expansion_linear.py) -> x_linear_fit -> fit_device, the Bayes grid driver.
- `bayes_xty_kernel` (x_linear/device.mojo:303): one THREAD per column walks all 1M rows (16 threads, latency-bound
  serial chains with strided loads): the likely dominant cost (100+ ms).
- Means + centered Gram from the moments tiles (device.mojo:3563, mg_means/mg_cross): taxi's 16 features are ONE
  tile, so ONE block walks 1M rows for 136 cells.
- `bayes_yparts`/`bayes_yvar_parts`/`bayes_part_kernel` (device.mojo:278, 456): one thread per 4,096-row block
  (245 threads, each 4,096 serial loads), the part pass every guarded iteration that is not trusted.
- Host loop: every iteration waits twice (witness `ok` synchronize, x_linear/witness.mojo:75, then the state read
  device.mojo:3826/3884) and runs one-thread step kernels.
- X upload 64 MB and the host scalar finiteness scan of 16M values (bindings/_mojolearn_x_linear.mojo:82) are
  shared by every x_linear row (not addressed here; a device finiteness check is a follow-up).
Candidates: MOJOLEARN_BAYES_FAST_CLS1_STATS (fast grid Gram gives means, Gram, X'y: kills both one-block passes;
same path as the never-A/B'd KERNEL_FAST_BAYES_STATS), _PARTS (block-per-row-block partials,
x_linear/cls1_fast.mojo c1_sq/sum/dev_parts), _BATCH (8 guarded iterations per wait, the stop / trust / count words
read on the device: device.mojo c1_bayes_* kernels, loop at the `BAYES_CLS1_BATCH:` block in fit_device), all three.

### ard taxi (algos: fit 100k x 16, max_iter 300; x_linear/ard_grid.mojo)
- Every iteration ends in `wit.ok` (ard_grid.mojo:333), a full synchronize, although the kernels already no-op
  after the stop word (the AG_BATCH design is defeated): ~4 launches + 1 wait per iteration.
- `ard_part_kernel` (ard_grid.mojo:161): 25 threads each walking 4,096 rows per iteration.
- Moments on one block (ard_grid.mojo:270; 16 features = one tile).
Candidates: MOJOLEARN_ARD_FAST_CLS1_BATCH (one wait per 8 iterations), _PARTS, _STATS (fast grid Gram), all three.

### ridge-clf taxi (algos: fit 1M x 16 binary, predict 100k)
The device fit is already the fast grid Gram (ridge taxi regression is 16.2 ms on the board), so ~100 ms is Python glue:
- `_classes` (_expansion_linear.py:160) builds a 1M Python float list from the codes, `codes.tolist()` again,
  then `Y = [1.0 if c == 1 else -1.0 ...]` (:960) and `Array.from_list` of 1M: three Python passes over 1M rows.
- predict: `scores.tolist()` + a Python threshold per row (:184) + label decode.
Candidates: MOJOLEARN_RIDGE_FAST_CLS1_CODES (int32 codes straight to the binding, +-1 targets built on the device:
cls1_fast.mojo c1_codes_targets_kernel, device.mojo:140), _PREDICT (device class code per row,
device.mojo:171 + `x_linear_decision_codes`, native gather), both. Python reads `x_linear_cls1_flags`.

### nearest-centroid taxi (algos: fit 1M x 16, predict + predict_proba 100k)
- `_labels_of` (_expansion_neighbors.py:183): y.tolist(), set, dict lookup per row (1M); the Python counting loop
  (:512); `_i32(codes)` from a list.
- predict: `idx.tolist()` + Python list per row + `_class_array` (:601).
The device statistics (nc_stats, chunked on FAST Apple) are not the gap.
Candidates: MOJOLEARN_NC_FAST_CLS1_LABELS (native encoder codes + device class counts, x_neighbors/nc_cls1.mojo),
_PREDICT (int32 indices to the native gather), both. Python reads `x_neighbors_cls1_flags`.

### knn istella (classical: kneighbors 4,000 x 400,000 x 220, k = 64; fit before the clock)
Route: OursKNN (tools/classical_two_datasets.py) -> NearestNeighbors.kneighbors -> `knn_search_resident`
(python/mojolearn/neighbors.py:833) -> neighbors/estimator.mojo `_knn_search_on_device_index` ->
`brute_force_knn_impl` (knn_brute_force.mojo) -> `fast_mma_knn` (knn_brute_force.mojo:1875) chunked arm
`fast_mma_bigd_kernel[64, 1]`. So the classical driver DOES reach the matrix-unit arm once KNN_FAST_MMA_K64 is on
(default on main since 857fd5804; fast_mma_knn.mojo:122). The board's 1,805 ms predates that merge (afb24-knn
re-timed knn-clf / knn-reg only); the K64 A/B on this exact lane was 1,581 -> 352 ms. Expect the row already
below sklearn's 566 at main; `gapcls1-k64chk-istella` re-times main (K64_OFF vs default).
Remaining cost: K = 64 sorted insertions in registers; every candidate is admitted until a lane's list fills
in every one of the ~441 (query block, slice) pairs.
Candidates: MOJOLEARN_KNN_FAST_CLS1_PRESEED (a 4,096-row first pass gives each query an exact upper bound on its
k-th distance; the main pass starts its thresholds there: fast_mma_knn.mojo:458), _SLICES2 (960 blocks target,
fast_mma_knn.mojo:720), both.

## A/B lines
See the lq tags gapcls1-* (keep_tags.txt). Quality column per row as the board's (r2/rmse, accuracy/logloss, recall@k).
