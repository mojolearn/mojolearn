# lane/python-hotpath: per-row Python in the NumPy-free layer, replaced by core host helpers

Written 2026-09-17 for a reader with no memory. Branch `lane/python-hotpath`,
worktree `/Users/andrewhendel/mojolearn-wt/python-hotpath`, branched from main
86d33fcdf. Evidence: `/Users/andrewhendel/mojolearn-evidence/python-hotpath/`
(Mac runs at the top level, the x86 pod's under `pod/remote/leg_out/`).
DEVIATIONS 3100 to 3107. NOTHING HERE IS A NUMERIC SEAM OF AN ESTIMATOR: the
helpers cast, compare, count, gather and copy, and every one is the same
function of the same bytes as the Python routine it stands in for.

## What was measured first (the attribution)

The Python surface walks rows in the interpreter in places that are invisible
at 10,000 rows and cost hundreds of milliseconds to tens of seconds at
1,000,000. The audit was a repo-wide grep of `tolist()`, `flat_view`,
`from_list`, `array.array` and per-row comprehensions over the package (the
raw list is 200 sites; the ones that scale with rows are in the three tables
below), then each hit timed ALONE on synthetic inputs at 1M and 4M rows, one
core, by `bench/speed/python_hotpath_cells.py` (owned hits) and
`bench/speed/python_hotpath_other_lanes.py` (hits in files this lane does not
own). The largest before the change, x86 EPYC 7713, one thread, 1M rows: the
`radius_neighbors` per-row split 18,022 ms and the weighted `_column_means`
7,597 ms (both in files this lane does not own), then, owned, the
`cross_val_score` glue 5,935 ms, `as_f32_c` of a list of lists 5,911 ms,
`_indices` plus the overlap check 4,897 ms, `_default_folds` stratified 2,799
ms and `_classification_pair` plus encode 1,829 ms. The per-element cost of
`array.array(code, memoryview)` (one Python object per element) is 20 to 33
ns on the M4 and 43 to 72 ns on the EPYC pod.

## What changed, and why no bit moves

`bindings/hotpath_helpers.mojo` (805 lines) exports fourteen helpers from the
core host binding (`bindings/_mojolearn_core_host.mojo`): `cast_elements`
(DEVIATION 3100), `reduce_stat` (3101), `equal_elements` (3102),
`encode_labels_{f32,f64,i32,i64,u32,u8}` and `gather_i32` (3103),
`check_indices_i64`, `indices_overlap_i64`, `fold_ids`, `select_fold_i64`
(3104). The Python callers in `_array.py`, `_buffer.py`, `_labels.py`,
`_metrics_impl.py` and `model_selection.py` take the helper when the loaded
binding carries it and otherwise run the Python routine, which stays in the
package as the DEFINITION and the fallback. `MOJOLEARN_HOTPATH=python` forces
the Python routine everywhere (the reference arm of every test and A/B here).
Two more seams are pure Python: `_flatten_fast` (3106, one `isinstance` walk
instead of two per leaf), the owned strided-view store (3105), and the
metrics label cache `_EncodedLabels` (3107).

Why no bit moves: `cast_elements` performs the ONE conversion per element that
`array.array`'s C item setter performs (int to float32 THROUGH float64, so an
int64 beyond 2**53 rounds twice on both paths, DEVIATION 2306); any element
the Python path would refuse makes the helper return 1 and the caller reruns
the Python path, which raises its own words. `reduce_stat` is SEQUENTIAL in
storage order on one thread because Python's `min`, `max`, `sum` and argmax
are and order is observable (`min([nan, 1.0])` is nan, `min([1.0, nan])` is
1.0, `min([0.0, -0.0])` is 0.0, a float64 sum rounds per addition): same
comparisons, same order, same answer. The label encoder is the ORDER RULE's
existing code (DEVIATION 2500) verbatim; the index and fold helpers are
integer bookkeeping. The parallel helpers (`cast_elements`, `equal_elements`,
`gather_i32`) split contiguous ranges across the host pool under
`MOJOLEARN_CPU_THREADS`; a range is written by one task and no element
depends on another, so the thread count moves no byte (the x86 pod's
default-thread A/B column below hashes equal to its one-thread column).

A side effect worth its own line: before this lane a CPU-ONLY install ran the
ORDER RULE's label encoder as a Python loop because the core host binding did
not export `encode_labels_*` (`_labels.py` documented the fallback honestly).
It exports them now, so `encode_labels(int64 buffer)` reads 310 ms before and
9.5 ms after on the M4; `MOJOLEARN_HOTPATH=python` cannot revert that one
because the switch guards the new helpers, not the encoder's export.

## The differential test and the sabotage arm

`python/mojolearn/tests/test_hotpath_native.py` (717 lines, 230 cases) runs
every seam down both arms on randomized and awkward inputs (empty, one
element, above the helpers' 65,536-element one-task threshold, NaN, -0.0,
infinities, out-of-range ints, wrong dtypes, F-order, strided views) and holds
them to the SAME RESULT byte for byte or the SAME REFUSAL, type and words. A
comparison of a routine with itself proves nothing, so every case pins which
path answered: the helpers a case expects are wrapped with a counter in
`_buffer._NATIVE`; the new arm must call them and the reference arm must not.

The negative control is `-D MOJOLEARN_HOTPATH_SABOTAGE=1` (commit 574b72574),
which sabotages THESE HELPERS ALONE so a divergence under it cannot be the
k-NN or k-means fold's (`-D MOJOLEARN_HOST_SABOTAGE=1`, the family's existing
define, also trips it); `core_host_sabotage()` reports either and `_backend`
refuses such a binary without `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`. Under it the
cast writes each run of eight elements reversed, min and max trade places, the
float sum folds descending, argmax takes the last maximum, equality is
inverted, the encoder's codes are reversed, the gather reads the next table
slot, a duplicate index is accepted, an overlap is denied and the fold ids are
rotated by one.

| run | binding | clean | sabotage build, strict | sabotage build, MOJOLEARN_HOTPATH_EXPECT_SABOTAGE=1 |
|---|---|---|---|---|
| M4, one core, 826e7836b sources (`host-a`, `host-sab`) | sha256 51345093 (clean) | 230 passed (`test_clean.txt`) | 111 failed, 119 passed (`test_sabotage_strict.txt`) | 230 passed (`test_sabotage_expect.txt`) |
| x86 EPYC 7713 pod c3zry4j7ttc7jy, tip 59307e477 | sha256 fb520851 (clean), 3f95c277 (sabotage) | 230 passed, and 230 passed under MOJOLEARN_CPU_THREADS=1 | 111 failed, 119 passed | 230 passed |
| M4, one core, tip 59307e477 rebuilt 2026-09-17 21:32 (`host-tip`, this session) | sha256 9397d56e, read from inside the process, `core_host_sabotage()` False | 230 passed in 2.35 s (`test_clean_tip.txt`) | not rerun (sources unchanged since the pod's run) | not rerun |

Under the strict sabotage run 18 of the 25 test functions fail (111 of 230
parametrized cases: 30 `test_astype_matches_the_item_setter`, 25
`test_default_folds_match`, 18 `test_classification_labels_match`, 17
`test_cluster_label_union_matches`, and the rest one to four each); the seven
that stay green are the pure-Python seams the define cannot reach
(`_flatten_fast`, the block copy, the plain-label paths, the fold sabotage
control) and `test_sabotage_build_diverges`, which asserts only under
EXPECT_SABOTAGE, where it saw every helper group diverge. The pod also ran the
related files (`test_labels_native`, `test_model_selection_numpy_free`,
`test_numpy_free_core`, `test_device_array_refusal`,
`test_classification_metrics`, `test_regression_metrics`): 196 passed, 11
skipped (`test_related.txt`).

## Identity (tools/identity_break.py, five fixtures, two repeats, CPU column)

Lanes: every lane whose CPU path reaches the changed files: knn, knn-clf,
knn-reg, kmeans, kmeans-random, kmeans-array, kmeans-weighted,
knn-sqeuclidean, knn-clf-distance, knn-reg-distance, knn-manhattan,
knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc, radius,
radius-manhattan, radius-chebyshev, radius-minkowski-p3, kmeans-sqrt,
kmeans-classic-pp, cross-val-folds (22 lanes, 110 train cells, 210
infer/model cells, 105 batch cells, 5 n/a for the function lane).

| comparison | columns | train | infer/model | batch | file |
|---|---|---|---|---|---|
| main 86d33fcdf (`base-src`, `host-before`) vs branch 826e7836b (`host-a`), M4 | cpu-apple-m4 x2 | IDENTICAL 110 | IDENTICAL 210, N/A 10 | IDENTICAL 105, N/A 5 | `identity_diff.txt` |
| branch, helpers vs `MOJOLEARN_HOTPATH=python`, tip 59307e477, x86 EPYC 7713 | cpu-amd-epyc-7713 x2 | IDENTICAL 110 | IDENTICAL 210, N/A 10 | IDENTICAL 105, N/A 5 | `pod/remote/leg_out/identity_after_x86.json` vs `identity_reference_arm_x86.json` |
| branch on the M4 (826e7836b) vs branch on the EPYC (tip) | cpu-apple-m4 vs cpu-amd-epyc-7713 | IDENTICAL 110 | IDENTICAL 210, N/A 10 | IDENTICAL 105, N/A 5 | diffed this session |
| branch vs the SABOTAGE binding (`host-sab`, `-D MOJOLEARN_HOTPATH_SABOTAGE=1`), M4 | cpu-apple-m4 x2 | DIVERGENT 53, IDENTICAL 57 | DIVERGENT 59, IDENTICAL 151 | DIVERGENT 14, BATCH_MOVED 35, IDENTICAL 56 | `identity_diff_sabotage.txt` |

The sabotage diverges exactly where the helpers are reached: the k-NN
classifiers (label encoding and the index widening), plain kneighbors and the
radius lanes (the index `astype`), and cross-val-folds (the fold ids); the
k-NN regressors and every k-means lane, which reach no helper, stay IDENTICAL
under it. That contrast is the reach proof. The Mac "after" column predates
the sabotage-define commit; the pod's columns are at the tip and the tip's
Mojo sources equal 574b72574's (59307e477 touched only `bench/`).

## The box

RunPod CPU pod `c3zry4j7ttc7jy` (`mojolearn-cpu-python-hotpath-20260917-203758`),
image runpod/base:1.3.1-ubuntu2204, AMD EPYC 7713 (8 vCPU requested, the
container reports one schedulable CPU to `nproc`), Mojo 1.0.0 (ed45d567),
Python 3.13 from the pixi `test` env, source at 59307e477 with zero dirty
tracked files, bindings built on the pod (26 s clean, 25 s sabotage, both
promoted to the R2 bincache). Command 1,800 s, billed 2,022 s, $0.1348 at
$0.24/h. Reaped by the leg itself: DELETE 204, GET 404, absent from the pod
listing (`pod_leg.log`). Binding mtime 20:40:57Z, source mtime 20:14:40Z: the
binary postdates the source. Runner `tools/runpod_cpu_leg.sh`; the command is
`pod/user_cmd.sh`.

## The A/B (bench/speed/python_hotpath_ab.py)

Arms alternate within one process: P = `MOJOLEARN_HOTPATH=python`, N = the
seams. Per arm the minimum and the spread (max/min) over the rounds; a side
whose spread exceeds 1.10 is `u` and no ratio is quoted from it; the result
hash of the two arms must be equal (`same`), and it was in EVERY cell of every
run (1M one thread, 1M default threads, 4M one thread on x86; 1M and 4M on the
M4). x86: 5 rounds, `MOJOLEARN_CPU_THREADS=1`. The Mac's runs (7 rounds at 1M,
5 at 4M) were taken at load average 14 to 77 on the shared M4 and are NOT
quotable for a ratio; they are listed in the second table as attribution only.
The three `tolist`, `__iter__` and list-of-lists cells were skipped at 4M
(memory). Qualified cells (both sides inside the gate): 31 of 52 at 1M, 28 of
49 at 4M. The sides that miss it are mostly N sides under 12 ms (timer noise
on a shared vCPU: 14 of the 18 `u` N sides at 1M), plus at 1M the P sides of
`Array.min() float32` (39 ms, 1.14), `_sample_weight_f32` (53 ms, 1.15), the
`kl_divergence` scan (47 ms, 1.17), `finite_integer_codes` (37 ms, 1.19) and
`tolist` (1,219 ms, 1.12), and at 4M the N sides of `_default_folds`
stratified (149 ms, 1.36), `encode_labels(list of int)` (194 ms, 1.27) and
`_classification_pair` on str (1,290 ms, 1.17); those cells' ratios are not
quoted, their minima are in the table. Geometric mean of the 31 qualified 1M
ratios: 6.60x; this lane's merge gate is P against N on the same pod and
needs no opponent.

### Table 1: the ranked audit, owned hits (ranked by the Python arm's ms at 1M rows on x86)

Mac column: `bench_before_1m.json`, main's code with `host-before`, best of
three, under load. x86 columns: `pod/remote/leg_out/ab_1m_x86_one_thread.json`
and `ab_4m_x86_one_thread.json`. Ratio only where both sides are inside the
1.10 gate. "-" is a cell that run did not include.

| # | cell | file:line | iterates | big-O | Mac P 1M ms (before, load) | x86 P 1M ms (spread) | x86 N 1M ms (spread) | x86 ratio 1M | x86 P 4M ms (spread) | x86 N 4M ms (spread) | x86 ratio 4M | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cross_val_score glue end to end: stratified folds + _indices x10 + overlap x5 | python/mojolearn/model_selection.py:453 | stratified folds, then `_indices` and the overlap check per fold | O(rows x folds) | - | 5934.6 (1.02) | 45.3 (1.03) | 131.04x | 25440.2 (1.01) | 266.4 (1.01) | 95.49x | fixed (DEVIATION 3104) |
| 2 | _buffer.as_f32_c(list of lists [n,10] float) | python/mojolearn/_buffer.py:660 (walk python/mojolearn/_array.py:891, :844) | two isinstance checks per leaf, then `array.array` per element | O(rows x cols) | 1564.9 | 5910.6 (1.02) | 697.9 (1.04) | 8.47x | - | - | - | fixed: `_flatten_fast` (DEVIATION 3106) |
| 3 | model_selection._indices x10 + overlap x5 (cross_val_score glue) | python/mojolearn/model_selection.py:99, :88 (:96) | `set(train.tolist()).intersection(test.tolist())` per fold | O(rows) per fold | 1866.6 | 4897.4 (1.03) | 275.8 (1.05) | 17.76x | 21216.9 (1.02) | 1126.0 (1.09) | 18.84x | fixed: `check_indices_i64` + `indices_overlap_i64` (DEVIATION 3104) |
| 4 | Array.__iter__ rows of f32 [n,10] (n/10 rows timed, x10) | python/mojolearn/_array.py:598 | one Array object per row | O(rows) | 1888.2 | 4863.0 (1.01) | 4866.7 (1.00) | 1.00x | - | - | - | not changed: unavoidable for this API |
| 5 | model_selection._default_folds stratified 5 (int64 y) | python/mojolearn/model_selection.py:168, :243 | one Python fold assignment per row, then `from_list` per index | O(rows) | 416.3 | 2799.1 (1.01) | 37.1 (1.08) | 75.49x | 11758.3 (1.02) | 148.8 (1.36 u) | u | fixed: `fold_ids` + `select_fold_i64` (DEVIATION 3104) |
| 6 | model_selection._default_folds kfold 5 (f32 y) | python/mojolearn/model_selection.py:168, :243 | one Python fold assignment per row, then `from_list` per index | O(rows) | 195.5 | 2320.7 (1.02) | 8.64 (1.11 u) | u | 9913.9 (1.02) | 103.5 (1.15 u) | u | fixed (DEVIATION 3104) |
| 7 | metrics._classification_pair+_encode (int64 x2) | python/mojolearn/_metrics_impl.py:1158, :1334 | flatten of both, a dict lookup per label to encode | O(rows) | 798.6 | 1828.8 (1.01) | 29.1 (1.10 u) | u | 7447.8 (1.02) | 115.5 (1.03) | 64.48x | fixed (DEVIATION 3107) |
| 8 | metrics._prepare_cluster_labels(int64 x2) | python/mojolearn/_metrics_impl.py:252 (:276-:278) | `tolist()` of both, `sorted_classes(tl + pl)` per label | O(rows) | 627.3 | 1447.9 (1.01) | 39.8 (1.02) | 36.37x | 5991.8 (1.01) | 164.5 (1.05) | 36.43x | fixed: `_EncodedLabels` (DEVIATION 3107) |
| 9 | Array.tolist() float32 [n,10] | python/mojolearn/_array.py:490 | one Python float per element: the output IS the objects | O(rows x cols) | 483.8 | 1219.3 (1.12 u) | 1334.0 (1.02) | u | - | - | - | not changed: unavoidable for this API |
| 10 | metrics._classification_pair+_encode (list of str x2) | python/mojolearn/_metrics_impl.py:1158, :1334 | flatten of both, a dict lookup per str label (the objects stay) | O(rows) | 460.4 | 1163.2 (1.01) | 342.8 (1.01) | 3.39x | 4801.4 (1.02) | 1290.3 (1.17 u) | u | partly |
| 11 | _labels.encode_labels(list of float) | python/mojolearn/_labels.py:216 | flatten walk, set and dict per label, `from_list` per code | O(rows) | - | 1011.8 (1.01) | 47.7 (1.05) | 21.21x | 4132.8 (1.04) | 207.3 (1.12 u) | u | fixed (DEVIATION 3103) |
| 12 | metrics._binary_ranking_inputs+encode (int64 y, f32 score) | python/mojolearn/_metrics_impl.py:1233, :1140 | a compare per label into an int32 list | O(rows) | 397.1 | 932.8 (1.01) | 13.6 (1.02) | 68.54x | 3790.9 (1.03) | 54.5 (1.04) | 69.61x | fixed: `_label_map` (DEVIATION 3107) |
| 13 | _labels.argmax_rows(F-order f32 [n,7], Python arm) | python/mojolearn/_labels.py:434 | one Python argmax per row when scores are F-order | O(rows x classes) | 369.1 | 805.1 (1.04) | 491.7 (1.04) | 1.64x | 3359.7 (1.01) | 1942.0 (1.05) | 1.73x | partly: C-order path native; F-order still Python per row |
| 14 | metrics._as_i32_1d(list of int) | python/mojolearn/_metrics_impl.py:164 | flatten walk then `array.array` per element | O(rows) | 313.4 | 767.6 (1.01) | 275.0 (1.01) | 2.79x | 3072.3 (1.02) | 1065.5 (1.04) | 2.88x | partly (DEVIATION 3106) |
| 15 | _buffer.as_f32_c(int32 buffer [n,10]) | python/mojolearn/_buffer.py:660, :498 | `array.array(code, memoryview)` per element | O(rows x cols) | 312.0 | 745.1 (1.03) | 25.5 (1.03) | 29.25x | 2829.1 (1.04) | 103.8 (1.06) | 27.26x | fixed (DEVIATION 3100) |
| 16 | _labels.encode_labels(list of int) | python/mojolearn/_labels.py:216 | flatten walk, then `sorted_classes`, then `Array.from_list` per code | O(rows) | 306.0 | 720.3 (1.02) | 47.8 (2.09 u) | u | 2974.9 (1.02) | 193.8 (1.27 u) | u | fixed (DEVIATION 3103) |
| 17 | _buffer.as_f32_c(strided f32 view [n,10] of [2n,10]) | python/mojolearn/_buffer.py:473 | a strided view copied element by element through `tolist` | O(rows x cols) | 236.6 | 533.4 (1.06) | 68.0 (1.04) | 7.84x | 2140.6 (1.01) | 284.9 (1.08) | 7.51x | fixed: owned store from a strided view (DEVIATION 3105) |
| 18 | _labels.encode_labels(list of str) | python/mojolearn/_labels.py:216 | flatten walk, then `sorted_classes`, then `from_list` per code; str objects stay | O(rows) | 217.9 | 528.3 (1.03) | 104.9 (1.05) | 5.04x | 2229.7 (1.03) | 413.8 (1.03) | 5.39x | partly (C-driven map) |
| 19 | _buffer.as_i64_c(list [n] int) | python/mojolearn/_buffer.py:704 | one isinstance walk per int leaf, then `array.array` per element | O(rows) | 193.0 | 481.6 (1.04) | 61.1 (1.02) | 7.88x | 1985.0 (1.03) | 251.7 (1.01) | 7.89x | fixed (DEVIATION 3106) |
| 20 | _labels.decode_labels(int classes, int32 codes) | python/mojolearn/_labels.py:371 | int32 codes widened per element before the gather | O(rows) | - | 396.4 (1.02) | 1.96 (1.21 u) | u | 1716.3 (1.03) | 11.9 (1.38 u) | u | fixed: `gather_i32` (DEVIATION 3103) |
| 21 | Array.__getitem__ a[:, 2:5] of f32 [n,10] (strided, not a block) | python/mojolearn/_array.py:617 | a strided column slice copied element by element | O(rows x cols) | - | 362.2 (1.01) | 364.1 (1.02) | 0.99x | 1584.1 (1.02) | 1588.3 (1.02) | 1.00x | not changed: strided copy still Python |
| 22 | _buffer.as_f32_c(flat list [n] float, ndim=1) | python/mojolearn/_buffer.py:660 | one isinstance walk per leaf, then `array.array` per element | O(rows) | 126.3 | 356.6 (1.05) | 49.0 (1.10 u) | u | 1518.7 (1.02) | 203.3 (1.06) | 7.47x | fixed (DEVIATION 3106) |
| 23 | Array.__getitem__ row range a[n//10:] of f32 [n,10] | python/mojolearn/_array.py:617 | a row block copied element by element | O(rows x cols) | - | 344.3 (1.03) | 88.4 (1.03) | 3.90x | 1493.1 (1.02) | 434.2 (1.06) | 3.44x | fixed: owned block store (DEVIATION 3105) |
| 24 | _labels.sorted_classes(list of int, 7 classes) | python/mojolearn/_labels.py:156 | a set insert and a dict lookup per label | O(rows) | 130.7 | 252.4 (1.04) | 67.6 (1.07) | 3.73x | 1059.8 (1.02) | 291.5 (1.00) | 3.64x | fixed: `encode_labels_i64` on the core host binding (DEVIATION 3103) |
| 25 | _labels.flatten_labels(int64 Array) | python/mojolearn/_labels.py:91 | `tolist()` of the whole buffer, then one isinstance walk per label | O(rows) | 53.9 | 125.1 (1.03) | 25.1 (1.05) | 4.97x | 539.1 (1.03) | 88.7 (1.03) | 6.08x | fixed: plain-label C path (DEVIATION 3103) |
| 26 | _labels.flatten_labels(list of int) | python/mojolearn/_labels.py:91 (walk python/mojolearn/_array.py:891, fast path :844) | one isinstance walk per label to flatten and vet a nested list | O(rows) | 50.1 | 117.9 (1.10) | 11.5 (1.41 u) | u | 493.8 (1.02) | 46.8 (1.03) | 10.56x | fixed: `_flatten_fast` (DEVIATION 3106) |
| 27 | metrics._as_i32_1d(int32 buffer) | python/mojolearn/_metrics_impl.py:164 | `array.array` per element with a range check | O(rows) | 62.0 | 113.8 (1.03) | 2.18 (1.15 u) | u | 450.5 (1.03) | 12.1 (1.29 u) | u | fixed (DEVIATION 3100) |
| 28 | Array._as_order('F') f32 [n,10] Python _reorder | python/mojolearn/_array.py:477, :178 | memoryview slice per column (C-driven) | O(rows x cols) | 44.7 | 108.4 (1.03) | 108.0 (1.03) | 1.00x | 451.7 (1.05) | 452.8 (1.05) | 1.00x | not changed: no per-row Python body |
| 29 | _buffer.as_f32_c(bool buffer [n]) | python/mojolearn/_buffer.py:660, :498 | `array.array` per element | O(rows) | 45.1 | 105.3 (1.01) | 44.0 (1.02) | 2.39x | 426.3 (1.03) | 178.4 (1.04) | 2.39x | fixed (DEVIATION 3100) |
| 30 | Array.min()+max() int64 [n] | python/mojolearn/_array.py:761, :767 | `tolist()` then builtin `min`/`max` | O(rows) | 33.2 | 89.0 (1.03) | 1.52 (1.07) | 58.57x | 390.1 (1.01) | 7.15 (1.29 u) | u | fixed: `reduce_stat` (DEVIATION 3101) |
| 31 | _labels.sorted_classes(list of str, 7 classes) | python/mojolearn/_labels.py:156 | a set insert and a dict lookup per str label (the objects stay) | O(rows) | 45.6 | 76.6 (1.03) | 46.2 (1.01) | 1.66x | 295.5 (1.02) | 172.5 (1.02) | 1.71x | partly: C-driven set and map, objects unavoidable |
| 32 | _buffer.as_f32_c(int64 buffer [n], ndim=1) | python/mojolearn/_buffer.py:660, cast at :498 | `array.array(code, memoryview)`: one Python object per element (20 to 33 ns on the M4, 43 to 72 ns on the EPYC pod) | O(rows) | 31.2 | 72.0 (1.01) | 0.70 (1.51 u) | u | 275.8 (1.03) | 3.08 (1.35 u) | u | fixed: `cast_elements` (DEVIATION 3100) |
| 33 | metrics._as_i32_1d(int64 buffer) | python/mojolearn/_metrics_impl.py:164 | `array.array` per element with a range check | O(rows) | 36.8 | 68.8 (1.01) | 1.87 (1.05) | 36.82x | 286.7 (1.03) | 10.5 (1.15 u) | u | fixed (DEVIATION 3100) |
| 34 | Array.__eq__(Array) int64 [n] | python/mojolearn/_array.py:692 | `tolist()` of both then one compare per element | O(rows) | 30.2 | 59.3 (1.04) | 1.01 (1.18 u) | u | 307.3 (1.09) | 4.56 (1.46 u) | u | fixed: `equal_elements` (DEVIATION 3102) |
| 35 | Array.astype('<i8') from float32 [n] | python/mojolearn/_array.py:503 | `array.array(code, values)` per element | O(rows) | 22.2 | 54.2 (1.03) | 0.66 (1.13 u) | u | 217.1 (1.08) | 3.63 (1.23 u) | u | fixed: `cast_elements` (DEVIATION 3100) |
| 36 | metrics._sample_weight_f32(f32 buffer) | python/mojolearn/_metrics_impl.py:212 (:228) | builtin `min` over a float view | O(rows) | 17.5 | 53.2 (1.15 u) | 1.65 (1.18 u) | u | 360.8 (1.04) | 6.54 (1.21 u) | u | fixed: `reduce_stat` (DEVIATION 3101) |
| 37 | Array.argmax() float32 [n] | python/mojolearn/_array.py:788 | `tolist()` then a Python scan | O(rows) | 26.6 | 52.5 (1.05) | 0.69 (1.03) | 76.04x | 247.7 (1.07) | 2.80 (1.20 u) | u | fixed (DEVIATION 3101) |
| 38 | _buffer.as_i32_c(int64 buffer [n]) | python/mojolearn/_buffer.py:700, :498 | `array.array` per element with a range check | O(rows) | 24.6 | 51.6 (1.00) | 0.41 (1.83 u) | u | 205.2 (1.11 u) | 3.48 (1.45 u) | u | fixed (DEVIATION 3100) |
| 39 | Array.sum() float32 [n] | python/mojolearn/_array.py:773 | `tolist()` then `math.fsum`-free sequential sum | O(rows) | 17.0 | 47.7 (1.06) | 1.01 (1.04) | 47.23x | 234.0 (1.03) | 4.08 (1.10) | 57.36x | fixed (DEVIATION 3101) |
| 40 | metrics.kl_divergence negativity scan (Array.min x2) | python/mojolearn/_metrics_impl.py:815 (:834) | `Array.min()` twice | O(rows) | 29.9 | 47.4 (1.17 u) | 1.39 (1.15 u) | u | 353.0 (1.04) | 5.60 (1.12 u) | u | fixed through `Array.min` (DEVIATION 3101) |
| 41 | _serialize write_npz+read_npz 1000 trees x n/1000 nodes (5 arrays) | python/mojolearn/_serialize.py:253 | one list index per ARRAY (not per row) | O(arrays) | 21.5 | 44.7 (1.04) | 44.8 (1.11 u) | u | 324.2 (1.04) | 321.8 (1.03) | 1.01x | control: no per-row Python found |
| 42 | Array.astype('<i4') from int64 [n] | python/mojolearn/_array.py:503 | `array.array(code, values)` per element with a range check | O(rows) | 19.6 | 42.6 (1.01) | 0.32 (1.77 u) | u | 171.0 (1.14 u) | 3.67 (1.28 u) | u | fixed (DEVIATION 3100) |
| 43 | Array.min() float32 [n] | python/mojolearn/_array.py:761 | `tolist()` then builtin `min` | O(rows) | 14.1 | 39.1 (1.14 u) | 0.70 (1.09) | u | 196.3 (1.07) | 2.78 (1.17 u) | u | fixed (DEVIATION 3101) |
| 44 | _labels.finite_integer_codes(float32 Array) | python/mojolearn/_labels.py:461 | `set(flat_view(arr))` over every element | O(rows) | 11.7 | 37.1 (1.19 u) | 43.7 (1.02) | u | 152.7 (1.02) | 152.5 (1.00) | 1.00x | not changed (C-driven set) |
| 45 | _labels.decode_labels(str classes, int64 codes) | python/mojolearn/_labels.py:337 | one list index per code; output is a Python list of str | O(rows) | 14.0 | 31.3 (1.04) | 31.5 (1.05) | 0.99x | 153.2 (1.02) | 151.5 (1.02) | 1.01x | not changed: output objects are the cost |
| 46 | _labels.decode_labels(bool classes, int64 codes) | python/mojolearn/_labels.py:337 | one list index per code; output is a Python list of bool | O(rows) | 13.7 | 31.0 (1.03) | 30.7 (1.04) | 1.01x | 146.0 (1.02) | 144.8 (1.02) | 1.01x | not changed |
| 47 | _buffer.as_f32_c(f64 buffer [n,10]) native cast (control) | python/mojolearn/_buffer.py:660 | native cast already | O(rows x cols) | 3.36 | 27.5 (1.03) | 27.1 (1.06) | 1.01x | 119.4 (1.05) | 117.2 (1.08) | 1.02x | control: already native |
| 48 | _labels.encode_labels(float32 buffer) | python/mojolearn/_labels.py:216, :269 | a CPU-only install ran the ORDER RULE encoder as a Python loop, float32 codes | O(rows) | 419.8 | 15.5 (1.04) | 15.5 (1.06) | 1.00x | 61.1 (1.01) | 61.1 (1.01) | 1.00x | fixed: export added |
| 49 | _labels.encode_labels(int64 buffer) | python/mojolearn/_labels.py:216, :269 | a CPU-only install ran the ORDER RULE encoder as a Python loop (no `encode_labels_i64` export on the core host binding) | O(rows) | 310.2 | 13.2 (1.01) | 13.1 (1.02) | 1.01x | 51.5 (1.02) | 51.4 (1.01) | 1.00x | fixed: export added; the P arm cannot revert it (see note) |
| 50 | _buffer.all_finite(f32 [n,10]) native (control) | python/mojolearn/_buffer.py:765 | native scan already | O(rows x cols) | 2.65 | 3.56 (1.13 u) | 3.87 (1.12 u) | u | 16.5 (1.29 u) | 16.6 (1.14 u) | u | control: already native |
| 51 | _labels.decode_labels(int classes, int64 codes) | python/mojolearn/_labels.py:337, :360 | gather of one class per code (already `gather_i64`) | O(rows) | 1.85 | 1.32 (1.32 u) | 1.24 (1.53 u) | u | 8.00 (3.41 u) | 26.9 (1.10 u) | u | control: already native |
| 52 | metrics._as_f32_1d(f32 buffer) (control) | python/mojolearn/_metrics_impl.py:186 | native already | O(rows) | 0.30 | 0.48 (1.29 u) | 0.53 (1.28 u) | u | 2.12 (1.26 u) | 2.34 (1.21 u) | u | control |

Reading the ratios by shape: the wins are structural, one C-level pass in
place of one Python object per element, and hold at 4M where the cell ran
(`_binary_ranking_inputs` 68.5x at 1M and 69.6x at 4M, `_prepare_cluster_labels`
36.4x and 36.4x, `as_f32_c` of an int32 matrix 29.3x and 27.3x, `_indices` plus
overlap 17.8x and 18.8x, `as_i64_c` of a list 7.9x and 7.9x, `sorted_classes`
of ints 3.7x and 3.6x). The cells that read 1.00x are the API-bound ones
(`tolist`, `__iter__`, the strided column slice, `_as_order`) and the controls,
which is what a control should read. No qualified cell regressed: the lowest
qualified ratio is `decode_labels(str classes)` at 0.99x, inside the noise.
The one cell that LOOKS slower, `finite_integer_codes` at 37 ms against 44 ms
at 1M, has its P side outside the gate (spread 1.19) and reads 152.7 against
152.5 ms at 4M with both sides inside it; the function did not change.

### Table 2: the same cells on the M4 (under load, attribution only) and on x86 with the default thread count

Mac: `ab_1m.json` (7 rounds) and `ab_4m.json` (5 rounds), one core, `host-a`
(sources of 574b72574), load average 14 to 77, NOT quotable for a ratio. x86
default threads: `ab_1m_x86_default_threads.json`, the pod's pool, hashes equal
to the one-thread run in every cell.

| cell | Mac P 1M (spread) | Mac N 1M (spread) | Mac P 4M (spread) | Mac N 4M (spread) | x86 P 1M default threads (spread) | x86 N 1M default threads (spread) |
|---|---|---|---|---|---|---|
| cross_val_score glue end to end: stratified folds + _indices x10 + overlap x5 | 2796.1 (1.61 u) | 32.5 (1.68 u) | 15195.4 (1.17 u) | 155.7 (1.72 u) | 5972.8 (1.02) | 44.5 (1.06) |
| _buffer.as_f32_c(list of lists [n,10] float) | 2472.8 (1.30 u) | 291.9 (1.14 u) | - | - | 5890.0 (1.00) | 708.3 (1.03) |
| model_selection._indices x10 + overlap x5 (cross_val_score glue) | 3779.6 (1.42 u) | 232.8 (1.34 u) | 11645.1 (1.35 u) | 820.3 (1.57 u) | 4902.7 (1.02) | 282.7 (1.03) |
| Array.__iter__ rows of f32 [n,10] (n/10 rows timed, x10) | 2231.3 (1.08) | 2274.5 (1.06) | - | - | 4808.2 (1.01) | 4814.1 (1.02) |
| model_selection._default_folds stratified 5 (int64 y) | 1437.1 (1.32 u) | 14.8 (1.60 u) | 6016.9 (2.63 u) | 61.5 (2.37 u) | 2787.3 (1.01) | 39.6 (1.03) |
| model_selection._default_folds kfold 5 (f32 y) | 1238.4 (1.63 u) | 6.18 (6.07 u) | 5058.5 (1.23 u) | 20.8 (1.46 u) | 2301.1 (1.03) | 8.89 (1.12 u) |
| metrics._classification_pair+_encode (int64 x2) | 1735.1 (1.68 u) | 22.0 (1.63 u) | 4168.7 (2.34 u) | 64.6 (3.46 u) | 1841.6 (1.02) | 29.8 (1.02) |
| metrics._prepare_cluster_labels(int64 x2) | 1931.0 (1.35 u) | 42.4 (5.01 u) | 3851.1 (1.42 u) | 85.0 (1.24 u) | 1401.0 (1.05) | 42.7 (1.02) |
| Array.tolist() float32 [n,10] | 421.0 (1.20 u) | 452.3 (1.43 u) | - | - | 1239.4 (1.12 u) | 1333.8 (1.09) |
| metrics._classification_pair+_encode (list of str x2) | 881.9 (1.91 u) | 278.9 (2.32 u) | 2742.4 (1.69 u) | 703.3 (2.08 u) | 1175.0 (1.02) | 340.5 (1.04) |
| _labels.encode_labels(list of float) | 660.7 (2.15 u) | 34.6 (1.64 u) | 2117.2 (1.61 u) | 100.8 (1.42 u) | 1022.9 (1.01) | 48.6 (1.02) |
| metrics._binary_ranking_inputs+encode (int64 y, f32 score) | 557.0 (1.45 u) | 11.1 (2.96 u) | 2645.9 (1.33 u) | 37.7 (2.38 u) | 939.8 (1.01) | 14.3 (1.02) |
| _labels.argmax_rows(F-order f32 [n,7], Python arm) | 394.8 (1.51 u) | 253.6 (1.11 u) | 1990.9 (1.80 u) | 1253.7 (1.65 u) | 802.4 (1.04) | 495.5 (1.04) |
| metrics._as_i32_1d(list of int) | 831.4 (1.57 u) | 314.2 (1.63 u) | 2122.6 (1.22 u) | 778.1 (1.37 u) | 755.5 (1.01) | 268.6 (1.04) |
| _buffer.as_f32_c(int32 buffer [n,10]) | 328.2 (1.05) | 2.87 (1.60 u) | 1457.8 (1.67 u) | 12.2 (1.34 u) | 737.8 (1.03) | 17.2 (1.04) |
| _labels.encode_labels(list of int) | 333.5 (1.12 u) | 20.0 (4.36 u) | 1366.9 (1.31 u) | 81.0 (2.14 u) | 723.1 (1.03) | 50.2 (1.99 u) |
| _buffer.as_f32_c(strided f32 view [n,10] of [2n,10]) | 258.0 (1.34 u) | 10.6 (1.37 u) | 1096.4 (1.71 u) | 53.0 (2.12 u) | 545.6 (1.05) | 68.3 (1.04) |
| _labels.encode_labels(list of str) | 246.8 (1.18 u) | 53.5 (1.04) | 1125.2 (1.37 u) | 220.1 (1.43 u) | 543.8 (1.10) | 111.6 (1.02) |
| _buffer.as_i64_c(list [n] int) | 231.1 (1.15 u) | 34.6 (1.08) | 983.5 (2.02 u) | 150.5 (1.50 u) | 484.7 (1.03) | 62.6 (1.03) |
| _labels.decode_labels(int classes, int32 codes) | 201.9 (1.10 u) | 2.64 (1.35 u) | 778.5 (2.23 u) | 11.0 (1.58 u) | 390.4 (1.03) | 2.37 (1.26 u) |
| Array.__getitem__ a[:, 2:5] of f32 [n,10] (strided, not a block) | 162.6 (1.31 u) | 169.4 (1.41 u) | 866.3 (1.88 u) | 1123.5 (1.75 u) | 360.4 (1.02) | 359.9 (1.04) |
| _buffer.as_f32_c(flat list [n] float, ndim=1) | 164.4 (1.03) | 23.6 (1.07) | 997.4 (1.32 u) | 136.9 (1.22 u) | 360.0 (1.04) | 49.6 (1.02) |
| Array.__getitem__ row range a[n//10:] of f32 [n,10] | 145.9 (1.05) | 38.9 (1.04) | 980.7 (2.02 u) | 239.6 (3.40 u) | 348.4 (1.02) | 91.9 (1.02) |
| _labels.sorted_classes(list of int, 7 classes) | 128.2 (1.21 u) | 31.8 (1.06) | 955.8 (1.44 u) | 244.6 (1.57 u) | 251.7 (1.02) | 68.0 (1.08) |
| _labels.flatten_labels(int64 Array) | 54.2 (1.02) | 10.9 (1.07) | 236.2 (1.29 u) | 49.9 (1.28 u) | 126.2 (1.02) | 24.9 (1.02) |
| _labels.flatten_labels(list of int) | 50.0 (1.30 u) | 6.88 (1.25 u) | 196.2 (1.28 u) | 28.1 (1.03) | 114.1 (1.07) | 11.4 (1.37 u) |
| metrics._as_i32_1d(int32 buffer) | 131.2 (1.14 u) | 3.81 (1.35 u) | 313.4 (1.94 u) | 9.53 (1.50 u) | 113.2 (1.07) | 2.97 (1.05) |
| Array._as_order('F') f32 [n,10] Python _reorder | 48.6 (1.33 u) | 48.0 (1.22 u) | 352.1 (1.38 u) | 344.7 (1.35 u) | 108.5 (1.03) | 109.5 (1.02) |
| _buffer.as_f32_c(bool buffer [n]) | 48.6 (1.21 u) | 20.7 (1.25 u) | 371.4 (1.71 u) | 156.9 (1.13 u) | 106.6 (1.03) | 44.6 (1.06) |
| Array.min()+max() int64 [n] | 34.2 (1.04) | 1.00 (1.32 u) | 220.5 (1.90 u) | 6.15 (1.32 u) | 88.0 (1.04) | 1.52 (1.29 u) |
| _labels.sorted_classes(list of str, 7 classes) | 38.2 (1.06) | 20.6 (1.08) | 167.0 (1.58 u) | 82.5 (1.47 u) | 86.5 (1.03) | 53.8 (1.01) |
| _buffer.as_f32_c(int64 buffer [n], ndim=1) | 33.0 (1.05) | 0.33 (25.05 u) | 136.8 (1.03) | 1.32 (2.18 u) | 71.0 (1.01) | 0.92 (1.68 u) |
| metrics._as_i32_1d(int64 buffer) | 91.2 (1.16 u) | 2.78 (1.50 u) | 172.7 (1.14 u) | 5.37 (1.52 u) | 68.8 (1.03) | 2.25 (1.50 u) |
| Array.__eq__(Array) int64 [n] | 30.2 (1.05) | 0.26 (2.79 u) | 248.7 (1.45 u) | 2.17 (2.58 u) | 60.3 (1.02) | 1.51 (1.15 u) |
| Array.astype('<i8') from float32 [n] | 22.6 (1.11 u) | 0.64 (1.36 u) | 160.0 (1.41 u) | 3.29 (1.41 u) | 54.4 (1.05) | 1.17 (2.12 u) |
| metrics._sample_weight_f32(f32 buffer) | 44.1 (1.86 u) | 2.10 (1.10 u) | 140.8 (1.07) | 5.66 (1.05) | 47.5 (1.36 u) | 1.54 (1.06) |
| Array.argmax() float32 [n] | 25.6 (1.09) | 1.00 (1.03) | 174.1 (1.47 u) | 6.83 (1.16 u) | 53.7 (1.04) | 0.70 (1.16 u) |
| _buffer.as_i32_c(int64 buffer [n]) | 25.6 (1.09) | 0.41 (1.39 u) | 218.3 (1.21 u) | 2.38 (1.42 u) | 51.1 (1.02) | 0.80 (1.16 u) |
| Array.sum() float32 [n] | 17.1 (1.10) | 0.66 (1.29 u) | 105.7 (1.09) | 4.04 (1.11 u) | 46.4 (1.07) | 1.02 (1.01) |
| metrics.kl_divergence negativity scan (Array.min x2) | 44.3 (1.03) | 1.73 (1.02) | 136.4 (1.10 u) | 4.45 (1.13 u) | 46.5 (1.06) | 1.38 (1.25 u) |
| _serialize write_npz+read_npz 1000 trees x n/1000 nodes (5 arrays) | 20.5 (1.75 u) | 21.2 (1.58 u) | 97.0 (1.37 u) | 86.2 (1.32 u) | 45.0 (1.18 u) | 45.4 (1.53 u) |
| Array.astype('<i4') from int64 [n] | 19.5 (1.08) | 0.30 (1.58 u) | 148.0 (1.42 u) | 1.93 (2.27 u) | 42.6 (1.07) | 0.77 (1.15 u) |
| Array.min() float32 [n] | 15.2 (1.05) | 0.50 (1.48 u) | 91.2 (1.11 u) | 3.08 (1.11 u) | 39.3 (1.16 u) | 0.69 (1.08) |
| _labels.finite_integer_codes(float32 Array) | 11.5 (1.63 u) | 11.5 (1.53 u) | 56.4 (1.13 u) | 57.8 (1.05) | 34.8 (1.04) | 34.7 (1.07) |
| _labels.decode_labels(str classes, int64 codes) | 12.6 (1.12 u) | 12.8 (1.16 u) | 49.4 (1.13 u) | 50.7 (1.24 u) | 31.1 (1.07) | 31.1 (1.17 u) |
| _labels.decode_labels(bool classes, int64 codes) | 12.9 (1.09) | 13.3 (1.07) | 59.5 (1.17 u) | 59.2 (1.14 u) | 29.6 (1.02) | 29.8 (1.21 u) |
| _buffer.as_f32_c(f64 buffer [n,10]) native cast (control) | 3.65 (1.26 u) | 3.68 (1.33 u) | 22.1 (1.73 u) | 22.6 (1.26 u) | 27.7 (1.03) | 27.7 (1.03) |
| _labels.encode_labels(float32 buffer) | 5.71 (1.05) | 5.74 (1.04) | 23.6 (1.08) | 23.9 (1.03) | 15.4 (1.03) | 15.3 (1.17 u) |
| _labels.encode_labels(int64 buffer) | 5.37 (1.05) | 5.48 (1.02) | 22.9 (1.08) | 23.6 (1.05) | 12.9 (1.02) | 12.9 (1.00) |
| _buffer.all_finite(f32 [n,10]) native (control) | 2.62 (1.03) | 2.61 (1.04) | 32.7 (1.39 u) | 25.8 (2.18 u) | 3.67 (1.23 u) | 4.00 (1.14 u) |
| _labels.decode_labels(int classes, int64 codes) | 2.00 (1.47 u) | 1.97 (1.18 u) | 7.11 (1.20 u) | 7.44 (1.12 u) | 1.54 (1.16 u) | 1.62 (1.43 u) |
| metrics._as_f32_1d(f32 buffer) (control) | 0.47 (1.13 u) | 0.47 (1.07) | 1.18 (1.09) | 1.19 (1.06) | 0.50 (1.26 u) | 0.46 (1.44 u) |

### Table 3: hits in files this lane does not own (ranked by the Python arm's ms at 1M rows on x86)

`bench/speed/python_hotpath_other_lanes.py`: each cell timed with
`MOJOLEARN_HOTPATH=python` (P) and with the shared-layer seams active (N),
interleaved, THREE rounds (attribution, not the five-round merge gate), min
and spread. Mac: `other_lanes_1m.json` (under load). x86:
`pod/remote/leg_out/other_lanes_1m_x86.json`. A cell whose N equals its P is a
hit the shared layer does not reach; a cell whose N is far below its P got its
gain through `Array`, `_labels` or `_buffer` without an edit to its own file.
The cells marked x10 time n/10 rows and scale; they hold a fitted-model-shaped
input the harness synthesizes.

| # | cell | file:line | iterates | big-O | Mac P 1M ms (spread, load) | Mac N 1M ms (spread, load) | x86 P 1M ms (spread) | x86 N 1M ms (spread) | status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | neighbors radius_neighbors per-row split (n/10 rows, 10 nbrs each, x10) | python/mojolearn/neighbors.py:1537-:1553 | two Array slices and an `astype` PER ROW | O(rows) | 7693.1 (1.07) | 7870.6 (1.38) | 18021.8 (1.05) | 17985.1 (1.06) | unfixed (not owned): the largest hit found |
| 2 | linear_model._column_means weighted [n,10] (LinearRegression.fit + sample_weight) | python/mojolearn/linear_model.py:319 (:347-:350) | one Python float product per CELL | O(rows x cols) | 3086.9 (1.14) | 960.0 (1.06) | 7597.4 (1.05) | 2337.3 (1.05) | OWED diff: `scale_rows_f32` |
| 3 | randomforest._class_weight_rows balanced (RF fit + class_weight) | python/mojolearn/randomforest.py:326 (:355), caller :660 | one scalar `Array.__getitem__` per row | O(rows) | 1252.7 (1.15) | 1109.5 (1.20) | 2994.3 (1.01) | 2592.5 (1.00) | OWED diff: table lookup by `map` |
| 4 | _gpc_impl binary predict_proba [[1-p,p]] from_list | python/mojolearn/_gpc_impl.py:420 | one two-element list per row | O(rows) | 1885.9 (1.25) | 199.0 (2.49) | 2760.3 (1.05) | 235.9 (1.29) | shared-layer gain (from_list); the list build unfixed |
| 5 | ensemble._pairs_arrays (PairLogit), n pairs | python/mojolearn/ensemble.py:212 (:220) | `pairs.tolist()` then one tuple per pair | O(pairs) | 579.1 (1.01) | 579.1 (1.02) | 1417.3 (1.01) | 1402.1 (1.02) | unfixed (not owned) |
| 6 | _gp_impl predict rescale (normalize_y) per-row _round_f32 x2 | python/mojolearn/_gp_impl.py:913, :1134 (also :917) | two `_round_f32` and two `_ftz` per row | O(rows) | 742.9 (1.25) | 551.3 (1.17) | 1011.8 (1.01) | 711.6 (1.01) | unfixed (not owned): a numeric seam, needs its own DEVIATION |
| 7 | _gbdt_adapters GradientBoostingClassifier.fit label prep | python/mojolearn/_gbdt_adapters.py:137 | a dict lookup per label | O(rows) | 389.0 (1.36) | 207.7 (1.45) | 858.5 (1.01) | 440.6 (1.01) | OWED diff: `_label_map` |
| 8 | linear_model: LogisticRegression.fit label prep (_labels_1d + sorted_classes + f32 codes) | python/mojolearn/linear_model.py:142, :1055, :1065 | `sorted_classes` per label then `float(c)` per code | O(rows) | 358.6 (1.08) | 80.4 (1.06) | 797.8 (1.02) | 192.5 (1.01) | shared-layer gain; `float(c)` loop unfixed |
| 9 | neighbors KNeighborsClassifier.fit label prep (min/max + tolist + from_list + sorted_classes) | python/mojolearn/neighbors.py:976-:995 | min/max, `tolist`, `from_list`, `sorted_classes` per label | O(rows) | 318.5 (1.12) | 59.8 (1.31) | 637.3 (1.01) | 111.7 (1.01) | shared-layer gain |
| 10 | linear_model._vector_mean weighted + _weight_total | python/mojolearn/linear_model.py:356 (:366), :304 (:309) | one product per row, one add per row | O(rows) | 227.3 (1.01) | 90.3 (1.08) | 557.9 (1.05) | 235.8 (1.04) | unfixed (not owned) |
| 11 | neighbors kneighbors index widen ind.astype('<i8') [n,10] u32 | python/mojolearn/neighbors.py:809-:810 (also :772-:773, :1090-:1092) | `astype` per element | O(rows x k) | 234.0 (1.05) | 6.85 (1.21) | 545.0 (1.02) | 55.2 (1.02) | fixed through `Array.astype` (DEVIATION 3100) |
| 12 | _forest_protocol.score classifier compare (flatten + zip + from_list) | python/mojolearn/_forest_protocol.py:271 | `int(a == b)` per row | O(rows) | 231.8 (1.05) | 47.4 (1.12) | 534.3 (1.01) | 115.7 (1.04) | shared-layer gain; the zip loop unfixed |
| 13 | _svm_impl._as_labels (SVC.fit), int64 y | python/mojolearn/_svm_impl.py:149 (:168) | `sorted_classes` per label | O(rows) | 290.8 (1.20) | 85.5 (1.52) | 505.6 (1.00) | 149.4 (1.02) | shared-layer gain |
| 14 | linear_model: binary predict codes + decode_labels (LogisticRegression.predict) | python/mojolearn/linear_model.py:1136 | `1 if s > 0.0 else 0` per row, then decode | O(rows) | 234.2 (1.04) | 77.8 (1.13) | 460.2 (1.01) | 143.2 (1.04) | partly (decode native); codes loop unfixed |
| 15 | _gbdt_adapters predict: decode_labels(int classes, int32 codes) | python/mojolearn/_gbdt_adapters.py:181 | int32 codes widened per element | O(rows) | 176.1 (1.01) | 2.93 (1.17) | 397.5 (1.03) | 1.73 (1.19) | fixed through `_labels` (DEVIATION 3103) |
| 16 | linear_model: sqrt(weights) root list (LinearRegression.fit:620) | python/mojolearn/linear_model.py:620 | `math.sqrt` and `_round_f32` per weight | O(rows) | 144.5 (1.07) | 145.9 (1.10) | 348.3 (1.04) | 344.0 (1.01) | unfixed (not owned) |
| 17 | ensemble FeatureFreq distinct per column set(xv[col]) [n,10] | python/mojolearn/ensemble.py:2101, :2116 | `set()` of every column | O(rows x cols) | 130.8 (1.06) | 136.9 (1.07) | 331.6 (1.00) | 331.2 (1.00) | unfixed (not owned) |
| 18 | density DBSCAN._store_core host pass (flags, idx, labels) [n] | python/mojolearn/density.py:343-:356 | `tolist` of flags and labels, a Python filter per row | O(rows) | 199.0 (1.27) | 106.9 (1.16) | 261.7 (1.01) | 106.2 (1.07) | partly (from_list faster); the filter unfixed |
| 19 | ensemble._has_nan X [n,10] (nan_mode=Forbidden, inf present) | python/mojolearn/ensemble.py:496 (:503) | `any(map(math.isnan, view))` over every cell | O(rows x cols) | 88.3 (1.05) | 89.7 (1.13) | 197.1 (1.00) | 196.4 (1.02) | unfixed (not owned) |
| 20 | linear_model._r2_sums (score of 6 regressors) | python/mojolearn/linear_model.py:238 (:251-:252) | `tolist()` of both, one residual per row in Python | O(rows) | 65.7 (1.06) | 66.9 (1.04) | 192.2 (1.08) | 192.4 (1.02) | unfixed (not owned) |
| 21 | ensemble OrderedRMSE permutation checks (min/max/set/astype u4) | python/mojolearn/ensemble.py:2268-:2270 | `min`, `max`, `set` over the order, then `astype` | O(rows) | 88.0 (1.38) | 78.5 (1.48) | 183.9 (1.08) | 166.3 (1.07) | partly (astype native) |
| 22 | linear_model._accuracy_host (LogisticRegression/SVC score), int64 y | python/mojolearn/linear_model.py:210 (:214) | `tolist()` then one compare per row | O(rows) | 76.6 (1.02) | 32.8 (1.05) | 155.9 (1.11) | 55.3 (1.06) | shared-layer gain only |
| 23 | extratrees predict_proba vote.astype('<f8') [n,3] | python/mojolearn/extratrees.py:481, :584 | `astype` per element | O(rows x classes) | 91.7 (1.04) | 1.91 (1.92) | 144.6 (1.00) | 1.71 (1.24) | fixed through `Array.astype` (DEVIATION 3100) |
| 24 | density KernelDensity.fit weights w.min() + w.sum() | python/mojolearn/density.py:726 | `Array.min()` and `Array.sum()` | O(rows) | 44.2 (1.15) | 1.53 (1.14) | 66.5 (1.05) | 1.69 (1.02) | fixed through `Array` (DEVIATION 3101) |
| 25 | extratrees fit codes.astype('<f4') [n] | python/mojolearn/extratrees.py:468, :471 | `astype` per element | O(rows) | 29.8 (1.06) | 0.44 (1.15) | 60.7 (1.02) | 0.21 (1.19) | fixed through `Array.astype` (DEVIATION 3100) |
| 26 | embedding ids int64 -> int32 (as_i32_c) | python/mojolearn/embedding.py:168 | `array.array` per id | O(ids) | 33.2 (1.07) | 0.53 (1.41) | 49.4 (1.03) | 0.30 (2.13) | fixed through `_buffer` (DEVIATION 3100) |
| 27 | density KernelDensity.score sum loop | python/mojolearn/density.py:795 | one add per row | O(rows) | 25.6 (1.09) | 24.6 (1.24) | 43.0 (1.03) | 42.9 (1.07) | unfixed (not owned) |
| 28 | linear_model._check_sample_weight | python/mojolearn/linear_model.py:441 | min scan over the weights | O(rows) | 18.7 (1.04) | 1.08 (1.76) | 37.5 (1.06) | 1.17 (1.08) | shared-layer gain |
| 29 | ensemble sample_weight min/max builtin scans (fit:1426) | python/mojolearn/ensemble.py:1425-:1426 | builtin `min` over a float view | O(rows) | 16.3 (1.03) | 16.5 (1.02) | 28.7 (1.07) | 28.7 (1.05) | unfixed (not owned); `reduce_stat` would retire it |

### Table 4: hits found by the static audit and NOT timed

These need a fitted model, a GPU binding or a family binding the pod did not
build; they are listed so the next lane starts from the list, not from a
grep. Line numbers are the branch's, which equal main's for every file this
lane did not edit.

| file:line | iterates | big-O | why not timed |
|---|---|---|---|
| python/mojolearn/_svm_impl.py:205 | `dual_coef` x `support_vectors.tolist()` rows | O(support vectors x cols) | needs a fitted SVC |
| python/mojolearn/_svm_impl.py:658 | `1 if v == label1 else 0` per row of `raw.tolist()` | O(rows) | needs the SVM binding |
| python/mojolearn/linear_model.py:851 | `w.tolist()` then a Python pass | O(rows) | inside a fit path |
| python/mojolearn/linear_model.py:1169 | `predict_proba(X).tolist()` per row | O(rows x classes) | needs a fitted LogisticRegression |
| python/mojolearn/_gp_impl.py:917 | `math.sqrt` and two `_round_f32` per row of `var.tolist()` | O(rows) | numeric seam of GP predict; needs its own DEVIATION |
| python/mojolearn/_gpc_impl.py:282, :470-:471 | `int(c)` per code; `tolist()` of pred and truth | O(rows) | needs a fitted GPC |
| python/mojolearn/randomforest.py:787, :797 | `min(flat_view(y32))` | O(rows) | C-driven builtin, one Python float per row; `reduce_stat` would retire it |
| python/mojolearn/neighbors.py:1072, :1209 | `uniq.tolist()`, multi-output `ya.tolist()` | O(rows x outputs) | multi-output fit path |
| python/mojolearn/model_selection.py:432-:433 | `[int(i) for i in ...tolist()]` per fold | O(rows x folds) | `split_descriptor`, a reproduction record, not a training path |
| python/mojolearn/ensemble.py:177 | `group_id.tolist()` | O(rows) | lane/gbdt-group-sizes owns it (ad200d2cd) |
| python/mojolearn/ensemble.py:524 | `member.tolist()` | O(rows) | serialization of a member |
| python/mojolearn/ensemble.py:1852, python/mojolearn/_gbdt_host.py:724 | `flat_view(p1, "d")` scan | O(rows) | GBDT predict path, needs the GBDT binding |
| python/mojolearn/density.py:346-:356 | timed in Table 3 | O(rows) | (timed) |
| python/mojolearn/_byte_lm_impl.py:328, :759, :879 | `any(...)` over every token or flag | O(tokens) | byte LM binding |
| python/mojolearn/_samba_impl.py:447, :501 | `sum(v != IGNORE for v in flat_view(y))` | O(tokens) | Samba binding |
| python/mojolearn/_mlp_impl.py:250 | `min` over the moments view | O(parameters) | MLP binding |
| python/mojolearn/tokenizer.py:269, :323; python/mojolearn/models/causal_lm.py:234, :510 | `tolist()` of ids and logits | O(tokens) | neural inference, another lane's column |

Sites the grep returned that are NOT per-row: every `meta.tolist()` unpack
(`_hierarchy_impl.py:454`, `_spectral_impl.py:796`, `kernel_methods.py:292,
:446, :583`), `_serialize.py:253` (one index per array), `_gp_impl.py:212,
:219, :807` (per hyperparameter).

## Owed diffs in shared files (the orchestrator applies them; this lane did not touch the files)

All three live in `/Users/andrewhendel/mojolearn-evidence/python-hotpath/`
and are reproduced by name here so a reader can find them:

1. `OWED_host_surface.diff` (python/mojolearn/host_surface.py, the core
   family): adds `bindings/hotpath_helpers.mojo` to `host_modules` and the
   fourteen helper names to `exports`. Without it
   `python/mojolearn/tests/test_host_surface.py` will name the exports the
   binding carries and the manifest does not.
2. `OWED_base_binding.diff` (bindings/_mojolearn.mojo, the GPU base binding):
   imports the eight non-encoder helpers and registers them with
   `m.def_function`. WITHOUT IT THE HELPERS ARE INERT ON A GPU INSTALL: the
   GPU wheel's Python resolves `_native("cast_elements")` against the base
   binding, misses, and runs the Python fallback, so a GPU user keeps every
   millisecond in Table 1's P column. The encoders are already there.
3. `OWED_other_lanes.diff` (randomforest.py `_class_weight_rows`,
   linear_model.py `_column_means`, _gbdt_adapters.py label prep): the three
   largest hits in files other lanes own. Byte-equality of old against new
   was checked on the M4 by `scripts/verify_owed_other_lanes.py` (class
   weights on three weightings, weighted column means on three shapes with
   exponents -20 to 20, the GBDT label prep on seven label kinds): all equal.
   NOT timed on the pod (the pod ran the package as committed); Table 3
   gives the cost they remove (2,592 ms, 2,337 ms and 441 ms at 1M).

## Documents found false or misleading, and what is true

- `python/mojolearn/_metrics_impl.py:181` (main :180) said the int64 to int32
  narrowing is "`array.array`'s C item loop ... no Python loop". There is no
  Python loop BODY, but the setter makes one Python int per element: 69 ms per
  1,000,000 rows on the EPYC pod, 37 ms on the M4. Corrected on the branch to
  name the helper and the measured cost.
- `bindings/hotpath_helpers.mojo:10` (this lane's own header, commit
  2c713c760) gave the per-element cost as "22 to 31 ns" without naming the
  machine; that is the M4's range (20 to 33 ns measured here), the EPYC pod
  reads 43 to 72 ns. Corrected on the branch.
- Commit 98992b58f's subject says "180 clean"; the file grew to 230 cases in
  826e7836b. A commit message is not a document, noted so the next reader is
  not puzzled by the count.
- `python/mojolearn/ensemble.py:2096-:2097` and `:1421-:1424` describe the
  `set()` and builtin `min`/`max` scans as "C-driven ... no Python loop
  body", which is true and, as measured, still 331 ms per 1M x 10 cells and
  29 ms per 1M rows (one Python float per element). The comments are not
  false; they are the reason those hits stay in the work list. Not this
  lane's file; no edit.

Nothing in `docs/` made a claim about these paths that the measurements
contradict.

## Work list: unfixed hits ranked by milliseconds still paid at 1M rows (x86, arm N)

Each row is a hit the shared layer did not remove. The three OWED rows are
fixed by the owed diff once applied. "Unavoidable for this API" means the
output is Python objects by contract (`tolist`, `__iter__`); the fix there is
in the CALLER not calling it, which is what the other rows are.

| rank | ms still paid at 1M rows (x86, arm N) | hit | file:line | state |
|---|---|---|---|---|
| 1 | 17985.1 | neighbors radius_neighbors per-row split (n/10 rows, 10 nbrs each, x10) | python/mojolearn/neighbors.py:1537-:1553 | unfixed (not owned): the largest hit found |
| 2 | 4866.7 | Array.__iter__ rows of f32 [n,10] (n/10 rows timed, x10) | python/mojolearn/_array.py:598 | not changed: unavoidable for this API |
| 3 | 2592.5 | randomforest._class_weight_rows balanced (RF fit + class_weight) | python/mojolearn/randomforest.py:326 (:355), caller :660 | OWED diff: table lookup by `map` |
| 4 | 2337.3 | linear_model._column_means weighted [n,10] (LinearRegression.fit + sample_weight) | python/mojolearn/linear_model.py:319 (:347-:350) | OWED diff: `scale_rows_f32` |
| 5 | 1402.1 | ensemble._pairs_arrays (PairLogit), n pairs | python/mojolearn/ensemble.py:212 (:220) | unfixed (not owned) |
| 6 | 1334.0 | Array.tolist() float32 [n,10] | python/mojolearn/_array.py:490 | not changed: unavoidable for this API |
| 7 | 711.6 | _gp_impl predict rescale (normalize_y) per-row _round_f32 x2 | python/mojolearn/_gp_impl.py:913, :1134 (also :917) | unfixed (not owned): a numeric seam, needs its own DEVIATION |
| 8 | 491.7 | _labels.argmax_rows(F-order f32 [n,7], Python arm) | python/mojolearn/_labels.py:434 | partly: C-order path native; F-order still Python per row |
| 9 | 440.6 | _gbdt_adapters GradientBoostingClassifier.fit label prep | python/mojolearn/_gbdt_adapters.py:137 | OWED diff: `_label_map` |
| 10 | 364.1 | Array.__getitem__ a[:, 2:5] of f32 [n,10] (strided, not a block) | python/mojolearn/_array.py:617 | not changed: strided copy still Python |
| 11 | 344.0 | linear_model: sqrt(weights) root list (LinearRegression.fit:620) | python/mojolearn/linear_model.py:620 | unfixed (not owned) |
| 12 | 342.8 | metrics._classification_pair+_encode (list of str x2) | python/mojolearn/_metrics_impl.py:1158, :1334 | partly |
| 13 | 331.2 | ensemble FeatureFreq distinct per column set(xv[col]) [n,10] | python/mojolearn/ensemble.py:2101, :2116 | unfixed (not owned) |
| 14 | 275.0 | metrics._as_i32_1d(list of int) | python/mojolearn/_metrics_impl.py:164 | partly (DEVIATION 3106) |
| 15 | 235.9 | _gpc_impl binary predict_proba [[1-p,p]] from_list | python/mojolearn/_gpc_impl.py:420 | shared-layer gain (from_list); the list build unfixed |
| 16 | 235.8 | linear_model._vector_mean weighted + _weight_total | python/mojolearn/linear_model.py:356 (:366), :304 (:309) | unfixed (not owned) |
| 17 | 196.4 | ensemble._has_nan X [n,10] (nan_mode=Forbidden, inf present) | python/mojolearn/ensemble.py:496 (:503) | unfixed (not owned) |
| 18 | 192.5 | linear_model: LogisticRegression.fit label prep (_labels_1d + sorted_classes + f32 codes) | python/mojolearn/linear_model.py:142, :1055, :1065 | shared-layer gain; `float(c)` loop unfixed |
| 19 | 192.4 | linear_model._r2_sums (score of 6 regressors) | python/mojolearn/linear_model.py:238 (:251-:252) | unfixed (not owned) |
| 20 | 166.3 | ensemble OrderedRMSE permutation checks (min/max/set/astype u4) | python/mojolearn/ensemble.py:2268-:2270 | partly (astype native) |
| 21 | 143.2 | linear_model: binary predict codes + decode_labels (LogisticRegression.predict) | python/mojolearn/linear_model.py:1136 | partly (decode native); codes loop unfixed |
| 22 | 115.7 | _forest_protocol.score classifier compare (flatten + zip + from_list) | python/mojolearn/_forest_protocol.py:271 | shared-layer gain; the zip loop unfixed |
| 23 | 108.0 | Array._as_order('F') f32 [n,10] Python _reorder | python/mojolearn/_array.py:477, :178 | not changed: no per-row Python body |
| 24 | 106.2 | density DBSCAN._store_core host pass (flags, idx, labels) [n] | python/mojolearn/density.py:343-:356 | partly (from_list faster); the filter unfixed |
| 25 | 104.9 | _labels.encode_labels(list of str) | python/mojolearn/_labels.py:216 | partly (C-driven map) |
| 26 | 46.2 | _labels.sorted_classes(list of str, 7 classes) | python/mojolearn/_labels.py:156 | partly: C-driven set and map, objects unavoidable |
| 27 | 43.7 | _labels.finite_integer_codes(float32 Array) | python/mojolearn/_labels.py:461 | not changed (C-driven set) |
| 28 | 42.9 | density KernelDensity.score sum loop | python/mojolearn/density.py:795 | unfixed (not owned) |
| 29 | 31.5 | _labels.decode_labels(str classes, int64 codes) | python/mojolearn/_labels.py:337 | not changed: output objects are the cost |
| 30 | 30.7 | _labels.decode_labels(bool classes, int64 codes) | python/mojolearn/_labels.py:337 | not changed |
| 31 | 28.7 | ensemble sample_weight min/max builtin scans (fit:1426) | python/mojolearn/ensemble.py:1425-:1426 | unfixed (not owned); `reduce_stat` would retire it |

## Commands (all one core on the Mac; the pod's are in `pod/user_cmd.sh`)

```
# build the core host family at the tip (through the slot, one core)
python3 tools/mac_slot.py run ~/mojolearn-evidence/python-hotpath/scripts/build_core.sh ~/mojolearn-evidence/python-hotpath/host-tip
# the sabotage arm
python3 tools/mac_slot.py run ~/mojolearn-evidence/python-hotpath/scripts/build_core.sh ~/mojolearn-evidence/python-hotpath/host-sab "-D MOJOLEARN_HOTPATH_SABOTAGE=1"
# the differential test, clean / strict sabotage / expected sabotage (from the worktree's python/)
env -u MOJOLEARN_NUMERIC_MODE -u MOJOLEARN_HOTPATH MOJOLEARN_HOST_DIR=$HOME/mojolearn-evidence/python-hotpath/host-tip OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 nice -n 19 ../.pixi/envs/test/bin/python -m pytest -q -p no:cacheprovider mojolearn/tests/test_hotpath_native.py
MOJOLEARN_HOST_ALLOW_SABOTAGE=1 MOJOLEARN_HOST_DIR=$HOME/mojolearn-evidence/python-hotpath/host-sab ... same
MOJOLEARN_HOTPATH_EXPECT_SABOTAGE=1 MOJOLEARN_HOST_ALLOW_SABOTAGE=1 MOJOLEARN_HOST_DIR=.../host-sab ... same
# identity, 22 lanes, five fixtures, two repeats (scripts/identity.sh before|after, scripts/identity_sab.sh sab)
MOJOLEARN_CPU_THREADS=1 MOJOLEARN_HOST_DIR=... python tools/identity_break.py --lanes "<the 22>" --fixtures base,ties,odd,dupes,wide --repeats 2 --json out.json
python tools/identity_break.py --diff before.json after.json
# the A/B and the other-lane cells
MOJOLEARN_CPU_THREADS=1 MOJOLEARN_HOST_DIR=... PYTHONPATH=python python bench/speed/python_hotpath_ab.py --sizes 1000000 --rounds 5 --json out.json
MOJOLEARN_CPU_THREADS=1 MOJOLEARN_HOST_DIR=... PYTHONPATH=python python bench/speed/python_hotpath_other_lanes.py 1000000 out.json
```

## What was rejected or not done, and what is owed

- Rejected: making the reductions parallel. `min`, `max`, `sum` and argmax
  are sequential on one thread because order is observable (see above); a
  parallel float sum would move bits and a parallel argmax would need a
  tie rule Python does not have.
- Rejected: a native `tolist` or row iterator. The output is Python objects;
  the cost is the objects. The fix is callers not calling them (Table 4).
- Not done: the `_gp_impl` rescale (Table 3 row 6) is a numeric seam
  (`_round_f32`, `_ftz` per row) and needs a DEVIATION of its own with the
  GP lane's fixtures; not this lane's file.
- Not done: the F-order `argmax_rows` path (Table 1) still walks rows in
  Python; the C-order path is native. A column-strided argmax helper is a
  small addition to `hotpath_helpers.mojo`.
- Owed by the orchestrator at landing: the three diffs above. Owed at the
  next release, as for every lane: the Apple and AMD GPU columns; this lane
  changed no GPU code and its CPU columns (M4 and EPYC) are IDENTICAL to each
  other and to main.
- The Mac A/B numbers are attribution only (load 14 to 77); the quotable A/B
  is the pod's.
