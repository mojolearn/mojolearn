# Opponent reference table

The opponents are measured once and stored here. A round measures OUR
IDENTICAL arm only and reads the opponent's number from this file. A new
opponent measurement is owed only when a row is missing for the tuple
(GPU model, driver, library version, dataset or shape, parameters), or when
a library pin changes. Never re-run an opponent to "refresh" a row that
already exists; never quote a row against a run of ours on a different GPU
model or a different dataset size.

Why a table and not published numbers (checked 2026-09-09): NVIDIA
gbm-bench is a harness with no results in its README; CatBoost's GPU
benchmark page stops at the V100 with no version or iteration count;
the cuML accelerator page gives speedup ratios against an unnamed CPU with
no GPU model, version or absolute time; cuBLAS blog charts cover Hopper in
FP16 and FP8. Nobody publishes absolute times on H100 or L40S at our sizes
with a pinned version, so published numbers are only a sanity check that a
row here is not misconfigured.

**HIGGS IS RETIRED (2026-09-11, ENGINEERING_RULES.md section 9).** Every
HIGGS row and every HIGGS ratio below is history and is never quoted as a
result again. The two benchmark datasets for trees and classical lanes are
NYC taxi (`taxi`, `taxireg`) and Istella-S LETOR (`istella`, `istellareg`);
their opponent rows are measured once per (GPU, driver, version, dataset)
on their first leg and land in this file under the GPU's section.

Every number is a median in milliseconds unless stated. Paths are relative
to the repository root. `e1g/<stamp>` means `bench/results/e1g/<stamp>/remote/logs/`.
"Config" is the opponent's FAST arm as pinned by `bench/speed/`; the
opponent's deterministic modes are not measured (see
`docs/` and the standing rule in the memory index).

## NVIDIA H100 80GB HBM3, driver 580.126.09, CUDA 12.4, torch 2.4.1+cu124

Trees, HIGGS (28 float32 features), 100 estimators, depth 6, lr 0.1, seed 7.
Source: `e1g/2026-08-28_030908-nvidia-speed-forest` (5 rounds, 1 warm-up
dropped) unless another stamp is named.

| opponent | version | config | 1M | 2M | 5M |
|---|---|---|---|---|---|
| CatBoost GPU symmetric | 1.2.10 | task_type GPU, SymmetricTree, l2 1.0, border_count 254, bootstrap No, Plain | 846.1 | 1227.2 | 2458.5 |
| CatBoost GPU depthwise | 1.2.10 | same, grow_policy Depthwise | 1232.5 | 1560.6 | 2559.7 |
| CatBoost GPU lossguide | 1.2.10 | same, grow_policy Lossguide | 1600.6 | 1954.5 | 2870.5 |
| XGBoost GPU depthwise | 3.2.0 | reg_lambda 1.0, max_bin 255, subsample 1.0 | 617.3 | 1039.4 | 2165.9 |
| XGBoost GPU lossguide | 3.2.0 | same, grow_policy lossguide | 816.9 | 1272.0 | 2413.4 |
| LightGBM CUDA lossguide | 4.7.0 | num_leaves 2^depth, max_bin 255 (`e1g/2026-08-28_030911-nvidia-speed-forest`) | 1313.7 | 1669.3 | not run |
| cuML RandomForest | 26.8.0 | n_estimators 100, max_depth 16, max_features sqrt, n_bins 128, bootstrap | 3231.5 | 4543.0 | 7284.7 |
| LightGBM CUDA extra_trees | 4.7.0 | INVALID: 181 s / 227 s single samples in `030911`; build refusal in `030908` | invalid | invalid | invalid |

Trees, HIGGS 1M, 2026-09-09 trees lane (`bench/results/trees_identical/h100_2026-09-09/speed/`,
runpod/pytorch:2.4.0-py3.11-cuda12.4.1 container, driver 580.126.09,
train = first 1M rows, test = last 500,000, same process as our identical
arm, 7 rounds for the symmetric cells, 5 for RF, ms median with min..max).
The CatBoost row here (900) and the Aug 28 row above (846.1) are the same
library on the same GPU model in different containers; quote whichever
matches the leg you are comparing against and name it.

| opponent | version | config | 1M | log |
|---|---|---|---|---|
| CatBoost GPU symmetric, Logloss | 1.2.10 | SymmetricTree, iters 100, depth 6, lr 0.1, l2 1, border_count 254, bootstrap No, Plain, seed 7 | 900 (864..939) | baseline.gbdt-symmetric.higgs.r1000000.full.log |
| CatBoost GPU symmetric, RMSE on the 0/1 label | 1.2.10 | same, loss RMSE (higgsreg) | 699 (680..759) | baseline.gbdt-symmetric.higgsreg.r1000000.full.log |
| cuML RandomForestClassifier | 26.08.00 | 100 trees, depth 16, sqrt features, 128 bins, bootstrap, seed 7, n_streams default; logloss 0.538814 | 3314 (3257..3950) | baseline.rf.higgs.r1000000.full.log |
| LightGBM CUDA rf boosting | 4.7.0, USE_CUDA=ON | 100 trees, depth 16, 32768 leaves, bagging 0.632/1, feature_fraction sqrt, max_bin 255; logloss 0.638510 (3 rounds) | 469654 (468691..471303) | same |

Our identical arm in the same process on that H100: symmetric Logloss 775
ms (697..1161), RMSE 806, RF 5762 (see `docs/lanes/HANDOFF_trees.md`;
450-600 ms of each round is host-side outside the fit).

Our identical arm, 2026-09-10 leg, same GPU model and image, driver
580.126.09, source deb01bcf, ours alone in the process (no opponent
re-measured; `bench/results/trees_identical/h100_2026-09-10/speed/`, 5 rounds
RF/ET, 7 symmetric, ms median with min..max): RF 1M 2475 (2385..2532) at the
shipped source and 2234 (2216..2277) at the flipped HIST_ITEMS_PER_THREAD 4
default (hash 3ffa2951595422d4 both, the Sep 9 hash); RF 2M 4037 (3890..4135)
shipped, 3743 (3719..3761) with the flip; ET 1M 3314 (3272..3430), ET 2M 6099
(5994..6335); symmetric Logloss 1M 478 (459..528), 2M 753 (741..926), RMSE 1M
396 (377..476). Quote the cuML 1M row above (3314) against the 1M cells and
the Aug 28 cuML 2M row (4543.0, different container) against the 2M cells,
naming each. Detail: `docs/lanes/HANDOFF_trees.md`, "2026-09-10 H100 leg".

Our identical arm, 2026-09-10 night leg, same GPU model and image, driver
580.126.09, source 7cebeecf (DEVIATION 2500 native label encoding on the
forest path), ours alone in the process, no opponent re-measured
(`bench/results/trees_identical/h100_2026-09-10b/speed/`, ms median with
min..max, 5 rounds RF/ET and the 2M cells, 7 rounds the 1M gbdt cells):
RF 1M 1516 (1476..1579) hash 3ffa2951595422d4, RF 2M 2283 (2231..2293) hash
67d883dc6079b90f; quote the cuML 1M row above (3314, Sep 9) against the 1M
cell and the Aug 28 cuML 2M row (4543.0, different container) against the
2M cell: 0.46x and 0.50x of their time. Depthwise Logloss 1M 1070
(987..1121), 2M 1917 (1905..1962): against the Aug 28 XGBoost GPU depthwise
rows (617.3, 1039.4) 1.73x and 1.84x, against the Aug 28 CatBoost depthwise
rows (1232.5, 1560.6) 0.87x and 1.23x. Lossguide Logloss 1M 1652
(1639..1690), 2M 2451 (2437..2509): against the Aug 28 XGBoost lossguide rows
(816.9, 1272.0) 2.02x and 1.93x, LightGBM CUDA lossguide (1313.7, 1669.3)
1.26x and 1.47x, CatBoost lossguide (1600.6, 1954.5) 1.03x and 1.25x.
Symmetric Logloss 1M 533 (490..570) against the Sep 9 CatBoost row (900)
0.59x and the Aug 28 row (846.1) 0.63x; the Sep 10 leg's 478 was on another
physical pod and the gap is unresolved. ET 1M 2578 (2543..2616), still no
valid NVIDIA opponent row. Fingerprints 81/81 IDENTICAL against the Sep 10
set. Detail: `docs/lanes/HANDOFF_trees.md`, "2026-09-10 night H100 leg".

Our identical arm, 2026-09-11 confirmation leg, same GPU model and image,
driver 580.126.09, source 352d9781 (DEVIATION 2502 pure-node leaf ON, as
it is in the shipped default since 2026-09-11 evening after one day as
opt-in; the 7cebeecf-class RF forest, 1516/2283 ms below, is the opt-out
arm; 2512 kernel zero), ours alone in the process, no opponent re-measured
(`bench/results/trees_identical/h100_2026-09-11/speed/`, ms median with
min..max, 5 rounds RF/ET, 7 rounds the gbdt cells; RF 1M and symmetric 1M
were each run twice, first and last, as the drift control): RF 1M 1088
(1071..1108) and 1081 (1051..1099), hash efd14ab2c09ff57c (the DEVIATION
2502 forest, equal to the Apple M4 hash); RF 2M 1760 (1716..1792), hash
7fd9fda29a4fa81d. Quote the cuML 1M row above (3314, Sep 9) against the 1M
cells and the Aug 28 cuML 2M row (4543.0, different container) against the
2M cell: 0.33x and 0.39x of their time (the Sep 10 night leg's 0.46x and
0.50x at the previous forest). Symmetric Logloss 1M 527 (463..602) and 493
(472..578), hash dac2cf366e219cec unchanged: against the Sep 9 CatBoost row
(900) 0.59x and 0.55x, against the Aug 28 row (846.1) 0.62x and 0.58x.
Depthwise Logloss 1M 1016 (986..1083), hash unchanged: against the Aug 28
XGBoost GPU depthwise row (617.3) 1.65x, CatBoost depthwise (1232.5) 0.82x.
Lossguide Logloss 1M 1610 (1602..1649), hash unchanged: against the Aug 28
XGBoost lossguide row (816.9) 1.97x, LightGBM CUDA lossguide (1313.7) 1.23x,
CatBoost lossguide (1600.6) 1.01x. ET 1M 2504 (2476..2585), hash unchanged,
still no valid NVIDIA opponent row. Fingerprints against the Sep 10 night
set: rf-clf moved 9 of 9 (the DEVIATION 2502 forest), 72 of 72 IDENTICAL on
the other eight lanes; rf-clf and rf-reg 18 of 18 IDENTICAL against the
Apple M4 2502 set. Detail: `docs/lanes/HANDOFF_trees.md`, "2026-09-11 H100
confirmation leg".

Extra trees therefore has NO valid NVIDIA opponent row. That measurement
is owed (a LightGBM build with USE_CUDA, or cuML RF with `split_criterion`
random thresholds if cuML admits it).

Accuracy alongside the timing (CatBoost GPU symmetric, HIGGS 1M): logloss
0.542398, AUC 0.800529, from the same logs (`FSPEED-ACC` lines).

Trees, Istella-S 1M (220 features), 2026-09-11 istella leg
(`bench/results/trees_identical/h100_2026-09-11_istella/speed/`, pod
5gvizdykv4gqwm, runpod/pytorch:2.4.0-py3.11-cuda12.4.1 container, driver
580.126.09, source a6d25306 plus the `_buffer.py` fix in
`lane/nvidia-istella-0911`). Istella-S LETOR (`tools/speed_gbdt_arm.py:
load_istella`): train = first 1M rows of sample/train.txt, test = first
500,000 rows of sample/test.txt, binary target relevance > 0 (11.4 percent
positive), the 1.797e308 missing marker clamped to the float32 maximum
(0.30 percent of cells). Same process as our identical arm, arms
alternating per round, 7 rounds for the boosting cells and 5 for RF, ms
median with min..max. Config per lane as the HIGGS rows above (100
estimators, depth 6, lr 0.1, seed 7; RF 100 trees, depth 16, sqrt
features, 128 bins, bootstrap). The depthwise and lossguide cells ran
twice: the first pass had no XGBoost on the pod (the setup pip line
lacked it, fixed in `tools/trees_identical_remote.sh`) and is kept as
`*.full.pass1.log`; the rows below are the complete-roster pass.

| opponent | version | config | 1M | 2M | log |
|---|---|---|---|---|---|
| cuML RandomForestClassifier | 26.08.00 | 100 trees, depth 16, sqrt features, 128 bins, bootstrap, seed 7, n_streams default; logloss 0.145504, AUC 0.964577 (2M: 0.144861, 0.964984) | 3500 (3440..3679) | 5751 (5623..6696) | baseline.rf.istella.r1000000.full.log, baseline.rf.istella.r2000000.full.log |
| CatBoost GPU symmetric, Logloss | 1.2.10 | SymmetricTree, iters 100, depth 6, lr 0.1, l2 1, border_count 254, bootstrap No, Plain, seed 7; logloss 0.139154, AUC 0.966720 | 1458 (1408..1527) | not run | baseline.gbdt-symmetric.istella.r1000000.full.log |
| CatBoost GPU depthwise, Logloss | 1.2.10 | same, grow_policy Depthwise; logloss 0.125909, AUC 0.972801 (pass1 without XGBoost: 1629 (1573..1778)) | 1664 (1633..1746) | not run | baseline.gbdt-depthwise.istella.r1000000.full.log |
| XGBoost GPU depthwise, Logloss | 3.2.0 | device cuda, reg_lambda 1.0, max_bin 255, subsample 1.0; logloss 0.125110, AUC 0.973774 | 1768 (1649..1951) | not run | same |
| CatBoost GPU lossguide, Logloss | 1.2.10 | same, grow_policy Lossguide; logloss 0.121318, AUC 0.975388 (pass1 without XGBoost: 2328 (2299..2433)) | 2358 (2325..2433) | not run | baseline.gbdt-lossguide.istella.r1000000.full.log |
| XGBoost GPU lossguide, Logloss | 3.2.0 | same, grow_policy lossguide, max_leaves 64; hash 377719029530ae62 and accuracy equal to its depthwise row (every depth-6 leaf is reached at 1M rows) | 1845 (1762..1940) | not run | same |
| LightGBM CUDA (rf and lossguide) | 4.7.0 pip wheel | REFUSED: CUDA Tree Learner not enabled in this build (USE_CUDA build skipped on this leg) | refused | refused | same logs |

Note (2026-09-11, AMD trees leg): the XGBoost lossguide row above is XGBoost's depthwise tree, because at depth 6 the 64-leaf cap never binds on any arm, ours included (our Lossguide stops a leaf at max_depth, `greedy_search_helper_depthwise.mojo` `is_terminal_leaf`); the config already matches our lossguide arm, and a lossguide cell where max_leaves binds needs max_depth above 6 on every arm.

Our identical arm in the same process on that H100, Istella-S, hash the same
on every round: RF 1M 3265 (3233..3482) hash 15e38312cb4bb870, logloss
0.145578, AUC 0.964548, 0.93x of the cuML row; RF 2M 5103 (4964..5738) hash
b1d9d40baca870b5, logloss 0.144875, AUC 0.964945, 0.89x of cuML. Symmetric
Logloss 1M 2407 (2326..2530) hash 238d3abce0cabf43, logloss 0.138653, AUC
0.966990, 1.65x of the CatBoost row. Depthwise 1M 3304 (3175..3483) hash
5d053cd086658072, logloss 0.126517, AUC 0.971896, 1.99x of CatBoost depthwise
and 1.87x of XGBoost depthwise (pass1, XGBoost absent: 3033 (2971..3477),
1.86x of that pass's CatBoost). Lossguide 1M 3908 (3778..4160) hash
6182fd2bee4fb941, logloss 0.122045, AUC 0.975037, 1.66x of CatBoost lossguide
and 2.12x of XGBoost lossguide (pass1: 3892 (3802..4037), 1.67x). ET 1M 6045
(6029..6148) hash 40b1c5b03ba40420, logloss 0.188191, AUC 0.938768, still no
valid NVIDIA opponent row. HIGGS RF 1M reference on the same pod, ours alone,
no opponent re-run: 1523 (1509..1628) hash 3ffa2951595422d4 (the shipped
default forest, the Sep 10 night hash), logloss 0.538850, AUC 0.809906.

DEVIATION 2502 (pure classification node is a leaf; this leg built source
a6d25306 where it was opt-in; since main 329cbeb4 the same evening it is ON
by default and this leg's "opt-in" arm is the shipped forest, its "default"
arm the `-D MOJOLEARN_2502_RETRY_PURE=1` opt-out), default vs opt-in on
the same pod, ours alone for the opt-in cells (`pureleaf.*.log`), ms median
(min..max), hash, logloss, AUC: Istella-S 1M default 3265 (3233..3482)
15e38312cb4bb870 0.145578 0.964548 vs opt-in 2136 (2064..2292)
574b24d0d7af51d0 0.145560 0.964538 (0.65x); Istella-S 2M default 5103
(4964..5738) b1d9d40baca870b5 0.144875 0.964945 vs opt-in 3714 (3670..4457)
cc25cb08f8b5a813 0.144845 0.964998 (0.73x); HIGGS 1M default 1523
(1509..1628) 3ffa2951595422d4 0.538850 0.809906 vs opt-in 1058 (1053..1097)
efd14ab2c09ff57c 0.538817 0.809830 (0.69x). Stage times (one untimed
replicate each, `*.stage.log`, Istella-S 1M): default device_wait 1.664 s,
host_enq_hist_retry 0.371 s, host_hist_zero 0.079 s, fit_total 2.519 s;
opt-in device_wait 0.857 s, host_enq_hist_retry 0.044 s, host_hist_zero
0.017 s, fit_total 1.233 s. Fingerprints: the default set is 81 of 81
IDENTICAL against the Sep 10 night H100 set (`ib/diff.sep10b_baseline.
baseline.txt`); the opt-in set moves rf-clf 9 of 9 and keeps rf-reg 9 of 9
against the default (`ib/diff.baseline.pureleaf.txt`) and is 18 of 18
IDENTICAL with the Apple M4 2502 set (`ib/diff.apple_rf2502.pureleaf.txt`).
Finding on this leg: with cuML in the process our RF arm was REFUSED
("expected LP__PyBuffer instance instead of pointer to _PyBuffer",
`baseline.rf.istella.r1000000.full.pass0.log`, cuML rows only) because
`treelite.model` assigns its own `argtypes` on the cached
`ctypes.pythonapi.PyObject_GetBuffer`; `python/mojolearn/_buffer.py` now
takes private function pointers through `PyDLL.__getitem__`.

Classical, cuML 26.8.0 / cuVS, FAST arm, `e1g/2026-08-28_040832-nvidia-speed-classical`
(3 rounds per arm run; PCA row is the only H100 PCA opponent that did not
refuse, the Aug 26 leg refused its solver).

| lane | shape | opponent | ms |
|---|---|---|---|
| knn | 400,000 x 32, 4,000 queries, k 10 | cuML NearestNeighbors | 9.542 |
| kmeans | 4M x 32, k 64, 20 iterations | cuML KMeans | 120.217 |
| ols | 4M x 32 | cuML LinearRegression | 118.933 |
| pca | 4M x 32, 8 components | cuML PCA | 119.669 |
| iforest | 500,000 x 32 | cuML IsolationForest (`e1g/2026-08-28_040244`, 12 rounds) | 85.159 |
| ivf | 512 x 8, 64 queries, 8 lists, 3 probes, k 8 | cuVS IVF-Flat (8 rounds) | 13.229 |
| dbscan | 4,000 x 16 | cuML DBSCAN | 1.082 |
| hdbscan | fixture | cuML HDBSCAN | 4.391 |
| cd | 2,048 x 16 | cuML coordinate descent | 1.258 |
| kde | 1,024 x 256 x 8 | cuML KernelDensity | 0.384 |
| krr | RBF 16 x 5 | cuML KernelRidge | 1.808 |
| linkage | fixture | cuML AgglomerativeClustering | 2.315 |
| svm | xor 240 x 2 | cuML SVC | 4.864 |
| holtwinters | 7 x 72, forecast 12 | cuML HoltWinters | 8.147 |
| kpss | 8 x 520 | cuML kpss | 0.424 |
| metrics | fixture | cuML metrics | 12.626 |
| cholesky | RBF 64 x 64, 4 rhs | torch linalg (fixture solve) | 0.160 |

The classical fixtures above are small (thousands of rows) except knn,
kmeans, ols, pca and iforest. Only those five are admissible as an
identical-vs-opponent row; the rest measure launch overhead.

kNN and UMAP, cuML 26.08.00, cupy 14.2.0, CUDA runtime 12.9, 2026-09-09,
worktree branch lane/knn-identical,
`bench/results/knn/2026-09-09-h100-identical-defaults/ref/summary.json` and
`umap/` (7 rounds). "request" includes host transfers, "device" excludes
them. k=10 and k=15 are separate rows; do not substitute one for the other.

| index | queries | k | cuML brute NearestNeighbors request | cuML device | Historical Sep10 request | Historical Sep10 device | historical request ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k x 32 | 32 | 10 | 1.125 | 0.669 | 0.528 | 0.262 | 0.47x |
| 100k x 32 | 128 | 10 | 0.914 | 0.459 | 0.657 | 0.388 | 0.72x |
| 100k x 32 | 1000 | 10 | 1.572 | 1.081 | 2.535 | 2.243 | 1.61x |
| 100k x 32 | 4000 | 10 | 3.480 | 2.904 | 9.091 | 8.711 | 2.61x |
| 400k x 32 | 32 | 10 | 1.546 | 1.200 | 1.959 | 1.000 | 1.27x |
| 400k x 32 | 128 | 10 | 1.531 | 1.159 | 2.476 | 1.521 | 1.62x |
| 400k x 32 | 1000 | 10 | 4.101 | 3.659 | 9.830 | 8.847 | 2.40x |
| 400k x 32 | 4000 | 10 | 10.225 | 9.632 | 35.486 | 34.402 | 3.47x |
| 100k x 32 | 32 | 15 | 1.113 | 0.675 | 0.571 | 0.304 | 0.51x |
| 100k x 32 | 128 | 15 | 0.922 | 0.470 | 0.703 | 0.431 | 0.76x |
| 100k x 32 | 1000 | 15 | 1.617 | 1.111 | 2.892 | 2.583 | 1.79x |
| 100k x 32 | 4000 | 15 | 3.571 | 2.957 | 10.515 | 10.077 | 2.94x |
| 400k x 32 | 32 | 15 | 1.573 | 1.241 | 2.124 | 1.168 | 1.35x |
| 400k x 32 | 128 | 15 | 1.532 | 1.197 | 2.652 | 1.688 | 1.73x |
| 400k x 32 | 1000 | 15 | 4.183 | 3.750 | 11.216 | 10.203 | 2.68x |
| 400k x 32 | 4000 | 15 | 10.817 | 9.756 | 40.956 | 39.814 | 3.79x |

Later 512-query batching supersedes the 400k/4000 rows: final request
27.525704 ms (k10) and 31.860726 ms (k15), or 2.69x/2.95x the cached
references. See "Sep10 bounded kNN query batches" below for provenance.
The full historical grid is retained rather than mixing captures.

The historical Sep10 IDENTICAL columns above use the eight-query register tile from
`edada38d` with `MOJOLEARN_KNN_IDENTICAL_ROWS8=1` (adopted as default in
`0e213d65`; original lane commits `8fdfd85f` and `c03bbcfc`). NVIDIA H100 80GB HBM3, driver580.126.09, Mojo1.0.0(ed45d567),
IDENTICAL, dyadic-v1, two warmups and seven rounds; evidence:
`bench/results/knn/2026-09-10-logical-width/h100/`. The archived cuML tuple
is reused unchanged. All16 new-grid index/distance fingerprints match the
previous corrected baseline. Exact396584-triplet and24-layout gates pass.
At400k/4000 queries, paired k10 request39.31 ->35.46ms (~9.8% less),
k15 44.80 ->40.93ms (~8.6% less). Eight-feature coverage is flat to0.9%
slower; this is not a universal shape-speedup claim. Logical-width extraction
alone was performance-neutral in alternating baseline/candidate measurements.
The final default build, without the experimental flag, passes the same oracle
and layouts and records k10 request35.473274/device34.374416ms and k15
request40.942684/device39.805605ms in seven rounds; those paired validation
samples are separate from the complete-grid medians tabulated above.

Historical previous Sep10 columns used source `8266f1c0`, NVIDIA H100 80GB HBM3,
driver 580.126.09, Mojo 1.0.0 (ed45d567), IDENTICAL mode, dyadic-v1,
two warmups and seven timed rounds. The opponent is the archived matched
hardware/driver reference above, not rerun. Evidence and all 16 price logs:
`bench/results/knn/2026-09-10-residual-final/h100/`.
Same-pod pristine `5011239a` controls at 400k/4000 queries measured
49.452488 -> 39.310253 ms (k10, 20.51% less) and 54.973115 -> 44.817340 ms
(k15, 18.48% less), with matching index and distance fingerprints.
The repeated final-grid rows above can differ slightly from those control-pair
medians. This is an own-before/after bit check; opponent admission still compares
indices only. Previous Sep9 absolute timings came from another pod/session;
the compiler version is the same, so no compiler regression is inferred.

The k=15 rows were backfilled from the same archived JSON on 2026-09-10,
without new opponent runs. All rows use dyadic-v1, two warmups and seven
timed rounds. Admission compares neighbour indices; it does not assert
bitwise equality of cuML distances. Request includes host transfers and
device excludes them, as specified in `tools/knn_cuml_reference.py`.

### torch byte-LM training step (target shape, enwik8 and Pile GitHub)

Source: `bench/results/e1g/2026-09-11_133041-nvidia-h100-attention-torch/remote/torch-lm-step/`
(`summary.tsv`, one JSON per column and corpus), RunPod pod quf7729vxu5q66,
driver 580.126.09, torch 2.4.1+cu124, CUDA 12.4, cuDNN 90100, Python 3.11.10,
harness `tools/torch_lm_step_opponent.py` (sha256 16b44393...) at commit
cd086f67. Shape: batch 1, length 2048, d_model 768, 12 heads (12 KV), head_dim
64, intermediate 2048, 12 layers, vocab 50257, 162,147,840 parameters, AdamW
(lr 1e-3, betas 0.9/0.999, eps 1e-8, weight decay 0.01) on every parameter,
the same init as our probe (seed 93261). SDPA backend `efficient` (flash is
not available in float32). Clock: synchronize; ids to device; zero_grad;
forward and mean cross entropy; backward; AdamW step; `loss.item()`;
synchronize. 2 warmups (compilation included for compile), then 7 timed steps
in one process per column and corpus; median seconds.

OUR IDENTICAL arm on the SAME pod: the shipped `stash_tiled` lean step from
`remote/attention-step/lm_summary.tsv`, same shape, parameters and corpora.

| column | enwik8 s | Pile GitHub s | our IDENTICAL / column (enwik8, Pile GitHub) |
|---|---|---|---|
| eager_tf32 (their fastest measured) | 0.03794 | 0.03776 | 10.11x, 10.15x |
| compile_fp32 (inductor, default mode) | 0.05744 | 0.05760 | 6.68x, 6.66x |
| eager_fp32 | 0.06174 | 0.06184 | 6.21x, 6.20x |
| ours IDENTICAL (`stash_tiled`, bits equal on every vendor) | 0.3835 | 0.3833 | 1 |

Not measured, so their true fast column may be faster still: compile with
TF32, and mixed precision (bf16 autocast). The harness now has those columns
(`compile_tf32`; `eager_bf16` and `compile_bf16`, which wrap forward and loss
in `torch.autocast` bfloat16 with float32 parameters and AdamW state, leave
the SDPA pick to torch under `auto`, and record the SDPA kernel that ran). They
are owed on this H100 and on the AMD MI300X (Hot Aisle, ROCm torch). The
MI300X rows are a new tuple and are never mixed with the H100 rows above.
Our number is the step time for
bitwise identical results across Apple, NVIDIA and AMD; theirs carries no
such property (`nondeterministic_label` true on every column).

#### Same pod, after the two NVIDIA default flips (2026-09-11 16:41Z)

Source: `bench/results/e1g/2026-09-11_164101-nvidia-h100-80gb-hbm3-new-defaults-torch/remote/`
(`attention-step/lm_summary.tsv`, `torch-lm-step/summary.tsv`), RunPod pod
mgpc9vhnkjre5x, driver 580.126.09 (1980 MHz clock reading), commit e629434d,
same harness, shape, init, AdamW, clock and corpora as the table above. OUR
IDENTICAL arm is the shipped NVIDIA default at that commit: GEMM `ksplit`
(DEVIATION 2595) and attention `stash_tiled_fgrid_r32_qres_pf` (DEVIATION
2534), resolved by the binding and confirmed by the shipped fused check on
the pod. SDPA observed in the profiled warmup: `efficient` for the float32
columns, `flash` for the bf16 columns.

| column | enwik8 s | Pile GitHub s | our IDENTICAL / column (enwik8, Pile GitHub) |
|---|---|---|---|
| compile_bf16 (their fastest measured) | 0.02038 | 0.02085 | 14.48x, 14.15x |
| compile_tf32 | 0.03189 | 0.03208 | 9.25x, 9.19x |
| eager_bf16 | 0.03705 | 0.03879 | 7.96x, 7.60x |
| eager_tf32 | 0.03788 | 0.03779 | 7.79x, 7.80x |
| compile_fp32 | 0.05775 | 0.05735 | 5.11x, 5.14x |
| eager_fp32 | 0.06197 | 0.06200 | 4.76x, 4.76x |
| ours IDENTICAL (shipped defaults, bits equal on every vendor) | 0.2950 | 0.2949 | 1 |

On the same pod our previous attention default (`stash_tiled`, with the GEMM
default already `ksplit`) ran 0.3414 / 0.3409 s. Every torch column is
nondeterministic by label; ours is bitwise identical across Apple, NVIDIA
and AMD.

#### Same pod, after the NVIDIA dk/dv flip (2026-09-11 19:32Z)

Source: `bench/results/e1g/2026-09-11_193203-nvidia-h100-80gb-hbm3-kv-default-torch/remote/`
(`attention-step/lm_summary.tsv`, `torch-lm-step/summary.tsv`), RunPod pod
2zjzt53vh0tpz1, driver 580.126.09 (1980 MHz clock reading), commit 272011ae,
same harness, shape, init, AdamW, clock and corpora as the tables above. OUR
IDENTICAL arm is the shipped NVIDIA default at that commit: GEMM `ksplit`
(DEVIATION 2595) and attention `stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`
(DEVIATIONS 2534 and 2597), named by the shipped fused check on the pod
(`DEFAULT column=nvidia ... word=52327`, `is_default=True`, dk/dv
`fused_bwd_dkdv_r2_kernel[64,32]`). Every step witness equals the previous
default's on both corpora. Operand dumps are outside the repository in
`~/mojolearn-evidence/` under the same leg name. SDPA observed: `efficient`
for the float32 columns, `flash` for the bf16 columns.

| column | enwik8 s | Pile GitHub s | our IDENTICAL / column (enwik8, Pile GitHub) |
|---|---|---|---|
| compile_bf16 (their fastest measured) | 0.02017 | 0.02355 | 14.47x, 12.34x |
| compile_tf32 | 0.03198 | 0.03192 | 9.13x, 9.10x |
| eager_tf32 | 0.03802 | 0.03803 | 7.68x, 7.64x |
| eager_bf16 | 0.04015 | 0.03806 | 7.27x, 7.64x |
| compile_fp32 | 0.05736 | 0.05689 | 5.09x, 5.11x |
| eager_fp32 | 0.06114 | 0.06136 | 4.77x, 4.74x |
| ours IDENTICAL (shipped defaults, bits equal on every vendor) | 0.2919 | 0.2906 | 1 |

On the same pod the previous attention default (`stash_tiled_fgrid_r32_qres_pf`)
ran 0.2941 / 0.2940 s, so the flip is 0.9925 / 0.9884 here (0.9908 on the
zdot leg's pod). compile_bf16 on Pile GitHub (0.02355) sits 1.17x above its
enwik8 cell and 1.13x above the 16:41Z pod's 0.02085; quote the enwik8 cell as
their fastest. Every torch column is nondeterministic by label.

#### Same pod, after the NVIDIA attention estash flip (2026-09-12 12:49Z)

Source: `bench/results/e1g/2026-09-12_124715-nvidia-h100-owed/remote/`
(`attention-step/lm_summary.tsv`, `torch-lm-step/summary.tsv`), RunPod pod
4pmnz0eju0iptm, driver 580.126.09, commit 5b7e1e41, same harness, shape,
init, AdamW, clock and corpora as the tables above. OUR IDENTICAL arm is the
shipped NVIDIA default at that commit, attention
`stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32` (DEVIATION 2657),
named by the shipped fused check on the pod itself (`DEFAULT column=nvidia
... word=6343783`, `is_default=True`, `arm_is_default=True` on both corpora,
`transformer_fused_check: PASS, 15 cases, every compared buffer
bit-identical`). Operand dumps are outside the repository in
`~/mojolearn-evidence/` under the same leg name. SDPA observed: `efficient`
for the float32 and tf32 columns, `flash` for the bf16 columns.

| column | enwik8 s | Pile GitHub s | our IDENTICAL / column (enwik8, Pile GitHub) |
|---|---|---|---|
| compile_bf16 (their fastest measured) | 0.02152 | 0.02291 | 11.20x, 10.46x |
| compile_tf32 | 0.03198 | 0.03180 | 7.54x, 7.53x |
| eager_bf16 | 0.03554 | 0.03448 | 6.78x, 6.95x |
| eager_tf32 | 0.03773 | 0.03778 | 6.39x, 6.34x |
| compile_fp32 | 0.05777 | 0.05773 | 4.17x, 4.15x |
| eager_fp32 | 0.06183 | 0.06148 | 3.90x, 3.90x |
| ours IDENTICAL (shipped defaults, bits equal on every vendor) | 0.2411 | 0.2396 | 1 |

WHAT MOVED AND WHAT DID NOT. Every ratio here is smaller than in the 19:32Z
table above, and the reason is OUR cell, not theirs: ours went 0.2919 /
0.2906 to 0.2411 / 0.2396 on the estash flip, while their columns sit where
they sat (compile_bf16 0.02017 -> 0.02152 on enwik8, inside the spread these
columns already show pod to pod). Every torch column is nondeterministic by
label; ours is bitwise identical across Apple, NVIDIA and AMD.

THIS TABLE DOES NOT INCLUDE DEVIATION 2649. It was measured at 5b7e1e41 and
the step glue flip merged after that pod died. The table below supersedes it.

#### Same pod, after the step glue flip as well (2026-09-12 13:45Z)

Source: `bench/results/e1g/2026-09-12_133007-nvidia-h100-owed-rest/remote/`
(`attention-step/lm_summary.tsv`, `torch-lm-step/summary.tsv`), RunPod pod
aylfazvhgd6g8d, driver 580.126.09, commit bb679f19 (main 00cd9a5f plus one
bench file), same harness, shape, init, AdamW, clock and corpora as the
tables above. This is the first opponent table measured at a commit carrying
BOTH neural flips of this pass, DEVIATION 2657 (attention estash) and
DEVIATION 2649 (byte LM step glue). OUR IDENTICAL arm is the shipped NVIDIA
default, named by the shipped fused check on the pod itself (`DEFAULT
column=nvidia arm=stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32
word=6343783 trial_hook=False`, `arm_is_default=True` on both corpora,
`transformer_fused_check: PASS, 15 cases, every compared buffer
bit-identical`). SDPA observed: `efficient` for the float32 and tf32
columns, `flash` for the bf16 columns.

| column | enwik8 s | Pile GitHub s | our IDENTICAL / column (enwik8, Pile GitHub) |
|---|---|---|---|
| compile_bf16 (their fastest measured) | 0.01955 | 0.02016 | 11.90x, 11.45x |
| compile_tf32 | 0.03183 | 0.03166 | 7.31x, 7.29x |
| eager_bf16 | 0.03620 | 0.03634 | 6.43x, 6.35x |
| eager_tf32 | 0.03780 | 0.03785 | 6.15x, 6.10x |
| compile_fp32 | 0.05682 | 0.05740 | 4.09x, 4.02x |
| eager_fp32 | 0.06153 | 0.06076 | 3.78x, 3.80x |
| ours IDENTICAL (shipped defaults, bits equal on every vendor) | 0.2326 | 0.2309 | 1 |

BOTH CELLS MOVED THIS TIME, and the ratios must not be read as one number
moving. Ours went 0.2411 / 0.2396 to 0.2326 / 0.2309, which is 3.5% and 3.6%
and is where DEVIATION 2649 lands. But `compile_bf16` moved too, 0.02152 /
0.02291 to 0.01955 / 0.02016, about 9% and 12% on a different physical H100,
so THAT ratio got WORSE (11.20x / 10.46x to 11.90x / 11.45x) even though our
own step improved. At the other end `eager_fp32` barely moved at all (0.06183
/ 0.06148 to 0.06153 / 0.06076) and that ratio improved, 3.90x / 3.90x to
3.78x / 3.80x. The reading is that the compiled bf16 column is the volatile
one pod to pod and the eager float32 column is the stable one, so a ratio
against `compile_bf16` carries pod noise that a ratio against `eager_fp32`
does not. The 12:49Z table's claim that only our cell moves was true of that
pair of runs and is not a general rule.

Every torch column is nondeterministic by label; ours is bitwise identical
across Apple, NVIDIA and AMD.

THIS TABLE DOES NOT INCLUDE THE STEP GLUE FLIP. The pod ran commit
5b7e1e41, which carries DEVIATION 2657 but NOT DEVIATION 2649
(`optskip_noshadow_rows16`, merged afterwards at 65a889bf, measured FLIP
geomean 0.9723 on its own leg). The next pod that measures this table should
read a smaller cell for ours again.

### kNN second kind (HIGGS rows)

Every kNN row above is dyadic-v1, a generator, and the gate's `large`
fixture is another generator of the same shape, so (ENGINEERING_RULES.md
section 9) a selection win timed on both was timed on ONE kind. The second
kind is REAL data, the HIGGS prefix `tools/knn_datasets.py::higgs_block`
loads (UCI 00280, the first 404,000 rows of the gzip stream, the 28 raw
float32 kinematic features, no shuffle, no scaling, no deduplication;
prefix rows 0..399,999 are the index, 400,000..403,999 the queries), the
same bytes `tools/knn_selection_gate.py`'s `higgs` fixture times. A second
dataset is a new opponent tuple, measured ONCE on its first leg and never
rerun; later gate legs pass the numbers through
`MOJOLEARN_KNN_SELECTION_CACHED_OPPONENT_HIGGS=k10=<ms>,k15=<ms>` and quote
them as a cached-reference ratio, never as a paired opponent measurement.

| index | queries | k | features | cuML brute NearestNeighbors request ms | cuML device ms | GPU, driver, cuML | sha256_block | evidence |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 400,000 (HIGGS rows 0..399,999) | 4,000 (rows 400,000..403,999) | 10 | 28 | 10.159 | 9.588 | H100 80GB HBM3, 580.126.09, cuML 26.08.00 (cupy 14.2.0, CUDA runtime 12.9) | 17806734cbf7c2b8 | `bench/results/e1g/2026-09-11_063726-nvidia-h100-knn-selection-secondkind/remote/knn-selection/opponent-higgs/cuml-reference-higgs.json` |
| 400,000 (HIGGS rows 0..399,999) | 4,000 (rows 400,000..403,999) | 15 | 28 | 10.270 | 9.704 | same | same | same |

The invocation that produces the tuple, on the box, after the gate phase
(`tools/knn_selection_gate.sh` runs it as its optional `opponent` phase
under `MOJOLEARN_KNN_SELECTION_OPPONENT=1`, in a venv built from
`numpy==2.4.6 cupy-cuda12x==14.2.0 cuml-cu12==26.8.0` with
`--extra-index-url https://pypi.nvidia.com`, the recipe of
`tools/knn_reference_leg.sh`), seven timed rounds after two warmups,
request and device regions as above.

```
GBM_BENCH_DATA=$HOME/datasets/gbm-bench python tools/knn_cuml_reference.py --dataset higgs \
    --index 400000 --queries 4000 --k 10 15 --rounds 7 \
    --out /root/gemm_leg_out/knn-selection/opponent-higgs
```

The JSON (`opponent-higgs/cuml-reference-higgs.json`) names the dataset,
`sha256_block`, `sha256_index`, `sha256_queries`, `index_rows` and
`query_rows` per result and the load record under
`environment.dataset_source`; the `sha256_block` must equal
`fixtures.higgs.sha256_block` in the gate JSON of the same leg. Fill the
table from `request_median_ms` and `device_median_ms`, with the leg
directory under `bench/results/e1g/` as evidence. Measured 2026-09-11 (the
leg above, seven rounds, sha256_block equal to the gate fixture of the same
leg); our IDENTICAL request on the same bytes, `warpbound_guard` default,
is 26.974 ms (k10, 2.66x the cached row) and 30.181 ms (k15, 2.94x).

| UMAP (32 features, 15 neighbors, 2 components, 200 epochs) | cuML UMAP ms |
|---|---:|
| 20k rows | 145.7 |
| 100k rows | 321.3 |

GEMM, cuBLAS through torch 2.4.1+cu124 (`torch.matmul`, `allow_tf32`
False for the fp32 column and True for the tf32 column), CUDA 12.4,
`e1g/2026-08-25_155542-nvidia-speed-gemmseq/remote/logs/gemm.gemm.cublas.log`
(5 rounds after 1 warm-up, medians). The repeat leg
`e1g/2026-08-25_160520-nvidia-speed-gemmseq` agrees within 0.002 ms on
every row except lm_head.t512 (10.820 / 1.387) and mlp_down.t512 tf32
(0.212); quote the 155542 leg. Transcribed 2026-09-09 from the logs, no
re-run. Shape names are `bench/gemm_shapes.mojo`'s; the t512 names
carry m as the log reports it.

| shape | cublas-fp32 | cublas-tf32 |
|---|---|---|
| gram.32x32x1M | 0.240 | 0.115 |
| gram.32x32x64K | 0.039 | 0.029 |
| gram.128sq.x100003 | 0.096 | 0.060 |
| ols.step1.16x16x64K | 0.038 | 0.028 |
| pca.transform.8192x4x4 | 0.024 | 0.019 |
| pca.transform.wide.8192x64x128 | 0.026 | 0.020 |
| kmeans.dist.4096x64x64 | 0.023 | 0.019 |
| ols.predict.gemv.64Kx16 | 0.020 | 0.019 |
| llama8b.qkv.t1 | 0.042 | 0.043 |
| llama8b.qkv.t8 | 0.061 | 0.048 |
| llama8b.qkv.t512 | 0.374 | 0.071 |
| llama8b.mlp_up.t1 | 0.096 | 0.097 |
| llama8b.mlp_up.t8 | 0.160 | 0.109 |
| llama8b.mlp_up.t512 | 1.379 | 0.187 |
| llama8b.mlp_down.t1 | 0.097 | 0.098 |
| llama8b.mlp_down.t8 | 0.198 | 0.110 |
| llama8b.mlp_down.t512 | 1.185 | 0.220 |
| llama8b.lm_head.t1 | 0.702 | 0.700 |
| llama8b.lm_head.t8 | 1.154 | 0.793 |
| llama8b.lm_head.t512 | 10.808 | 1.368 |

That is every cuBLAS row the Aug 25 legs measured (20 shapes). cuBLAS
does not see our plans, so these rows serve any round of ours on an H100
of this driver and torch pin; the OUR side of the H100 table is in
`docs/lanes/HANDOFF_gemm_splitk.md`.

Sequence models, torch 2.4.1+cu124, `e1g/2026-08-25_160520-nvidia-speed-gemmseq`
(5 rounds), Llama-8B shapes at 512 tokens:

| op | torch fp32 | torch tf32 | sdpa math fp32 | sdpa efficient fp32 |
|---|---|---|---|---|
| attention t512 | 1.418 | 0.684 | 1.275 | 1.198 |
| mlp t512 | 4.068 | 0.726 | | |
| rmsnorm t512 | 0.100 | 0.090 | | |
| mamba130m prefill t512 (torch reference scan) | 29.556 | 29.957 | | |
| selective_scan t512 | 30.064 | 31.151 | | |

### Classical KDE and SVC on Istella-S against cuML (September 11, pod 44j9e1zik7r8qr)

Pod `44j9e1zik7r8qr` (`mojolearn-ctdk-2026-09-11_180850`) on RunPod machine
`l4vyjngwksom`, $3.49 per hour, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 26 vCPUs and 251
GB, kernel 6.8.0-90. Ours is IDENTICAL at commit 94c82db8, which carries the
NVIDIA GEMM ksplit default (DEVIATION 2595), timed from a host float32 array
to host results. cuML 26.08.00 (cuml-cu12 26.8.0, cupy 14.2.0) and NumPy
2.4.6 in the same Python. 1 warm-up plus 3 rounds, arms interleaved, ms
median (min..max), quality computed by `tools/classical_two_datasets.py`.
Shapes match the MI300X classical section below, kde 100,000 standardized
Istella-S fit rows x 2,000 queries with Scott bandwidth and svc 10,000
standardized fit rows with RBF, C 1 and gamma 1/d. GPU path only, so no
scikit-learn arm runs here. Evidence
`bench/results/classical_h100_2026-09-11_istella/`.

| lane | dataset | opponent | device | opponent ms | opponent quality | ours IDENTICAL ms | ours quality | ours / opponent |
|---|---|---|---|---|---|---|---|---|
| kde | Istella-S | cuML KernelDensity | GPU, H100 | 6.79 (6.72..7.30) | mean log-lik -212.117 | 219 (194..227) | mean log-lik -212.117 | 32.25x |
| svc | Istella-S | cuML SVC | GPU, H100 | 20.64 (19.91..24.56) | accuracy 0.9222, 2401 SV | 70.7 (65.7..73.1), with DEVIATION 2623 | accuracy 0.9222, 2400 SV | 3.43x |
| svc | taxi | cuML SVC | GPU, H100 | 415 (414..415) | accuracy 0.7675, 5586 SV | 857 (857..860), with DEVIATION 2623 | accuracy 0.7675, 5527 SV | 2.06x |

Every arm held one digest across its rounds.

At 94c82db8 our SVC failed on this GPU for every fit with more than 512
training rows. `n_ws = min(1024, n_train)` selects
`smo_block_solve_kernel[1024]`, and CUDA refused DEVIATION 2491's warp-fold
kernel at that width with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, while the
MI300X and the Apple M4 launch it. The first Istella-S race recorded that
failure (cuML 20.96 ms in it). DEVIATION 2623 adds a kernel-matrix row that
sends NVIDIA above width 512 back to the pre-2491 halving trees. The SVC rows
above are a second race, 19:10Z on the same pod, with the SVM binding rebuilt
from 94c82db8 plus that change (branch `lane/svc-cuda-1024`, 11928fec).
Taxi was prepped for this race only. The scikit-learn KDE row on Istella-S
in the MI300X section is still owed.

### Classical KDE on NYC taxi against cuML (September 11, pod dn8er13wjuxtax)

Pod `dn8er13wjuxtax` (`kde-speed-2026-09-11_203033`), NVIDIA H100 80GB
HBM3, driver 580.126.09, the image above, cuML 26.08.00. Ours is IDENTICAL
at commit 36ca51fd (staged path), the kde lane's shape and harness as in
the Istella-S rows above (100,000 standardized taxi fit rows x 11 numeric
columns, 2,000 queries, Scott bandwidth). 1 warm-up plus 5 interleaved
rounds, ms median (min..max). Evidence
`bench/results/kde_speed_h100_2026-09-11/`. Both arms held one digest.

| lane | dataset | opponent | device | opponent ms | opponent quality | ours IDENTICAL ms | ours quality | ours / opponent |
|---|---|---|---|---|---|---|---|---|
| kde | taxi | cuML KernelDensity | GPU, H100 | 2.37 (2.19..9.20) | mean log-lik -9.58205 | 39.2 (35.8..40.4) | mean log-lik -9.58207 | 16.55x |

### Classical KDE on both datasets, before and after DEVIATIONS 2625, 2626 and 2660 (September 11, pod ur95zh3h9qbx2p)

Pod `ur95zh3h9qbx2p` (`kde-finish-2026-09-11_213144`), NVIDIA H100 80GB HBM3,
driver 570.124.06, the image above, cuML 26.08.00 (cuml-cu12 26.8.0, cupy
14.2.0), NumPy 2.4.6, scikit-learn 1.9.1 in the image's Python 3.11. Both of
our arms and every gate ran with `MODULAR_NVPTX_COMPILER_PATH` at the CUDA
12.9.86 wheel's `ptxas`, because MAX asks for driver 580 otherwise and this
box has 570; the image's 12.4 `ptxas` refuses PTX `.version` 8.5. The kde
lane's shape and harness as in the rows above, 100,000 standardized fit rows
x 2,000 queries, gaussian kernel, euclidean metric, Scott bandwidth, 1
warm-up plus 5 interleaved rounds, ms median (min..max). `ours-base` is
`origin/main` 2c64a778 (the staged path) built on this same pod and raced
round by round beside `ours` (lane `lane/kde-finish`), so it is OURS, never
an opponent row. Evidence `bench/results/kde_finish_2026-09-11/`.

| lane | dataset | opponent | device | opponent ms | opponent quality | ours IDENTICAL ms | ours quality | ours / opponent |
|---|---|---|---|---|---|---|---|---|
| kde | taxi | cuML KernelDensity | GPU, H100 | 2.02 (1.98..2.06) | mean log-lik -9.58205 | 28.94 (28.92..28.97) | mean log-lik -9.58207 | 14.30x |
| kde | Istella-S | cuML KernelDensity | GPU, H100 | 6.95 (6.92..7.14) | mean log-lik -212.117 | 67.94 (67.46..68.79) | mean log-lik -212.117 | 9.78x |

Ours before this lane, same pod, same window, same harness: taxi 35.69
(35.64..35.99) ms, 17.64x; Istella-S 218.73 (217.91..221.95) ms, 31.49x. The
after/before ratios are 0.811 and 0.311, geometric mean 0.502, and the score
digests are equal between the two arms on both datasets (taxi
aa8ac4159ad2cbfa, Istella-S 81d11ed7fcd9eb38), so the three deviations stay
ON by default under ENGINEERING_RULES.md section 9. cuML held one digest per
dataset as well (cdce01475f9e6977, 3f4541591da35fc6). The cuML taxi row is
the first of the two races on this pod; in the second, cuML's taxi arm took
a 22.1 ms round and read 3.56 ms median, against which ours reads 8.14x.
## NVIDIA H200, driver 570.211.01, CUDA 12.8 (ptxas 12.9.86 override), cuml-cu12 26.8.0

RunPod pod `4oih8bhjepzlmm` (`svm-finish-2026-09-11_213129`), image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 96 vCPUs. THE HOST
DRIVER IS BELOW MOJO'S CUDA FLOOR (it asks for 580 or newer, CUDA 13), so
every run in this section exports `MODULAR_NVPTX_COMPILER_PATH` pointing at the
pod's ptxas 12.9.86, which is Modular's documented older-driver path. The fits
it produces hash equal to the H100 and MI300X rows (n=400/600/2000
457e29b82bca9df9, 733a383c5699f427, 2b66bc991a9c9ed0), so the assembler moves
no bits. H200 is its own tuple: these rows are never mixed with the H100 rows
above.

### SVC on taxi and Istella-S (September 11, svm-finish lane)

1 warm-up plus 5 interleaved rounds through
`tools/classical_two_datasets_leg.sh` with `MOJOLEARN_CTD_LANES=svc`, blocks of
10,000 standardized fit rows and 10,000 eval rows, RBF, C 1, gamma 1/d, tol
1e-3, quality computed by `tools/classical_two_datasets.py`. Ours is IDENTICAL
at `lane/svm-finish` (DEVIATIONS 2665 and 2666). Evidence
`bench/results/svm_finish_2026-09-11/`.

| lane | dataset | opponent | device | opponent ms | opponent quality | ours IDENTICAL ms | ours quality | ours / opponent |
|---|---|---|---|---|---|---|---|---|
| svc | taxi | cuML SVC | GPU, H200 | 418.4 (417.9..418.9) | accuracy 0.7675, 5586 SV | 768.7 (767.9..769.5) | accuracy 0.7675, 5527 SV | 1.84x |
| svc | Istella-S | cuML SVC | GPU, H200 | 20.29 (19.95..20.35) | accuracy 0.9222, 2401 SV | 61.3 (60.4..66.6) | accuracy 0.9222, 2400 SV | 3.02x |

The same race carried a before arm, origin/main 2c64a778 built on this pod and
raced in the same conductor: taxi 865.0 ms (2.07x) and Istella-S 72.2 ms
(3.56x). Every arm held one digest across its five rounds.

## NVIDIA L40S, driver 580.126.09, CUDA 12.4, torch 2.4.1+cu124, cuBLAS 120402, cupy 14.2.0

Trees, HIGGS 1M, 2026-09-09 trees lane, CatBoost GPU symmetric 1.2.10, same
config as the H100 trees rows above, console capture
`bench/results/trees_identical/l40s_2026-09-09/speed/profile_cells_console.txt`
(the pod's log files expired before they were fetched).

| opponent | lane | 1M |
|---|---|---|
| CatBoost GPU symmetric | Logloss | 781 (771..788) |
| CatBoost GPU symmetric | RMSE on the 0/1 label | 915 (849..941) |

Our identical arm in the same process on that L40S: 426 ms Logloss, 318 ms RMSE.

GEMM, 2026-09-09, worktree branch lane/gemm-identical,
`bench/results/e1g/2026-09-09_123601-nvidia-l40s-identical-gemm-merged/opponents_l40s2.log`
(5 rounds plus warm-up). The earlier `2026-09-09_092558` leg agrees within
noise; quote the 123601 leg.

| shape | cublas-fp32 (torch) | cublas-tf32 (torch) | cublasSgemm fp32 (cupy) | torch-fp32 |
|---|---|---|---|---|
| llama8b.qkv.t512 | 0.536 | 0.153 | 0.507 | 0.518 |
| llama8b.mlp_up.t512 | 1.828 | 0.725 | 1.835 | 1.840 |
| llama8b.mlp_down.t512 | 1.686 | 0.571 | 1.681 | 1.683 |
| llama8b.lm_head.t512 | 16.89 | 5.251 | 17.37 | 17.31 |
| pca.transform.wide.8192x64x128 | 0.019 | 0.017 | 0.017 | 0.019 |
| kmeans.dist.4096x64x64 | 0.015 | 0.015 | 0.015 | 0.015 |
| gram.128sq.x100003 | 0.123 | 0.083 | 0.248 | 0.133 |
| gram.32x32x1M | 0.378 | 0.363 | 0.374 | 0.377 |
| ols.step1.16x16x64K | 0.024 | 0.023 | 0.023 | 0.024 |

Note the 2x disagreement on gram.128sq between the torch route and the
direct cublasSgemm route; both are cuBLAS. Quote the torch route (the one
users hit) and say so.

Attention, 2026-09-09, worktree branch lane/samba-attention,
`bench/results/attnlane_2026-09-09/window_timing.log` (3 rounds after 1
warm-up). d_model 1024, 16 heads, 4 kv heads, head_dim 64, window 2048,
sequence 4096, batch 4, TF32 off, explicit sliding-window mask.

| arm | forward | forward+backward |
|---|---|---|
| torch eager fp32 SDPA | 33.6 | 106.9 |
| torch.compile | 34.7 | 93.6 |

UMAP at 1M rows, 2026-09-09, worktree branch lane/umap-optimizer, pod
ao1rwg13e6uph4 (NVIDIA L40S 46 GB, driver 580.126.09), cuML 26.08.00, cupy
14.2.0, CUDA runtime 12.9, `bench/results/umap/2026-09-09-l40s-device-optimizer/cuml-1m.log`
and `cuml-umap-1m.json` (5 rounds after one warm-up, `tools/umap_cuml_reference.py`,
same dyadic-v1 fixture, n_neighbors 15, 2 components, 200 epochs, spectral
init, cuML's default approximate build). This is an L40S row; the 20k and
100k cuML rows above are H100 rows and are NOT its siblings.

| UMAP (32 features, 15 neighbors, 2 components, 200 epochs) | cuML UMAP ms (L40S) |
|---|---:|
| 1,000,000 rows | 7991.1 (samples 7946.4, 7979.7, 7991.1, 8008.7, 8031.6) |

The same block at sequence 16384 (everything else as above), worktree
branch lane/fused-attention, pod mmzcdqdbg27c6w, driver 580.126.09,
`bench/results/attnlane_fused_2026-09-09/timing_16384_torch.log` (3 rounds
after 1 warm-up). Ours is REFUSED at this length by name (the 8192
absolute-position ceiling, DEVIATION 812), so no row of ours exists to
quote against it; the torch.compile arm ran out of memory (64 GiB asked).

| arm | forward | forward+backward |
|---|---|---|
| torch eager fp32 SDPA, seq 16384 | 291.6 | 931.6 |
| torch.compile, seq 16384 | OOM | OOM |

The same block at sequence 1024 (everything else as above), worktree
branch lane/fused-attention, pod n5h6et424b5oj6, driver 580.159.03,
`bench/results/attnlane_fused_2026-09-09/round2_8b996d6d/timing_1024_torch.log`
(3 rounds after 1 warm-up).

| arm | forward | forward+backward |
|---|---|---|
| torch eager fp32 SDPA, seq 1024 | 5.7 | 16.8 |
| torch.compile, seq 1024 | 5.3 | 13.8 |

## NVIDIA RTX 4090, driver 580.126.20, torch 2.4.1+cu124

Public-API host-array comparison (transfers included), 2026-09-05,
`bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/final-supplement/torch-knn-*/results.json`
and `remote/campaign/public-*/results.json` (7 rotating rounds). The kNN
opponent here is torch `cdist` + `topk`, NOT cuML.

| lane | shape | opponent | ms |
|---|---|---|---|
| knn q32 | 100k x 32, k 10 | torch cdist+topk fp32 | 1.108 |
| knn q128 | same | same | 1.386 |
| knn q1000 | same | same | 5.639 |
| gemv | 2048 x 2048 | torch | 1.067 |
| nt | 16384 x 64 x 64 | torch | 0.884 |
| gram | 65536 x 32 | torch | 1.097 |
| umap | 15 neighbors, 2 components, 50 epochs | cuML UMAP | REFUSED (no cupy in the venv) |

## AMD Instinct MI300X (Hot Aisle), ROCm 6.4.1 image on a ROCm 7.2.4 host, torch 2.6.0+rocm6.4.1

### torch byte-LM training step, PROVISIONAL (shared 2x VM)

Source: `bench/results/e1g/2026-09-11_165905-amd-mi300x-2gpu-vm-hotaisle-torch-lm-step/remote/torch-lm-step/`,
Hot Aisle VM enc1-gpuvm005 (deployment 210d0e97), commit 593e580d, one GPU of
a 2x MI300X VM whose other GPU ran our attention leg at the same time on the
shared 26 cores (label `mi300x-2gpu-vm`). torch 2.6.0+rocm6.4.1.git1ded221d,
HIP 6.4.43483, Python 3.12.11 in a uv venv inside
rocm/dev-ubuntu-22.04:6.4.1-complete; device "AMD Instinct MI300X VF". Same
harness, shape, init, AdamW, clock and corpora as the H100 table. TF32 does
not exist on ROCm (not_applicable, exit 4).

| column | enwik8 s | Pile GitHub s |
|---|---|---|
| compile_bf16 | 0.03228 | 0.02054 |
| eager_bf16 | 0.03375 | 0.03920 |
| compile_fp32 | 0.04510 | 0.04527 |
| eager_fp32 | 0.05005 | 0.04999 |

NOT a row to quote: the two compile_bf16 corpora disagree by 1.6x, which fits
host contention from the concurrent body. No same-box ratio to our IDENTICAL
step is stated here; the clean 1x MI300X row is owed (item 5 below), with our
step on the same VM.

## Rows that do not exist yet (owed, in priority order)

1. LightGBM CUDA extra_trees, valid build, HIGGS 1M/2M/5M (the Sep 9 trees
   lane built USE_CUDA=ON but its ET rungs never ran; its rf-boosting row
   at 470 s per fit is not an ET row).
1b. Trees at 2M and 5M for the Sep 9 same-process protocol (only 1M exists).
2. (closed 2026-09-09: the 20 H100 cuBLAS rows above were transcribed
   from the Aug 25 logs by the GEMM split-K lane; no re-run.)
3. (closed 2026-09-11 by lane linear-cluster-istella on H100 pod
   1yxsotvvcbxtuu: cuML PCA at 4,000,000 x 11 (19.54 ms) and 2,043,304 x 220
   (81.91 ms), and cuML DBSCAN at 1,000,000 rows on both datasets (taxi
   13631.0 ms at eps 0.177, Istella-S 55240 ms at eps 4.17), plus cuML
   HDBSCAN at 100,000 rows on both. The small fixtures above are superseded
   for these two.)
4. cuML UMAP on the H100 at 1M rows, and on the L40S at 20k and 100k rows
   (the H100 has 20k and 100k, the L40S has 1M; ours IDENTICAL was measured
   on the L40S at all three on 2026-09-09, so no same-GPU ratio exists yet
   below 1M).
5. (closed 2026-09-11 on the H100: the "torch byte-LM training step" table
   in the H100 section, eager fp32, eager TF32 and compile fp32 on enwik8 and
   Pile GitHub. Still owed: the same row on AMD with ROCm torch (the leg now
   targets the Hot Aisle MI300X, a new tuple), and the harness's
   `compile_tf32`, `eager_bf16` and `compile_bf16` columns on the H100 and on
   the MI300X.)
6. cuML brute-force kNN on the taxi and Istella-S prefixes (DEVIATION 2524
   moved the kNN tooling off HIGGS, which is retired; the "kNN second kind
   (HIGGS rows)" table above is history), H100, k 10 and 15, measured once
   by `tools/knn_selection_gate.sh` under `MOJOLEARN_KNN_SELECTION_OPPONENT=1`.

## Rows never to quote

- Anything under `bench/results/fast_speed/mac-*`: Apple M4 CPU and MPS
  numbers, some with NVIDIA-sounding arm names (`torch-gpu-fp32` on MPS).
- Aug 26 and Aug 27 H100 forest legs: 3 rounds, and three different driver
  builds across what reads as one campaign (580.159.04, 580.126.09,
  580.126.20). The Aug 28 030908 leg is the row.
- Our own FAST or DETERMINISTIC arms, on any vendor.

## How to add a row

Run the opponent through the existing arm in `bench/speed/` or
`tools/nvidia_public_compare.py --external-mode fast`, on a rented box, 5
rounds minimum, with the versions printed in the log. Commit the log under
`bench/results/e1g/<stamp>/` (top-level files only) and add the row here
with the stamp, GPU, driver and versions. Then delete the opponent from the
round's command line; the round measures us alone.

### L40S attention, driver 580.159.03 (2026-09-09)

A new driver tuple, measured once on the same pod as the register-tiled
IDENTICAL candidate. Torch 2.4.1+cu124, eager FP32 SDPA, TF32 off, explicit
sliding-window mask; B=4, L=4096, d_model=1024, heads=16, kv_heads=4,
head_dim=64, window=2048, intermediate=4096. One warmup, median of three
rounds: **35.9 ms forward; 114.3 ms forward+backward**.

Source: `bench/results/attnlane_regblock_2026-09-09/jobs/torch_reference.log`;
GPU/version records and the full harness are in the same directory.


## H100 Mamba-3 reference scan, driver 580.126.09 (September 7 archive)

Transcribed 2026-09-10 without rerunning the opponent. Torch 2.4.1+cu124,
CUDA 12.4, eager FP32, TF32 off, deterministic algorithms off. The shipped
seed-7 public fixture uses state size 128, head dimension 64, expansion 2,
and chunk size 64. Five untimed warmups plus one logged warmup, then
five timed rounds; median milliseconds.
Timing scope is the existing torch reference arm in
the September 7 `tools/speed_torch_seq.py` snapshot; compare only with the corresponding
public `bench/speed/seq_py_speed_arm.py` shape from the sibling
`mojolearn-grid` checkout. Torch weights/input are already on device and
the timed call returns device y; our public call includes host transfers,
per-call weight checks and four public reports. These scopes remain
explicit when quoting the ratio. This is the FP32 reference scan.

| shape | torch FP32 ms | numerical admission | IDENTICAL public ms Sep10 | ratio |
|---|---:|---|---:|---:|
| lane.b2_l4_d32 | 1.477005 | PASS | 1.219134 | 0.83x |
| narrow.b8_l4096_d512 | 16.396241 | PASS | 55.910172 | 3.41x |
| wide.b8_l1024_d2048 | 19.383267 | PASS | 101.634823 | 5.24x |

Latest Sep10 public medians use the guarded NVIDIA increment and yintra
tiles, plus the existing fresh-prefill/caller-copy path. Final yintra source
`28a32835`, five timed rounds after one logged warmup. The final reverse-order
same-pod controls are 59.503218 -> 55.910172 ms narrow and
105.693202 -> 101.634823 ms wide. The initial order also improves:
60.212171 -> 56.820129 ms and 106.248042 -> 102.802685 ms.
All three complete output hashes are unchanged; native and public exact gates
pass. Apple and tiny calls retain the previous yintra default.

This is a NEW physical H100 80GB HBM3, driver 580.126.09, GPU UUID
GPU-27c60f95-99ff-5839-38e2-a46406c86af6. Ratios above reuse the archived
numerically admitted Torch reference, with its original resident-device scope;
they are cached-reference ratios, not a newly paired opponent run.
Do not attribute changes from the preceding pod's absolute timings entirely
to this optimization. Prior increment-only medians were 65.868374/120.395977;
the current optimization's before/after denominators are the same-pod controls
above. Source/runtime/CPU metadata and every sample are in
`bench/results/identity_continuation_2026-09-10/final-performance/`;
first-order evidence is in `bench/results/mamba3/2026-09-10-yintra-tile/`.

The earlier Sep10 measurements remain archived in
`bench/results/mamba3/2026-09-10-fresh/`; do not use their different physical
pod as this optimization's before/after control.
The small lane row measures overhead; do not extrapolate it to large shapes.

Source directory:
`bench/results/mamba3/2026-09-09-statepass/opponent-admission/`.
`seq.mamba3.torch.log` contains every sample and original output gates
(rtol 0.0005, atol 0.00001); `september7-gpu.txt` identifies the GPU/driver;
`september7-source_sha256.txt` pins source provenance. The original run
omitted `--mojo-log`; `input-admission.json` closes that missing witness
check by read-only replay of all 30 input tensors. It does not retime torch.

## L40S transformer end-to-end, driver 580.159.03 (September 9 archive)

Transcribed 2026-09-10 without rerunning. Torch 2.4.1+cu124, CUDA 12.4,
eager FP32, TF32 off, shipped seed-7 public shapes, head dimension 128.
Five untimed warmups plus one logged warmup, then five rounds. These
measurements are retained to avoid
repeating a known failed comparison; **neither row qualifies a speed ratio**.

| shape | torch FP32 ms | numerical admission |
|---|---:|---|
| narrow.b8_l4096_d512 | 63.849 | FAIL, max absolute difference 0.0138197 |
| wide.b8_l1024_d2048 | 43.787 | FAIL, max absolute difference 0.0278463 |

Source: `bench/results/transformer_e2e_2026-09-09/torch_fp32.log`, harness
`grid_torch.py` and `attn_grid_torch.sh` in the same directory. All 20 input
witnesses passed. The original tolerance was rtol 0.0005, atol 0.00001.
A numerical-only investigation can reuse these logs; a changed comparator
needs its own provenance and admission, and must not inherit these timings.

### Reuse and admission details for future rows

Before timing an opponent, search this table and its referenced raw logs.
Backfill an existing missing row before considering a new run. Match GPU,
driver, library/CUDA pins, fixture or input witness, shape, parameters,
precision switches, and timing scope (including transfer policy). A
shape-only match is a timing context, not an admitted identical-input ratio.
Record the command/harness revision, warmup count, every timed sample,
median, input/output admission result and source log with every new row.
Keep failed admission and failed runs visible and explicitly unqualified.
Correctness-only opponent runs need no fresh timing; if the comparator or
fixture changes, create a new row rather than overwriting its history.

### H100 transformer admission diagnosis (2026-09-10, no new timing)

`bench/results/transformer_admission_2026-09-10/` reproduces the original
H100 input/output witnesses and failed admission. Different pinned RoPE
inverse constants explain most of the original discrepancy. Supplying
exact complete production RoPE tables still leaves both shapes outside
the original tolerance; both FP32 implementations also fail that tolerance
against a matched FP64 diagnostic with comparable error magnitudes. This
is numerical evidence, not a new timing row or a qualified speed ratio.
The comparator defaults and tolerance remain unchanged. Reuse this
investigation before spending another opponent run on the same mismatch.

### H100 transformer caller-transfer measurements (2026-09-10)

These are own public IDENTICAL timings, not admitted opponent ratios. The
original Torch admission failure above remains unchanged; no opponent was
retimed and no tolerance or comparator default changed. H100 80GB HBM3,
driver 580.126.09, Mojo 1.0.0 (ed45d567), Python 3.11.10/NumPy 1.26.3.
Original seed-7 large fixtures, one logged warmup and seven timed rounds per
build/order. Each call includes host inputs/weights, refusal checks, device
work, output and cache transfers. Final source `f31508bb`.

| shape | legacy first ms | candidate first ms | legacy reverse ms | final default reverse ms |
|---|---:|---:|---:|---:|
| narrow.b8_l4096_d512 | 265.319614 | 245.827597 | 255.330876 | 245.223133 |
| wide.b8_l1024_d2048 | 261.891093 | 246.565180 | 260.435883 | 245.318148 |

All82 small full-array records (including backward gradients and carried/ring
caches) match across builds, as do both full16,777,216-cell large output SHA256
values. Final flag-absent defaults pass on Apple and NVIDIA. The reversed
order confirms approximately4%/6% less request time; the initial narrow
baseline was noisier. Apple paired samples show a consistent small-HD64 win
and variable HD128 timing. Evidence, raw samples and source hashes:
`bench/results/transformer_transfer_2026-09-10/`. No qualified Torch ratio
may be inferred from these own-versus-own measurements.

### H100 stateless transformer and identity continuation (September 10)

Own public IDENTICAL medians below use the original seed-7 large fixtures on
H100 80GB HBM3, driver 580.126.09, GPU UUID
GPU-27c60f95-99ff-5839-38e2-a46406c86af6. Mojo 1.0.0 (ed45d567),
Python 3.11.10/NumPy 1.26.3; one logged warmup and seven timed rounds.
The final flag-absent NVIDIA default omits only the cache that a stateless
caller discards. Host input/weight transfers, refusal checks, device work and
host output remain inside the call. Explicit-state/backward behavior is unchanged.

| shape | baseline first ms | forced fresh first ms | baseline reverse ms | final default reverse ms |
|---|---:|---:|---:|---:|
| narrow.b8_l4096_d512 |211.354129|204.134570|210.893420|204.793565|
| wide.b8_l1024_d2048 |212.392121|187.488862|214.515634|188.905248|

The reverse order confirms 2.9%/11.9% less request time. Source `f02511bd`;
Apple retains its previous default because local timing evidence was mixed.
All 82 array hashes and both complete large output SHA256s are unchanged,
including comparison with the earlier caller-transfer artifact. Torch's
original numerical admission still fails: **no qualified opponent ratio**.
No opponent was retimed in this pass. Initial and final evidence:
`bench/results/transformer_fresh_2026-09-10/` and
`bench/results/identity_continuation_2026-09-10/final-performance/`.

IVF k/probe extensions, fused CDNA logical groups, wide full PCA, and UMAP
portable host math were correctness/capability work, not new opponent timing
rows. Their executable gates and limitations are recorded in
`docs/lanes/HANDOFF_identity_continuation_2026-09-10.md`. Existing cuML/cuBLAS
prices and comparison qualifications remain unchanged.

### Sep10 GEMM operand staging (IDENTICAL, cached opponents)

Source `c89a73d8` moves only input FTZ into the shared-operand load stage;
per-step rounded FMA/FTZ and the reduction tree are unchanged. NVIDIA H100
80GB HBM3, driver580.126.09, Mojo1.0.0(ed45d567), one warmup and seven rounds.
The same-pod control and candidate run in both orders; the reverse-order
medians below are representative of both. Complete output fingerprints match,
and the seven existing device gates plus798 adversarial plan cases pass on
Apple/H100. NVIDIA adopts it; Apple retains the previous default.

| Shape | Same-pod baseline ms | Staged ms | Cached cuBLAS FP32 ms | Staged / cached |
|---|---:|---:|---:|---:|
| llama8b.qkv.t512 |1.940172|1.523785|0.374|4.07x|
| llama8b.mlp_up.t512 |7.581844|6.177937|1.379|4.48x|
| llama8b.mlp_down.t512 |6.791881|5.356788|1.185|4.52x|
| pca.transform.wide.8192x64x128 |0.032999|0.025159|0.026|0.97x|
| kmeans.dist.4096x64x64 |0.022839|0.018727|0.023|0.81x|

Opponent prices above are the existing Aug25 H100 rows, unchanged. These are
cached-reference ratios, not fresh paired opponent trials; physical GPU/host
variation must not be counted as a source speedup. The paired dense savings
are18.5–21.6%. All raw samples, hardware UUID and scripts:
`bench/results/staging_performance_2026-09-10/gemm-stage/`.

The rejected kNN coalesced-column experiment also measured400k/4000/d32/k10,
k15;400k/1000/d8/k10;10k/32/d32/k15;65537/129/d17/k10. Full outputs match,
but the target workloads do not improve. The three additional control shapes
have no opponent price assigned; no opponent was run. The raw measurements
and complete-output hashes are retained in the same evidence directory under
`knn-stage/`, so they need not be repeated merely to recover their values.

### Sep10 bounded kNN query batches (IDENTICAL, cached cuML)

Main `de042700` uses512-query batches on NVIDIA through400k index rows;
`78248bb7` wires Python's automatic request to that planner. Larger indices
and explicit starting tiles above512 keep historical budgeting. Apple keeps
256. Known batch-dependent scratch in the admitted scope is <=522.6MiB;
this is not a total-memory guarantee. Source changes do not alter row
arithmetic, tie ordering, distance layout or selection.

H100 GPU-504d7226-23e4-42fe-6ed9-64586e4da2e2, driver580.126.09,
Mojo1.0.0(ed45d567), dyadic-v1, two warmups. Five rounds per arm in each
order; a final default build then takes seven rounds. All five experiment
shapes' complete outputs match across arms/orders; Apple/H100 native and
rebuilt public Python gates pass.

| index / queries / features / k | Paired baseline request ms | Paired512 request ms | Final default request ms | Cached cuML request ms | Final / cached |
|---|---:|---:|---:|---:|---:|
|400000 /4000 /32 /10|29.194590|27.499942|27.525704|10.225|2.69x|
|400000 /4000 /32 /15|33.625961|31.865536|31.860726|10.817|2.95x|
|400000 /1000 /8 /10|5.950702|5.595354|—|—|—|
|10000 /32 /32 /15|0.149774|0.148186|—|—|—|
|65537 /129 /17 /10|0.366868|0.370769|—|—|—|

Final k10/k15 device medians are26.369018/30.658421ms. The target request
savings are5.8%/5.2% against same-pod baselines. Old cuML rows retain their
original provenance and are not newly paired measurements. The last two
controls clamp to identical batches, so their small timing changes are not
attributed to the new policy. Evidence: `knn-batch/` and `knn-final/` under
`bench/results/staging_performance_2026-09-10/`. Earlier16-row Sep10 prices
above are retained as historical captures from their recorded physical GPU.

### Sep10 end-to-end effect of GEMM staging

Same H100 as the staging rows above, IDENTICAL, original seed7 fixtures,
complete output SHA unchanged. Both run orders use the same physical device.
Mamba has one warmup plus five rounds; transformer one plus seven. This
changes only shared GEMM scheduling; Mamba scratch zero-initialization stays.

| Workload | Baseline ms (order0) | Default ms (order0) | Baseline ms (order1) | Default ms (order1) |
|---|---:|---:|---:|---:|
|Mamba3 B8/L1024/D2048|102.724629|95.521010|103.468360|93.319228|
|Transformer B8/L4096/D512|206.074415|202.123621|206.111947|200.625729|
|Transformer B8/L1024/D2048|191.636596|172.340987|189.356467|168.650821|

The first paired wide-Mamba trials show7.0–9.8% less time, but a later
same-binary repeat also exposes an unstable wide baseline. No stable new
Mamba price or opponent ratio is qualified. The19.383267ms Torch reference
is retained unchanged.
Transformer saves1.9–2.7% narrow and10.1–10.9% wide; all82 full-array checks
and both large full-output hashes match. Its original Torch numerical gate
still fails, so no qualified Torch ratio is supplied and Torch is not retimed.

Narrow Mamba has no qualified new performance claim: the *same* original
baseline library measured56.767391ms in the first run but229.788203 and
225.317191ms later. New default was222.376963/222.875569ms in those later
runs. Tiny Mamba measured1.220500/1.231913ms baseline versus1.243062/1.260955ms
default; no tiny improvement is claimed. The original scratch experiment's
236.131446ms narrow result cannot establish source-caused regression when
that baseline also enters the slow regime. Scratch is archived for lack of
stable positive evidence. Raw times and every full-output SHA are retained;
no faster regime is selected to manufacture a current ratio.

Evidence: `final-stage/`, `mamba-stage/` and `mamba-repeat/` under
`bench/results/staging_performance_2026-09-10/`.

The final same-binary repeat (baseline/default/baseline, all full hashes
unchanged) measured narrow248.337356/251.352344/250.333956ms and
wide102.489213/97.158484/283.757282ms. This invalidates a stable new
Mamba performance claim for either large shape; it is not evidence of a
source-caused4x regression. Baseline library SHA256 equality and GPU
before/after/process snapshots accompany `mamba-repeat/`. No cause is
established, and no additional opponent run would resolve this own-side issue.


### 2026-09-10 GEMM throughput interpretation and probe-label correction

No new opponent or device measurements in the runtime byte-LM pass. Reusing
`staging_performance_2026-09-10/gemm-stage/summary.json`, useful throughput
`2*m*n*k/(ms*1e9)` for the two staged run medians is QKV 11.274–11.280,
MLP-up 9.733–9.743, MLP-down 11.225–11.227 TFLOP/s. Source, raw timings and
H100 80GB HBM3 provenance are the existing staging rows above; cached cuBLAS
prices are unchanged. These are IDENTICAL-arm throughput values, not ratios
against our FAST arm. Against the published H100 SXM FP32 non-tensor reference
of 67 TFLOP/s, they are 14.5–16.8% (nominal reference, not measured sustained peak).
[NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).

The older `performance_residual_2026-09-10/gemm-residual/64tile-price.log`
used generic `untuned`/`dispatch` labels while actually comparing current 128×128
against a forced 64×64 candidate. Its 0.747–0.841 ratio documents the rejected
candidate, not a current dispatch regression. Original logs remain immutable;
new probe output names baseline/candidate and their selectors explicitly.
See `docs/lanes/HANDOFF_speed_gemm_2026-09-10.md` for the corrected next steps.


## 2026-09-10 continuation: own-arm M4 measurement, no new opponent price

`bench/results/gemm_swizzle_2026-09-10/` records Apple M4, IDENTICAL FP32,
actual shapes below, four host-synchronized samples per arm with alternating
order in a 9.49-second window. Values are medians; raw arithmetic means and
first/last drift are retained separately. Baseline is current plan 10;
candidate is forced transpose-swizzle plan 19. All output digests agree and
all seven stronger device gates pass. No dispatch change or peak utilization
claim follows from these samples. No external opponent was rerun.

| Actual m,n,k | Baseline ms | Candidate ms | Baseline TFLOP/s | Candidate TFLOP/s |
|---|---:|---:|---:|---:|
|512,4096,4096|110.375|108.139|0.15565|0.15887|
|512,14336,4096|370.7255|369.4800|0.16219|0.16274|
|512,4096,14336|380.8295|370.1255|0.15789|0.16246|

The small kNN phase smoke (`knn_phase_2026-09-10`) and Mamba tiny smoke
(`mamba_regime_2026-09-10`) are diagnostics, not replacement target-shape
prices. Transformer remains numerically unadmitted. Reuse existing qualified
opponent rows only within their recorded shape, arithmetic and hardware scope.

### Sep10 Apple large-target preflight audit (own arms only)

No opponent was rerun. Current Apple IDENTICAL preflight versus safe
NO_PREFLIGHT control, 400k rows / 4000 queries / d32, five rounds after two
warmups in both orders, phase instrumentation disabled:

| k | Default request ms, orders 0/1 | Control request ms, orders 0/1 | Request reduction |
|---|---:|---:|---:|
|10|1380.890 / 1305.022|1432.033 / 1409.590|3.6–7.4%|
|15|1332.921 / 1351.913|1389.374 / 1442.922|4.1–6.3%|

Complete outputs match both arms and both orders, including low-feature and
ragged controls. Keep the existing preflight; no new default changed. Normal
desktop activity and small-control request noise limit generalization. These
are own-arm comparisons, not Apple/cuML opponent ratios. Raw samples, phase
diagnostics, binary hashes and scope: `bench/results/knn_large_gate_audit_2026-09-10/`.
Historical Apple admission used 400k rows with 1000 queries; these new runs
cover the full 4000-query target. Decision history: `docs/lanes/PERFORMANCE_GATE_AUDIT_2026-09-10.md`.

### Sep10 Apple kNN request-local metadata: own-arm update, no new opponent

Apple M4 IDENTICAL, dyadic-v1, 400000 index rows / 4000 queries / 32 features,
Euclidean return_sqrt. This is a paired implementation comparison; cuML was not
rerun and these Apple timings do not qualify a cuML ratio. Existing matched
NVIDIA opponent prices remain reusable and unchanged.

| k | Comparison / order | Safe existing preflight request ms | Metadata request ms | Request saving |
|---|---|---:|---:|---:|
| 10 | Forced candidate, existing first | 849.940 | 654.728 | 23.0% |
| 10 | Forced candidate, metadata first | 880.852 | 654.823 | 25.7% |
| 15 | Forced candidate, existing first | 871.992 | 670.155 | 23.1% |
| 15 | Forced candidate, metadata first | 914.559 | 695.273 | 24.0% |
| 10 | Actual scoped default first | 856.831 | 642.981 | 25.0% |
| 10 | Disabled arm first | 890.776 | 748.700 | 15.9% |
| 15 | Actual scoped default first | 875.495 | 668.901 | 23.6% |
| 15 | Disabled arm first | 1071.877 | 903.554 | 15.7% |

Five ordinary request rounds after two warmups per arm, both execution orders;
phase instrumentation disabled, preparation/allocation included. All complete
selected index/distance outputs match within and across windows. Independent
integer oracle passed 396584 cases/four arms, plus 24 full distance-layout cases
and in-place metadata mutation. Large actual-default reverse-pass device drift
(0.876–1.116 last/first) limits absolute price claims; all four large request pairs
still favor metadata. Small controls do not support a broader default.

Adopted only for the exact large Apple IDENTICAL metric/layout/shapes above.
Other shapes retain existing preflight, and explicit experimental/disable flags
remain. Source: forced aeddd83e; final aeddd83e plus retained working patch.
Raw samples, full outputs, source/binary hashes, flags and final policy:
`bench/results/knn_metadata_2026-09-10/README.md`.


### NVIDIA kNN vector-load component diagnostic, September 10

`bench/results/knn_loads_2026-09-10/` retains current batch-512 phase profiles
and same-process distance-tile trials. On H100 80GB HBM3 / driver 580.126.09,
the aligned interior candidate takes 0.314308 ms vs scalar 0.335547 ms on
512×65,536/d32 (6.33% less component time). This excludes preparation,
selection and request work. It is not a new opponent tuple, ordinary request
price, promoted default, or replacement for the qualified 2.69×/2.95× rows.
No opponent was timed. Exact fixture, source, toolchain, UUID and output
checks are recorded with the evidence.


### Sep10 NVIDIA aligned kNN loads: qualified public requests

Evidence: `bench/results/knn_vector_request_2026-09-10`. H100 80GB HBM3,
GPU-fcd67bc9-4348-93cc-aba6-22a2076f2fdc, driver 580.126.09, Mojo ed45d567.
IDENTICAL L2SqrtExpanded, dyadic-v1, 400000 index / 4000 queries / 32 features,
query batch 512 and index partition 65536. Same-process alternating scalar
and aligned-load arms, two shape-order windows, 31 samples per arm, save
3.1–3.6% complete request time. Full selected distances/indices match;
large ragged controls and candidate sabotage establish correctness and reach.
The aligned path is now default for this measured NVIDIA scope.

| k | Ordinary final default request ms | Cached cuML request ms | Default / cached |
|---|---:|---:|---:|
| 10 | 26.661871 | 10.225 | 2.61x |
| 15 | 31.126293 | 10.817 | 2.88x |

Final prices use 15 ordinary request samples after two warmups, without the
compile-time test override. Every output byte also matches the scalar
three-arm check artifact. Cached cuML entries are unchanged: these are
same-model/driver/shape/fixture/scope cached-reference ratios from different
physical rentals, not a fresh paired opponent comparison. No opponent ran
and no new opponent tuple was introduced. The earlier 6.33% isolated
component gain does not describe complete requests.

## AMD Instinct MI325X, amdgpu 6.12.12, ROCm 6.4.0, DigitalOcean tor1

Droplet `gpu-mi325x1-256gb` (the card reports AMD Instinct Mi325X VF,
gfx942), AMD EPYC 9575F with 20 vCPUs, Ubuntu 24.04.2, Python 3.12.3, Mojo
1.0.0 (ed45d567). ENGINEERING_RULES.md section 10 applies here. An opponent
with an AMD GPU path runs on the GPU, one without (cuML, CatBoost GPU) runs
on this box's CPU on all 20 cores, and each row names its device. Ours is
the IDENTICAL tier at the default build (DEVIATION 2502 on), source 92b4bf9b
with the harness of aca4c256. Same process as our arm, arms alternating per
round, opponents imported before our binding, 1 warm-up plus 5 rounds, ms
median (min..max). Configs are the H100 Istella-S rows' (boosting 100
estimators, depth 6, lr 0.1, l2 1, 254 borders or max_bin 255, no bagging,
Plain, seed 7, max_leaves 64 for lossguide; forests 100 trees, depth 16,
sqrt features, bootstrap for RF only, seed 7). Accuracy is logloss / AUC on
the fixed test tail. LightGBM 4.7.0 refused the lossguide cell on both
datasets (the PyPI wheel on both, and a USE_GPU (OpenCL) build on taxi) with
"Check failed: (best_split_info.left_count) > (0)" at warm-up under the
harness's LightGBM params, so no LightGBM row exists on this box.
Evidence: `bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/`.

Identity caveat for this box. Our IDENTICAL GBDT arms (symmetric, depthwise,
lossguide) returned a different prediction hash in every round on taxi, and
`identity_break` here moves the gbdt `ties` fixture between its two repeats
on all four gbdt lanes (77 of 81 cells stable; the H100 set is 81 of 81).
RF and ET held one hash (RF once gave a second hash in 30 taxi rounds). The
GBDT times are real fits, but the IDENTICAL contract does not hold for them
on AMD until that is fixed, and their accuracy column is the last round's
model. Istella-S says the same and more. The GBDT hashes move every round
there too, and the models are not the H100's: logloss 0.147018 against the
H100's 0.138653 (symmetric), 0.133018 against 0.126517 (depthwise), 0.126223
against 0.122045 (lossguide) at the same config. The symmetric
`use_pointwise_searcher=True` arm on this box reproduces the H100 default
arm's 0.138653 / 0.966990 exactly (still two hashes across five rounds), which
points at the greedy searcher path on AMD. RF and ET hashes equal the H100
hashes at 1M and 2M (574b24d0d7af51d0, cc25cb08f8b5a813, 40b1c5b03ba40420).

Cause found and fixed 2026-09-11 (DEVIATION 2600, lane/amd-gbdt-identity-fix).
The greedy searcher's binary, half-byte and hist_2 5-/6-bit histogram kernels
peeled a partition's head and tail with a loop that gave threads at or past
128 or 256 of a 512-thread block no trip, so part of the block skipped a
threadgroup barrier that only the 64-lane AMD column issues
(`gbdt/methods/greedy_subsets_searcher/kernel/lane_sync.mojo`). Taxi and
Istella-S have low-cardinality columns that land in those kernels; the
128-border float fixtures do not. Verified on a Hot Aisle MI300X (VF, gfx942,
8 cores, ROCm 6.4.1 userland on a 7.2.4 host) with the same source built with
and without the fix: identity_break 36/36 gbdt cells equal to the H100 with it
(32/36, `ties` MOVED, without); `checks/gbdt_sub_byte_identity_check.py` 16/16
with it (0/16 without, and restoring one kernel file at a time failed exactly
that file's fixtures); taxi 1M symmetric hash 90c3558501933f47 in 10 of 10
rounds with logloss 0.525735 / AUC 0.619460, equal to the H100 cell of the same
source (without it five hashes in five rounds, twice). On the H100 both builds
gave the same bits. The GBDT rows and accuracy above were measured BEFORE the
fix and are not re-run; they are not IDENTICAL results. The shipped 0.8.1 wheel
moves on `ties` on the MI300X (it carries the defect). The pointwise arm's
second hash is not explained by this fix and was not re-measured.

The pointwise arm's second hash has its own cause, found and fixed
2026-09-11 (DEVIATION 2624, lane/pointwise-hash-drift). The three pointwise
histogram launchers split each part's documents `multiplier` ways (a count
derived from the column's multiprocessor count) and every document block
added its partial histogram into the same float cell in completion order,
so the cells jittered and a near-tied split could flip. In the deterministic
and identical tiers the multiplier is now 1 (`checks/kernel_matrix.mojo::pointwise_doc_split_for`),
gated by `checks/pointwise_identical_multiplier_check.mojo` (896 grids,
55,290 cells bit-equal; it fails with the old multiplier). Verified on an
H100 (three fits in one process trace-identical on Istella 1M and two
synthetic fixtures), on a Hot Aisle MI300X (the check, and pointwise and
greedy model hashes on five 1M-row synthetic fixtures in 3 of 3 rounds all
equal to the H100's) and on the Apple M4 (the check). It costs the opt-in
pointwise arm time on the H100 (Istella 1M about 1690 to 2750 ms, taxi 1M
about 810 to 1805 ms) and changes its taxi model (logloss 0.525668 to
0.525925); greedy bits and times do not move. Every pointwise time and
verdict in this file was measured before it.

What 2624 cost can be won back, but NOT with 2624's bits (lane
pointwise-speed, 2026-09-11 night, RunPod H100 `eqxtzdcpctpnkh`).
**DEVIATION 2669 is impossible and the lane proved it rather than trying it:**
a fixed-order fold of per-document-block partials is not the multiplier-1
sequential sum, because each partial rounds without the other blocks'
documents (float32 `x = (1e8, 1, -1e8, 1)`: sequential 1, two-block fold 2),
and the only exact schedule chains the blocks and so serializes them.
**DEVIATION 2670 (opt-in, `-D MOJOLEARN_2670_PW_PRIVATE_DOC_SLOTS=1`, NOT
flipped, NOT merged)** takes the speed back with NEW bits: the multiplier is
`EstimateBlockPerFeatureMultiplier` at a PINNED SM count (`PW_2670_PINNED_SM
= 128`, never the device's), every document block stores into its own scratch
slot, and one launch folds slots 0..M-1 onto `binSums` in block order, so
there is no atomic and no dependence on `MULTIPROCESSOR_COUNT`. On the H100,
IDENTICAL, both trees built on that pod and interleaved at the process level
(three outer rounds, three timed rounds each, medians of nine): the opt-in
pointwise arm on taxi 1M went 1803.3 to 795.4 ms (0.441x) with logloss
0.525925 to 0.525668, while the greedy control held 309.3 to 308.9 ms
(0.999x) at hash 90c3558501933f47. **The Istella-S 1M cell is RUN OWED**: the
pod's Istella download was killed by its own timeout at 464 of 472 MB, the
speed harness then fell back to its synthetic fixture and labeled the rows
`shape=synthclf-720000x100`, and those rows were deleted rather than quoted.
So 2670 has NO flip verdict: one dataset is not a result (section 9).
Identity traces name where the new bits come from, 20-tree fits, main against
2670: `onebyte` IDENTICAL at 182 of 182 stages (that family accumulates in
Int32 fixed point, DEVIATION 93, so its cells fold exactly in any order),
while `halfbyte`, `binary`, `mixed` and `wide220` all diverge at
`tree001.depth00.hist`. Gates on the H100: `check-pointwise-identical-multiplier-2670`
G1 (896 grids, the multiplier independent of `sm_count` at every one, 832 of
them splitting) and G2 (55,290 cells bit-equal across sm 1/16/132/4096 and
three repeats), plus `check-pointwise-dispatch-2670` F1-F7 on integer stats
against a host tally. Merging it needs the Apple M4 and an AMD box first;
evidence in `bench/results/pointwise_speed_2026-09-11/README.md` on branch
`lane/pointwise-speed`.

Taxi, 1,000,000 training rows (2,000,000 for RF 2M), leg 1 (droplet
599636038):

| lane | opponent | version | device | config | opponent ms | opponent logloss / AUC | ours IDENTICAL ms | ours logloss / AUC | ours / opponent | log |
|---|---|---|---|---|---|---|---|---|---|---|
| RF 1M | scikit-learn RandomForestClassifier | 1.9.1 | CPU, 20 cores (n_jobs=-1) | exact thresholds (no bins) | 4069 (4054..4130) | 0.525336 / 0.617420 | 1180 (1172..1182) | 0.525910 / 0.617154 | 0.29x | baseline.rf.taxi.r1000000.full.log |
| RF 2M | scikit-learn RandomForestClassifier | 1.9.1 | CPU, 20 cores | same | 8871 (8858..9541) | 0.523834 / 0.622621 | 1734 (1723..1743) | 0.524221 / 0.622648 | 0.20x | baseline.rf.taxi.r2000000.full.log |
| ET 1M | scikit-learn ExtraTreesClassifier | 1.9.1 | CPU, 20 cores | no bootstrap | 2513 (2512..2558) | 0.527011 / 0.611206 | 3837 (3833..3840) | 0.527541 / 0.608084 | 1.53x | baseline.et.taxi.r1000000.full.log |
| symmetric 1M | CatBoost SymmetricTree | 1.2.10 | CPU, 20 threads | task_type CPU | 1035 (1030..1044) | 0.525674 / 0.617966 | 457 (452..461) | 0.529216 / 0.608483 | 0.44x | baseline.gbdt-symmetric.taxi.r1000000.full.log |
| depthwise 1M | CatBoost Depthwise | 1.2.10 | CPU, 20 threads | task_type CPU | 2204 (2150..2229) | 0.525755 / 0.620508 | 802 (799..804) | 0.527313 / 0.616813 | 0.36x | baseline.gbdt-depthwise.taxi.r1000000.full.log |
| depthwise 1M | XGBoost, AMD ROCm build `amd_xgboost` (pypi.amd.com rocm-6.4.4) | 3.1.1 | GPU (USE_HIP; setup probe config device cuda:0, rocm-smi 21 and 56 percent) | tree_method hist, device cuda | 337 (329..343) | 0.525221 / 0.620149 | 802 (799..804) | 0.527313 / 0.616813 | 2.38x | same |
| lossguide 1M | CatBoost Lossguide | 1.2.10 | CPU, 20 threads | task_type CPU | 3650 (3543..3692) | 0.526113 / 0.619758 | 1519 (1505..1539) | 0.526799 / 0.614677 | 0.42x | baseline.gbdt-lossguide.taxi.r1000000.full.log |
| lossguide 1M | XGBoost `amd_xgboost` | 3.1.1 | GPU | grow_policy lossguide; hash 348bf22bf60a14bc, equal to its depthwise row | 559 (537..560) | 0.525221 / 0.620149 | 1519 (1505..1539) | 0.526799 / 0.614677 | 2.72x | same |
| lossguide 1M | LightGBM | 4.7.0 | CPU | refused (above) | refused | - | - | - | - | same, and *.full.lgbmwheel.log |

A second lossguide pass with the PyPI LightGBM wheel
(`baseline.gbdt-lossguide.taxi.r1000000.full.lgbmwheel.log`) gave ours 1516
(1489..1529), CatBoost CPU 3567 (3550..3606), XGBoost GPU 549 (538..551).

Our FAST tier beside IDENTICAL on taxi, ours only, same box, interleaved in
one process (`*.ours.fastab2.log`), FAST then IDENTICAL: symmetric 509
(494..524) and 438 (416..448); depthwise 882 (868..885) and 785 (773..794);
lossguide 1627 (1607..1637) and 1498 (1449..1529); RF 1178 (1174..1181) and
1174 (1169..1182); ET 3838 (3833..3840) and 3836 (3835..3839). Against the
opponent rows above, FAST is 0.49x of CatBoost symmetric, 0.40x of CatBoost
and 2.62x of XGBoost depthwise, 0.45x of CatBoost and 2.91x of XGBoost
lossguide, 0.29x of scikit-learn RF and 1.53x of scikit-learn ET.

Istella-S, 1,000,000 training rows (2,000,000 for RF 2M), leg 2 (droplet
599649142; the XGBoost venv was rebuilt after apt fetched stale packages and
its GPU probe re-run, config device cuda:0, rocm-smi 20 percent):

| lane | opponent | version | device | config | opponent ms | opponent logloss / AUC | ours IDENTICAL ms | ours logloss / AUC | ours / opponent | log |
|---|---|---|---|---|---|---|---|---|---|---|
| RF 1M | scikit-learn RandomForestClassifier | 1.9.1 | CPU, 20 cores (n_jobs=-1) | exact thresholds (no bins) | 13698 (13642..13802) | 0.146188 / 0.964092 | 1937 (1929..1946) | 0.145560 / 0.964538 | 0.14x | baseline.rf.istella.r1000000.full.log |
| RF 2M | scikit-learn RandomForestClassifier | 1.9.1 | CPU, 20 cores | same | 29436 (29159..29783) | 0.145257 / 0.964663 | 3109 (3102..3115) | 0.144845 / 0.964998 | 0.11x | baseline.rf.istella.r2000000.full.log |
| ET 1M | scikit-learn ExtraTreesClassifier | 1.9.1 | CPU, 20 cores | no bootstrap | 8944 (8863..8983) | 0.187901 / 0.939528 | 14538 (14532..14563) | 0.188191 / 0.938768 | 1.63x | baseline.et.istella.r1000000.full.log |
| symmetric 1M | CatBoost SymmetricTree | 1.2.10 | CPU, 20 threads | task_type CPU | 4373 (4342..4400) | 0.138897 / 0.966996 | 1312 (1284..1464) | 0.147018 / 0.962675 | 0.30x | baseline.gbdt-symmetric.istella.r1000000.full.log |
| depthwise 1M | CatBoost Depthwise | 1.2.10 | CPU, 20 threads | task_type CPU | 6671 (6645..6740) | 0.128097 / 0.971895 | 1987 (1979..2007) | 0.133018 / 0.969407 | 0.30x | baseline.gbdt-depthwise.istella.r1000000.full.log |
| depthwise 1M | XGBoost `amd_xgboost` | 3.1.1 | GPU | tree_method hist, device cuda | 1575 (1569..1597) | 0.124822 / 0.973880 | 1987 (1979..2007) | 0.133018 / 0.969407 | 1.26x | same |
| lossguide 1M | CatBoost Lossguide | 1.2.10 | CPU, 20 threads | task_type CPU | 9045 (9021..9095) | 0.127841 / 0.972012 | 2607 (2582..2790) | 0.126223 / 0.973382 | 0.29x | baseline.gbdt-lossguide.istella.r1000000.full.log |
| lossguide 1M | XGBoost `amd_xgboost` | 3.1.1 | GPU | grow_policy lossguide; hash 4d80a2001a82f486, equal to its depthwise row | 1880 (1866..1941) | 0.124822 / 0.973880 | 2607 (2582..2790) | 0.126223 / 0.973382 | 1.39x | same |
| lossguide 1M | LightGBM | 4.7.0 | CPU | refused (above) | refused | - | - | - | - | same |

Our FAST tier beside IDENTICAL on Istella-S, ours only, same box, interleaved
in one process (`*.istella.r1000000.ours.fastab.log`), FAST then IDENTICAL:
symmetric 1042 (1024..1045) and 1146 (1127..1169); depthwise 1783
(1772..1791) and 1861 (1854..1893); lossguide 2702 (2687..2756) and 2635
(2601..2647); RF 1951 (1947..1957) and 1928 (1922..1937); ET 14545
(14534..14555) and 14552 (14538..14573). Against the Istella-S opponent rows,
FAST is 0.24x of CatBoost symmetric, 0.27x of CatBoost and 1.13x of XGBoost
depthwise, 0.30x of CatBoost and 1.44x of XGBoost lossguide, 0.14x of
scikit-learn RF and 1.63x of scikit-learn ET. FAST ET returns the IDENTICAL
hash; FAST RF holds one hash of its own; FAST GBDT hashes move every round.

Verdicts on this box (tools/flip_verdict.py, both datasets): symmetric
`use_pointwise_searcher=True` NO FLIP geomean=1.816 taxi=2.342
istella=1.408 reason=time (quality better on both; measured before
DEVIATION 2624, which makes that arm slower still); DEVIATION 2512 on
against off, RF NO FLIP geomean=1.001 taxi=0.999 istella=1.004 (bits equal),
symmetric NO FLIP geomean=1.011 taxi=0.999 istella=1.023 (its quality delta
is inside the round-to-round movement above), so 2512 is neutral on AMD.

## AMD Instinct MI300X, amdgpu 6.16.13, ROCm 6.4.1, Hot Aisle (13-core VM)

Hot Aisle 1x MI300X VM on the 13-core spec (the card reports AMD Instinct
MI300X VF, gfx942), Intel Xeon Platinum 8470 with 13 cores and 224 GB. The
host runs Ubuntu 24.04.4 with ROCm 7.2.4, and the leg body runs in the
runner's default container `rocm/dev-ubuntu-22.04:6.4.1-complete` (Ubuntu
22.04.5, glibc 2.35, ROCm 6.4.1 userland), CPython 3.12.14, Mojo 1.0.0
(ed45d567). ENGINEERING_RULES.md section 10 applies, and these rows are a new
tuple, never mixed with the MI325X rows above or with any NVIDIA row. CatBoost
and scikit-learn have no AMD GPU path and run on this VM's CPU on all 13
cores. Ours is the IDENTICAL tier at the default build, source 8d7e129c, which
carries the AMD GBDT identity fix (6ad945c6, DEVIATION 2600). Same process as
our arm, arms alternating per round, opponents imported before our binding, 1
warm-up plus 5 rounds, ms median (min..max), configs as in the MI325X section.
Accuracy is logloss / AUC on the fixed test tail. Ours FAST is the
`numeric_mode='fast'` arm interleaved with IDENTICAL in its own ours-only cell
(`*.ours.fastab.log`), set against the opponent row beside it. Evidence is in
`bench/results/trees_identical/mi300x_hotaisle_2026-09-11/taxi/` (build logs
over 100 KB stay outside the repository).

Two opponents with an AMD GPU path did not run on the GPU here, and each row
says so. XGBoost is the PyPI 3.4.1 build on the CPU, because AMD ships
`amd_xgboost` only as manylinux_2_39 wheels and this container has glibc 2.35,
so pip found no distribution. The amd_xgboost GPU rows at the end of the
table were measured on a second VM with `rocm/dev-ubuntu-24.04:6.4.1-complete`
(glibc 2.39). LightGBM 4.7.0 ran on the CPU with `min_child_weight=1e-3`
(`MOJOLEARN_SPEED_LGBM_PARAMS`), the one retry after the harness's 0.0 was
refused on the MI325X. Its OpenCL build compiled and still failed its probe
with "Check failed: (best_split_info.right_count) > (0)" under
`gpu_use_dp=True` (one attempt, 80 s).

Identity on this box. With the fix, each of our IDENTICAL GBDT lanes returned
one prediction hash across all ten fits in both of its cells (symmetric
90c3558501933f47, depthwise 40c1683b9e0eb151, lossguide 0dd8bcfc3c3a4a1d). Our
RF and ET hashes equal the MI325X taxi hashes (RF 1M d8f64dae01de00bd, RF 2M
f1240292e3b1dd3f, ET e683f121d11f59dd). FAST GBDT hashes move every round, FAST
RF holds one hash of its own, and FAST ET returns the IDENTICAL hash.

Taxi, 1,000,000 training rows (2,000,000 for RF 2M), VM 9b86604d:

| lane | opponent | version | device | config | opponent ms | opponent logloss / AUC | ours IDENTICAL ms | ours logloss / AUC | IDENTICAL / opponent | ours FAST ms | FAST / opponent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF 1M | scikit-learn RandomForestClassifier | 1.9.1 | CPU, 13 cores (n_jobs=-1) | exact thresholds (no bins) | 12529 (12478..12593) | 0.525336 / 0.617420 | 1291 (1280..1372) | 0.525910 / 0.617154 | 0.10x | 1280 (1273..1295) | 0.10x |
| RF 2M | scikit-learn RandomForestClassifier | 1.9.1 | CPU, 13 cores | same | 27955 (27811..28089) | 0.523834 / 0.622621 | 1896 (1888..1961) | 0.524221 / 0.622648 | 0.07x | not run (FAST cells are 1M) | - |
| ET 1M | scikit-learn ExtraTreesClassifier | 1.9.1 | CPU, 13 cores | no bootstrap | 10224 (10190..10246) | 0.527011 / 0.611206 | 4093 (4074..4134) | 0.527541 / 0.608084 | 0.40x | 4081 (4075..4091) | 0.40x |
| symmetric 1M | CatBoost SymmetricTree | 1.2.10 | CPU, 13 threads | task_type CPU | 2180 (2168..2212) | 0.525674 / 0.617966 | 507 (492..525) | 0.525735 / 0.619460 | 0.23x | 543 (541..557) | 0.25x |
| depthwise 1M | CatBoost Depthwise | 1.2.10 | CPU, 13 threads | task_type CPU | 4362 (4317..4423) | 0.525755 / 0.620508 | 1047 (1038..1063) | 0.525086 / 0.621421 | 0.24x | 1097 (1086..1129) | 0.25x |
| depthwise 1M | XGBoost (PyPI) | 3.4.1 | CPU, 13 threads (amd_xgboost not installable on the glibc 2.35 image) | tree_method hist, device cpu | 1043 (1013..1483) | 0.525332 / 0.620103 | 1047 (1038..1063) | 0.525086 / 0.621421 | 1.00x | 1097 (1086..1129) | 1.05x |
| lossguide 1M | CatBoost Lossguide | 1.2.10 | CPU, 13 threads | task_type CPU | 6943 (6868..6986) | 0.526113 / 0.619758 | 1774 (1741..1779) | 0.525504 / 0.619386 | 0.26x | 1837 (1821..1859) | 0.26x |
| lossguide 1M | XGBoost (PyPI) | 3.4.1 | CPU, 13 threads (same reason) | grow_policy lossguide; hash 98e44c1ffb4f3ad5, equal to its depthwise row | 1034 (982..1491) | 0.525332 / 0.620103 | 1774 (1741..1779) | 0.525504 / 0.619386 | 1.72x | 1837 (1821..1859) | 1.78x |
| lossguide 1M | LightGBM | 4.7.0 | CPU, 13 threads | min_child_weight=1e-3 (retry); OpenCL build failed its probe | 952 (930..988) | 0.525029 / 0.620694 | 1774 (1741..1779) | 0.525504 / 0.619386 | 1.86x | 1837 (1821..1859) | 1.93x |
| depthwise 1M | XGBoost `amd_xgboost` (pypi.amd.com rocm-6.4.4) | 3.1.1 | GPU (USE_HIP, config device cuda:0, rocm-smi peak 43 percent) | tree_method hist, device cuda; VM 96560ef4 on `rocm/dev-ubuntu-24.04:6.4.1-complete` (glibc 2.39) | 594 (561..612) | 0.525221 / 0.620149 | 1030 (1020..1046) | 0.525086 / 0.621421 | 1.73x | 1097 (1086..1129), VM 9b86604d | 1.85x |
| lossguide 1M | XGBoost `amd_xgboost` | 3.1.1 | GPU (rocm-smi peak 99 percent) | grow_policy lossguide; hash 348bf22bf60a14bc, equal to its depthwise row and to the MI325X amd_xgboost row | 792 (756..797) | 0.525221 / 0.620149 | 1727 (1717..1783) | 0.525504 / 0.619386 | 2.18x | 1837 (1821..1859), VM 9b86604d | 2.32x |

The two amd_xgboost rows come from a second VM of the same 13-core spec
(96560ef4, verified gone), ours IDENTICAL interleaved with amd_xgboost only,
rocm-smi sampled beside each cell. Our hashes there equal the taxi VM's. Their
FAST column is the taxi VM's FAST cell. Evidence is in
`bench/results/trees_identical/mi300x_hotaisle_2026-09-11/xgbgpu/`.

Istella-S against amd_xgboost on the same VM 96560ef4. No Istella-S FAST cell
ran on Hot Aisle; the Istella-S FAST cells below belong to the RunPod pod and
are not set against these rows. Our hashes here equal the RunPod pod's, and the
amd_xgboost hash and logloss equal the MI325X Istella-S row's.

| lane | opponent | version | device | config | opponent ms | opponent logloss / AUC | ours IDENTICAL ms | ours logloss / AUC | IDENTICAL / opponent | ours FAST ms | FAST / opponent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| depthwise 1M | XGBoost `amd_xgboost` | 3.1.1 | GPU (rocm-smi peak 99 percent) | tree_method hist, device cuda | 2851 (2827..2864) | 0.124822 / 0.973880 | 3376 (3332..3501) | 0.126517 / 0.971896 | 1.18x | not run on this box | - |
| lossguide 1M | XGBoost `amd_xgboost` | 3.1.1 | GPU (rocm-smi peak 100 percent) | grow_policy lossguide; hash 4d80a2001a82f486, equal to its depthwise row | 3297 (3285..3327) | 0.124822 / 0.973880 | 4172 (4126..4311) | 0.122045 / 0.975037 | 1.27x | not run on this box | - |

### AMD Instinct MI300X on RunPod

Istella-S ran on one RunPod AMD pod (mdv9clyq6r73bz, verified gone with HTTP
404) because no Hot Aisle slot was free (ENGINEERING_RULES.md section 10 box
order). This pod is its own box and its rows are never set against the Hot
Aisle table above. The card reports AMD Instinct MI300X, gfx942, amdgpu
6.10.5. The pod shows 192 logical CPUs of two AMD EPYC 9474F 48-core
processors and allots 21 of them (`joblib.cpu_count` 21), so scikit-learn ran
on 21 cores and CatBoost on its default thread count. The host is shared, and
the load average reached 22 to 24 during the GBDT cells, so absolute times
are noisy. Every arm alternates round by round in the same process, so each
ratio below holds within this pod. The body ran in
`rocm/dev-ubuntu-22.04:6.4.1-complete` (Ubuntu 22.04.5, glibc 2.35, ROCm 6.4.1
userland), CPython 3.12.14, source 8af3b8f0 (the AMD GBDT identity fix
included), same configs and protocol as the Hot Aisle section. The same glibc
2.35 kept `amd_xgboost` from installing, so XGBoost is the PyPI build on the
CPU and the amd_xgboost GPU row is owed on this pod. LightGBM 4.7.0 ran on the
CPU with `min_child_weight=1e-3`; the one OpenCL attempt belonged to the taxi
leg and was not repeated here. Evidence is in
`bench/results/trees_identical/mi300x_hotaisle_2026-09-11/runpod_istella/`.

Identity on this pod. Each of our IDENTICAL GBDT lanes returned one prediction
hash across all ten fits in both of its cells (symmetric 238d3abce0cabf43,
depthwise 5d053cd086658072, lossguide 6182fd2bee4fb941), and their logloss
equals the H100 values quoted in the MI325X section (0.138653, 0.126517,
0.122045). Our RF and ET hashes are the ones that section records as equal to
the H100's (RF 1M 574b24d0d7af51d0, RF 2M cc25cb08f8b5a813, ET
40b1c5b03ba40420). FAST GBDT held one hash of its own for symmetric and two
across five rounds for depthwise and lossguide, FAST RF one of its own, and
FAST ET returns the IDENTICAL hash.

Istella-S, 1,000,000 training rows (2,000,000 for RF 2M), pod mdv9clyq6r73bz:

| lane | opponent | version | device | config | opponent ms | opponent logloss / AUC | ours IDENTICAL ms | ours logloss / AUC | IDENTICAL / opponent | ours FAST ms | FAST / opponent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF 1M | scikit-learn RandomForestClassifier | 1.9.1 | CPU, 21 cores (n_jobs=-1) | exact thresholds (no bins) | 21217 (20864..21759) | 0.146188 / 0.964092 | 2360 (2343..2458) | 0.145560 / 0.964538 | 0.11x | 2402 (2372..2425) | 0.11x |
| RF 2M | scikit-learn RandomForestClassifier | 1.9.1 | CPU, 21 cores | same | 47043 (46123..47921) | 0.145257 / 0.964663 | 3862 (3838..3907) | 0.144845 / 0.964998 | 0.08x | not run (FAST cells are 1M) | - |
| ET 1M | scikit-learn ExtraTreesClassifier | 1.9.1 | CPU, 21 cores | no bootstrap | 14195 (13802..15482) | 0.187901 / 0.939528 | 15457 (15435..15460) | 0.188191 / 0.938768 | 1.09x | 15483 (15463..15536) | 1.09x |
| symmetric 1M | CatBoost SymmetricTree | 1.2.10 | CPU, default threads | task_type CPU | 7063 (6784..7171) | 0.138897 / 0.966996 | 2382 (2359..2456) | 0.138653 / 0.966990 | 0.34x | 1876 (1832..2480) | 0.27x |
| depthwise 1M | CatBoost Depthwise | 1.2.10 | CPU, default threads | task_type CPU | 10625 (9915..11212) | 0.128097 / 0.971895 | 3034 (2907..3095) | 0.126517 / 0.971896 | 0.29x | 2554 (2510..2651) | 0.24x |
| depthwise 1M | XGBoost (PyPI) | 3.4.1 | CPU (amd_xgboost not installable on the glibc 2.35 image) | tree_method hist, device cpu | 4592 (4044..4929) | 0.125270 / 0.973728 | 3034 (2907..3095) | 0.126517 / 0.971896 | 0.66x | 2554 (2510..2651) | 0.56x |
| lossguide 1M | CatBoost Lossguide | 1.2.10 | CPU, default threads | task_type CPU | 16190 (16047..16764) | 0.127841 / 0.972012 | 4304 (4262..4380) | 0.122045 / 0.975037 | 0.27x | 4371 (4116..4430) | 0.27x |
| lossguide 1M | XGBoost (PyPI) | 3.4.1 | CPU (same reason) | grow_policy lossguide; hash 0d844a0ee7f63cb5, equal to its depthwise row | 6133 (5530..6731) | 0.125270 / 0.973728 | 4304 (4262..4380) | 0.122045 / 0.975037 | 0.70x | 4371 (4116..4430) | 0.71x |
| lossguide 1M | LightGBM | 4.7.0 | CPU, default threads | min_child_weight=1e-3 (retry) | 4687 (4318..4776) | 0.125178 / 0.973695 | 4304 (4262..4380) | 0.122045 / 0.975037 | 0.92x | 4371 (4116..4430) | 0.93x |
| depthwise, lossguide 1M | XGBoost `amd_xgboost` | - | GPU | - | not on this pod (glibc 2.35 image); measured on Hot Aisle VM 96560ef4 above, a different box whose ratios are not set against this pod's | - | - | - | - | - | - |

## AMD Instinct MI300X on RunPod, amdgpu 6.10.5, ROCm 6.4.1 userland, torch 2.6.0+rocm6.4.1

A RunPod pod is not a Hot Aisle VM. Its CPU rows are comparable only within
this section. Its torch GPU rows are the same GPU model as a Hot Aisle
MI300X, on a different provider and host driver.

### Classical lanes on taxi and Istella-S (September 11, pod uncjlirh5elvmp)

Pod `uncjlirh5elvmp` (`mojolearn-gemm-amd-2026-09-11_125346`) on RunPod
machine `j03rnq2tcsxu`, $2.39 per hour, image
`rocm/dev-ubuntu-22.04:6.4.1-complete` (ROCm 6.4.1-83 userland) on host
amdgpu 6.10.5, kernel 6.8.0-138. GPU AMD Instinct MI300X (SR-IOV SKU
MI3SRIOV, gfx942). CPU AMD EPYC 9474F 48-Core Processor, 192 CPUs visible
to the pod and a CFS quota of 20.4 CPUs (2040000 over 100000; the API allots
24 vCPUs and 283 GB). Ours is IDENTICAL at commit 37480b23 (Mojo 1.0.0
ed45d567, pixi Python 3.14), timed from a host float32 array to host
results. torch 2.6.0+rocm6.4.1.git1ded221d from repo.radeon.com
rocm-rel-6.4.1 (HIP 6.4.43483, setup probe ran eigh, cdist, topk, lstsq and
addmm on the GPU), inputs uploaded before the clock, the clock ends at
`torch.cuda.synchronize()`. scikit-learn 1.9.1, SciPy 1.18.1 and NumPy 2.5.3
in a Python 3.12 venv run in two arms. `sklearn-cpu` is as installed, no
thread variables, OpenBLAS 0.3.34 (NumPy's) and 0.3.31.dev (SciPy's) at 64
threads each and OpenMP at 192, `n_jobs=-1` for kNN. `sklearn-cpu-quota` is
the same call under `threadpool_limits(20)` with kNN `n_jobs=20` (DEVIATION
2573). KernelDensity scoring and libsvm are single-threaded by design in
both arms. cuML has no ROCm path. 1 warm-up plus 5 rounds, arms interleaved
with the order rotated every round, ms median (min..max), quality computed by
one float64 NumPy function per lane (`tools/classical_two_datasets.py`).
Shapes are taxi 4,000,000 x 11 and Istella-S 2,043,304 x 220 for kmeans (k
64, 20 iterations from a shared init), pca (8 components) and ols; kNN
400,000 index rows x 4,000 queries, k 10; kde 100,000 standardized fit rows x
2,000 queries, Scott bandwidth; svc 10,000 standardized fit rows, RBF, C 1,
gamma 1/d. Taxi and Istella-S bytes matched the Mac's sha256. Evidence
`bench/results/classical_hotaisle_2026-09-11/runpod_mi300x/`.

| lane | dataset | opponent | device, threading, BLAS | opponent ms | opponent quality | ours IDENTICAL ms | ours quality | ours / opponent |
|---|---|---|---|---|---|---|---|---|
| kmeans | taxi | scikit-learn KMeans lloyd | CPU, OpenBLAS 64 + OpenMP 192 on a 20.4-CPU quota | 1176 (1171..1225) | inertia 1.20198e8, 20 iter | 129 (122..155) | inertia 1.20628e8, 21 iter | 0.11x |
| kmeans | taxi | scikit-learn KMeans lloyd | CPU, threadpool_limits(20) | 1310 (1183..1370) | inertia 1.20197e8 | 129 (122..155) | same | 0.10x |
| kmeans | taxi | torch Lloyd (addmm, index_add_) | GPU, MI300X | 237 (235..248) | inertia 1.20192e8, 20 iter | 129 (122..155) | same | 0.55x |
| kmeans | Istella-S | scikit-learn KMeans lloyd | CPU, OpenBLAS 64 + OpenMP 192 | 1796 (1791..1947) | inertia 1.28328e17 | 5239 (5215..5254) | inertia 1.31285e17, 21 iter | 2.92x |
| kmeans | Istella-S | scikit-learn KMeans lloyd | CPU, threadpool_limits(20) | 1835 (1829..1984) | inertia 1.28328e17 | 5239 (5215..5254) | same | 2.86x |
| kmeans | Istella-S | torch Lloyd | GPU, MI300X | 472 (466..488) | inertia 1.28553e17 | 5239 (5215..5254) | same | 11.09x |
| pca | taxi | scikit-learn PCA covariance_eigh | CPU, OpenBLAS 64 + OpenMP 192 | 118 (118..120) | EVR sum 0.997809 | 38.1 (22.8..39.0) | EVR sum 0.997861 | 0.32x |
| pca | taxi | scikit-learn PCA covariance_eigh | CPU, threadpool_limits(20) | 120 (118..120) | EVR sum 0.997809 | 38.1 (22.8..39.0) | same | 0.32x |
| pca | taxi | torch covariance eigh | GPU, MI300X | 12.5 (12.2..12.7) | EVR sum 0.997861 | 38.1 (22.8..39.0) | same | 3.04x |
| pca | Istella-S | scikit-learn PCA covariance_eigh | CPU, OpenBLAS 64 + OpenMP 192 | 1361 (1350..1363) | EVR sum 1.0 | 686 (658..773) | EVR sum 1.0 | 0.50x |
| pca | Istella-S | scikit-learn PCA covariance_eigh | CPU, threadpool_limits(20) | 1355 (1352..1358) | EVR sum 1.0 | 686 (658..773) | same | 0.51x |
| pca | Istella-S | torch covariance eigh | GPU, MI300X | 20.4 (20.4..20.6) | EVR sum 1.0 | 686 (658..773) | same | 33.55x |
| ols | taxi | scikit-learn LinearRegression (gelsd) | CPU, OpenBLAS 64 + OpenMP 192 | 632 (614..688) | R2 0.908824 | 221 (206..233) | R2 0.908837 | 0.35x |
| ols | taxi | scikit-learn LinearRegression | CPU, threadpool_limits(20) | 487 (479..499) | R2 0.908824 | 221 (206..233) | same | 0.45x |
| ols | taxi | torch.linalg.lstsq | GPU, MI300X | 315 (183..621) | R2 0.908839 | 221 (206..233) | same | 0.70x |
| ols | taxi | torch normal equations eigh | GPU, MI300X | 36.9 (36.4..38.7) | R2 0.908822 | 221 (206..233) | same | 5.99x |
| ols | Istella-S | scikit-learn LinearRegression (gelsd) | CPU, OpenBLAS 64 + OpenMP 192 | 10087 (9694..10414) | R2 0.164077 | 1828 (1774..1846) | R2 -115.603 | not quoted (quality) |
| ols | Istella-S | scikit-learn LinearRegression | CPU, threadpool_limits(20) | 6670 (6603..6716) | R2 0.164151 | 1828 (1774..1846) | R2 -115.603 | not quoted (quality) |
| ols | Istella-S | torch.linalg.lstsq | GPU, MI300X | 3109 (2999..3189) | R2 NaN | 1828 (1774..1846) | R2 -115.603 | not quoted (quality) |
| ols | Istella-S | torch normal equations eigh | GPU, MI300X | 44.8 (44.6..44.9) | R2 0.151604 | 1828 (1774..1846) | R2 -115.603 | not quoted (quality) |
| knn | taxi | scikit-learn NearestNeighbors brute | CPU, n_jobs=-1, OpenBLAS 64 + OpenMP 192 | 246 (240..257) | recall@10 1.0 | 44.1 (28.9..45.5) | recall@10 0.99915 | 0.18x |
| knn | taxi | scikit-learn NearestNeighbors brute | CPU, n_jobs=20, threadpool_limits(20) | 244 (234..249) | recall@10 1.0 | 44.1 (28.9..45.5) | same | 0.18x |
| knn | taxi | torch cdist + topk | GPU, MI300X | 28.7 (28.6..33.1) | recall@10 0.999325 | 44.1 (28.9..45.5) | same | 1.54x |
| knn | Istella-S | scikit-learn NearestNeighbors brute | CPU, n_jobs=-1, OpenBLAS 64 + OpenMP 192 | 1076 (1071..1081) | recall@10 1.0 | 154 (137..244) | recall@10 0.923025 | 0.14x |
| knn | Istella-S | scikit-learn NearestNeighbors brute | CPU, n_jobs=20, threadpool_limits(20) | 1087 (1086..1090) | recall@10 1.0 | 154 (137..244) | same | 0.14x |
| knn | Istella-S | torch cdist + topk | GPU, MI300X | 34.2 (34.1..41.8) | recall@10 0.932375 | 154 (137..244) | same | 4.49x |
| kde | taxi | scikit-learn KernelDensity, rtol = atol = 0 | CPU, one thread (score_samples) | 7634 (7629..7643) | mean log-lik -9.58204 | 62.6 (47.7..64.8) | mean log-lik -9.58207 | 0.0082x |
| kde | taxi | scikit-learn KernelDensity | CPU, one thread, capped arm | 7631 (7594..7672) | mean log-lik -9.58204 | 62.6 (47.7..64.8) | same | 0.0082x |
| kde | Istella-S | scikit-learn KernelDensity | CPU, one thread | RUN OWED | - | RUN OWED | - | - |
| svc | taxi | scikit-learn SVC (libsvm) | CPU, one thread | 2776 (2729..2819) | accuracy 0.7675, 5675 SV | 1559 (1539..1585) | accuracy 0.7675, 5527 SV | 0.56x |
| svc | taxi | scikit-learn SVC (libsvm) | CPU, one thread, capped arm | 2735 (2726..2800) | accuracy 0.7675 | 1559 (1539..1585) | same | 0.57x |
| svc | Istella-S | scikit-learn SVC (libsvm) | CPU, one thread | 1508 (1318..1535) | accuracy 0.9222, 2400 SV | 241 (138..502) | accuracy 0.9222, 2400 SV | 0.16x |
| svc | Istella-S | scikit-learn SVC (libsvm) | CPU, one thread, capped arm | 1348 (1330..1519) | accuracy 0.9222 | 241 (138..502) | same | 0.18x |

Every ours arm held one digest across its five rounds.

Quality findings. Ours OLS on Istella-S returns R2 -115.603 (RMSE 9.011)
where scikit-learn returns 0.164 (RMSE 0.763), so no ratio is quoted for
that cell. Ours mirrors cuML `algorithm='eig'`, a float32 eigendecomposition
of the centered normal equations, and the torch arm of that class with a
d x eps32 x max|eig| cutoff reaches R2 0.152. That points at our eigenvalue
cutoff on Istella's near-constant columns and is not yet measured. torch
lstsq on Istella-S returns NaN. Ours kNN recall@10 on Istella-S is 0.923
against torch's 0.932 and scikit-learn's 1.0. Ours kmeans inertia is 0.36
percent above scikit-learn's on taxi and 2.3 percent above on Istella-S, and
it reports 21 iterations because the binding refuses tol 0 and the arm falls
back to tol 1e-7 while both opponents run exactly 20.

kde on Istella-S is RUN OWED. The race hit its 900 s bound (rc 124) before
the uncapped scikit-learn arm's fifth round, so no JSON exists. Each
scikit-learn `score_samples` call took about 79 s on both arms (78,860 to
79,599 ms over the rounds that ran), and ours took 608 to 771 ms over rounds
1 to 5. The rerun needs a race bound near 1,200 s or one scikit-learn arm,
since both are one thread there.

Capping scikit-learn at the quota moved ols outside its range (632 to 487
ms on taxi, 10,087 to 6,670 ms on Istella-S). Every other cell's capped and
uncapped ranges overlap, including kmeans on taxi, whose capped median is
higher (1,310 against 1,176 ms).

An earlier pod on the same machine (`m73ut5pd92r1rk`, commit ba49747e)
measured the same cells before the kmeans fix (evidence in
`superseded_pod_m73ut5pd92r1rk/`). Its taxi medians differ from the table
by 1 percent or less for ours pca, knn, kde and svc, 10 percent for ours ols
(201 against 221 ms), 13 percent for scikit-learn kmeans (1,333 against
1,176 ms) and 28 percent for torch lstsq (403 against 315 ms, whose rounds
span 168 to 621 ms). It was reaped at 16:57Z, before this pod's first race
at 17:00Z. Both pods were verified gone (HTTP 404). This pod's gemm device card
matched the Apple card at all 60 stages.

### Classical OLS, PCA and k-means on taxi against cuML (September 11, pod 22up9vbhj3tbeg, lane linear-cluster-speed)

RunPod H100 80GB HBM3, driver 580.126.09, cuML 26.08 (cuml-cu12 26.8.0) in the
same Python 3.11; ours IDENTICAL built on the pod (sm_90a) from main 36ca51fd.
`tools/classical_two_datasets.py race`, 1 warm-up plus 5 interleaved rounds, ms
median (min..max). Taxi 4,000,000 x 11; k-means k 64, 20 iterations, shared init,
tol 1e-7 on both arms (cuVS refuses tol 0); cuML PCA `svd_solver='full'`, ours
`covariance_eigh`; cuML OLS `algorithm='eig'`. Istella-S not measured (RUN OWED in
`bench/results/linear_cluster_speed_2026-09-11/README.md`, with the stage
breakdown and logs).

| lane | dataset | opponent | opponent ms | opponent quality | ours IDENTICAL ms (36ca51fd) | ours quality | ours / opponent | ours with DEVIATIONS 2632, 2633 (lane branch) |
|---|---|---|---|---|---|---|---|---|
| ols | taxi | cuML LinearRegression eig | 23.12 (21.92..26.83) | R2 0.908836 | 263.5 (251.8..275.9) | R2 0.908837 | 11.40x | 177.8 (146.0..180.0), 7.60x of that race's cuML 23.39, same coefficient digest |
| pca | taxi | cuML PCA full | 20.99 (20.08..21.73) | EVR sum 0.99786 | 28.69 (25.78..38.83) | EVR sum 0.997861 | 1.37x | unchanged |
| kmeans | taxi | cuML KMeans | 128.2 (127.4..129.0) | inertia 1.20192e8, 20 iter | 304.0 (279.6..368.6) | inertia 1.20628e8 | 2.37x | 216.2 (196.2..222.7), 1.68x of that race's cuML 128.9, same centroid digest |

cuML k-means returned a different centroid digest in every round; ours held one.

Istella-S, measured by the orchestrator the same night on RunPod pod
n03kul7dt759n0 (NVIDIA H200, driver 580.95.05; H100 stock was out, same sm_90a
build), cuML 26.08, both trees built on that pod, the same harness, 1 warm-up plus
5 interleaved rounds. Istella-S 2,043,304 x 220. Evidence
`~/mojolearn-evidence/linear-cluster-speed-2026-09-11/pod2-h200-istella/`,
summary `bench/results/linear_cluster_speed_2026-09-11/istella_h200_summary.tsv`.

| lane | dataset | opponent | opponent ms | opponent quality | ours before (36ca51fd) ms | ours after (2632, 2633) ms | after / before | ours after / opponent | ours quality, digest before = after |
|---|---|---|---|---|---|---|---|---|---|
| ols | Istella-S | cuML LinearRegression eig | 83.48 (83.29..83.54) | R2 -6473.68 | 4110.0 (4068.5..4320.3) | 2565.1 (2524.0..2585.7) | 0.624 | 30.73x | R2 0.331944, 6f12cfc209ecd1f9 |
| pca | Istella-S | cuML PCA full | 81.32 (81.26..116.80) | EVR sum 1.0000000156 | 687.8 (687.7..689.4) | 687.4 (684.2..688.6) | 0.999 (unchanged code) | 8.45x | EVR sum 1.0000000146, e43f2f52f20f511a |
| kmeans | Istella-S | cuML KMeans | 173.80 (173.60..176.06) | inertia 1.28555e17 (0.979 of ours), 20 iter, digest different every round | 2021.1 (2015.4..2023.3) | 805.7 (796.0..824.2) | 0.399 | 4.64x | inertia 1.31285e17, 21 iter, 7f720b0b76896308 |

Flip verdict (ENGINEERING_RULES section 9, geomean of after/before over taxi and
Istella-S): OLS sqrt(0.67 x 0.624) = 0.65, k-means sqrt(0.71 x 0.399) = 0.53, both
below 1 with bits unchanged on both datasets, so DEVIATIONS 2632 and 2633 are the
default (merged into main). cuML's eig OLS again returns a broken fit on Istella-S
(R2 -6473.68), the failure DEVIATIONS 2620 and 2621 fixed in ours.
### kNN on taxi and Istella-S against cuML, same pod (September 11, knn-speed lane, pod 62dlwtf4amlt2s)

H100 80GB HBM3, driver 580.159.04, cuML 26.8.0 (pypi.nvidia.com), source
origin/main 36ca51fd. Ours IDENTICAL through the public `NearestNeighbors`
(host in, host out, upload inside the clock); cuML brute NearestNeighbors
with cupy inputs uploaded before the clock (`tools/classical_two_datasets.py
race --lane knn`, 1 warm-up plus 5 interleaved rounds). Index rows
[0, 400,000), queries [400,000, 404,000), k 10, raw columns (Istella's
float32-max sentinel set to 0.0). Evidence:
`bench/results/knn_speed_2026-09-11/`.

| lane | dataset | opponent | ms | quality | ours IDENTICAL ms | quality | ours / opponent |
|---|---|---|---:|---|---:|---|---:|
| knn | taxi | cuML NearestNeighbors brute | 8.19 (8.17..8.60) | recall@10 0.99925 | 24.79 (24.41..25.25) | recall@10 0.99915 | 3.03x |
| knn | Istella-S | cuML NearestNeighbors brute | 51.11 (median of 4 races, 50.51..52.13) | recall@10 0.92205 | 124.71 (median of 2 races, 119.93..128.71) | recall@10 0.923025 | 2.44x |
| knn | dyadic-v1 400k x 4k x d32, k10 | cuML NearestNeighbors brute (`tools/knn_cuml_reference.py`, request) | 10.05 | 4000 of 4000 rows ordered-equal | 23.66 | same | 2.36x |
| knn | dyadic-v1, k15 | same | 10.22 | 4000 of 4000 rows ordered-equal | 26.21 | same | 2.57x |

### kNN on taxi and Istella-S against cuML, same pod (September 11, knn-finish lane, pod zwmta1li2twxx2)

A NEW TUPLE: NVIDIA H200 143,771 MiB, driver 580.159.03, cuML 26.8.0. The
H100 80GB HBM3 and H100 NVL pools were empty, so these are H200 rows and are
never mixed with the H100 rows above. Same harness and shape as that section
(`race --lane knn`, index 400,000 x queries 4,000, k10, 1 warm-up plus 5
interleaved rounds); ours is the IDENTICAL arm at DEVIATION 2631's flipped
query tile. Evidence: `bench/results/knn_finish_2026-09-11/`.

| lane | dataset | opponent | opponent ms | opponent quality | ours ms | ours quality | ratio |
|---|---|---|---:|---|---:|---|---:|
| knn | taxi | cuML NearestNeighbors brute | 8.90 (8.56..9.23) | recall@10 0.99925 | 19.92 (19.79..20.37) | recall@10 0.99915 | 2.24x |
| knn | Istella-S | cuML NearestNeighbors brute | 52.10 (51.61..132.92) | recall@10 0.92205 | 95.51 (94.62..96.01) | recall@10 0.923025 | 1.83x |

Ours before DEVIATION 2631 on the same pod, in the same races, was 22.14 ms
(taxi) and 100.29 ms (Istella-S), so the ratios there were 2.49x and 1.93x.
### H100 forests same-pod baseline, taxi and Istella-S (2026-09-11 night, lane forest-speed)

Pod f2zzlf4dj4it1p (verified gone, HTTP 404), NVIDIA H100 80GB HBM3, driver
580.126.09, GPU-539422f5-3a31-9491-3564-da52a8f316de, runpod/pytorch:2.4.0-py3.11-cuda12.4.1
container, Intel Xeon Platinum 8480+ (224 logical CPUs visible, cgroup quota
23.8 CPUs, joblib cpu_count 24). Source 36ca51fd (main), IDENTICAL tier, the
setup-built bindings plus the svm extension. Same process, arms alternating,
1 warm-up plus 3 rounds, ms median (min..max), 1,000,000 training rows,
logloss / AUC on the fixed test tail. Configs as the Istella-S rows above
(100 trees, depth 16, sqrt features, 128 bins for RF, bootstrap for RF only,
seed 7). Evidence: `bench/results/forest_speed_2026-09-11/`.

| family | dataset | opponent | device | opponent ms | opponent logloss / AUC | ours IDENTICAL ms | ours logloss / AUC | ours / opponent |
|---|---|---|---|---|---|---|---|---|
| RF | taxi | cuML RandomForestClassifier 26.08.00 | GPU, H100 | 1980 (1942..1984) | 0.525800 / 0.617582 | 877 (876..914), hash d8f64dae01de00bd | 0.525910 / 0.617154 | 0.44x |
| RF | Istella-S | cuML RandomForestClassifier 26.08.00 | GPU, H100 | 3668 (3633..3855) | 0.145504 / 0.964577 | 2113 (2100..2170), hash 574b24d0d7af51d0 | 0.145560 / 0.964538 | 0.58x |
| ET | taxi | scikit-learn ExtraTreesClassifier 1.9.1 (cuML has none) | CPU, 24-core quota, n_jobs=-1 | 4075 (3799..4179) | 0.527011 / 0.611206 | 2058 (2015..2115), hash e683f121d11f59dd | 0.527541 / 0.608084 | 0.51x |
| ET | Istella-S | scikit-learn ExtraTreesClassifier 1.9.1 | CPU, 24-core quota | 16259 (15756..16302) | 0.187901 / 0.939528 | 6073 (5977..6208), hash 40b1c5b03ba40420 | 0.188191 / 0.938768 | 0.37x |
| IsolationForest | taxi | cuML IsolationForest 26.08.00 (`cuml.ensemble.isolation_forest`) | GPU, H100 | 63 (59..125) | proxy AUC 0.553631 | REFUSED by the harness (below) | - | owed |
| IsolationForest | Istella-S | cuML IsolationForest 26.08.00 | GPU, H100 | 1526 (1477..1911) | proxy AUC 0.821218 | REFUSED by the harness | - | owed |

cuML 26.08.00 does ship IsolationForest; the Aug 28 `cuml-iforest-gpu` row
(85.159 ms) was that estimator against our arm labeled FAST. Our iforest arm
was refused here by `verify_our_arm` ("native compiled-mode readback missing
for _mojolearn": the class inherits `_BINDING = "_mojolearn"` while its code
is in `_mojolearn_svm`), fixed on lane/forest-speed and not re-run. The
iforest AUC is a proxy (anomaly score against the minority class; neither
dataset has planted anomalies). Every ours hash held in 3 of 3 rounds; RF
and ET hashes equal the AMD MI325X and MI300X rows above (and the Sep 11
H100 Istella-S rows), so these four forests are cross-vendor identical on
both datasets. `identity_break` rf-clf, rf-reg, et-clf, et-reg, iforest: 45
of 45 cells stable, and the forest lanes IDENTICAL against the Sep 11 H100
confirmation set. Setup's `check_buffer_foreign_argtypes.py --real-cuml`
exited 0 on this pod (the 0.8.1 check owed on NVIDIA).

### H100 forests BEFORE and AFTER DEVIATIONS 2637 and 2638, taxi and Istella-S (2026-09-11 night, lane forest-finish)

Pod `8gsem9f3thnhvu` (NVIDIA H100 80GB HBM3, driver 580.126.09,
`GPU-b645d4a6-3c99-a75a-6492-0b26a9929031`, Intel Xeon Platinum 8470, 208
logical CPUs, cgroup quota 22.1 CPUs, joblib `cpu_count` 23),
runpod/pytorch:2.4.0-py3.11-cuda12.4.1 container, cuML 26.08.00, scikit-learn
1.9.1, numpy 2.4.6. IDENTICAL tier. BEFORE is main `4dc4346a` built in a SECOND
CHECKOUT ON THIS POD; AFTER is lane/forest-finish `600dcec0` (DEVIATIONS 2637
and 2638). 1,000,000 training rows, 100 trees, depth 16, sqrt features, 128
bins for RF, bootstrap for RF only, seed 7. One warm-up plus 3 rounds, arms
alternating in one process, ms median (min..max). Each opponent column is the
run that carried the AFTER arm, so ours/opponent is a same-process ratio.
Evidence: `bench/results/forest_finish_2026-09-11/`.

| family | dataset | opponent | device | opponent ms | ours BEFORE ms | ours AFTER ms | after/before | ours/opponent AFTER | quality ours (before = after) | opponent quality |
|---|---|---|---|---|---|---|---|---|---|---|
| RF | taxi | cuML RandomForestClassifier 26.08.00 | GPU, H100 | 1860 (1856..1938) | 826 (822..850) | 811 (806..817) | 0.98 | 0.44x | logloss 0.525910 / AUC 0.617154 | 0.525800 / 0.617582 |
| RF | Istella-S | cuML RandomForestClassifier 26.08.00 | GPU, H100 | 3885 (3735..3943) | 1977 (1972..2040) | 1333 (1323..1346) | 0.67 | 0.34x | logloss 0.145560 / AUC 0.964538 | 0.145504 / 0.964577 |
| ET | taxi | scikit-learn ExtraTreesClassifier 1.9.1 (cuML has none) | CPU, 23-core quota | 3295 (3254..3310) | 1894 (1883..1937) | 1831 (1830..1904) | 0.97 | 0.56x | logloss 0.527541 / AUC 0.608084 | 0.527011 / 0.611206 |
| ET | Istella-S | scikit-learn ExtraTreesClassifier 1.9.1 | CPU, 23-core quota | 15502 (15428..15688) | 5782 (5693..5816) | 4947 (4915..4961) | 0.86 | 0.32x | logloss 0.188191 / AUC 0.938768 | 0.187901 / 0.939528 |
| IsolationForest | taxi | cuML IsolationForest 26.08.00 | GPU, H100 | 54.4 (48.9..55.4) | 290 (277..344) | 95.4 (94.3..97.7) | 0.33 | 1.75x | proxy AUC 0.553631 | 0.553631 |
| IsolationForest | Istella-S | cuML IsolationForest 26.08.00 | GPU, H100 | 1001 (999..1062) | 4456 (4409..4629) | 155 (154..162) | 0.035 | 0.16x | proxy AUC 0.821218 | 0.821218 |

Section 9 geometric means of after/before over the two datasets: RF 0.81, ET
0.91, IsolationForest 0.11.
BOTH PASSES, because one pass is one measurement. Each cell was run twice: an
interleaved pass against the opponent (the table above) and an ours-only pass
in the reverse set order (ABBA). after/before per pass, interleaved then
ours-only: RF taxi 0.98 / 0.93, RF Istella-S 0.67 / 0.60, ET taxi 0.97 / 1.03,
ET Istella-S 0.86 / 0.86, iforest taxi 0.33 / 0.36, iforest Istella-S
0.035 / 0.034. Every cell holds its hash in both passes.

ET ON TAXI IS FLAT, AND THE TWO PASSES DISAGREE ON ITS SIGN (0.97 against
1.03), so it is noise around 1 and not a win: 16 columns of staging is not
where a taxi ExtraTrees fit spends its time. ET's geometric mean is below 1
either way (0.91 with the interleaved pass, 0.94 with the ours-only one)
because Istella-S carries it at 0.86 in both. RF's taxi cell is the same story
one notch milder (0.98 / 0.93).
 Quality is BYTE-EQUAL before and after in every
cell, so all three flip; 2637 and 2638 are the shipped path on this branch
rather than opt-in switches, and these rows are what keeps them.

THE ISOLATION FOREST ROWS ARE THIS LANE'S FIRST for our arm. The Sep 11
baseline leg could not measure it at all -- `verify_our_arm` asked
`_mojolearn` for the mode of a class whose code lives in `_mojolearn_svm` --
so the section above carries "owed" in both cells and this section replaces
them. cuML's own row moved between the two runs (taxi 56.0 then 54.4,
Istella-S 828 then 1001), which is why every ratio here is quoted against the
opponent measured in the SAME process as the arm it is compared with.

Every model hash held one value in 3 of 3 rounds and is EQUAL before and
after: RF taxi `d8f64dae01de00bd`, RF Istella-S `574b24d0d7af51d0`, ET taxi
`e683f121d11f59dd`, ET Istella-S `40b1c5b03ba40420`, iforest taxi
`6f68d48431290524`, iforest Istella-S `a1902225f8730abf`. `identity_break`
rf-clf, rf-reg, et-clf, et-reg, iforest read 45 of 45 cells stable on EACH
set, with no DIVERGENT, MOVED or REFUSED row in the diff; `check-if` passed
under IDENTICAL; the non-finite refusal raised through the Python surface is
byte-equal between the sets for C-order and F-order input alike.

The iforest proxy AUC is an anomaly score against the minority class; neither
dataset has planted anomalies, so it is a sanity column and not a benchmark.

### Classical OLS, PCA, k-means and DBSCAN on both datasets (September 11, pod 1yxsotvvcbxtuu, lane linear-cluster-istella)

RunPod H100 80GB HBM3, driver 570.195.03, cuML 26.08 (cuml-cu12 26.8.0) in the
image's Python 3.11; ours IDENTICAL built on the pod (sm_90a) from this lane and
from origin/main 8dc33f00 (which carries DEVIATIONS 2632 and 2633), both trees on
the SAME pod, raced interleaved. A driver below 580 cannot take Mojo's own PTX
path, so every run here exports `MODULAR_NVPTX_COMPILER_PATH` at a CUDA 12.8
`ptxas` (`nvidia-cuda-nvcc-cu12`); the image's own 12.4 ptxas refuses Mojo's PTX
8.5. `tools/classical_two_datasets.py race`, 1 warm-up plus 5 interleaved rounds,
ms median. Taxi 4,000,000 x 11 and Istella-S 2,043,304 x 220; k-means k 64, 20
iterations, shared init, tol 1e-7 on both arms; cuML PCA `svd_solver='full'`,
ours `covariance_eigh`; cuML OLS `algorithm='eig'`.

| lane | dataset | opponent | opponent ms | opponent quality | ours before (8dc33f00) ms | ours after (2671, 2672) ms | after / before | ours after / opponent | ours quality, digest before = after |
|---|---|---|---|---|---|---|---|---|---|
| ols | taxi | cuML LinearRegression eig | 21.67 | R2 0.90883616 | 139.65 | 135.76 | 0.972 | 6.26x | R2 0.90883698, `fb86358654367fa0` |
| ols | Istella-S | cuML LinearRegression eig | 84.85 | R2 -6473.68 | 2529.06 | 2402.44 | 0.950 | 28.31x | R2 0.33194438, one digest, equal |
| pca | taxi | cuML PCA full | 19.54 | EVR sum 0.99786046 | 28.46 | 26.32 | 0.925 | 1.35x | EVR sum 0.99786071, `c790338770a4c120` |
| pca | Istella-S | cuML PCA full | 81.91 | EVR sum 1.0000000156 | 713.09 | 686.27 | 0.962 | 8.38x | EVR sum 1.0000000146, one digest, equal |
| kmeans | taxi | cuML KMeans | 129.28 | inertia 1.2019161e8, 20 iter, digest different every round | 216.59 | 211.99 | 0.979 | 1.64x | inertia 1.2062766e8, 21 iter, `89520efe99a08d5f` |
| kmeans | Istella-S | cuML KMeans | 171.81 | inertia 1.2855462e17, 20 iter, digest different every round | 1267.6 (pooled median of 5 instances) | 1260.4 (pooled) | 0.994 | 7.56x | inertia 1.3128483e17, 21 iter, `7f720b0b76896308` |

OLS and PCA reach the device Jacobi and k-means does not, so those rows isolate
DEVIATION 2671 (two barriers per rotation instead of four, same bits) and the
k-means row isolates DEVIATION 2672 (host staging removed from the k-means fit).
Flip verdicts under ENGINEERING_RULES section 9: OLS geomean 0.9610, PCA geomean
0.9434, and k-means geomean 0.9449 over pooled race instances, all below 1 with
quality equal and bits equal on both datasets, so BOTH deviations are the
default.

The k-means numbers are pooled because ONE race instance is not enough at this
shape: the Istella-S A/B read 1.066, 0.950, 0.9574, 1.0171 and 0.9986 across
five instances whose rounds were each tight, so the spread lives between race
instances (about 80 ms) and swamps a change the stage probe sizes at about 5 ms
of host work. Pooled medians of instances are taxi 183.5 after against 204.4
before (0.8979, four instances) and Istella-S 1260.4 against 1267.6 (0.9943,
five instances). The taxi instances that race only ours against ours-base run
faster than the three-arm race in the table above (181 to 185 ms against 212),
which is the cuML worker's own GPU work showing up in its neighbours.

**Where the Istella-S time goes** (stage probe, IDENTICAL, same pod): the OLS
device solve is 1744 to 1775 ms, of which the 220-column Jacobi is 1656 ms at 12
sweeps and the Gram is 23.6 ms; PCA is upload 34.9, covariance 30.0 and Jacobi
418.3 at 3 sweeps. On the same matrices the four-phase kernel takes 1825.8 and
466.6 ms, and the two kernels agree at every one of 48,400 matrix cells and
48,400 eigenvector cells.

#### DBSCAN and HDBSCAN at 1,000,000 rows (first rows for this pair)

Blocks are 1,000,000 rows of each dataset's train split, sentinel cleaned and
standardized by their own float64 mean and standard deviation, so one eps means
the same thing on every column. eps is the p75 quantile of the min_samples-th
nearest neighbor distance measured on the block itself (`dbscan_eps.py`), the
SAME value for every arm, min_samples 10. Ours runs its default ball cover;
`cuml-gpu` is cuML's default brute force. 1 warm-up plus 3 rounds.

| lane | dataset | eps | opponent | opponent ms | ours IDENTICAL ms | ours / opponent | agreement |
|---|---|---|---|---|---|---|---|
| dbscan | taxi (11 features) | 0.177 | cuML DBSCAN brute | 13631.0 | 1129.8 | 0.083x | 2900 clusters and noise fraction 0.2161 on both arms, adjusted Rand index 0.99999999908, noise agreement 1.000 |
| dbscan | Istella-S (220 features) | 4.17 | cuML DBSCAN brute | pending | pending | | |
| hdbscan | taxi, 100,000 rows | - | cuML HDBSCAN (min_samples 10, min_cluster_size 100, eom) | 191.2 | no arm | - | 159 clusters, noise fraction 0.1310; this library ships no HDBSCAN |

cuML's `algorithm='rbc'` arm refused both datasets at this size ("An overflow
occurred with the current choice of precision and the number of samples"), so
cuML's own ball cover has no row here.
### H100 ExtraTrees frontier batch width, DEVIATION 2663 (2026-09-11 night, lane forest-finish)

Same pod as the section above (`8gsem9f3thnhvu`, NVIDIA H100 80GB HBM3, driver
580.126.09), IDENTICAL tier, 1,000,000 training rows, 100 trees, depth 16,
seed 7. THIS IS OURS AGAINST OURS, not an opponent race: `ctl` is this lane's
build at cuML's shipped `max_batch_size` of 4096 (`decisiontree.hpp:86-95`) and
the trial sets are the same source built with
`-D MOJOLEARN_ET_DEVICE_BATCH_16384=1` and `_32768=1`. Two passes per cell in
rotated order, 3 rounds each (2 for istellareg), pooled medians. The opponent
rows for these datasets are in the section above and did not change.

| cell | columns sampled | ctl (4096) ms | 16384 ms | ratio | 32768 ms | ratio | hash (all widths) |
|---|---|---|---|---|---|---|---|
| et taxi | 4 of 16 | 1907.7 | 1616.3 | 0.847 | 1587.3 | 0.832 | `e683f121d11f59dd` |
| et Istella-S | 14 of 220 | 5026.6 | 4596.3 | 0.914 | 4519.0 | 0.899 | `40b1c5b03ba40420` |
| et taxireg | 11 of 11 | 5350.0 | 5292.7 | 0.989 | 5285.8 | 0.988 | `9844a40ba74bc375` |
| et istellareg | 220 of 220 | 67051.0 | 67476.6 | 1.006 | 67293.4 | 1.004 | `59547e3a9db9ecd8` |

Geometric means: the classification pair 0.880 / 0.865, ALL FOUR CELLS 0.9371
at 16384 and 0.9280 at 32768, with quality equal in every cell (one model hash
per cell across all three widths; `flip_verdict` deltas +0.000000). 16384 IS
NOW THE DEFAULT; 32768 stays a measurement arm, because it wins by 0.97
percent -- inside this lane's pre-registered 2 percent margin -- while doubling
the level workspace (about 800 MB at 220 columns against 400 MB, and 100 MB at
4096) on every vendor, for time measured on an 80 GB H100.
`-D MOJOLEARN_ET_DEVICE_BATCH_4096=1` restores cuML's width for an A/B.

THE ISTELLAREG CELL IS A LOSS, about half a percent at both widths, and it is
reported rather than averaged away. The switch reaches four cells because the
ExtraTrees regressor takes the same plan, and section 9 decides it on the
geometric mean over all of them.

WHY IT PAYS, AND ONLY ON CLASSIFICATION. The width was widened on the theory
that a 4096-node frontier runs many level cycles, each ending in a drain and a
host pass. `-D MOJOLEARN_ET_CYCLE_STATS=1` refuted that: a 100-tree Istella-S
forest runs 266 cycles (67 trees then 33, DEVIATION 211's grouping), 3,836
nodes per cycle, 94 percent of capacity, and the whole host family is about 3
percent of the loop. What pays is DEVIATION 205's rescue. Taxi samples 4
columns of 16, so 5.0 percent of its nodes draw an all-constant sample
(Istella-S, sampling 14 of 220, sees 0.8 percent), and EVERY cycle carrying a
retry runs two extra staged sub-batches -- the survey over all columns, then
the k=1 rescue -- each restaging and draining. That cost scales with CYCLES, so
a four-times wider batch pays it a quarter as often. At `max_features=1.0` a
rescue needs every column constant at once, so the regression cells are flat.

Identity: `identity_break` rf-clf, rf-reg, et-clf, et-reg, iforest read 45 of
45 cells stable at 4096, at 16384, at 32768 AND at the rebuilt default, every
diff against the lane build carrying 46 IDENTICAL rows with no DIVERGENT,
MOVED or REFUSED, and the ExtraTrees fingerprints equal at every width
(et-clf/base `c586b27a3b049614`, et-clf/wide `ee8b318d6bf698b2`, et-reg/base
`754d8c127ecfc04d`, et-reg/wide `b745e53515f59cac`). `device_batched_check`
PASSED with both sabotages moving thousands of nodes (scalar-tree 2688 / 2052
clf/reg, shared-row-base 2499 / 2598). The width is a scheduling parameter and
this is the evidence, not the argument.

The shipped default was rebuilt with NO defines and re-timed: et taxi 1617.9 ms
and et Istella-S 4634.7 ms, hashes `e683f121d11f59dd` and `40b1c5b03ba40420`,
logloss 0.527541 and 0.188191 -- where the trial width landed.

### GBDT host-work switches, before and after on one H100 (September 12, pod n2ltmel2optel5)

OURS AGAINST OURS, not an opponent row: each pair differs by exactly one
compile define and nothing else, which the four distinct `_mojolearn_gbdt.so`
prove (baseline 069feb45, a2634 0d12ac1d, both 20633cd7, all 8c20150a). NVIDIA
H100 80GB HBM3, driver 580.126.09, 224 host cores; our IDENTICAL arm only,
1,000,000 rows, 100 trees, depth 6, 5 timed fits per process, 3 rounds
alternating PROCESSES over the sets (two `.so` sets cannot share a process, and
the order rotates each round so a set is never compared across heat windows).
Quality is equal in all 24 cells and no model bit moves in any of them.
Evidence `bench/results/gbdt_finish_2026-09-11/` and
`~/mojolearn-evidence/gbdt-redo-2026-09-12/`.

Medians in ms, by grow policy:

| chain step | switch | sym taxi | dep taxi | loss taxi | sym Istella-S | dep Istella-S | loss Istella-S |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | none | 341.1 | 462.9 | 999.0 | 1370.5 | 1924.6 | 2539.4 |
| a2634 | 2634 on | 312.6 | 434.2 | 962.5 | 1303.8 | 1912.4 | 2544.8 |
| both | 2634, 2635 on | 311.3 | 434.2 | 965.1 | 1309.1 | 1840.5 | 2502.8 |
| all | 2634, 2635, 2636 on | 303.4 | 438.2 | 938.3 | 1341.5 | 1867.7 | 2526.1 |

Per-switch after/before and the section 9 verdict (a shared switch has ONE
default, gated on the geometric mean over all six (policy, dataset) cells):

| switch | pair | sym/dep/loss taxi | sym/dep/loss Istella-S | six-cell geomean | verdict |
|---|---|---|---|---:|---|
| 2634 | baseline -> a2634 | 0.916 / 0.938 / 0.964 | 0.951 / 0.994 / 1.002 | 0.9603 | FLIP, ships on |
| 2635 | a2634 -> both | 0.996 / 1.000 / 1.003 | 1.004 / 0.962 / 0.984 | 0.9913 | FLIP, ships on |
| 2636 | both -> all | 0.975 / 1.009 / 0.972 | 1.025 / 1.015 / 1.009 | 1.0006 | NO FLIP, now OPT-IN |
| total | baseline -> all | 0.889 / 0.947 / 0.939 | 0.979 / 0.970 / 0.995 | 0.9526 | FLIP |

DEVIATION 2636 is the one that lost, and the shape says why: it parallelizes the
host fills of a ring revolution, but the uploads and binarize kernels still
enqueue serially in the same order, so the device stays the bottleneck and the
880 MB of single-thread memcpy it removes was never on the critical path.
gbdt-depthwise is a clear 1.012 loss. 2634 and 2635 carry the win without it,
0.9603 x 0.9913 being essentially all of the 0.9526 measured end to end, so
2636 moved to `-D MOJOLEARN_2636_PARALLEL_STAGING=1` (commit 31f0142f).

DEVIATION 2661 (non-symmetric group width), the same pod's phase 2, `all` ->
`a2661`, 3 rounds, its own heat window (so its `all` medians differ slightly
from the table above, which is why each pair is read only within its phase):

| policy | dataset | before ms | after ms | after/before |
|---|---|---:|---:|---:|
| symmetric | taxi | 305.1 | 305.8 | 1.0023 |
| depthwise | taxi | 430.3 | 429.8 | 0.9988 |
| lossguide | taxi | 949.2 | 939.5 | 0.9898 |
| symmetric | Istella-S | 1291.0 | 1288.3 | 0.9979 |
| depthwise | Istella-S | 1873.7 | 1849.5 | 0.9871 |
| lossguide | Istella-S | 2501.0 | 2513.8 | 1.0051 |

Six-cell geomean 0.9968, quality equal in every cell, and it moves no bit
(`ib_diff all vs a2661` IDENTICAL=36, `gbdt_sub_byte_identity_check` 16/16
PASS). It CLEARS the section 9 threshold, and it STAYS OPT-IN anyway, for two
reasons worth writing down rather than flipping on a technicality. The margin
is 0.3 percent with a mixed per-policy split (gbdt-symmetric is NO FLIP at
1.000, only depthwise 0.993 and lossguide 0.997 win), and the base it was
measured against is no longer the shipped default: this pod's `all` set was
built before DEVIATION 2636 became opt-in, so 2661 was timed with 2636 ON
underneath. Re-measure it against the new default before making it one.

### GBDT opponents on this tuple (September 12, pod u4elzj1eo486ps)

The row above was RUN OWED because I reaped the first pod ~20 minutes before
these cells finished; a second pod (below) repaid the whole setup to get them.
ENGINEERING_RULES 11 exists because of that.

NVIDIA H100 80GB HBM3, driver 580.126.09, 81,559 MiB, Xeon Platinum 8480+ (224
cores); CatBoost 1.2.10, XGBoost 3.2.0, scikit-learn 1.9.1, NumPy 2.4.6. Our
IDENTICAL arm against their FAST arms, 1,000,000 rows, 100 trees, depth 6, 5
interleaved rounds in one process per cell, medians (min..max). Ours is the
`all` set, which after commit 31f0142f means DEVIATIONS 2634 and 2635 on with
2636 opt-in, i.e. the shipped default. XGBoost has no oblivious grower, so it
has no gbdt-symmetric cell. Evidence
`~/mojolearn-evidence/gbdt-redo-2026-09-12/opp_evidence.tgz`.

| policy | dataset | ours ms | CatBoost GPU ms | ours/CB | XGBoost GPU ms | ours/XGB |
|---|---|---:|---:|---:|---:|---:|
| symmetric | taxi | 310.1 (307.8..325.6) | 709.0 (676.3..734.1) | **0.437x** | - | - |
| symmetric | Istella-S | 1288.9 (1243.1..1302.2) | 1553.4 (1503.7..1690.8) | **0.830x** | - | - |
| depthwise | taxi | 436.6 (433.4..470.6) | 842.4 (831.2..922.3) | **0.518x** | 365.1 (351.9..378.5) | 1.196x |
| depthwise | Istella-S | 1838.2 (1813.1..1894.7) | 1784.8 (1734.0..1822.3) | 1.030x | 1769.2 (1679.7..1861.0) | 1.039x |
| lossguide | taxi | 950.0 (945.2..1043.9) | 1094.6 (1091.8..1144.3) | **0.868x** | 476.2 (469.6..491.2) | 1.995x |
| lossguide | Istella-S | 2504.3 (2463.8..2782.2) | 2510.8 (2426.0..2608.0) | 0.997x | 1994.7 (1946.5..2516.4) | 1.256x |

Quality, same fits: symmetric ours logloss 0.138653 / AUC 0.966990 against
CatBoost 0.139154 / 0.966719 (ours very slightly better on both); depthwise
ours 0.126517 / 0.971896, CatBoost 0.125832 / 0.972819, XGBoost 0.125110 /
0.973774; lossguide ours 0.122045 / 0.975037, CatBoost 0.121096 / 0.975462,
XGBoost 0.125110 / 0.973774.

READ THIS HONESTLY. We beat CatBoost on four of its six cells, and the two
symmetric cells are the strongest (0.437x on taxi, 0.830x on Istella-S) on
CatBoost's OWN policy. On Istella-S depthwise we are 1.030x, slightly behind,
and on Istella-S lossguide 0.997x is parity, not a win. XGBoost is FASTER THAN
US wherever it competes: 1.196x and 1.039x on depthwise, and 1.995x on taxi
lossguide, which is the largest single gap on the board. Quality is within a
thousandth of both opponents everywhere, ahead of CatBoost on symmetric and
marginally behind on the other two policies.

The identity column no opponent has: ours returned ONE model hash per cell
across all 5 rounds (symmetric Istella-S `238d3abce0cabf43`), while CatBoost
returned a DIFFERENT hash in every round of the same cell (`1827fc2260f91628`,
`0169524ba0e5364e`, `6c122f43cbd65ad0`, `725dc8116ae6e6cd`, `b38f6ba81e7fa984`).

### GBDT on criteo, the CATEGORICAL set: our CTR path measured for the first time (September 12, pod 0zl2hrxqq26b0t)

criteo is NOT a section 9 gating dataset. taxi and Istella-S remain the two
kinds every flip verdict is computed over; criteo exists to reach the
categorical and CTR code, which neither of them touches. A number here alone is
a one-kind number.

WHY IT IS THE FIRST TIME. `cat_features` reached no benchmark before
2026-09-12, so DEVIATION 2634 ("skip the CTR target prep when no column is
categorical") had only ever executed its SKIP branch. Its 0.9603 flip verdict
was measured on taxi and Istella-S, neither of which declares a categorical
column. Everything below is therefore a NEW measurement, not a confirmation.

NVIDIA H100 80GB HBM3, driver 580.126.09; CatBoost 1.2.10; our IDENTICAL arm;
1,000,000 train rows of criteo (13 integer + 26 hashed categorical, 3.22%
positive), 100 trees, depth 6, 3 rounds, medians. Category codes re-ranked
WITHIN the train slice (see the density note below).

| policy | arm | median ms | AUC | logloss | ours / CatBoost |
|---|---|---:|---|---|---:|
| SymmetricTree | ours (categorical) | 11,913 | 0.742155 | 0.128135 | 1.21x |
| SymmetricTree | CatBoost GPU | 9,883 | 0.741531 | 0.128211 | - |
| SymmetricTree | ours-nocat (codes as numbers) | 425 | 0.723855 | 0.130338 | - |

Read it honestly: on the CTR path we are 1.21x CatBoost's time, and slightly
AHEAD of it on both quality figures (AUC 0.742155 against 0.741531, logloss
0.128135 against 0.128211). The `ours-nocat` row is the more useful one: the
same 26 columns split as ORDERED NUMBERS run 28x faster (425 ms) and score
materially worse (AUC 0.7239), which both prices the categorical path and
proves it is genuinely reached -- disabling it moves the time by 28x and the
quality by 0.018 AUC.

THE DECODE BUG THIS EXPOSED, and the refusal that caught it.
`bench/speed/forest_speed_arm.py` cannot fit criteo on our arm at all as the
loader first shipped: `gbdt/train.mojo:1231` refuses by name,

    cat_features column 13 is not densely coded: category 1 is absent from 0..621909

and the refusal is CORRECT. `_decode_criteo` ranked each category over every
decoded row, then `load_criteo` fits a row SLICE, so any category living only
in withheld rows is a hole in the codes that reach fit.
`tools/criteo_density_audit.py` counts it rather than arguing about it:

| slice | rows | non-dense columns | missing codes | worst column |
|---|---:|---:|---:|---:|
| all decoded rows | 3,061,005 | 0 of 26 | 0 | 0 |
| train uncapped | 2,561,005 | 19 of 26 | 386,091 | 100,064 |
| train 1,000,000 | 1,000,000 | 21 of 26 | 1,702,570 | 429,601 |
| train 200,000 | 200,000 | 22 of 26 | 2,592,446 | 628,467 |
| test tail | 500,000 | 22 of 26 | 2,163,064 | 537,863 |

Dense over the matrix and dense over NO slice of it, the uncapped train split
included -- so this is a property of the decode's global ranking, not a
`--rows` artifact. Global sorted-unique ranking was chosen to make codes
reproducible between runs; it does not make them dense in the slice that
reaches fit, which is what our surface requires. The numbers above come from
`tools/criteo_dense_arms.py`, which re-ranks within the train slice.

A second wiring bug, same lane: `xgboost_arms._frame` built the pandas category
dtype independently for fit and for predict, so XGBoost 3.2.0 raised while
scoring ("Found a category not in the training set for the 32th column:
679286"). Timings survived, quality did not, which is why the XGBoost criteo
cells carry times and no quality. Both frames must share one
`CategoricalDtype(range(k))`.

The refusal probes behaved as designed: `feature_fraction < 1` with
cat_features, `permutation_count` without cat_features, and
`ctr_estimation_permutation_id` without cat_features all raised BY NAME, while
`cat_features` alone was accepted.
