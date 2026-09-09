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

Extra trees therefore has NO valid NVIDIA opponent row. That measurement
is owed (a LightGBM build with USE_CUDA, or cuML RF with `split_criterion`
random thresholds if cuML admits it).

Accuracy alongside the timing (CatBoost GPU symmetric, HIGGS 1M): logloss
0.542398, AUC 0.800529, from the same logs (`FSPEED-ACC` lines).

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
them. k 15 rows are within 2% of k 10 (full table in the handoff).

| index | queries | k | cuML brute NearestNeighbors request | device |
|---:|---:|---:|---:|---:|
| 100k x 32 | 32 | 10 | 1.125 | 0.669 |
| 100k x 32 | 128 | 10 | 0.914 | 0.459 |
| 100k x 32 | 1000 | 10 | 1.572 | 1.081 |
| 100k x 32 | 4000 | 10 | 3.480 | 2.904 |
| 400k x 32 | 32 | 10 | 1.546 | 1.200 |
| 400k x 32 | 128 | 10 | 1.531 | 1.159 |
| 400k x 32 | 1000 | 10 | 4.101 | 3.659 |
| 400k x 32 | 4000 | 10 | 10.225 | 9.632 |

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

## Rows that do not exist yet (owed, in priority order)

1. LightGBM CUDA extra_trees, valid build, HIGGS 1M/2M/5M (the Sep 9 trees
   lane built USE_CUDA=ON but its ET rungs never ran; its rf-boosting row
   at 470 s per fit is not an ET row).
1b. Trees at 2M and 5M for the Sep 9 same-process protocol (only 1M exists).
2. (closed 2026-09-09: the 20 H100 cuBLAS rows above were transcribed
   from the Aug 25 logs by the GEMM split-K lane; no re-run.)
3. cuML DBSCAN and PCA at 1M rows or more (the fixtures above are small).
4. cuML UMAP on the H100 at 1M rows, and on the L40S at 20k and 100k rows
   (the H100 has 20k and 100k, the L40S has 1M; ours IDENTICAL was measured
   on the L40S at all three on 2026-09-09, so no same-GPU ratio exists yet
   below 1M).
5. torch byte-LM training step time on H100 (the Sep 7 comparison in this
   tree is a correctness record with no torch timing).

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
