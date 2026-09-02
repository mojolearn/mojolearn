# Speed summary, 2026-08-29. Every timing comparison on disk, per lane, per vendor

Compiled 2026-08-29 from the boards already under `bench/results/`. Nothing
here was rerun. Every number carries the file it came from and the commit or
date that file records. Where a file and a newer file disagree on one cell,
both are shown and the newer one is marked current.

## 1. How to read this

Three vendor columns. Apple M4 (the MacBook Pro on the desk, 10 cores, 16 GB),
NVIDIA H100 80GB SXM (RunPod), AMD Instinct MI325X (DigitalOcean `tor1`).
The peer on NVIDIA and AMD is the competitor's own GPU arm and nothing else
(cuML, cuVS, CatBoost-GPU, XGBoost-GPU, LightGBM-CUDA, cuBLAS, hipBLASLt,
torch on CUDA or ROCm). The harness refuses the CPU arms on those boxes by
name (`GPU-PATH-ONLY` in every refusal table). On Apple the peer is the CPU
arm of the same library, because CatBoost's `task_type="GPU"` raises on Apple
silicon, LightGBM ships no Metal learner, and cuML and cuVS are CUDA-only
(`bench/results/fast_speed/2026-08-26-APPLE-trees.md`). The one exception the
boards record is the gemm and sequence lanes, where torch on MPS is a real
Apple GPU peer and is used as such
(`bench/results/fast_speed/2026-08-28-APPLE-gemmseq.md`). Every AMD row is our
arm alone, because no peer library runs on the MI325X
(`bench/results/BOARD_2026-08-28_three-vendor.md` section 2.3).

Two tiers. FAST is the default build and promises speed only. IDENTICAL is
the `-D MOJOLEARN_NUMERIC_IDENTICAL=1` build and promises the same bits on
three vendors. Every `fast_speed/` board is FAST on both sides; every leg
records `ours_headers_identical=0`. The IDENTICAL timings live in
`bench/results/SPEED_LANE_2026-08-25.md` and `bench/results/lanes_price/`, and
are labeled IDENTICAL wherever they appear below. Boards older than
2026-08-23 predate the tier split and do not record a mode; they are labeled
"mode not recorded". A row never mixes tiers without saying so.

`ratio` is median(ours) over median(theirs) as the boards compute it, restated
here as "Nx faster" or "Nx slower". Medians are over the timed rounds after
one untimed warm-up, ours and the peer alternating inside one process. On the
M4 that alternation is the only defense against the thermal drift that file
after file records (section 5).

## 2. Per family, per vendor, from the boards

Column key. ours ms and peer ms are medians. tier is the mode of OUR arm. src
is the board and the commit or date it records.

### 2.1 Gradient boosting, symmetric trees (CatBoost pair only)

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| higgs 1,000,000 x 28, 5 rounds | 856.116 | catboost-gpu 864.690 | 1.01x faster | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-forest-A.md`, a8d838e, 2026-08-28 (current) |
| higgs 2,000,000 x 28, 5 rounds | 1827.444 | catboost-gpu 1379.246 | 1.32x slower | FAST | NVIDIA H100 | same |
| higgs 1,000,000 x 28, 5 rounds | 3666.967 (min 2622.429) | catboost-gpu 846.102 | 4.33x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28_030908-nvidia-forest.md`, 3d65d70, 2026-08-28 earlier leg; superseded by forest-A above, see section 5 |
| higgs 2,000,000 x 28, 5 rounds | 1304.338 | catboost-gpu 1227.208 | 1.06x slower | FAST | NVIDIA H100 | same |
| higgs 5,000,000 x 28, 5 rounds | 5227.542 | catboost-gpu 2458.494 | 2.13x slower | FAST | NVIDIA H100 | same (the only 5M symmetric row at a 2026-08-28 commit) |
| higgs 1M / 2M / 5M, 3 rounds | 668.142 / 1185.877 / 2760.228 | catboost-gpu 994.561 / 1312.121 / 2543.586 | 1.49x faster / 1.11x faster / 1.09x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-26-nvidia-forest-postfix.md`, 132d754, 2026-08-26 (older) |
| higgs 1M / 5M, 3 rounds | 636.590 / 2998.741 | catboost-gpu 1182.268 / 2687.290 | 1.86x faster / 1.12x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-26_040049-nvidia-trees-ladder.md`, c637e30, 2026-08-26 (older) |
| year 463,715 x 90, 3 rounds | 1931.296 | catboost-cpu 2197.476 | 1.14x faster | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-forest.md`, a8d838e, 2026-08-28 (current) |
| year 463,715 x 90, 3 rounds | 2005.700 | catboost-cpu 2411.320 | 1.20x faster | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-trees-evening.md`, 20729ea, 2026-08-27 (older) |
| higgs 1,000,000 x 28, 3 rounds | 4409.991 | catboost-cpu 5762.869 | 1.31x faster | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-trees-evening.md`, 20729ea, 2026-08-27 (current for this cell) |
| higgs 1,000,000 x 28, 3 rounds | 3762.358 | catboost-cpu 3806.136 | 1.01x faster, parity | FAST | Apple M4 | `fast_speed/2026-08-26-APPLE-HIGGS-1M.md`, 5b728e2, 2026-08-26 (older) |
| year, gbm-bench 500 rounds depth 8 | 11.95 s | cat-cpu 13.59 s | 1.14x faster | mode not recorded | Apple M4 | `THREE_SUITE_QUIET_2026-08-22.md`, dfa41bb, 2026-08-22; same row in `README.md` benchmark table |
| higgs 8.8M, gbm-bench 500 rounds | 131.5 s | cat-cpu 177.8 s | 1.35x faster, AUC 0.822 vs 0.830 open gap | mode not recorded | Apple M4 | same |
| synthclf 720,000 x 100, 10 rounds | 1041.722 | none | no peer on AMD | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest.md`, 88b918d, 2026-08-28; synthetic fallback fixture, see section 3 |
| higgs 1M / 2M | unrun | unrun | | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest-higgs.md`, 4f6a17a: our arm raised at warm-up (DEVIATION 1910, 64-lane half-byte histograms) |

### 2.2 Gradient boosting, depthwise

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| higgs 1M x 28, 5 rounds | 1382.644 | catboost-gpu 1139.637 | 1.21x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-forest-A.md`, a8d838e (current) |
| higgs 1M x 28, 5 rounds | 1382.644 | xgboost-gpu 743.058 | 1.86x slower | FAST | NVIDIA H100 | same |
| higgs 2M x 28, 5 rounds | 2483.338 | catboost-gpu 1496.789 | 1.66x slower | FAST | NVIDIA H100 | same |
| higgs 2M x 28, 5 rounds | 2483.338 | xgboost-gpu 1310.138 | 1.90x slower | FAST | NVIDIA H100 | same |
| higgs 5M x 28, 5 rounds | 5287.681 | catboost-gpu 2559.735 / xgboost-gpu 2165.906 | 2.07x / 2.44x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28_030908-nvidia-forest.md`, 3d65d70 (only 5M depthwise row) |
| higgs 1M / 2M, 3 rounds | 1316.996 / 3112.093 | xgboost-gpu 685.180 / 1141.118 | 1.92x / 2.73x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-27_171939-nvidia-depthwise.md`, bc2a3b4, 2026-08-27 (older) |
| year 463,715 x 90, 3 rounds | 4543.493 | catboost-cpu 4189.724 | 1.08x slower | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-forest.md`, a8d838e (current) |
| year 463,715 x 90, 3 rounds | 6224.937 | catboost-cpu 4880.259 | 1.28x slower | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-trees-evening.md`, 20729ea (older) |
| synthclf 720,000 x 100, 10 rounds | 1446.487 | none | no peer on AMD | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest.md`, 88b918d |
| higgs 1M / 2M | unrun | | | FAST | AMD MI325X | our arm raised at warm-up, DEVIATION 1910 |

### 2.3 Gradient boosting, lossguide (leaf-wise)

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| higgs 1M x 28, 5 rounds | 1751.063 | catboost-gpu 1533.086 / xgboost-gpu 771.236 | 1.14x / 2.27x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-forest-B.md`, a8d838e (current) |
| higgs 2M x 28, 5 rounds | 2690.113 | catboost-gpu 1864.952 / xgboost-gpu 1185.849 | 1.44x / 2.27x slower | FAST | NVIDIA H100 | same |
| higgs 1M x 28, 5 rounds | 1906.192 | lightgbm-cuda 1313.705 | 1.45x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28_030911-nvidia-forest.md`, 3d65d70 (the only leg where lightgbm-cuda ran) |
| higgs 2M x 28, 5 rounds | 2936.414 | lightgbm-cuda 1669.264 | 1.76x slower | FAST | NVIDIA H100 | same |
| higgs 5M x 28, 5 rounds | 6301.633 | catboost-gpu 2870.462 / xgboost-gpu 2413.373 | 2.20x / 2.61x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28_030908-nvidia-forest.md`, 3d65d70 (current 5M row) |
| higgs 5M x 28, 3 rounds | 14800.158 (min 6030.534) | xgboost-gpu 2647.444 | 5.59x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-27_031904-nvidia-forest.md`, 69f2503, 2026-08-27 (older; the 5M rung, section 5) |
| year 463,715 x 90, 3 rounds | 8660.290 | catboost-cpu 5894.057 / lightgbm-cpu 3886.938 | 1.47x / 2.23x slower | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-forest.md`, a8d838e (current) |
| year 463,715 x 90, 3 rounds | 10822.723 | catboost-cpu 6267.296 | 1.73x slower | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-trees-evening.md`, 20729ea (older) |
| synthclf 720,000 x 100, 10 rounds | 2437.570 | none | no peer on AMD | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest.md`, 88b918d |
| higgs 1M / 2M | unrun | | | FAST | AMD MI325X | our arm raised at warm-up, DEVIATION 1910 |

### 2.4 Random forest

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| higgs 1M x 28, 5 rounds | 5309.306 | cuml-rf-gpu 3177.072 | 1.67x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-forest-A.md`, a8d838e (current) |
| higgs 2M x 28, 5 rounds | 7308.012 | cuml-rf-gpu 4445.280 | 1.64x slower | FAST | NVIDIA H100 | same |
| higgs 5M x 28, 5 rounds | 12224.530 | cuml-rf-gpu 7284.710 | 1.68x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28_030908-nvidia-forest.md`, 3d65d70 |
| higgs 1M / 2M / 5M, 3 rounds | 6323.739 / 8286.882 / 14047.177 | cuml-rf-gpu 3486.530 / 4839.604 / 7598.814 | 1.81x / 1.71x / 1.85x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-27_031904-nvidia-forest.md`, 69f2503 (older) |
| higgs 1M / 5M, 3 rounds | 6188.417 / 14698.891 | cuml-rf-gpu 3079.464 / 7057.536 | 2.01x / 2.08x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-26_040049-nvidia-trees-ladder.md`, c637e30 (older) |
| covtype 522,911 x 54, 3 rounds | 15281.972 | sklearn-rf-cpu 12430.019 | 1.23x slower | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-forest.md`, a8d838e (current) |
| covtype 522,911 x 54 | 15281.972 | lightgbm-cpu 258148.588 (n=1, budget exceeded) | 16.89x faster | FAST | Apple M4 | same |
| covtype 522,911 x 54, 3 rounds | 19875.178 | sklearn-rf-cpu 13721.545 | 1.45x slower | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-trees-evening.md`, 20729ea (older) |
| higgs 1M x 28, 3 rounds | 26774.701 | sklearn-rf-cpu 82120.529 | 3.07x faster | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-trees-evening.md`, 20729ea (current for this cell) |
| higgs 1M x 28, 3 rounds | 26909.764 | sklearn-rf-cpu 84216.725 | 3.13x faster | FAST | Apple M4 | `fast_speed/2026-08-26-APPLE-HIGGS-1M.md`, 5b728e2 (older) |
| covtype, gbm-bench 100 trees | 4.02 s | sklearn RF 10 cores 5.26 s | 1.31x faster | mode not recorded | Apple M4 | `THREE_SUITE_QUIET_2026-08-22.md`, dfa41bb; `README.md` table reads 4.0 s vs 5.3 s, 1.33x |
| year, gbm-bench 100 trees | 14.14 s | sklearn RF 10 cores 385.1 s | 27.2x faster (sleep-straddled invocation, flagged) | mode not recorded | Apple M4 | same |
| higgs 8.8M, gbm-bench 100 trees | 47.05 s | sklearn RF 10 cores 310.3 s | 6.6x faster | mode not recorded | Apple M4 | same; `README.md` 47.1 s vs 310.3 s |
| covtype, gbm-bench pairs | 4.62 s | lgbm-rf-cpu 12.70 s | 2.75x faster | mode not recorded | Apple M4 | `RF_ET_2026-08-22_lightgbm-pairs.md`, 2026-08-22 |
| year, gbm-bench pairs | 23.09 s | lgbm-rf-cpu 7.69 s | 3.0x slower | mode not recorded | Apple M4 | same |
| higgs 8.8M, gbm-bench pairs | 68.12 s | lgbm-rf-cpu 29.53 s | 2.3x slower, ours wins every accuracy column | mode not recorded | Apple M4 | same |
| higgs 1M x 28, 5 rounds | 4525.381 | none | no peer on AMD; H100 same source 5309.306, AMD 1.17x faster than H100 | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest-higgs.md`, 4f6a17a; cross-box row in `BOARD_2026-08-28_three-vendor.md` 2.3 |
| higgs 2M x 28, 5 rounds | 6268.397 | none | H100 7308.012, AMD 1.17x faster | FAST | AMD MI325X | same |
| synthclf 720,000 x 100, 10 rounds | 3057.926 | none | | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest.md`, 88b918d |

### 2.5 Extra trees

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| higgs 1M x 28 | 4935.292 (5 rounds) | lightgbm-cuda 181070.935 (n=1, budget exceeded) | 36.69x faster, and better logloss 0.622 vs 0.644 | FAST | NVIDIA H100 | `fast_speed/2026-08-28_030911-nvidia-forest.md`, 3d65d70 |
| higgs 2M x 28 | 7449.510 (5 rounds) | lightgbm-cuda 227402.966 (n=1) | 30.53x faster | FAST | NVIDIA H100 | same |
| higgs 1M / 2M, 5 rounds | 4160.298 / 7661.135 | none ran (cuML has no ExtraTrees; lightgbm-cuda build refused in this leg) | no peer | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-forest-B.md`, a8d838e |
| higgs 5M x 28, 5 rounds | 14826.190 | none ran | no peer | FAST | NVIDIA H100 | `fast_speed/2026-08-28_030908-nvidia-forest.md`, 3d65d70 |
| covtype 522,911 x 54, 3 rounds | 9826.842 | sklearn-et-cpu 11006.799 | 1.12x faster, acc 0.6796 vs 0.6768 | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-forest.md`, a8d838e (current) |
| covtype 522,911 x 54 | 9826.842 | lightgbm-cpu 6625.966 | 1.48x slower, lgbm acc 0.5417 | FAST | Apple M4 | same |
| covtype 522,911 x 54, 3 rounds | 14648.080 | sklearn-et-cpu 12972.721 | 1.13x slower | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-trees-evening.md`, 20729ea (older) |
| covtype 522,911 x 54, 3 rounds | 9843.1 | sklearn-et-cpu 10454.7 | 1.06x faster | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-covtype-et-flip.md`, f732d11 (older) |
| higgs 1M x 28, 3 rounds | 23754.352 | sklearn-et-cpu 29630.341 | 1.25x faster | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-trees-evening.md`, 20729ea (current for this cell) |
| higgs 1M x 28, 3 rounds | 20944.393 | sklearn-et-cpu 28761.257 | 1.37x faster | FAST | Apple M4 | `fast_speed/2026-08-26-APPLE-HIGGS-1M.md`, 5b728e2 (older) |
| covtype, gbm-bench 100 trees | 3.70 s | sklearn ET 10 cores 5.44 s | 1.47x faster | mode not recorded | Apple M4 | `THREE_SUITE_QUIET_2026-08-22.md`, dfa41bb; `README.md` 3.7 s vs 5.4 s, 1.46x |
| year, gbm-bench 100 trees | 18.84 s | sklearn ET 10 cores 54.64 s | 2.90x faster (flagged invocation) | mode not recorded | Apple M4 | same |
| higgs 8.8M, gbm-bench 100 trees | 39.02 s | sklearn ET 10 cores 132.4 s | 3.39x faster | mode not recorded | Apple M4 | same; `README.md` 39.0 s vs 132.4 s, 3.4x |
| covtype, gbm-bench pairs | 4.10 s | lgbm-et-cpu 6.84 s | 1.67x faster | mode not recorded | Apple M4 | `RF_ET_2026-08-22_lightgbm-pairs.md` |
| year, gbm-bench pairs | 15.60 s | lgbm-et-cpu 6.17 s | 2.5x slower, MSE 98.0 vs 92.8 ours behind | mode not recorded | Apple M4 | same |
| higgs 1M x 28, 5 rounds | 18294.084 | none | H100 same source 4160.298, AMD 4.40x slower; NOT REPRODUCED at 9c8ffc23 (61772 ms, 53.9 s of it host push, DEVIATION 1945); the two GPU passes 17.0 s -> 6.2 s (DEVIATION 1943) | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest-higgs.md`, 4f6a17a; `e1/2026-08-29_204736-mojolearn-e2-amd/lanes/et_profile/` |
| higgs 2M x 28, 5 rounds | 35521.428 | none | H100 7661.135, AMD 4.64x slower; NOT REPRODUCED at 9c8ffc23 (88751 ms, 74.8 s host push) | FAST | AMD MI325X | same |
| synthclf 720,000 x 100, 10 rounds | 24199.518 | none | | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest.md`, 88b918d |

### 2.6 Isolation forest

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| anomaly 500,000 x 32, 10 rounds | 232.422 | cuml-iforest-gpu 85.159 | 2.73x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-forest-B.md`, a8d838e |
| anomaly 500,000 x 32, 3 rounds | 219.910 | sklearn-iforest-cpu 147.265 | 1.49x slower | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-forest.md`, a8d838e |
| anomaly 500,000 x 32, 10 rounds | 155.077 | none | H100 232.422, AMD 1.50x faster than H100 | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest-higgs.md`, 4f6a17a |
| anomaly 500,000 x 32, 10 rounds | 153.199 | none | | FAST | AMD MI325X | `fast_speed/2026-08-28-AMD-forest.md`, 88b918d |

### 2.7 k-means

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| 4,000,000 x 32, k=64, 20 iter, 5 rounds | 155.353 | cuml-gpu 120.217 | 1.29x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91, 2026-08-28 (current) |
| same, 5 rounds | 155.067 | cuml-gpu 236.789 | 1.53x faster | FAST | NVIDIA H100 | `fast_speed/2026-08-26_040100-nvidia-classical.md`, 2026-08-26 (older; the cuML arm moved 236.8 to 120.2 ms between legs) |
| same, 5 rounds | 167.145 | cuml-gpu 233.434 | 1.40x faster | FAST | NVIDIA H100 | `fast_speed/2026-08-25_235543-nvidia-classical.md`, 2026-08-25 (older) |
| 4,000,000 x 32, k=64, 20 iter, 3 rounds | 748.774 | none ran (cuML not on this box) | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| 4M x 32, k=64, 20 iter | 759.49 | sklearn KMeans 2055.22 | 2.71x faster, ranges disjoint | mode not recorded | Apple M4 | `BOARD_2026-08-20_algorithm-matched.md`, 2026-08-20 |
| any | unrun | | | | AMD MI325X | no AMD classical FAST speed leg exists under `fast_speed/` |

### 2.8 k-NN (brute force) and IVF

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| knn 400,000 x 32, 4,000 q, k=10, 5 rounds | 26.404 | cuml-gpu 9.555 | 2.76x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 (current; DEV 1921-1923 landed) |
| same | 233.827 / 243.862 | cuml-gpu 10.822 / 10.224 | 21.61x / 23.85x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-26_040100-nvidia-classical.md` and `2026-08-25_235543-nvidia-classical.md` (older; quoted as the target row in `LANE_knn-speed-campaign_2026-08-28.md`) |
| knn own arms, H100 | ours 26.404, ours-fused 26.409, ours-tiled 53.926 | | AUTO matches fused | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md` |
| knn 400,000 x 32, 4,000 q, k=10, 3 rounds | 692.947 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| knn own arms, Apple | ours 692.947, ours-fused 699.921, ours-tiled 616.570 (hash moved 3 of 3 rounds) | | tiled 1.12x faster than shipped choice | FAST | Apple M4 | same |
| knn 400k idx, 4k q, d=32, k=10 | 695.94 | sklearn 943.53 | 1.36x faster | mode not recorded | Apple M4 | `BOARD_2026-08-20_algorithm-matched.md`, 2026-08-20 |
| ivf 512x8, q64, L8, p3, k8, 5 rounds | 7.143 | cuvs-gpu 14.584 | 2.04x faster (fixed-cost row) | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| ivf same, 3 rounds | 86.763 | none ran (cupy missing) | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md` |
| knn or ivf | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

### 2.9 DBSCAN and HDBSCAN

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| dbscan 4,000 x 16, 5 rounds | 0.682 | cuml-gpu 1.093 | 1.60x faster (fixed-cost row) | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| hdbscan blobs96 96 x 4, 5 rounds | 1.162 | cuml-gpu 4.417 | 3.80x faster (fixed-cost row, no size knob) | FAST | NVIDIA H100 | same |
| dbscan 4,000 x 16, 3 rounds | 22.431 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| hdbscan 96 x 4, 3 rounds | 21.551 | none ran | no peer | FAST | Apple M4 | same |
| dbscan 4k x 16 | 11.75 | sklearn 9.08 | 0.77x, INDISTINGUISHABLE (ranges overlap) | mode not recorded | Apple M4 | `BOARD_2026-08-20_algorithm-matched.md` |
| dbscan d=8, 50k to 800k, n_jobs=-1 | 172.9 to 7930.5 | sklearn 93.1 to 4009.6 | 0.43x to 0.66x, a loss above ~20k points | mode not recorded | Apple M4 | `DBSCAN_RBC_2026-08-19.md`, 2026-08-19 late |
| dbscan or hdbscan | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

### 2.10 PCA and OLS

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| pca 4,000,000 x 32, 8 comp, 5 rounds | 9.278 | cuml-gpu (jacobi) 119.669 | 12.90x faster | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| ols 4,000,000 x 32, 5 rounds | 8.555 | cuml-gpu (eig) 118.837 | 13.89x faster | FAST | NVIDIA H100 | same |
| ols 4,000,000 x 32, 5 rounds | 8.607 | cuml-gpu 176.938 | 20.56x faster | FAST | NVIDIA H100 | `fast_speed/2026-08-26_040100-nvidia-classical.md` (older; pca refused there, `svd_solver` name) |
| pca 4M x 32 c8, 3 rounds | 36.315 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| ols 4M x 32, 3 rounds | 33.500 | none ran | no peer | FAST | Apple M4 | same |
| pca 4M x 32, 8 comp | 37.63 | sklearn covariance_eigh 132.53 | 3.52x faster | mode not recorded | Apple M4 | `BOARD_2026-08-20_algorithm-matched.md`, 2026-08-20 |
| ols 4M x 32 | 34.79 | sklearn Ridge cholesky 149.22 | 4.29x faster | mode not recorded | Apple M4 | same |
| pca or ols | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

sklearn's own GPU path on Apple was measured and is slower than its CPU path
(PCA full on MPS 1484.6 ms against covariance_eigh CPU 136.6 ms;
`SKLEARN_GPU_BASELINE_2026-08-20.md`), so the CPU arm is their best arm on
that box.

### 2.11 Ridge, logistic, lasso and elasticnet (cd), cholesky, krr

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| cd 2048 x 16, 5 rounds | 1.137 | cuml-gpu 1.283 | 1.13x faster (fixed-cost row) | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| cholesky RBF 64x64 r4, 5 rounds | 0.341 | torch-gpu (cuSOLVER) 0.162 | 2.11x slower (fixed-cost row) | FAST | NVIDIA H100 | same |
| krr RBF 16x5, 5 rounds | 0.300 | cuml-gpu 1.825 | 6.08x faster (fixed-cost row) | FAST | NVIDIA H100 | same |
| cd 2048 x 16, 3 rounds | 7.630 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| cholesky, 3 rounds | 4.737 | none ran (torch not installed in that env) | no peer | FAST | Apple M4 | same |
| krr, 3 rounds | 4.962 | none ran | no peer | FAST | Apple M4 | same |
| ridge, logistic | unrun on any speed board | | | | all | no `ridge` or `logreg` lane exists in the classical speed harness; they have identity cards only (`CHANGELOG.md` 0.2.0) |
| cd, cholesky, krr | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

### 2.12 SVM

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| F2 xor 240 x 2, fit only, 5 rounds | 1.333 | cuml-gpu 4.864 | 3.65x faster (fixed-cost row) | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| same, 3 rounds | 12.044 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| svm | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

### 2.13 KDE

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| 1024 x 256 x 8, 5 rounds | 0.203 | cuml-gpu 0.385 | 1.89x faster (fixed-cost row) | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| same, 3 rounds | 2.691 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| kde | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

### 2.14 Linkage (single linkage hierarchy)

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| blobs_dups 102 x 5, 5 rounds | 0.963 | cuml-gpu 2.318 | 2.41x faster (fixed-cost row) | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| same, 3 rounds | 14.035 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| linkage | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

### 2.15 Spectral, GMM, GP, Nystroem, RBFSampler, resample

RAPIDS ships none of these, so on NVIDIA and AMD they have no legal peer.
On Apple the peer is scikit-learn or SciPy on the CPU, and the boards mark
every one of these rows FIXED-COST at tiny shipped fixtures (24x2 to 48x4),
"NOT a speed claim".

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| spectral 48 x 4 c3, 3 rounds | 75.866 | sklearn-cpu 3.648 | 20.80x slower, fixed cost | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 (current) |
| spectral, 6 rounds | 84.643 | sklearn-cpu 3.180 | 26.62x slower | FAST | Apple M4 | `fast_speed/2026-08-27-APPLE-classical-six-small-lanes.md`, 9a4e759 (older) |
| gmm SEPARATED 24 x 2 K3, 3 rounds | 32.283 | sklearn-cpu 1.115 | 28.95x slower, fixed cost | FAST | Apple M4 | `2026-08-28-APPLE-classical.md` (current); 35.52x on 2026-08-27 |
| gp ard 12 x 3 s6, 3 rounds | 6.125 | sklearn-cpu 0.290 (float64, no optimizer both sides) | 21.14x slower, fixed cost | FAST | Apple M4 | same (current); 24.78x on 2026-08-27 |
| nystroem RBF 16 x 5 q8, 3 rounds | 5.816 | sklearn-cpu 0.483 | 12.03x slower, fixed cost | FAST | Apple M4 | same (current); 13.87x on 2026-08-27 |
| rbfsampler 3 x 5 q8, 3 rounds | 2.580 | sklearn-cpu 0.167 | 15.41x slower, fixed cost | FAST | Apple M4 | same (current); 19.04x on 2026-08-27 |
| resample 200 x 2, r10000, 3 rounds | 7.673 | scipy-cpu 35.149 | 4.58x faster | FAST | Apple M4 | same (current); 4.01x on 2026-08-27 |
| spectral, gmm, gp, nystroem, rbfsampler, resample | ours only, 2.456 / 0.464 / 0.418 / 0.193 / 0.529 / 6.428 | none | no peer, RAPIDS ships none | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md` |
| all six | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

### 2.16 Time series (holtwinters, kpss); ARIMA

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| holtwinters 7 x 72 f12, 5 rounds | 10.399 | cuml-gpu 8.156 | 1.27x slower (fixed-cost row) | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| kpss 8 x 520, 5 rounds | 0.130 | cuml-gpu 0.436 | 3.35x faster (fixed-cost row) | FAST | NVIDIA H100 | same |
| holtwinters, 3 rounds | 17.816 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| kpss, 3 rounds | 0.681 | none ran (statsmodels missing) | no peer | FAST | Apple M4 | same |
| holtwinters, kpss | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |
| arima | unrun on every column | | | | all | `CHANGELOG.md` 0.2.0, `ARIMA` has no `fit`; `estimate_x0` and the batched L-BFGS driver are not ported. *(Overtaken 2026-09-01: `ARIMA.fit` shipped. Still unrun on every column -- no arima timing exists anywhere.)* |

### 2.17 Metrics

| dataset and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| eleven metrics, lab2053 flt2053 sil521x4 tru301x6, 5 rounds | 0.687 | cuml-gpu 12.626 | 18.37x faster (fixed-cost row) | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-classical.md`, 65d5e91 |
| same, 3 rounds | 9.491 | none ran | no peer | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-classical.md`, cc499f7 |
| metrics | unrun | | | | AMD MI325X | no AMD classical FAST speed leg |

### 2.18 GEMM and matmul

FAST rows first. DEVIATION 1885 records that the FAST vendor GEMM on NVIDIA
is a TF32-class kernel (MAX `linalg.matmul` measured 200 TFLOP/s beside
cuBLAS TF32 207.5 and cuBLAS FP32 44.4), so on NVIDIA the fair FAST peer is
`cublas-tf32` (`fast_speed/2026-08-26-DEVIATION-1885-fast-gemm-is-tf32.md`,
`SPEED_LANE_2026-08-25.md` 1.1).

| shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| gram.128sq.x100003, 5 rounds | 0.142 | cublas-fp32 0.093 / cublas-tf32 0.058 | 1.53x / 2.47x slower | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-gemmseq.md`, a8d838e |
| gram.32x32x1M | 0.271 | cublas-fp32 0.237 / tf32 0.108 | 1.14x / 2.50x slower | FAST | NVIDIA H100 | same |
| gram.32x32x64K | 0.047 | 0.037 / 0.027 | 1.27x / 1.74x slower | FAST | NVIDIA H100 | same |
| kmeans.dist.4096x64x64 | 0.025 | 0.020 / 0.017 | 1.25x / 1.45x slower | FAST | NVIDIA H100 | same |
| ols.step1.16x16x64K | 0.045 | 0.036 / 0.027 | 1.27x / 1.65x slower | FAST | NVIDIA H100 | same |
| ols.predict.gemv.64Kx16 | 0.016 | 0.018 / 0.017 | 1.12x / 1.10x faster | FAST | NVIDIA H100 | same |
| pca.transform.8192x4x4 | 0.021 | 0.019 / 0.018 | 1.11x / 1.16x slower | FAST | NVIDIA H100 | same |
| pca.transform.wide.8192x64x128 | 0.029 | 0.023 / 0.019 | 1.28x / 1.51x slower | FAST | NVIDIA H100 | same |
| llama8b.qkv t1 / t8 / t512 | 0.031 / 0.054 / 0.090 | tf32 0.040 / 0.044 / 0.066 | 1.29x faster / 1.22x slower / 1.37x slower | FAST | NVIDIA H100 | same |
| llama8b.mlp_up t1 / t8 / t512 | 0.083 / 0.115 / 0.211 | tf32 0.093 / 0.105 / 0.185 | 1.11x faster / 1.10x slower / 1.14x slower | FAST | NVIDIA H100 | same |
| llama8b.mlp_down t1 / t8 / t512 | 0.085 / 0.110 / 0.232 | tf32 0.094 / 0.103 / 0.211 | 1.11x faster / 1.07x slower / 1.10x slower | FAST | NVIDIA H100 | same |
| llama8b.lm_head t8 / t512 | 0.775 / 1.407 | tf32 0.751 / 1.357 | 1.03x / 1.04x slower | FAST | NVIDIA H100 | same; lm_head.t1 ours refused (m=1 gemv path, `SPEED_LANE` section 4) |
| gram.32x32x1M, 3 rounds | 8.041 | mps-default 8.340 | 1.04x faster | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-gemmseq.md`, 8fd467f |
| gram.128sq.x100003 | 4.212 | mps-default 1.894 | 2.22x slower | FAST | Apple M4 | same |
| gram.32x32x64K | 0.856 | 0.756 | 1.13x slower | FAST | Apple M4 | same |
| kmeans.dist.4096x64x64 | 0.279 | 0.260 | 1.07x slower | FAST | Apple M4 | same |
| ols.step1.16x16x64K | 0.705 | 0.761 | 1.08x faster | FAST | Apple M4 | same |
| ols.predict.gemv.64Kx16 | 0.503 | 0.239 | 2.11x slower | FAST | Apple M4 | same |
| pca.transform.8192x4x4 | 0.209 | 0.220 | 1.05x faster | FAST | Apple M4 | same |
| pca.transform.wide.8192x64x128 | 0.431 | 0.329 | 1.31x slower | FAST | Apple M4 | same |
| llama8b.qkv t1 / t8 / t512 | 0.947 / 2.067 / 17.047 | mps-default 0.990 / 1.195 / 6.508 | 1.05x faster / 1.73x slower / 2.62x slower | FAST | Apple M4 | same |
| llama8b.mlp_up t1 / t8 / t512 | 3.052 / 9.404 / 36.506 | 2.976 / 3.892 / 22.494 | 1.03x / 2.42x / 1.62x slower | FAST | Apple M4 | same |
| llama8b.mlp_down t1 / t8 / t512 | 2.997 / 4.853 / 53.140 | 2.763 / 3.539 / 22.665 | 1.08x / 1.37x / 2.34x slower | FAST | Apple M4 | same |
| llama8b.lm_head t1 / t8 / t512 | 24.148 / 40.071 / 251.424 | 22.179 / 29.002 / 238.973 | 1.09x / 1.38x / 1.05x slower | FAST | Apple M4 | same |
| any FAST gemm shape | unrun | | | FAST | AMD MI325X | no AMD gemmseq FAST speed leg under `fast_speed/`; the only AMD gemm timings are IDENTICAL, below |

IDENTICAL rows, strict FP32 on our side, from `SPEED_LANE_2026-08-25.md`
(commits ba35096 through f5b905a, 2026-08-25). Read that file's section 1
first; the M4 and MI325X columns are single-round and the H100 column is a
band across two single-round legs.

| shape | ours | peer | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| llama8b.qkv.t1 | 2.21 ms | MAX linalg.matmul 1.21 / MPS 1.41 | 1.8x / 1.6x slower | IDENTICAL vs vendor FAST | Apple M4 | `SPEED_LANE_2026-08-25.md` 3.1 |
| llama8b.qkv.t8 | 4.50 | 1.41 / 2.72 | 3.2x / 1.7x slower | IDENTICAL | Apple M4 | same |
| llama8b.qkv.t512 | 218.8 | 11.6 / 8.7 | 18.9x / 25.1x slower | IDENTICAL | Apple M4 | same |
| llama8b.mlp_up.t512 | 771.5 | 39.8 / 25.8 | 19.4x / 29.9x slower | IDENTICAL | Apple M4 | same |
| llama8b.mlp_down.t512 | 770.4 | 68.9 / 29.0 | 11.2x / 26.6x slower | IDENTICAL | Apple M4 | same |
| llama8b.lm_head.t8 | 113.1 | 39.5 / 32.3 | 2.9x / 3.5x slower | IDENTICAL | Apple M4 | same |
| peak achieved | ours 78 GF/s | MPS 2332 GF/s | about 2 percent of what the box gives torch | IDENTICAL | Apple M4 | same |
| llama8b.qkv.t512 | 1.95 to 3.51 TF/s | cuBLAS strict FP32 44.4 TF/s (TF32 207.5) | 12.7x to 22.8x slower | IDENTICAL vs FP32 | NVIDIA H100 | `SPEED_LANE_2026-08-25.md` 3.2 |
| llama8b.mlp_up.t512 | 2.00 to 3.55 | 42.8 (TF32 291.3) | 12.1x to 21.4x slower | IDENTICAL | NVIDIA H100 | same |
| llama8b.mlp_down.t512 | 1.96 to 3.47 | 49.8 (TF32 263.6) | 14.4x to 25.4x slower | IDENTICAL | NVIDIA H100 | same |
| llama8b.qkv.t512 | 4.31 TF/s | hipBLASLt strict FP32 77.1 | 17.9x slower | IDENTICAL | AMD MI325X | `SPEED_LANE_2026-08-25.md` 3.3 (torch 2.9.1+rocm6.4, 10 repeats, median) |
| llama8b.mlp_up.t512 | 4.52 | 70.9 | 15.7x slower | IDENTICAL | AMD MI325X | same |
| llama8b.mlp_down.t512 | 4.37 | 50.4 | 11.5x slower | IDENTICAL | AMD MI325X | same |
| llama8b.lm_head.t8 | 3.81 | 12.9 | 3.4x slower | IDENTICAL | AMD MI325X | same |
| share of FP32 vector peak | 2 to 5 percent on all three vendors | | | IDENTICAL | all three | `SPEED_LANE_2026-08-25.md` 3.4 |

### 2.19 Sequence lanes (attention, mlp, rmsnorm, transformer, mamba)

These are FAST. On NVIDIA every lane routing a GEMM took the TF32 cut of
DEVIATION 1885 and the file says to quote its speedups only with "part of
this is a TF32 precision cut" attached. Best and worst peer per shape from
the 2026-08-28 boards; full per-arm rows are in the source files.

| lane and shape | ours ms | peer and ms | ratio | tier | vendor | src |
|---|---|---|---|---|---|---|
| attention lane.b2_l4_d32_kv2 | 0.284 | torch-gpu-sdpa-efficient-fp32 0.295 to torch-gpu-fp32 0.398 | 1.04x to 1.40x faster | FAST | NVIDIA H100 | `fast_speed/2026-08-28-NVIDIA-gemmseq.md`, a8d838e |
| attention llama8b.prefill.t512 | 2.171 | torch-gpu-tf32 0.642 to torch-gpu-fp32 1.425 | 3.38x to 1.52x slower | FAST | NVIDIA H100 | same |
| attention llama8b.prefill.t8 | 1.314 | sdpa-efficient 0.311 | 4.23x slower (worst) | FAST | NVIDIA H100 | same |
| mlp llama8b.prefill.t512 | 0.684 | torch-gpu-fp32 4.046 / tf32 0.716 | 5.92x faster / 1.05x faster | FAST | NVIDIA H100 | same |
| mlp llama8b.prefill.t8 | 0.338 | fp32 0.513 / tf32 0.306 | 1.51x faster / 1.11x slower | FAST | NVIDIA H100 | same |
| rmsnorm lane.b2_l4_d32_kv2 | 0.010 | torch-gpu-fp32 0.053 | 5.40x faster | FAST | NVIDIA H100 | same |
| rmsnorm llama8b.prefill.t128 | 1.052 | tf32 0.062 | 16.88x slower (worst row on the board) | FAST | NVIDIA H100 | same |
| transformer lane.b2_l4_d32_kv2 | 0.396 | torch-gpu-fp32 0.540 | 1.36x faster | FAST | NVIDIA H100 | same |
| transformer llama8b.prefill.t512 | 8.880 | tf32 1.380 / fp32 5.542 | 6.44x / 1.60x slower | FAST | NVIDIA H100 | same |
| transformer llama8b.decode.t1.ctx512 | 3.594 | sdpa-math 0.623 | 5.77x slower | FAST | NVIDIA H100 | same |
| mamba mamba130m.prefill.t512 | 2.470 | torch-ref-scan-gpu 21.236 | 8.60x faster (peer is the pure-PyTorch reference scan; mamba_ssm not importable) | FAST | NVIDIA H100 | same |
| mamba mamba130m.prefill.t1 / t8 / t128 | 0.237 / 0.350 / 1.042 | torch-ref-scan-gpu 0.414 / 0.754 / 5.691 | 1.75x / 2.16x / 5.46x faster | FAST | NVIDIA H100 | same |
| transformer llama8b.prefill.t1, before and after DEV 1873-1877 | 1009.778 then 1.885 | best torch arm | 686x slower became 7.12x slower at t512 | FAST | NVIDIA H100 | `fast_speed/2026-08-26-DEVIATION-1885-fast-gemm-is-tf32.md`, 2026-08-26 |
| attention lane.b2_l4_d32_kv2 | 4.282 | torch-gpu-fp32 2.302 to flash-bf16 0.557 | 1.86x to 7.68x slower | FAST | Apple M4 | `fast_speed/2026-08-28-APPLE-gemmseq.md`, 8fd467f |
| attention llama8b.prefill.t512 | 88.574 | fp32 27.636 to flash-bf16 18.524 | 3.21x to 4.78x slower | FAST | Apple M4 | same |
| attention llama8b.prefill.t8 | 20.762 | flash-bf16 1.565 | 13.27x slower (worst) | FAST | Apple M4 | same |
| mlp llama8b.decode.t1.ctx512 | 7.744 | fp32 7.374 / tf32 7.062 | 1.05x / 1.10x slower | FAST | Apple M4 | same |
| mlp llama8b.prefill.t512 | 111.150 | fp32 71.586 | 1.55x slower | FAST | Apple M4 | same |
| rmsnorm lane.b2_l4_d32_kv2 | 0.269 | fp32 0.494 / tf32 0.440 | 1.84x / 1.64x faster | FAST | Apple M4 | same |
| rmsnorm llama8b.decode.t1.ctx512 | 3.277 | tf32 0.226 | 14.48x slower (worst) | FAST | Apple M4 | same |
| transformer llama8b.prefill.t512 | 221.114 | fp32 101.787 / flash-bf16 80.126 | 2.17x / 2.76x slower | FAST | Apple M4 | same |
| transformer lane.b2_l4_d32_kv2 | 8.707 | flash-fp32 1.086 | 8.02x slower (worst) | FAST | Apple M4 | same |
| mamba, all five shapes | 4.480 to 7.749 | none ran (einops missing, mamba_ssm missing) | no peer | FAST | Apple M4 | same |
| every sequence lane | unrun | | | FAST | AMD MI325X | no AMD gemmseq FAST speed leg |

Row counts as the boards state them. NVIDIA gemmseq 2026-08-28 has 53 rows
with an opponent, all fixed-cost by the undeclared-scale rule; the three
vendor board counts the same leg as 130 comparison pairs, 55 wins and 75
losses (`BOARD_2026-08-28_three-vendor.md` 2.1). Apple gemmseq has 44 rows
with an opponent; the board calls it 116 pairs.

## 3. What is unrun, and why the boards say so

| cell | why |
|---|---|
| AMD MI325X, gbdt symmetric, depthwise, lossguide on higgs 1M and 2M | Our arm raised at warm-up in the 4f6a17a leg ("half-byte histograms on a 64-lane column ... DEVIATION 1910"), `fast_speed/2026-08-28-AMD-forest-higgs.md`. FAST gbdt builds and runs on the MI325X since DEVIATIONS 1906 and 1910 (commits f853e8df and 19b319c7, `e1/2026-08-29_093711-mojolearn-e2-amd/p9_fast_build_gbdt.log`, `stability/` gbdt lanes STABLE 6/6) but the speed rows have not been rerun (`BOARD_2026-08-28_three-vendor.md` 2.3). |
| AMD MI325X, any peer in any lane | No legal GPU opponent exists on AMD. cuBLAS, cuML, cuVS, CatBoost-GPU, XGBoost-GPU, LightGBM-CUDA are CUDA-only; each arm prints `FSPEED-REFUSED` (`BOARD_2026-08-28_three-vendor.md` 2.3). catboost-gpu on the MI325X died with "CUDA driver version is insufficient" and lightgbm-cuda with "CUDA Tree Learner was not enabled" (`2026-08-28-AMD-forest-higgs.md`). |
| AMD MI325X, all 21 classical lanes and all gemm and sequence lanes, FAST | No AMD classical or gemmseq FAST speed leg exists under `bench/results/fast_speed/`; the only AMD leg directories are `do-2026-08-28_*-amd-forest`. |
| AMD MI325X, gemm XF32 arm | `SPEED_LANE_2026-08-25.md` 3.3 states the harness does not touch `allow_tf32` on the hip backend, so the CDNA3 XF32 arm is owed the way TF32 is measured on NVIDIA. |
| AMD synthetic-fixture leg (88b918d) | The leg had no dataset download step, so every higgs rung refused and fell back to `synthclf-720000x100`; "60 REAL timed rounds, none on the dataset the comparison needs" (`BOARD_2026-08-28_three-vendor.md` defect 9). The 4f6a17a leg fixed it for rf, et and iforest. |
| NVIDIA H100, extra trees against cuML | cuML has no ExtraTrees estimator (refusal text in every NVIDIA forest board). The only GPU peer that ever ran is lightgbm-cuda, once, at 3d65d70, with n=1 because it exceeded the 300 s per-arm budget. |
| NVIDIA H100, iforest at higgs 1M or 2M | The iforest lane ships the anomaly 500k fixture only; no higgs rung exists for it on any board. |
| NVIDIA H100, gemm llama8b.lm_head.t1 | Our FAST arm refused; MAX `linalg.matmul` crashes at m=1, n=128256 on an H100 (`SPEED_LANE_2026-08-25.md` section 4, DEVIATION 1093). |
| NVIDIA H100, llama8b.lm_head.t512 in the IDENTICAL lane | Above the 5e10 MAC cap on both sides, skipped on both (`SPEED_LANE_2026-08-25.md` section 2). |
| NVIDIA H100, torch flash arms for attention and transformer | `No available kernel` at every shape on that box (`2026-08-28-NVIDIA-gemmseq.md` refusals). |
| NVIDIA H100, mamba against mamba-ssm | `mamba_ssm` not importable on any box; the only peer is the pure-PyTorch reference scan, "NOT what anyone deploys" (both gemmseq boards). |
| NVIDIA H100, gmm, gp, nystroem, rbfsampler, resample, spectral | RAPIDS ships none of them; the sklearn and scipy CPU arms are refused under GPU-PATH-ONLY (`2026-08-28-NVIDIA-classical.md`). |
| NVIDIA or AMD, gbm-bench harness (year, covtype, higgs at 500 rounds; lgbm-*-gpu six-arm forest interleave) | `RF_ET_2026-08-22_lightgbm-pairs.md` states the NVIDIA leg was "ready and waiting on a box"; no `gbm_bench_*` result under `bench/results/` carries an NVIDIA or AMD device record. Every gbm-bench number on disk is the M4. |
| Apple M4, 15 of 21 classical lanes (cd, cholesky, dbscan, hdbscan, holtwinters, ivf, kde, kmeans, knn, kpss, krr, linkage, metrics, ols, pca, svm) | The peer is cuML, cuVS or torch CUDA, none of which runs on Apple; `NO OPPONENT ON THIS BOX` (`2026-08-28-APPLE-classical.md`). The sklearn comparison for ols, pca, kmeans, knn, dbscan is the 2026-08-20 algorithm-matched board, which is a different harness and predates the tier split. |
| Apple M4, cholesky | torch was not installed in the classical env for that leg (`ModuleNotFoundError("No module named 'torch'")`). |
| Apple M4, kpss | cuML absent and `statsmodels` not installed. |
| Apple M4, mamba | `einops` missing took the torch reference arm down (`BOARD_2026-08-28_three-vendor.md` defect 7). |
| Apple M4, depthwise and lossguide on higgs; any 2M or 5M rung | Only the year fixture has been run for those two lanes on the Mac. "No 5M rung on the Mac by policy, that load goes to rented boxes" (`2026-08-26-APPLE-HIGGS-1M.md`). |
| Apple M4, rf against lightgbm-cpu with more than one round | The lightgbm-cpu arm ran 258 s per fit and hit the per-arm budget after one round (`2026-08-28-APPLE-forest.md`). |
| Apple M4, xgboost CPU or GPU, cuML, lightgbm-cuda | Not installed or no Metal build; refused by name on every Apple tree board. |
| ridge, logistic, arima, svr on any column | No speed lane exists. `ARIMA` has no `fit` and `SVR` is absent (`CHANGELOG.md` 0.2.0). *(Overtaken 2026-09-01: both shipped -- `SVR` and a batched `ARIMA.fit`. The verdict of this row stands: still no speed lane, no timing has been taken for either.)* |
| Any lane, IDENTICAL against a peer other than gemm | The only IDENTICAL-versus-vendor timings on disk are the GEMM shapes in `SPEED_LANE_2026-08-25.md`. Every other lane's IDENTICAL cost is ours-versus-ours (section 4). |
| Scale label on every forest board | Every 2026-08-28 forest board prints "Scale UNKNOWN for every row in this run" because the arms did not emit `scale=`; the rows are listed as fixed-cost "because that is the conservative bucket", not because they are small. |

## 4. The IDENTICAL price, where it was measured

All of these are ours against ours, same box, same shapes, two binaries,
except the SPEED_LANE section 3 rows already listed in 2.18.

### 4.1 GEMM fold pin, `gemm/mojo_only/gemm_unpinned_price.mojo`

pinned over unpinned at the tiled `llama8b` t512 rows, 2026-08-25
(`SPEED_LANE_2026-08-25.md` 1.2, 1.2a, 1.2c).

| shape | Apple M4 | NVIDIA H100 | AMD MI325X |
|---|---|---|---|
| qkv.t512 | 1.522x | 2.33x | 2.21x |
| mlp_up.t512 | 1.538x | 2.32x | 2.20x |
| mlp_down.t512 | 1.546x | 2.31x | 2.17x |
| lm_head.t512 | 1.542x | 2.32x | 2.06x |
| seams share of the pin (pin over strict, t512) | 1.24x of 1.55x | 1.83x of 2.31x | 2.12x of 2.21x |
| mlp_down.t1 pin over unpinned | not listed | 1.02x | 2.30x |

The file's own sentences to carry. "The pin costs 2.31x on an H100 against
1.55x on an M4." "AMD sits with NVIDIA, not with Apple." "Report the range
with the table, never a scalar." The M4 prediction on record was 1.10x to
1.45x and it was wrong on the low side. Confounds are listed in 1.2b.

### 4.2 Per-lane price, `bench/results/lanes_price/`

FAST versus IDENTICAL, ours only, Apple M4, 5 rounds, commit 26eb8ba,
2026-08-28 (`lanes_price/2026-08-28_163353_26eb8ba/ratio.tsv`). The M4 drift
caveat applies; two modes are two binaries and cannot interleave inside one
process.

| lane | FAST median s | IDENTICAL median s | ratio (min to max) | bits |
|---|---|---|---|---|
| gemm (`gram.32x32x1M`, split-K 240 chunks against pinned 128) | 0.000710 | 0.003297 | 4.644x (3.961 to 5.849) | fast == ident |
| cd | 0.007365 | 0.009967 | 1.353x (1.199 to 1.714) | fast != ident |
| kde | 0.003200 | 0.003799 | 1.187x (0.984 to 1.343) | fast != ident |
| svm | 0.018298 | 0.019268 | 1.053x (0.791 to 1.206) | fast != ident |
| linkage | 0.011320 | 0.011176 | 0.987x (0.703 to 1.009) | fast == ident |
| metrics | 0.012427 | 0.011196 | 0.901x (0.721 to 1.194) | fast != ident |

The 2026-08-23 smoke at e69db89 (1 round, `lanes_price/2026-08-23_125704_e69db89_SMOKE/ratio.tsv`) read gemm 1.545x, cd 1.530x, kde 1.007x, linkage 1.014x, svm 0.944x, metrics 0.788x; the 26eb8ba run is current.

### 4.3 Two named single-shape prices

`gemm/IDENTICAL_FP32_CONTRACT.md` clause 5 records "the 2.85x pinned-distance
cost" (IDENTITY_PATHS row 24, the k-NN distance step) and "the 4.7x measured
on `nt.4096x64x64`" (the linalg lane's `gemm_nt`, MAX matmul under FAST
against `pinned_gemm_nt_kernel` under IDENTICAL, DEVIATION 526), and says
not to present either as universal. `bench/linalg_price_main.mojo` and
`bench/gemm_price_main.mojo` carry the same two figures in their docstrings.
No dated board under `bench/results/` records either measurement; the driver
text places them on the M4.

### 4.4 What the flag buys, same box, same run

`BOARD_2026-08-28_three-vendor.md` 1.3. gemm versus its oracle on Apple, FAST
31 shapes differ, IDENTICAL 0 differ; on AMD FAST 41 differ, IDENTICAL 0.
Transformer clause (d) fails under FAST and passes under IDENTICAL. k-NN on
one tie-rich fixture returns three different sorted index sets in three FAST
runs and one under IDENTICAL. `stability/RESULTS.md` (2026-08-29) reads
FAST moved in 8 of 10 attempts on the M4, 24 different answers in 24 calls
on an RTX 4090, 6 answers in 24 on the MI325X; deterministic and identical
stable on all three.

## 5. Anomalies the boards themselves flag

- Extra trees on the MI325X read 4.40x slower than on the H100 at higgs 1M
  (18294.084 versus 4160.298 ms) and 4.64x at 2M on 2026-08-28. Profiled
  2026-08-29 (`extratrees/DEVIATIONS.md` 1943): the range and score passes
  were 99% of device time at cuML's 128-thread block, and a `WARP_SIZE`-keyed
  512-wide block takes them from 17.0 s to 6.2 s at 1M with the identical
  hashes unchanged. The whole-fit number is now dominated by a 53.9 s host
  `NodeQueue.push` the 2026-08-28 arm cannot have contained (DEVIATION 1945,
  OPEN); the 2.3 rows carry both numbers.
- The 5M rung. Lossguide read 14800.158 ms with min 6030.534 at 69f2503 on
  2026-08-27 (5.59x behind xgboost-gpu) and 6301.633 ms at 3d65d70 the next
  morning (2.61x). The symmetric arm bent superlinear between 2M and 5M at
  374875a (marginal 273.6 then 1389.6 ms per 1M rows) and straightened to
  518 and 525 after 132d754 (`2026-08-26-nvidia-HIGGS-ladder-2M.md`).
- The same cell on the same day, two H100 hosts. gbdt-symmetric higgs 1M
  read 3666.967 ms (min 2622.429, warm-up 2125.632) at 3d65d70 in the
  030908 leg and 856.116 ms at a8d838e in the 040157 leg. The 2M row in the
  030908 leg is 1304.338, faster than its own 1M row. The ladder file's
  finding 1 explains the mechanism, "RunPod hosts differ in CPU allocation
  and our arm is host-prep-heavy"; the 1.86x-faster crossover at c637e30
  became 1.19x slower at 374875a on a different host, so cross-leg
  comparisons are "for direction only".
- The M4 drifts. "This box has been measured drifting 1.7x inside twenty
  minutes when heat pins the GPU governor" (`2026-08-26-APPLE-trees.md`);
  the same binary measured 21.1 then 12.1 ms per tree nineteen minutes apart
  (`PERF_2026-08-20_fixed-cost.md`). Only ratios from arms alternated inside
  one process are quoted; the six interleaved year runs on 2026-08-22 ranged
  1.14x to 1.68x (`README.md`). The 2026-08-27 covtype rf window is not
  claimed for the same reason (sklearn 9.1 s to 17.4 s day over day,
  `2026-08-27-APPLE-covtype-et-flip.md`).
- The H100 IDENTICAL GEMM variance. Two single-round legs an hour apart at
  the same commit read 4.90 and 8.799 ms at `llama8b.qkv.t512`, a 1.8x
  spread, so every H100 row in `SPEED_LANE_2026-08-25.md` is a band and the
  M4 and MI325X rows are single-round; "Re-run before publication".
- The FAST GEMM is TF32 on NVIDIA and was not declared until DEVIATION 1885.
  MAX `linalg.matmul` measured 200 TFLOP/s against an H100 FP32 peak of 67.
  The 535x transformer speedup at t1 carries a 47x to 239x worsening of
  max_abs_diff against the fp32 reference, with rmsnorm (no GEMM) unchanged
  as the control (`2026-08-26-DEVIATION-1885-fast-gemm-is-tf32.md`).
- `check_device_is_batch_invariant` fails under FAST on NVIDIA and AMD and
  holds on Apple, "a real defect" and not a FAST wobble; round 13 verdict
  NOT-CLOSED (`BOARD_2026-08-28_three-vendor.md` 1.4).
  `check_ols_is_launch_invariant` fails on the H100 and MI325X in both modes
  (`CHANGELOG.md` known issues).
- k-NN own-arm dispatch on Apple. `ours-tiled` is 1.12x faster than the
  shipped `ours` on the M4 and its hash moved in 3 of 3 rounds; on the H100
  the tiled arm is 2.04x slower than shipped (`2026-08-28-APPLE-classical.md`,
  `2026-08-28-NVIDIA-classical.md`). The kernel-matrix rows are per vendor.
- The knn H100 row moved 21.6x to 23.9x slower (2026-08-25 and 26) to 2.76x
  slower (2026-08-28) after DEV 1921-1923; the older files are superseded.
- The cuML k-means arm moved 236.789 ms to 120.217 ms between the 2026-08-26
  and 2026-08-28 H100 legs while ours held at 155 ms, which flipped the cell
  from 1.53x faster to 1.29x slower.
- Fixed-cost rows are unclaimable in both directions. Every 2026-08-28
  classical board ranks its small fixtures under "NOT a speed claim"; the
  Apple six-lane file says the 12x to 29x sklearn wins there are "the same
  mechanism as our 2-16x wins over cuML-GPU on small fixtures on the H100,
  with the roles reversed".
- Peers with n=1. lightgbm-cuda in the ET lane (181 s and 227 s per fit) and
  lightgbm-cpu in the Apple rf lane (258 s) each completed one timed round
  before the per-arm budget cut them off; the 36.69x, 30.53x and 16.89x
  ratios stand on a single peer sample.
- Nine silent harness defects on 2026-08-28 (gemm_nt_gram IDENTICAL not
  compiling since 2026-08-25, `git rev-parse` killing the NVIDIA matrix,
  iforest cards written to a scratch path, einops, the seq torch arm
  refusing non-CUDA boxes, the AMD leg with no dataset download) all
  produced boards "full of NO OPPONENT or plausible rows rather than
  errors" (`BOARD_2026-08-28_three-vendor.md` part 3).
- 2026-08-25 NVIDIA tree numbers (`2026-08-25-nvidia-trees-PARTIAL.md`,
  4cbead3) were transcribed from a live terminal after the leg script died
  before its fetch; they are not card-backed and every later leg supersedes
  them. Its CPU rows are struck.
- Accuracy where it is not fine. higgs gbdt AUC 0.822 versus CatBoost 0.830
  on the gbm-bench harness is an open parity gap (`README.md`,
  `THREE_SUITE_QUIET_2026-08-22.md`); year ET on the LightGBM pairs suite
  is behind on MSE (98.0 versus 92.8). Every 2026-08-28 higgs row on the
  H100 is at parity or better on logloss and AUC per the boards' accuracy
  tables.

## 6. Source files read

- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/BOARD_2026-08-28_three-vendor.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/SPEED_LANE_2026-08-25.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/BOARD_2026-08-20_algorithm-matched.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/SCOREBOARD_2026-08-19.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/SKLEARN_GPU_BASELINE_2026-08-20.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/SCALING_2026-08-19.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/SCALING_2026-08-20_catboost.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/DBSCAN_RBC_2026-08-19.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/GEMM_ROUND_2026-08-19.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/PERF_2026-08-20_fixed-cost.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/PERF_2026-08-20_estimation-bill.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/WINDOW_2026-08-20_interleaved-200k.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/COVTYPE_MULTICLASS_2026-08-21.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/EPSILON_2026-08-21_interleaved.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/EPSILON_DEPTH8_2026-08-21.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/EPSILON_LOGLOSS_2026-08-21.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/SHAPE_SWEEP_2026-08-21_epsilon.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/EXTRATREES_SKLEARN_2026-08-21.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/RF_2026-08-21_pipelined-forest.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/RF_2026-08-22_attribution.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/RF_ET_2026-08-22_lightgbm-pairs.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/WINDOW_2026-08-22_extratrees-batched.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/GBM_BENCH_2026-08-22_year.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/THREE_SUITE_2026-08-22.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/THREE_SUITE_QUIET_2026-08-22.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/LANE_knn-speed-campaign_2026-08-28.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-25-nvidia-trees-PARTIAL.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-25_155542-nvidia-gemmseq.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-25_160520-nvidia-gemmseq.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-25_235543-nvidia-classical.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26-APPLE-HIGGS-1M.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26-APPLE-trees.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26-DEVIATION-1885-fast-gemm-is-tf32.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26-nvidia-HIGGS-ladder.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26-nvidia-HIGGS-ladder-2M.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26-nvidia-forest-postfix.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26_040049-nvidia-trees-ladder.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26_040054-nvidia-gemmseq.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-26_162556-nvidia-forest.md` (via its summary file `2026-08-26-nvidia-HIGGS-ladder-2M.md`)
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-27-APPLE-classical-six-small-lanes.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-27-APPLE-covtype-et-flip.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-27-APPLE-trees-evening.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-27_031904-nvidia-forest.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-27_171939-nvidia-depthwise.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-AMD-forest.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-AMD-forest-higgs.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-APPLE-classical.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-APPLE-forest.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-APPLE-gemmseq.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-NVIDIA-classical.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-NVIDIA-forest-A.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-NVIDIA-forest-B.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28-NVIDIA-gemmseq.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28_030908-nvidia-forest.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/2026-08-28_030911-nvidia-forest.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-higgs-2026-08-26/table.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/do-2026-08-28_125022-amd-forest/run/leg.txt`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/do-2026-08-28_130709-amd-forest/run/leg.txt`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/do-2026-08-28_140119-amd-forest/run/leg.txt`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_040202-forest/leg.txt`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_122551-classical/leg.txt`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_122711-gemmseq/leg.txt`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/fast_speed/mac-2026-08-28_131915-gemmseq/leg.txt`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/e1g/2026-08-2*-nvidia-speed-*/leg.txt` (the ten NVIDIA legs, for commit and round provenance)
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/gemm_ladder/2026-08-23_095619_fe00e8a-dirty/ladder.card`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/gemm_ladder/2026-08-23_101159_ec944d8-dirty/ladder.card`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/gemm_ladder/2026-08-23_101329_5c6932b-dirty/clean.card` and `sab.card` (identity cards, no timings)
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/lanes_price/2026-08-23_125704_e69db89_SMOKE/env.txt`, `summary.tsv`, `ratio.tsv`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/lanes_price/2026-08-28_163353_26eb8ba/env.txt`, `summary.tsv`, `ratio.tsv`
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/stability/RESULTS.md`
- `/Users/andrewhendel/CascadeProjects/mojolearn/README.md` (Hardware, Benchmarks, Limitations sections)
- `/Users/andrewhendel/CascadeProjects/mojolearn/CHANGELOG.md` (0.2.0 cross-vendor standing, third tier, Performance, Known issues)
- `/Users/andrewhendel/CascadeProjects/mojolearn/gemm/IDENTICAL_FP32_CONTRACT.md` (clause 5, for the 2.85x and 4.7x figures)
- `/Users/andrewhendel/CascadeProjects/mojolearn/bench/linalg_price_main.mojo` and `bench/gemm_price_main.mojo` (docstrings, same two figures)

Skimmed for ours-versus-peer cells and found none. The twenty
`LANE_*_2026-08-19` and `LANE_*_2026-08-20` files (ball-cover, covariance,
dbscan, decomposition, dispatch-audit, gram, kmeans, knn, pca, rbc, splitk,
target-keying, vendor-correctness, warpsort) are Apple work logs whose only
ratios are ours-against-ours kernel A/Bs (for example the ball-cover launch
shape sweep, "1.54x slower at n = 16,000" for their tpb=64 against one query
per block, `LANE_rbc-build_2026-08-19.md`); their peer comparisons were
folded into `SCOREBOARD_2026-08-19.md` and the 2026-08-20 algorithm-matched
board. The `WINDOW_2026-08-20_*`, `FIRST_RUN`, `FUSED_ROUND`,
`SUBSTITUTION_ROUND`, `VENDOR_PATH`, `GRAM_PROFILE`, `MODULAR_UPSTREAM`,
`PREP_BILL`, `CONTAMINATION`, `DEFECT`, `RF_2026-08-21_accuracy-and-profile`
and `RF_2026-08-21_control-plane` files are pre-tier Apple windows whose
cells all have a newer source in the tables above.
