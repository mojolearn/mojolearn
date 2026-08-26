# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-26_170628-nvidia-speed-forest/remote/logs`, 33 arm logs.

Device(s) reported by the arms themselves: NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| et | ours | higgs-1000000x28 | 3 | 4099.810 | 4049.800 | 4500.205 | 5242.798 | fdce1a43cb984497 |
| et | ours | higgs-2000000x28 | 3 | 6803.474 | 6770.348 | 7587.814 | 8205.027 | 04071a5fab79ffe8 |
| et | ours | higgs-5000000x28 | 3 | 14208.357 | 14191.491 | 14384.285 | 15713.478 | 437913e2aacc6c01 |
| gbdt-symmetric | catboost-gpu | higgs-1000000x28 | 3 | 994.561 | 928.552 | 1006.163 | 1041.961 | MOVED(3) |
| gbdt-symmetric | catboost-gpu | higgs-2000000x28 | 3 | 1312.121 | 1275.755 | 1354.357 | 1357.410 | MOVED(3) |
| gbdt-symmetric | catboost-gpu | higgs-5000000x28 | 3 | 2543.586 | 2535.110 | 2587.139 | 2654.700 | MOVED(3) |
| gbdt-symmetric | ours | higgs-1000000x28 | 3 | 668.142 | 625.166 | 1013.140 | 1598.269 | c3b1ab8c7f136526 |
| gbdt-symmetric | ours | higgs-2000000x28 | 3 | 1185.877 | 1170.809 | 1193.692 | 1866.405 | f9cbed2393cc83e3 |
| gbdt-symmetric | ours | higgs-5000000x28 | 3 | 2760.228 | 2745.802 | 2767.198 | 3601.965 | c16d69d2a8d2458a |
| rf | cuml-rf-gpu | higgs-1000000x28 | 3 | 3005.881 | 2977.088 | 3148.732 | 6686.781 | a372de7ab27df595 |
| rf | cuml-rf-gpu | higgs-2000000x28 | 3 | 4486.053 | 4372.384 | 4556.685 | 4663.376 | dab1ee10bc85da38 |
| rf | cuml-rf-gpu | higgs-5000000x28 | 3 | 7042.403 | 6965.404 | 7426.759 | 8155.701 | 5e93788f17785841 |
| rf | ours | higgs-1000000x28 | 3 | 6140.287 | 6137.877 | 6207.423 | 7176.222 | 3ffa2951595422d4 |
| rf | ours | higgs-2000000x28 | 3 | 8886.001 | 8623.851 | 9079.805 | 10374.126 | 67d883dc6079b90f |
| rf | ours | higgs-5000000x28 | 3 | 15404.731 | 15359.082 | 16471.402 | 17319.900 | 498a272bd12aea30 |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| et | higgs-1000000x28 | (none ran) | 4099.810 | - | - | NO OPPONENT ON THIS BOX |
| et | higgs-2000000x28 | (none ran) | 6803.474 | - | - | NO OPPONENT ON THIS BOX |
| et | higgs-5000000x28 | (none ran) | 14208.357 | - | - | NO OPPONENT ON THIS BOX |
| gbdt-symmetric | higgs-1000000x28 | catboost-gpu | 668.142 | 994.561 | 0.67 | we are 1.49x FASTER |
| gbdt-symmetric | higgs-2000000x28 | catboost-gpu | 1185.877 | 1312.121 | 0.90 | we are 1.11x FASTER |
| gbdt-symmetric | higgs-5000000x28 | catboost-gpu | 2760.228 | 2543.586 | 1.09 | we are 1.09x SLOWER |
| rf | higgs-1000000x28 | cuml-rf-gpu | 6140.287 | 3005.881 | 2.04 | we are 2.04x SLOWER |
| rf | higgs-2000000x28 | cuml-rf-gpu | 8886.001 | 4486.053 | 1.98 | we are 1.98x SLOWER |
| rf | higgs-5000000x28 | cuml-rf-gpu | 15404.731 | 7042.403 | 2.19 | we are 2.19x SLOWER |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| et | ours | logloss | 0.621923 |
| et | ours | auc | 0.762823 |
| et | ours | logloss | 0.623010 |
| et | ours | auc | 0.761916 |
| et | ours | logloss | 0.624324 |
| et | ours | auc | 0.761727 |
| gbdt-symmetric | ours | logloss | 0.542247 |
| gbdt-symmetric | ours | auc | 0.800447 |
| gbdt-symmetric | catboost-gpu | logloss | 0.542424 |
| gbdt-symmetric | catboost-gpu | auc | 0.800373 |
| gbdt-symmetric | ours | logloss | 0.542096 |
| gbdt-symmetric | ours | auc | 0.800573 |
| gbdt-symmetric | catboost-gpu | logloss | 0.542353 |
| gbdt-symmetric | catboost-gpu | auc | 0.800336 |
| gbdt-symmetric | ours | logloss | 0.542133 |
| gbdt-symmetric | ours | auc | 0.800668 |
| gbdt-symmetric | catboost-gpu | logloss | 0.542201 |
| gbdt-symmetric | catboost-gpu | auc | 0.800665 |
| rf | ours | logloss | 0.538850 |
| rf | ours | auc | 0.809906 |
| rf | cuml-rf-gpu | logloss | 0.538814 |
| rf | cuml-rf-gpu | auc | 0.809834 |
| rf | ours | logloss | 0.537060 |
| rf | ours | auc | 0.811483 |
| rf | cuml-rf-gpu | logloss | 0.537039 |
| rf | cuml-rf-gpu | auc | 0.811519 |
| rf | ours | logloss | 0.535736 |
| rf | ours | auc | 0.812881 |
| rf | cuml-rf-gpu | logloss | 0.535689 |
| rf | cuml-rf-gpu | auc | 0.812976 |

## Scale UNKNOWN for every row in this run

No arm in these logs emitted `scale=` in its FSPEED header, so
this run is OLDER than the throughput/fixed-cost declaration and
the split below could not be made. The rows are listed under
FIXED-COST because that is the conservative bucket, but the label
is NOT a measurement here -- some of these lanes are genuinely
large (kmeans ships 4,000,000 x 32) and some are genuinely tiny
(krr ships 16 rows). Check each shape tag by hand before quoting
anything from this file, or re-run so the arms declare it.

## Rows, ranked -- scale UNDECLARED, check each shape by hand

Every row here is a lane whose fixture is small enough that both
arms are dominated by launch and dispatch latency, plus the Python
call overhead the vendor arm pays inside the clock and ours does
not. Read each as WHAT ONE FIT COSTS END TO END on this box.

A ratio here is not wrong, it is UNCLAIMABLE: it does not tell you
whose kernel is faster. Some of these lanes cannot be made bigger
for stated reasons -- `hdbscan`'s dense mutual-reachability arm
materializes an m x m matrix, and inventing a larger fixture to
make the number look like throughput would be inventing a dataset.
Others simply have no size knob yet, and that is owed work.

They are ranked and kept rather than deleted because the fixed
cost is a real thing a user pays on a small problem.

| rank | lane | shape | ours ms | best opponent | their ms | we are |
|---|---|---|---|---|---|---|
| 1 | rf | higgs-5000000x28 | 15404.731 | cuml-rf-gpu | 7042.403 | **2.19x SLOWER** |
| 2 | rf | higgs-1000000x28 | 6140.287 | cuml-rf-gpu | 3005.881 | **2.04x SLOWER** |
| 3 | rf | higgs-2000000x28 | 8886.001 | cuml-rf-gpu | 4486.053 | **1.98x SLOWER** |
| 4 | gbdt-symmetric | higgs-5000000x28 | 2760.228 | catboost-gpu | 2543.586 | **1.09x SLOWER** |
| 5 | gbdt-symmetric | higgs-2000000x28 | 1185.877 | catboost-gpu | 1312.121 | **1.11x FASTER** |
| 6 | gbdt-symmetric | higgs-1000000x28 | 668.142 | catboost-gpu | 994.561 | **1.49x FASTER** |

6 rows in total have an opponent: 0 throughput, 6 fixed-cost.

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| et | cuml-et-gpu | RuntimeError: cuML has no ExtraTrees estimator: its RandomForest searches quantile splits, not the uniform-random thresholds that define ExtraTrees. The like-for-like comparator for `et` is sklearn. |
| et | sklearn-et-cpu | GPU-PATH-ONLY: sklearn-et-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| et | cuml-et-gpu | RuntimeError: cuML has no ExtraTrees estimator: its RandomForest searches quantile splits, not the uniform-random thresholds that define ExtraTrees. The like-for-like comparator for `et` is sklearn. |
| et | sklearn-et-cpu | GPU-PATH-ONLY: sklearn-et-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| et | cuml-et-gpu | RuntimeError: cuML has no ExtraTrees estimator: its RandomForest searches quantile splits, not the uniform-random thresholds that define ExtraTrees. The like-for-like comparator for `et` is sklearn. |
| et | sklearn-et-cpu | GPU-PATH-ONLY: sklearn-et-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | sklearn-rf-cpu | GPU-PATH-ONLY: sklearn-rf-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| rf | sklearn-rf-cpu | GPU-PATH-ONLY: sklearn-rf-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| rf | sklearn-rf-cpu | GPU-PATH-ONLY: sklearn-rf-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

| lane | arm | shape | distinct hashes across rounds |
|---|---|---|---|
| gbdt-symmetric | catboost-gpu | higgs-1000000x28 | 3 |
| gbdt-symmetric | catboost-gpu | higgs-2000000x28 | 3 |
| gbdt-symmetric | catboost-gpu | higgs-5000000x28 | 3 |

## Notes the arms printed

- `forest.et.r1000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.et.r2000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.et.r5000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r1000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r2000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r5000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r1000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r2000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r5000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so

