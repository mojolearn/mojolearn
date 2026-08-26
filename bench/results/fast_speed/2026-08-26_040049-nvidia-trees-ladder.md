# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-26_040049-nvidia-speed-forest/remote/logs`, 30 arm logs.

Device(s) reported by the arms themselves: NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| et | ours | higgs-1000000x28 | 3 | 4146.711 | 4031.117 | 4220.394 | 4952.079 | fdce1a43cb984497 |
| et | ours | higgs-5000000x28 | 3 | 14446.117 | 14294.401 | 14855.969 | 14985.499 | 437913e2aacc6c01 |
| gbdt-symmetric | catboost-gpu | higgs-1000000x28 | 3 | 1182.268 | 1141.883 | 1191.638 | 1223.755 | MOVED(3) |
| gbdt-symmetric | catboost-gpu | higgs-5000000x28 | 3 | 2687.290 | 2661.616 | 2860.248 | 2522.231 | MOVED(3) |
| gbdt-symmetric | ours | higgs-1000000x28 | 3 | 636.590 | 631.735 | 668.418 | 1287.314 | c3b1ab8c7f136526 |
| gbdt-symmetric | ours | higgs-5000000x28 | 3 | 2998.741 | 2990.244 | 3002.917 | 3572.009 | c16d69d2a8d2458a |
| rf | cuml-rf-gpu | higgs-1000000x28 | 3 | 3079.464 | 3008.900 | 3452.264 | 7122.780 | a372de7ab27df595 |
| rf | cuml-rf-gpu | higgs-5000000x28 | 3 | 7057.536 | 7016.384 | 7852.648 | 7982.659 | 5e93788f17785841 |
| rf | ours | higgs-1000000x28 | 3 | 6188.417 | 6117.652 | 6241.723 | 6899.724 | 3ffa2951595422d4 |
| rf | ours | higgs-5000000x28 | 3 | 14698.891 | 14647.822 | 14909.752 | 15358.537 | 498a272bd12aea30 |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| et | higgs-1000000x28 | (none ran) | 4146.711 | - | - | NO OPPONENT ON THIS BOX |
| et | higgs-5000000x28 | (none ran) | 14446.117 | - | - | NO OPPONENT ON THIS BOX |
| gbdt-symmetric | higgs-1000000x28 | catboost-gpu | 636.590 | 1182.268 | 0.54 | we are 1.86x FASTER |
| gbdt-symmetric | higgs-5000000x28 | catboost-gpu | 2998.741 | 2687.290 | 1.12 | we are 1.12x SLOWER |
| rf | higgs-1000000x28 | cuml-rf-gpu | 6188.417 | 3079.464 | 2.01 | we are 2.01x SLOWER |
| rf | higgs-5000000x28 | cuml-rf-gpu | 14698.891 | 7057.536 | 2.08 | we are 2.08x SLOWER |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| et | ours | logloss | 0.621923 |
| et | ours | auc | 0.762823 |
| et | ours | logloss | 0.624324 |
| et | ours | auc | 0.761727 |
| gbdt-symmetric | ours | logloss | 0.542247 |
| gbdt-symmetric | ours | auc | 0.800447 |
| gbdt-symmetric | catboost-gpu | logloss | 0.542424 |
| gbdt-symmetric | catboost-gpu | auc | 0.800372 |
| gbdt-symmetric | ours | logloss | 0.542133 |
| gbdt-symmetric | ours | auc | 0.800668 |
| gbdt-symmetric | catboost-gpu | logloss | 0.542202 |
| gbdt-symmetric | catboost-gpu | auc | 0.800664 |
| rf | ours | logloss | 0.538850 |
| rf | ours | auc | 0.809906 |
| rf | cuml-rf-gpu | logloss | 0.538814 |
| rf | cuml-rf-gpu | auc | 0.809834 |
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
| 1 | rf | higgs-5000000x28 | 14698.891 | cuml-rf-gpu | 7057.536 | **2.08x SLOWER** |
| 2 | rf | higgs-1000000x28 | 6188.417 | cuml-rf-gpu | 3079.464 | **2.01x SLOWER** |
| 3 | gbdt-symmetric | higgs-5000000x28 | 2998.741 | catboost-gpu | 2687.290 | **1.12x SLOWER** |
| 4 | gbdt-symmetric | higgs-1000000x28 | 636.590 | catboost-gpu | 1182.268 | **1.86x FASTER** |

4 rows in total have an opponent: 0 throughput, 4 fixed-cost.

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
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
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
| gbdt-symmetric | catboost-gpu | higgs-5000000x28 | 3 |

## Notes the arms printed

- `forest.et.r1000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.et.r5000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r1000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r5000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r1000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r5000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so

