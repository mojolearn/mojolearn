# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-28_040244-nvidia-speed-forest/remote/logs`, 30 arm logs.

Device(s) reported by the arms themselves: NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| et | ours | higgs-1000000x28 | 5 | 4160.298 | 4135.233 | 4321.462 | 4856.111 | 2c192f6b12dbb6c5 |
| et | ours | higgs-2000000x28 | 5 | 7661.135 | 7626.871 | 7826.470 | 8585.331 | a9a9ff528583d3bd |
| gbdt-lossguide | catboost-gpu | higgs-1000000x28 | 5 | 1533.086 | 1511.423 | 1547.420 | 1514.878 | MOVED(5) |
| gbdt-lossguide | catboost-gpu | higgs-2000000x28 | 5 | 1864.952 | 1836.571 | 1882.574 | 1845.068 | MOVED(5) |
| gbdt-lossguide | ours | higgs-1000000x28 | 5 | 1751.063 | 1737.114 | 1776.182 | 2407.055 | MOVED(3) |
| gbdt-lossguide | ours | higgs-2000000x28 | 5 | 2690.113 | 2670.758 | 2697.366 | 3321.229 | MOVED(2) |
| gbdt-lossguide | xgboost-gpu | higgs-1000000x28 | 5 | 771.236 | 765.636 | 791.931 | 912.684 | 13505f62978a068d |
| gbdt-lossguide | xgboost-gpu | higgs-2000000x28 | 5 | 1185.849 | 1107.396 | 1215.747 | 1259.312 | 335671ec1071de24 |
| iforest | cuml-iforest-gpu | anomaly-500000x32 | 10 | 85.159 | 82.782 | 90.768 | 2533.445 | 30e11bef8c3eaef8 |
| iforest | ours | anomaly-500000x32 | 10 | 232.422 | 204.761 | 278.655 | 931.978 | 29d540e571cd7eec |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| et | higgs-1000000x28 | (none ran) | 4160.298 | - | - | NO OPPONENT ON THIS BOX |
| et | higgs-2000000x28 | (none ran) | 7661.135 | - | - | NO OPPONENT ON THIS BOX |
| gbdt-lossguide | higgs-1000000x28 | catboost-gpu | 1751.063 | 1533.086 | 1.14 | we are 1.14x SLOWER |
| gbdt-lossguide | higgs-1000000x28 | xgboost-gpu | 1751.063 | 771.236 | 2.27 | we are 2.27x SLOWER |
| gbdt-lossguide | higgs-2000000x28 | catboost-gpu | 2690.113 | 1864.952 | 1.44 | we are 1.44x SLOWER |
| gbdt-lossguide | higgs-2000000x28 | xgboost-gpu | 2690.113 | 1185.849 | 2.27 | we are 2.27x SLOWER |
| iforest | anomaly-500000x32 | cuml-iforest-gpu | 232.422 | 85.159 | 2.73 | we are 2.73x SLOWER |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| et | ours | logloss | 0.622379 |
| et | ours | auc | 0.762400 |
| et | ours | logloss | 0.624325 |
| et | ours | auc | 0.758920 |
| gbdt-lossguide | ours | logloss | 0.524971 |
| gbdt-lossguide | ours | auc | 0.813691 |
| gbdt-lossguide | catboost-gpu | logloss | 0.525838 |
| gbdt-lossguide | catboost-gpu | auc | 0.812883 |
| gbdt-lossguide | xgboost-gpu | logloss | 0.526563 |
| gbdt-lossguide | xgboost-gpu | auc | 0.812395 |
| gbdt-lossguide | ours | logloss | 0.525067 |
| gbdt-lossguide | ours | auc | 0.813511 |
| gbdt-lossguide | catboost-gpu | logloss | 0.524903 |
| gbdt-lossguide | catboost-gpu | auc | 0.813675 |
| gbdt-lossguide | xgboost-gpu | logloss | 0.526273 |
| gbdt-lossguide | xgboost-gpu | auc | 0.812587 |
| iforest | ours | auc | 1.000000 |
| iforest | cuml-iforest-gpu | auc | 1.000000 |
| iforest | ours | auc | 1.000000 |
| iforest | cuml-iforest-gpu | auc | 1.000000 |

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
| 1 | iforest | anomaly-500000x32 | 232.422 | cuml-iforest-gpu | 85.159 | **2.73x SLOWER** |
| 2 | gbdt-lossguide | higgs-1000000x28 | 1751.063 | xgboost-gpu | 771.236 | **2.27x SLOWER** |
| 3 | gbdt-lossguide | higgs-2000000x28 | 2690.113 | xgboost-gpu | 1185.849 | **2.27x SLOWER** |

3 rows in total have an opponent: 0 throughput, 3 fixed-cost.

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
| gbdt-lossguide | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| gbdt-lossguide | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| iforest | sklearn-iforest-cpu | GPU-PATH-ONLY: sklearn-iforest-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| iforest | sklearn-iforest-cpu | GPU-PATH-ONLY: sklearn-iforest-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

| lane | arm | shape | distinct hashes across rounds |
|---|---|---|---|
| gbdt-lossguide | catboost-gpu | higgs-1000000x28 | 5 |
| gbdt-lossguide | catboost-gpu | higgs-2000000x28 | 5 |
| gbdt-lossguide | ours | higgs-1000000x28 | 3 |
| gbdt-lossguide | ours | higgs-2000000x28 | 2 |

## Notes the arms printed

- `forest.et.r1000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.et.r2000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-lossguide.r1000000.log`: lane=gbdt-lossguide arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-lossguide.r2000000.log`: lane=gbdt-lossguide arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.iforest.r1000000.log`: lane=iforest arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.iforest.r2000000.log`: lane=iforest arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so

