# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-28_040157-nvidia-speed-forest/remote/logs`, 30 arm logs.

Device(s) reported by the arms themselves: NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| gbdt-depthwise | catboost-gpu | higgs-1000000x28 | 5 | 1139.637 | 1125.143 | 1189.524 | 1216.265 | MOVED(5) |
| gbdt-depthwise | catboost-gpu | higgs-2000000x28 | 5 | 1496.789 | 1473.356 | 1622.626 | 1604.891 | MOVED(5) |
| gbdt-depthwise | ours | higgs-1000000x28 | 5 | 1382.644 | 1304.459 | 1641.320 | 2002.561 | MOVED(2) |
| gbdt-depthwise | ours | higgs-2000000x28 | 5 | 2483.338 | 2462.019 | 3001.882 | 3800.881 | 96a1d3b8a783c2ac |
| gbdt-depthwise | xgboost-gpu | higgs-1000000x28 | 5 | 743.058 | 729.962 | 819.834 | 893.677 | 13505f62978a068d |
| gbdt-depthwise | xgboost-gpu | higgs-2000000x28 | 5 | 1310.138 | 1137.903 | 1341.307 | 1490.065 | 335671ec1071de24 |
| gbdt-symmetric | catboost-gpu | higgs-1000000x28 | 5 | 864.690 | 838.083 | 982.374 | 896.475 | MOVED(5) |
| gbdt-symmetric | catboost-gpu | higgs-2000000x28 | 5 | 1379.246 | 1276.950 | 1602.072 | 1343.974 | MOVED(5) |
| gbdt-symmetric | ours | higgs-1000000x28 | 5 | 856.116 | 820.897 | 1029.519 | 1586.812 | c3b1ab8c7f136526 |
| gbdt-symmetric | ours | higgs-2000000x28 | 5 | 1827.444 | 1790.869 | 2882.452 | 2416.144 | f9cbed2393cc83e3 |
| rf | cuml-rf-gpu | higgs-1000000x28 | 5 | 3177.072 | 3126.078 | 4055.870 | 7554.787 | a372de7ab27df595 |
| rf | cuml-rf-gpu | higgs-2000000x28 | 5 | 4445.280 | 4286.824 | 4683.788 | 5090.048 | dab1ee10bc85da38 |
| rf | ours | higgs-1000000x28 | 5 | 5309.306 | 4985.528 | 5704.045 | 6064.447 | 3ffa2951595422d4 |
| rf | ours | higgs-2000000x28 | 5 | 7308.012 | 6913.741 | 8221.634 | 8811.583 | 67d883dc6079b90f |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| gbdt-depthwise | higgs-1000000x28 | catboost-gpu | 1382.644 | 1139.637 | 1.21 | we are 1.21x SLOWER |
| gbdt-depthwise | higgs-1000000x28 | xgboost-gpu | 1382.644 | 743.058 | 1.86 | we are 1.86x SLOWER |
| gbdt-depthwise | higgs-2000000x28 | catboost-gpu | 2483.338 | 1496.789 | 1.66 | we are 1.66x SLOWER |
| gbdt-depthwise | higgs-2000000x28 | xgboost-gpu | 2483.338 | 1310.138 | 1.90 | we are 1.90x SLOWER |
| gbdt-symmetric | higgs-1000000x28 | catboost-gpu | 856.116 | 864.690 | 0.99 | we are 1.01x FASTER |
| gbdt-symmetric | higgs-2000000x28 | catboost-gpu | 1827.444 | 1379.246 | 1.32 | we are 1.32x SLOWER |
| rf | higgs-1000000x28 | cuml-rf-gpu | 5309.306 | 3177.072 | 1.67 | we are 1.67x SLOWER |
| rf | higgs-2000000x28 | cuml-rf-gpu | 7308.012 | 4445.280 | 1.64 | we are 1.64x SLOWER |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| gbdt-depthwise | ours | logloss | 0.525307 |
| gbdt-depthwise | ours | auc | 0.813564 |
| gbdt-depthwise | catboost-gpu | logloss | 0.524737 |
| gbdt-depthwise | catboost-gpu | auc | 0.814025 |
| gbdt-depthwise | xgboost-gpu | logloss | 0.526563 |
| gbdt-depthwise | xgboost-gpu | auc | 0.812395 |
| gbdt-depthwise | ours | logloss | 0.524605 |
| gbdt-depthwise | ours | auc | 0.814061 |
| gbdt-depthwise | catboost-gpu | logloss | 0.524873 |
| gbdt-depthwise | catboost-gpu | auc | 0.813941 |
| gbdt-depthwise | xgboost-gpu | logloss | 0.526273 |
| gbdt-depthwise | xgboost-gpu | auc | 0.812587 |
| gbdt-symmetric | ours | logloss | 0.542247 |
| gbdt-symmetric | ours | auc | 0.800447 |
| gbdt-symmetric | catboost-gpu | logloss | 0.542524 |
| gbdt-symmetric | catboost-gpu | auc | 0.800431 |
| gbdt-symmetric | ours | logloss | 0.542096 |
| gbdt-symmetric | ours | auc | 0.800573 |
| gbdt-symmetric | catboost-gpu | logloss | 0.542078 |
| gbdt-symmetric | catboost-gpu | auc | 0.800788 |
| rf | ours | logloss | 0.538850 |
| rf | ours | auc | 0.809906 |
| rf | cuml-rf-gpu | logloss | 0.538814 |
| rf | cuml-rf-gpu | auc | 0.809834 |
| rf | ours | logloss | 0.537060 |
| rf | ours | auc | 0.811483 |
| rf | cuml-rf-gpu | logloss | 0.537039 |
| rf | cuml-rf-gpu | auc | 0.811519 |

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
| 1 | gbdt-depthwise | higgs-2000000x28 | 2483.338 | xgboost-gpu | 1310.138 | **1.90x SLOWER** |
| 2 | gbdt-depthwise | higgs-1000000x28 | 1382.644 | xgboost-gpu | 743.058 | **1.86x SLOWER** |
| 3 | rf | higgs-1000000x28 | 5309.306 | cuml-rf-gpu | 3177.072 | **1.67x SLOWER** |
| 4 | rf | higgs-2000000x28 | 7308.012 | cuml-rf-gpu | 4445.280 | **1.64x SLOWER** |
| 5 | gbdt-symmetric | higgs-2000000x28 | 1827.444 | catboost-gpu | 1379.246 | **1.32x SLOWER** |
| 6 | gbdt-symmetric | higgs-1000000x28 | 856.116 | catboost-gpu | 864.690 | **1.01x FASTER** |

6 rows in total have an opponent: 0 throughput, 6 fixed-cost.

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| gbdt-depthwise | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
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
| gbdt-depthwise | catboost-gpu | higgs-1000000x28 | 5 |
| gbdt-depthwise | catboost-gpu | higgs-2000000x28 | 5 |
| gbdt-depthwise | ours | higgs-1000000x28 | 2 |
| gbdt-symmetric | catboost-gpu | higgs-1000000x28 | 5 |
| gbdt-symmetric | catboost-gpu | higgs-2000000x28 | 5 |

## Notes the arms printed

- `forest.gbdt-depthwise.r1000000.log`: lane=gbdt-depthwise arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-depthwise.r2000000.log`: lane=gbdt-depthwise arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r1000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r2000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r1000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r2000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so

