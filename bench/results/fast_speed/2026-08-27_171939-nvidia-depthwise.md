# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/e1g/2026-08-27_171939-nvidia-speed-forest/remote/logs`, 26 arm logs.

Device(s) reported by the arms themselves: NVIDIA_H100_80GB_HBM3

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| gbdt-depthwise | catboost-gpu | higgs-1000000x28 | 3 | 1258.061 | 1248.577 | 1305.710 | 1178.622 | MOVED(3) |
| gbdt-depthwise | catboost-gpu | higgs-2000000x28 | 3 | 1604.819 | 1593.171 | 1617.214 | 1577.233 | MOVED(3) |
| gbdt-depthwise | ours | higgs-1000000x28 | 3 | 1316.996 | 1282.239 | 1637.147 | 2267.704 | f521273542894631 |
| gbdt-depthwise | ours | higgs-2000000x28 | 3 | 3112.093 | 3107.702 | 3280.221 | 3820.032 | b199b56c2d852302 |
| gbdt-depthwise | xgboost-gpu | higgs-1000000x28 | 3 | 685.180 | 659.272 | 691.701 | 806.483 | 13505f62978a068d |
| gbdt-depthwise | xgboost-gpu | higgs-2000000x28 | 3 | 1141.118 | 1110.911 | 1161.125 | 1326.571 | 335671ec1071de24 |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| gbdt-depthwise | higgs-1000000x28 | catboost-gpu | 1316.996 | 1258.061 | 1.05 | we are 1.05x SLOWER |
| gbdt-depthwise | higgs-1000000x28 | xgboost-gpu | 1316.996 | 685.180 | 1.92 | we are 1.92x SLOWER |
| gbdt-depthwise | higgs-2000000x28 | catboost-gpu | 3112.093 | 1604.819 | 1.94 | we are 1.94x SLOWER |
| gbdt-depthwise | higgs-2000000x28 | xgboost-gpu | 3112.093 | 1141.118 | 2.73 | we are 2.73x SLOWER |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| gbdt-depthwise | ours | logloss | 0.525307 |
| gbdt-depthwise | ours | auc | 0.813565 |
| gbdt-depthwise | catboost-gpu | logloss | 0.524860 |
| gbdt-depthwise | catboost-gpu | auc | 0.813947 |
| gbdt-depthwise | xgboost-gpu | logloss | 0.526563 |
| gbdt-depthwise | xgboost-gpu | auc | 0.812395 |
| gbdt-depthwise | ours | logloss | 0.524605 |
| gbdt-depthwise | ours | auc | 0.814061 |
| gbdt-depthwise | catboost-gpu | logloss | 0.524631 |
| gbdt-depthwise | catboost-gpu | auc | 0.814230 |
| gbdt-depthwise | xgboost-gpu | logloss | 0.526273 |
| gbdt-depthwise | xgboost-gpu | auc | 0.812587 |

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
| 1 | gbdt-depthwise | higgs-2000000x28 | 3112.093 | xgboost-gpu | 1141.118 | **2.73x SLOWER** |
| 2 | gbdt-depthwise | higgs-1000000x28 | 1316.996 | xgboost-gpu | 685.180 | **1.92x SLOWER** |

2 rows in total have an opponent: 0 throughput, 2 fixed-cost.

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| gbdt-depthwise | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

| lane | arm | shape | distinct hashes across rounds |
|---|---|---|---|
| gbdt-depthwise | catboost-gpu | higgs-1000000x28 | 3 |
| gbdt-depthwise | catboost-gpu | higgs-2000000x28 | 3 |

## Notes the arms printed

- `forest.gbdt-depthwise.r1000000.log`: lane=gbdt-depthwise arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-depthwise.r2000000.log`: lane=gbdt-depthwise arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so

