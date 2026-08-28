# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/fast_speed/mac-2026-08-28_040202-forest/logs`, 6 arm logs.

Device(s) reported by the arms themselves: Darwin_arm64

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| et | lightgbm-cpu | covtype-522911x54 | 3 | 6625.966 | 6398.072 | 8229.048 | 6636.495 | fd371fb58677b768 |
| et | ours | covtype-522911x54 | 3 | 9826.842 | 9302.075 | 9915.576 | 9222.908 | 9516ac844dc1228e |
| et | sklearn-et-cpu | covtype-522911x54 | 3 | 11006.799 | 10896.473 | 11513.837 | 11308.375 | a660a67fe34b09c9 |
| gbdt-depthwise | catboost-cpu | year-463715x90 | 3 | 4189.724 | 4181.316 | 4407.742 | 4812.655 | 6aa4425d40e37a74 |
| gbdt-depthwise | ours | year-463715x90 | 3 | 4543.493 | 4447.952 | 4962.074 | 5632.738 | 255260b3b9d45b32 |
| gbdt-lossguide | catboost-cpu | year-463715x90 | 3 | 5894.057 | 5306.061 | 6581.007 | 5392.276 | 068100038b41b984 |
| gbdt-lossguide | lightgbm-cpu | year-463715x90 | 3 | 3886.938 | 3528.623 | 4320.358 | 3510.354 | 284ae9f4c4099ffe |
| gbdt-lossguide | ours | year-463715x90 | 3 | 8660.290 | 8324.922 | 9373.657 | 8337.984 | cddaf3167ed7d5a8 |
| gbdt-symmetric | catboost-cpu | year-463715x90 | 3 | 2197.476 | 1999.346 | 2729.256 | 2111.800 | 6ecb58fe0b522afa |
| gbdt-symmetric | ours | year-463715x90 | 3 | 1931.296 | 1901.922 | 2031.870 | 2376.059 | 2868341a9401e697 |
| iforest | ours | anomaly-500000x32 | 3 | 219.910 | 216.375 | 230.444 | 772.255 | d62cbc1f23e3cbee |
| iforest | sklearn-iforest-cpu | anomaly-500000x32 | 3 | 147.265 | 145.245 | 154.044 | 162.816 | e73871a3ebb1aae1 |
| rf | lightgbm-cpu | covtype-522911x54 | 1 | 258148.588 | 258148.588 | 258148.588 | 253905.276 | 9377475b6c286200 |
| rf | ours | covtype-522911x54 | 3 | 15281.972 | 15172.996 | 15537.910 | 18370.200 | d17ccce2fde38b00 |
| rf | sklearn-rf-cpu | covtype-522911x54 | 3 | 12430.019 | 12042.706 | 12823.441 | 8346.919 | 4edc4cefb648d129 |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| et | covtype-522911x54 | lightgbm-cpu | 9826.842 | 6625.966 | 1.48 | we are 1.48x SLOWER |
| et | covtype-522911x54 | sklearn-et-cpu | 9826.842 | 11006.799 | 0.89 | we are 1.12x FASTER |
| gbdt-depthwise | year-463715x90 | catboost-cpu | 4543.493 | 4189.724 | 1.08 | we are 1.08x SLOWER |
| gbdt-lossguide | year-463715x90 | catboost-cpu | 8660.290 | 5894.057 | 1.47 | we are 1.47x SLOWER |
| gbdt-lossguide | year-463715x90 | lightgbm-cpu | 8660.290 | 3886.938 | 2.23 | we are 2.23x SLOWER |
| gbdt-symmetric | year-463715x90 | catboost-cpu | 1931.296 | 2197.476 | 0.88 | we are 1.14x FASTER |
| iforest | anomaly-500000x32 | sklearn-iforest-cpu | 219.910 | 147.265 | 1.49 | we are 1.49x SLOWER |
| rf | covtype-522911x54 | lightgbm-cpu | 15281.972 | 258148.588 | 0.06 | we are 16.89x FASTER |
| rf | covtype-522911x54 | sklearn-rf-cpu | 15281.972 | 12430.019 | 1.23 | we are 1.23x SLOWER |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| et | ours | accuracy | 0.679592 |
| et | sklearn-et-cpu | accuracy | 0.676804 |
| et | lightgbm-cpu | accuracy | 0.541712 |
| gbdt-depthwise | ours | rmse | 9.086366 |
| gbdt-depthwise | catboost-cpu | rmse | 9.104583 |
| gbdt-lossguide | ours | rmse | 9.091499 |
| gbdt-lossguide | catboost-cpu | rmse | 9.126961 |
| gbdt-lossguide | lightgbm-cpu | rmse | 9.085702 |
| gbdt-symmetric | ours | rmse | 9.215942 |
| gbdt-symmetric | catboost-cpu | rmse | 9.230145 |
| iforest | ours | auc | 1.000000 |
| iforest | sklearn-iforest-cpu | auc | 1.000000 |
| rf | ours | accuracy | 0.707750 |
| rf | sklearn-rf-cpu | accuracy | 0.709506 |
| rf | lightgbm-cpu | accuracy | 0.601831 |

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
| 1 | gbdt-lossguide | year-463715x90 | 8660.290 | lightgbm-cpu | 3886.938 | **2.23x SLOWER** |
| 2 | iforest | anomaly-500000x32 | 219.910 | sklearn-iforest-cpu | 147.265 | **1.49x SLOWER** |
| 3 | et | covtype-522911x54 | 9826.842 | lightgbm-cpu | 6625.966 | **1.48x SLOWER** |
| 4 | rf | covtype-522911x54 | 15281.972 | sklearn-rf-cpu | 12430.019 | **1.23x SLOWER** |
| 5 | gbdt-depthwise | year-463715x90 | 4543.493 | catboost-cpu | 4189.724 | **1.08x SLOWER** |
| 6 | gbdt-symmetric | year-463715x90 | 1931.296 | catboost-cpu | 2197.476 | **1.14x FASTER** |

6 rows in total have an opponent: 0 throughput, 6 fixed-cost.

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| et | cuml-et-gpu | ModuleNotFoundError: No module named 'cuml' |
| gbdt-depthwise | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-depthwise | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| iforest | cuml-iforest-gpu | ModuleNotFoundError: No module named 'cuml' |
| rf | cuml-rf-gpu | ModuleNotFoundError: No module named 'cuml' |
| rf | lightgbm-cpu | per-arm budget 300s exceeded after 512.1s; remaining rounds skipped |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

No arm's output hash moved across its rounds in this run.

## Notes the arms printed

- `forest.et.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-depthwise.log`: lane=gbdt-depthwise arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-lossguide.log`: lane=gbdt-lossguide arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.iforest.log`: lane=iforest arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so

