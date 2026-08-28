# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/fast_speed/do-2026-08-28_130709-amd-forest/run/logs`, 12 arm logs.

Device(s) reported by the arms themselves: Linux_x86_64

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| et | ours | synthclf-720000x100 | 10 | 24199.518 | 24096.432 | 24311.228 | 24609.746 | 5071ee6045f6636a |
| gbdt-depthwise | ours | synthclf-720000x100 | 10 | 1446.487 | 1435.672 | 1509.244 | 1864.350 | b80ce535c1b3adfd |
| gbdt-lossguide | ours | synthclf-720000x100 | 10 | 2437.570 | 2420.247 | 2512.326 | 2894.822 | b9b7c3175c36bb4d |
| gbdt-symmetric | ours | synthclf-720000x100 | 10 | 1041.722 | 1028.042 | 1114.132 | 1496.314 | 190ff3121b206445 |
| iforest | ours | anomaly-500000x32 | 10 | 153.199 | 151.556 | 178.334 | 506.360 | 33d86f1901c5acce |
| rf | ours | synthclf-720000x100 | 10 | 3057.926 | 3037.275 | 3118.788 | 3376.952 | MOVED(2) |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| et | synthclf-720000x100 | (none ran) | 24199.518 | - | - | NO OPPONENT ON THIS BOX |
| gbdt-depthwise | synthclf-720000x100 | (none ran) | 1446.487 | - | - | NO OPPONENT ON THIS BOX |
| gbdt-lossguide | synthclf-720000x100 | (none ran) | 2437.570 | - | - | NO OPPONENT ON THIS BOX |
| gbdt-symmetric | synthclf-720000x100 | (none ran) | 1041.722 | - | - | NO OPPONENT ON THIS BOX |
| iforest | anomaly-500000x32 | (none ran) | 153.199 | - | - | NO OPPONENT ON THIS BOX |
| rf | synthclf-720000x100 | (none ran) | 3057.926 | - | - | NO OPPONENT ON THIS BOX |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| et | ours | logloss | 0.549266 |
| et | ours | auc | 0.977263 |
| et | ours | logloss | 0.549266 |
| et | ours | auc | 0.977263 |
| gbdt-depthwise | ours | logloss | 0.033278 |
| gbdt-depthwise | ours | auc | 0.999634 |
| gbdt-depthwise | ours | logloss | 0.033278 |
| gbdt-depthwise | ours | auc | 0.999634 |
| gbdt-lossguide | ours | logloss | 0.029903 |
| gbdt-lossguide | ours | auc | 0.999642 |
| gbdt-lossguide | ours | logloss | 0.029903 |
| gbdt-lossguide | ours | auc | 0.999642 |
| gbdt-symmetric | ours | logloss | 0.051836 |
| gbdt-symmetric | ours | auc | 0.999468 |
| gbdt-symmetric | ours | logloss | 0.051836 |
| gbdt-symmetric | ours | auc | 0.999468 |
| iforest | ours | auc | 1.000000 |
| iforest | ours | auc | 1.000000 |
| rf | ours | logloss | 0.260290 |
| rf | ours | auc | 0.994939 |
| rf | ours | logloss | 0.260290 |
| rf | ours | auc | 0.994939 |

## Scale UNKNOWN for every row in this run

No arm in these logs emitted `scale=` in its FSPEED header, so
this run is OLDER than the throughput/fixed-cost declaration and
the split below could not be made. The rows are listed under
FIXED-COST because that is the conservative bucket, but the label
is NOT a measurement here -- some of these lanes are genuinely
large (kmeans ships 4,000,000 x 32) and some are genuinely tiny
(krr ships 16 rows). Check each shape tag by hand before quoting
anything from this file, or re-run so the arms declare it.

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| et | cuml-et-gpu | ModuleNotFoundError: No module named 'cuml' |
| et | sklearn-et-cpu | GPU-PATH-ONLY: sklearn-et-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cpu | ModuleNotFoundError: No module named 'lightgbm' |
| et | lightgbm-cuda | ModuleNotFoundError: No module named 'lightgbm' |
| et | cuml-et-gpu | ModuleNotFoundError: No module named 'cuml' |
| et | sklearn-et-cpu | GPU-PATH-ONLY: sklearn-et-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cpu | ModuleNotFoundError: No module named 'lightgbm' |
| et | lightgbm-cuda | ModuleNotFoundError: No module named 'lightgbm' |
| gbdt-depthwise | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | catboost-cpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-depthwise | catboost-gpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-depthwise | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-depthwise | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-depthwise | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | catboost-cpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-depthwise | catboost-gpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-depthwise | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-depthwise | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | catboost-cpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-lossguide | catboost-gpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-lossguide | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | lightgbm-cpu | ModuleNotFoundError: No module named 'lightgbm' |
| gbdt-lossguide | lightgbm-cuda | ModuleNotFoundError: No module named 'lightgbm' |
| gbdt-lossguide | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | catboost-cpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-lossguide | catboost-gpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-lossguide | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | lightgbm-cpu | ModuleNotFoundError: No module named 'lightgbm' |
| gbdt-lossguide | lightgbm-cuda | ModuleNotFoundError: No module named 'lightgbm' |
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-symmetric | catboost-cpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-symmetric | catboost-gpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-symmetric | catboost-cpu | ModuleNotFoundError: No module named 'catboost' |
| gbdt-symmetric | catboost-gpu | ModuleNotFoundError: No module named 'catboost' |
| iforest | cuml-iforest-gpu | ModuleNotFoundError: No module named 'cuml' |
| iforest | sklearn-iforest-cpu | GPU-PATH-ONLY: sklearn-iforest-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| iforest | cuml-iforest-gpu | ModuleNotFoundError: No module named 'cuml' |
| iforest | sklearn-iforest-cpu | GPU-PATH-ONLY: sklearn-iforest-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | cuml-rf-gpu | ModuleNotFoundError: No module named 'cuml' |
| rf | sklearn-rf-cpu | GPU-PATH-ONLY: sklearn-rf-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | ModuleNotFoundError: No module named 'lightgbm' |
| rf | lightgbm-cuda | ModuleNotFoundError: No module named 'lightgbm' |
| rf | cuml-rf-gpu | ModuleNotFoundError: No module named 'cuml' |
| rf | sklearn-rf-cpu | GPU-PATH-ONLY: sklearn-rf-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | ModuleNotFoundError: No module named 'lightgbm' |
| rf | lightgbm-cuda | ModuleNotFoundError: No module named 'lightgbm' |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

| lane | arm | shape | distinct hashes across rounds |
|---|---|---|---|
| rf | ours | synthclf-720000x100 | 2 |

## Notes the arms printed

- `forest.et.r1000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.et.r2000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-depthwise.r1000000.log`: lane=gbdt-depthwise arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-depthwise.r2000000.log`: lane=gbdt-depthwise arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-lossguide.r1000000.log`: lane=gbdt-lossguide arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-lossguide.r2000000.log`: lane=gbdt-lossguide arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r1000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r2000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.iforest.r1000000.log`: lane=iforest arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.iforest.r2000000.log`: lane=iforest arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r1000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r2000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=gpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so

