# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/fast_speed/do-2026-08-28_140119-amd-forest/run/logs`, 12 arm logs.

Device(s) reported by the arms themselves: Linux_x86_64

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| et | ours | higgs-1000000x28 | 5 | 18294.084 | 18279.575 | 18322.684 | 18647.365 | 2c192f6b12dbb6c5 |
| et | ours | higgs-2000000x28 | 5 | 35521.428 | 35465.419 | 35528.708 | 35854.525 | a9a9ff528583d3bd |
| iforest | ours | anomaly-500000x32 | 10 | 155.077 | 152.974 | 177.432 | 509.834 | 33d86f1901c5acce |
| rf | ours | higgs-1000000x28 | 5 | 4525.381 | 4492.729 | 4591.560 | 4874.337 | cf0e938bee35499e |
| rf | ours | higgs-2000000x28 | 5 | 6268.397 | 6196.748 | 6297.358 | 6573.895 | 273ada48b6515f6b |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| et | higgs-1000000x28 | (none ran) | 18294.084 | - | - | NO OPPONENT ON THIS BOX |
| et | higgs-2000000x28 | (none ran) | 35521.428 | - | - | NO OPPONENT ON THIS BOX |
| iforest | anomaly-500000x32 | (none ran) | 155.077 | - | - | NO OPPONENT ON THIS BOX |
| rf | higgs-1000000x28 | (none ran) | 4525.381 | - | - | NO OPPONENT ON THIS BOX |
| rf | higgs-2000000x28 | (none ran) | 6268.397 | - | - | NO OPPONENT ON THIS BOX |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| et | ours | logloss | 0.622379 |
| et | ours | auc | 0.762400 |
| et | ours | logloss | 0.624325 |
| et | ours | auc | 0.758920 |
| iforest | ours | auc | 1.000000 |
| iforest | ours | auc | 1.000000 |
| rf | ours | logloss | 0.538715 |
| rf | ours | auc | 0.810021 |
| rf | ours | logloss | 0.536984 |
| rf | ours | auc | 0.811575 |

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
| et | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| et | cuml-et-gpu | ModuleNotFoundError: No module named 'cuml' |
| et | sklearn-et-cpu | GPU-PATH-ONLY: sklearn-et-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| et | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| gbdt-depthwise | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-depthwise | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-depthwise | ours | Exception during warm-up: half-byte histograms on a 64-lane column: the binary/half-byte accumulator template is 32-lane by construction (DEVIATION 1910); this build excludes the sub-byte families -- write the wide-wavefront layout or promo |
| gbdt-depthwise | catboost-gpu | CatBoostError during warm-up: catboost/cuda/cuda_lib/cuda_base.h:265: CUDA error 35: CUDA driver version is insufficient for CUDA runtime version |
| gbdt-depthwise | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-depthwise | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-depthwise | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-depthwise | ours | Exception during warm-up: half-byte histograms on a 64-lane column: the binary/half-byte accumulator template is 32-lane by construction (DEVIATION 1910); this build excludes the sub-byte families -- write the wide-wavefront layout or promo |
| gbdt-depthwise | catboost-gpu | CatBoostError during warm-up: catboost/cuda/cuda_lib/cuda_base.h:265: CUDA error 35: CUDA driver version is insufficient for CUDA runtime version |
| gbdt-lossguide | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | ours | Exception during warm-up: half-byte histograms on a 64-lane column: the binary/half-byte accumulator template is 32-lane by construction (DEVIATION 1910); this build excludes the sub-byte families -- write the wide-wavefront layout or promo |
| gbdt-lossguide | catboost-gpu | CatBoostError during warm-up: catboost/cuda/cuda_lib/cuda_base.h:265: CUDA error 35: CUDA driver version is insufficient for CUDA runtime version |
| gbdt-lossguide | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| gbdt-lossguide | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | xgboost-cpu | GPU-PATH-ONLY: xgboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | xgboost-cpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | xgboost-gpu | ModuleNotFoundError: No module named 'xgboost' |
| gbdt-lossguide | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-lossguide | ours | Exception during warm-up: half-byte histograms on a 64-lane column: the binary/half-byte accumulator template is 32-lane by construction (DEVIATION 1910); this build excludes the sub-byte families -- write the wide-wavefront layout or promo |
| gbdt-lossguide | catboost-gpu | CatBoostError during warm-up: catboost/cuda/cuda_lib/cuda_base.h:265: CUDA error 35: CUDA driver version is insufficient for CUDA runtime version |
| gbdt-lossguide | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-symmetric | ours | Exception during warm-up: half-byte histograms on a 64-lane column: the binary/half-byte accumulator template is 32-lane by construction (DEVIATION 1910); this build excludes the sub-byte families -- write the wide-wavefront layout or promo |
| gbdt-symmetric | catboost-gpu | CatBoostError during warm-up: catboost/cuda/cuda_lib/cuda_base.h:265: CUDA error 35: CUDA driver version is insufficient for CUDA runtime version |
| gbdt-symmetric | catboost-cpu | GPU-PATH-ONLY: catboost-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| gbdt-symmetric | ours | Exception during warm-up: half-byte histograms on a 64-lane column: the binary/half-byte accumulator template is 32-lane by construction (DEVIATION 1910); this build excludes the sub-byte families -- write the wide-wavefront layout or promo |
| gbdt-symmetric | catboost-gpu | CatBoostError during warm-up: catboost/cuda/cuda_lib/cuda_base.h:265: CUDA error 35: CUDA driver version is insufficient for CUDA runtime version |
| iforest | cuml-iforest-gpu | ModuleNotFoundError: No module named 'cuml' |
| iforest | sklearn-iforest-cpu | GPU-PATH-ONLY: sklearn-iforest-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| iforest | cuml-iforest-gpu | ModuleNotFoundError: No module named 'cuml' |
| iforest | sklearn-iforest-cpu | GPU-PATH-ONLY: sklearn-iforest-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | cuml-rf-gpu | ModuleNotFoundError: No module named 'cuml' |
| rf | sklearn-rf-cpu | GPU-PATH-ONLY: sklearn-rf-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |
| rf | cuml-rf-gpu | ModuleNotFoundError: No module named 'cuml' |
| rf | sklearn-rf-cpu | GPU-PATH-ONLY: sklearn-rf-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cpu | GPU-PATH-ONLY: lightgbm-cpu is a CPU arm and this box has an accelerator. On NVIDIA and AMD we compare against the vendor's GPU path only; their CPU path is the MacBook's. |
| rf | lightgbm-cuda | LightGBMError during warm-up: CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs) |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

No arm's output hash moved across its rounds in this run.

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

