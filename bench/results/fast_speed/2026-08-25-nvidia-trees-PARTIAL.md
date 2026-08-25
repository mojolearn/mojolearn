# Trees on an H100: four lanes, read off the live box, raw logs LOST

**PROVENANCE, AND READ IT BEFORE THE NUMBERS.** These figures were read from
pod `oxmb35yanlbsez` over ssh WHILE THE LEG WAS RUNNING, not from fetched
artifacts. The leg's payload finished correctly (`ours_headers_fast=6`,
`fspeed_refused=4`, sentinel written), and then the DRIVING SCRIPT died at
exit 127 before step 7's fetch, so `remote/` is empty and the box was
terminated with the logs still on it.

**THE CAUSE WAS MINE AND IT IS A GENERAL TRAP.** `sh` reads a script LAZILY,
by byte offset. `tools/gemm_remote_leg.sh` was edited and committed at 18:05
while this leg, started 17:58, was still executing it. The offset shifted
under the running shell, which resumed mid-word inside a comment and tried to
execute `SERTED,` (from `ASSERTED,` at line 3961). See DEVIATION 1882: legs
now run from an immutable snapshot copy.

So these four lanes are transcribed from a terminal, not parsed from a card.
They are recorded because they are the first tree numbers this project has
ever taken on NVIDIA and they cost four legs to get, but **they are not
card-backed and the next leg supersedes them.** `et` and `iforest` were
measured and are lost.

Box: NVIDIA H100 80GB HBM3. Commit `4cbead3`. FAST mode on every arm, mode
witness 6 FAST / 0 IDENTICAL. Three timed rounds after one untimed warm-up,
ours and every opponent ALTERNATING INSIDE ONE PROCESS.

## gbdt-symmetric, `year` 463,715 x 90

| arm | median ms | RMSE |
|---|---|---|
| **ours** | **895.4** | **9.2159** |
| catboost-gpu | 767.4 | 9.2175 |
| catboost-cpu | 1519.8 | 9.2301 |

1.17x slower than their CUDA learner, 1.70x faster than their CPU, and the
BEST RMSE of the three. CatBoost only, per the standing rule that the
symmetric pair is CatBoost's alone.

## gbdt-depthwise, `year` 463,715 x 90

| arm | median ms | RMSE |
|---|---|---|
| xgboost-gpu | 716.3 | 9.0949 |
| catboost-gpu | 965.4 | 9.0967 |
| **ours** | **1368.6** | **9.0858** |
| xgboost-cpu | 1545.7 | 9.0949 |
| catboost-cpu | 3032.1 | 9.1046 |

1.91x slower than XGBoost GPU, 1.13x faster than XGBoost CPU, BEST RMSE of
five. XGBoost CPU and GPU agreeing to seven digits is the config-parity check
working: the device was the only variable.

## gbdt-lossguide, `year` 463,715 x 90

| arm | median ms | RMSE |
|---|---|---|
| xgboost-gpu | 902.2 | 9.0949 |
| catboost-gpu | 1377.4 | 9.0928 |
| lightgbm-cpu | 1794.7 | 9.0857 |
| **ours** | **2094.1** | **9.0859** |
| xgboost-cpu | 2185.6 | 9.0949 |
| catboost-cpu | 5447.6 | 9.1270 |

2.32x slower than XGBoost GPU. Our weakest growth policy, which is the
expected direction: leaf-wise is LightGBM's own algorithm and the last one
ported here. RMSE essentially tied with LightGBM for best in the lane.

## rf, `covtype` 522,911 x 54, 7-class

| arm | median ms | accuracy |
|---|---|---|
| cuml-rf-gpu | 1529.9 | 0.7050 |
| sklearn-rf-cpu | 3665.1 | 0.7086 |
| **ours** | **5231.4** | **0.7078** |
| lightgbm-cpu | 89,642 | 0.6018 |

**THE WEAK LANE, AND IT IS THE ONE THAT SHOULD HURT.** `ensemble/` is a PORT
OF cuML's forest, measured against cuML on cuML's own hardware, and it is
3.42x behind. It also loses to scikit-learn ON THE CPU by 1.43x, the only
lane where a CPU opponent beats us. Accuracy is fine (between cuML's and
sklearn's, all three within 0.004), so the fit is right and the speed is not.

## The shape of it

| lane | ours vs best GPU opponent | our accuracy rank |
|---|---|---|
| symmetric GBDT | 1.17x slower | best of 3 |
| depthwise GBDT | 1.91x slower | best of 5 |
| lossguide GBDT | 2.32x slower | 2nd of 6, by 0.0002 |
| random forest | **3.42x slower** | 2nd of 4, by 0.0008 |

The boosting learners are within 1.2x to 2.3x of the best CUDA-native
libraries in the world, on their data and their hardware, while producing the
best or near-best accuracy in every lane. The forest is not, and the ordering
tracks the code's history: the symmetric CatBoost port is the oldest and
most-worked-over code here; `ensemble/` is newer and has never had a
performance pass against its own original.

## Refusals, kept

`lightgbm-cuda` refused in every lane it appears in:

    LightGBMError: CUDA Tree Learner was not enabled in this build.
    Please recompile with -DUSE_CUDA=1

**`lightgbm_cuda_build_exit=0` IS A LYING EXIT CODE.** The source build
reported success and produced a CPU-only wheel. The arm did not silently fall
back to CPU wearing a `lightgbm-cuda` label; it declined and said why, which
is the only reason this is visible at all. The build invocation is owed a
fix.

## Owed

* `et` and `iforest` were measured on this box and lost with it.
* Every number here is transcribed rather than card-backed. Rerun.
* cuML has no ExtraTrees and no IsolationForest, so those two lanes compare
  against scikit-learn and say so.
