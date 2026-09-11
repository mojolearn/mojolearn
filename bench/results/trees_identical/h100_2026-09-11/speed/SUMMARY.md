# H100 confirmation leg, 2026-09-11 (branch `lane/nvidia-identical-trees-0911`, source 352d9781)

Box: RunPod NVIDIA H100 80GB HBM3, driver 580.126.09, kernel 6.8.0-106,
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04, pod bai1webdyjdqtx
04:00 to 04:26 UTC, reaped with HTTP 404 verified. Ours IDENTICAL only
(MOJOLEARN_NUMERIC_MODE=identical), alone in the process, no opponent ran.
Same GPU model, image and driver as the 2026-09-10 night leg (7cebeecf),
a different physical pod. `logs/batchP.sh` is the whole run in order;
`logs/ab.txt` the exit-code ledger; `logs/bins_sha256.txt` the four .so.

Timing, HIGGS first-N rows, ms median (min..max), `logs/summarize_speed.py speed/`:

| set | lane | dataset | rows | pass | rounds | median ms | min..max | hash | FSPEED-ACC (ours) | log |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | et | higgs | 1000000 |  | 5 | 2504 | 2476..2585 | 2c192f6b12dbb6c5 | metric=logloss value=0.622379 metric=auc value=0.762400 | baseline.et.higgs.r1000000.ours |
| baseline | gbdt-depthwise | higgs | 1000000 |  | 7 | 1016 | 986..1083 | 592afa74b0d96982 | metric=logloss value=0.525450 metric=auc value=0.813518 | baseline.gbdt-depthwise.higgs.r1000000.ours |
| baseline | gbdt-lossguide | higgs | 1000000 |  | 7 | 1610 | 1602..1649 | 4f02e5cb8088b281 | metric=logloss value=0.525348 metric=auc value=0.813238 | baseline.gbdt-lossguide.higgs.r1000000.ours |
| baseline | gbdt-symmetric | higgs | 1000000 | pass1 | 7 | 527 | 463..602 | dac2cf366e219cec | metric=logloss value=0.542067 metric=auc value=0.800716 | baseline.gbdt-symmetric.higgs.r1000000.ours.pass1 |
| baseline | gbdt-symmetric | higgs | 1000000 | pass2 | 7 | 493 | 472..578 | dac2cf366e219cec | metric=logloss value=0.542067 metric=auc value=0.800716 | baseline.gbdt-symmetric.higgs.r1000000.ours.pass2 |
| baseline | rf | higgs | 1000000 | pass1 | 5 | 1088 | 1071..1108 | efd14ab2c09ff57c | metric=logloss value=0.538817 metric=auc value=0.809830 | baseline.rf.higgs.r1000000.ours.pass1 |
| baseline | rf | higgs | 1000000 | pass2 | 5 | 1081 | 1051..1099 | efd14ab2c09ff57c | metric=logloss value=0.538817 metric=auc value=0.809830 | baseline.rf.higgs.r1000000.ours.pass2 |
| baseline | rf | higgs | 2000000 |  | 5 | 1760 | 1716..1792 | 7fd9fda29a4fa81d | metric=logloss value=0.536959 metric=auc value=0.811575 | baseline.rf.higgs.r2000000.ours |

Order run (interleaved so drift is visible): rf 1M pass1, symmetric 1M pass1,
rf 2M, symmetric 1M pass2, depthwise 1M, lossguide 1M, et 1M, rf 1M pass2,
then rf 1M stage (one untimed replicate, MOJOLEARN_STAGE_TIMES=1).

Opponent rows quoted by name from `bench/OPPONENT_REFERENCE.md`, never
re-run: cuML RF 3314 (Sep 9 row) at 1M, 4543.0 (Aug 28 row) at 2M; CatBoost
GPU symmetric 900 (Sep 9) / 846.1 (Aug 28); XGBoost GPU depthwise 617.3,
lossguide 816.9 (Aug 28); CatBoost depthwise 1232.5, lossguide 1600.6 and
LightGBM CUDA lossguide 1313.7 (Aug 28).

Fingerprints (`ib/`): vs the Sep 10 night H100 set `ib/sep10b_baseline.json`,
rf-clf DIVERGENT 9 of 9 (parts predict and proba), 72 of 72 IDENTICAL on
rf-reg, et-clf, et-reg, gbdt-symmetric, gbdt-depthwise, gbdt-lossguide,
gbdt-rmse, kmeans (`ib/diff.sep10b_baseline.baseline.txt`). vs the Apple M4
DEVIATION 2502 set `ib/apple_rf2502.json`: rf-clf and rf-reg 18 of 18
IDENTICAL (`ib/diff.apple_rf2502.baseline.txt`).
