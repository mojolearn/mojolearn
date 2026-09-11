| set | lane | dataset | rows | mode | arm | rounds | median ms | min..max | hash | FSPEED-ACC | log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | gbdt-depthwise | istella | 1000000 | full | ours | 5 | 3376 | 3332..3501 | 5d053cd086658072 | metric=logloss value=0.126517 metric=auc value=0.971896 | baseline.gbdt-depthwise.istella.r1000000.full.xgbgpu |
| baseline | gbdt-depthwise | istella | 1000000 | full | xgboost-gpu | 5 | 2851 | 2827..2864 | 4d80a2001a82f486 | metric=logloss value=0.124822 metric=auc value=0.973880 | baseline.gbdt-depthwise.istella.r1000000.full.xgbgpu |
| baseline | gbdt-depthwise | taxi | 1000000 | full | ours | 5 | 1030 | 1020..1046 | 40c1683b9e0eb151 | metric=logloss value=0.525086 metric=auc value=0.621421 | baseline.gbdt-depthwise.taxi.r1000000.full.xgbgpu |
| baseline | gbdt-depthwise | taxi | 1000000 | full | xgboost-gpu | 5 | 594 | 561..612 | 348bf22bf60a14bc | metric=logloss value=0.525221 metric=auc value=0.620149 | baseline.gbdt-depthwise.taxi.r1000000.full.xgbgpu |
| baseline | gbdt-lossguide | istella | 1000000 | full | ours | 5 | 4172 | 4126..4311 | 6182fd2bee4fb941 | metric=logloss value=0.122045 metric=auc value=0.975037 | baseline.gbdt-lossguide.istella.r1000000.full.xgbgpu |
| baseline | gbdt-lossguide | istella | 1000000 | full | xgboost-gpu | 5 | 3297 | 3285..3327 | 4d80a2001a82f486 | metric=logloss value=0.124822 metric=auc value=0.973880 | baseline.gbdt-lossguide.istella.r1000000.full.xgbgpu |
| baseline | gbdt-lossguide | taxi | 1000000 | full | ours | 5 | 1727 | 1717..1783 | 0dd8bcfc3c3a4a1d | metric=logloss value=0.525504 metric=auc value=0.619386 | baseline.gbdt-lossguide.taxi.r1000000.full.xgbgpu |
| baseline | gbdt-lossguide | taxi | 1000000 | full | xgboost-gpu | 5 | 792 | 756..797 | 348bf22bf60a14bc | metric=logloss value=0.525221 metric=auc value=0.620149 | baseline.gbdt-lossguide.taxi.r1000000.full.xgbgpu |
