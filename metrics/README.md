# Metrics

GPU implementations of selected cuML metrics, their RAFT statistics primitives,
native Float32 MSE/MAE/RMSE, unweighted classification counts/scores, log loss
and binary ROC-AUC/precision-recall curves.

GPU log loss validation is limited to build/smoke
checks, with full numerical and cross-vendor qualification pending.
Binary ranking metrics (ROC-AUC and precision-recall) also have local build/smoke-only
validation and do not implement learning-to-rank objectives.

`NOT_IMPLEMENTED.tsv` is the explicit refusal ledger. Integer reductions can be order independent;
floating-point reductions must use a specified schedule when identity is promised.

## Verify

```bash
pixi run check-metrics-labels
pixi run check-regression-errors  # native + rebuilt public errors, all three modes
pixi run check-classification-metrics  # native + rebuilt public classification, all modes
pixi run check-metrics-regression
pixi run check-metrics-silhouette
pixi run check-metrics-trust
```

New metrics need a reference mapping, an adversarial fixture, and a documented reduction policy.
