# SGD-family quality on small unscaled data (lane sgd-quality, 2026-10-17)

Follow-up to `SGD_BATCH_AB_2026-10-17.md`: with `batch_size='auto'` (n // 32 rows a step) SGDClassifier
trailed per-sample updates and scikit-learn on raw breast_cancer (0.687 vs 0.807 vs 0.849, Apple M4 live test).

## Diagnosis (float64 model of `x_linear/sgd.mojo` `sgd_mb_one`, numpy, no binding)

Scripts and outputs: `~/mojolearn-evidence/sgd-quality-20261017/` (sim*.py, sim*.out). The model's per-sample
fit matches scikit-learn (synthetic unscaled 10k: 0.872 vs 0.885; breast_cancer 0.858 vs 0.844), so the
gap is the minibatch form, not the schedule constants (eta0, optimal t0, power_t, tol and n_iter_no_change
read the same as scikit-learn's `_plain_sgd`).

1. Mean step with the rate schedule counting batches: a batch of b rows moves the weights about as far as one
   per-sample update, so an epoch makes b times less progress and `n_iter_no_change` stops the fit early.
2. Steps per epoch: on unscaled data quality rises with the step count at every size.

Raw breast_cancer, 3-fold cv accuracy, mean of 8 seeds (sim5.out):

| steps an epoch | SGDClassifier mean | SGDClassifier summed | Perceptron summed | PA mean | SGDRegressor (diabetes R2) mean |
|---|---:|---:|---:|---:|---:|
| 32 (0.8.51) | 0.747 | 0.860 | 0.878 | 0.891 | 0.158 |
| 128 | 0.818 | 0.856 | 0.826 | 0.887 | 0.325 |
| 256 | 0.858 | 0.858 | 0.860 | 0.860 | 0.392 |
| per-sample | 0.858 | 0.858 | 0.860 | 0.860 | 0.392 |
| scikit-learn | 0.844 | | 0.881 | 0.898 | 0.392 |

Synthetic 10k rows, 32 features scaled up to 1000x, hinge, 4 seeds (sim4.out): 32 steps 0.717 mean / 0.751
summed, 128 steps 0.711 / 0.789, 512 steps 0.802 / 0.863, per-sample 0.872, scikit-learn 0.885.
Standard-normal data (sim3a.out): every form within 0.025 of scikit-learn at 1k / 10k / 100k (one seed).

## Changes

- `x_linear/sgd.mojo` `SGD_AUTO_STEPS` 32 -> 256 (`-D MOJOLEARN_SGD_AUTO_STEPS_256_OFF` restores 32). Cost: small
  batches (<= XG_TPB rows) run 64 steps a launch on the device's chunk arm, so 256 steps are four launches an
  epoch; between 65,536 and 1,048,576 rows the SGD pair's batch exceeds one block and pays up to 8x the old
  launches (speed A/B owed).
- `python/mojolearn/_expansion_linear.py` `_sgd_auto_sum`: SGDClassifier with `batch_size='auto'` and a
  bounded-gradient loss (hinge, log_loss, modified_huber, perceptron; not the pa1/pa2 rates) takes the summed
  step (`mb_step` bsum, the schedule counting rows), as Perceptron already did. An explicit batch size keeps the
  mean step (cuML MBSGD's form; the board pins `batch_size`). `MOJOLEARN_SGD_BATCH_SUM=0` restores the mean step.
- `x_linear/sgd.mojo` `sgd_mb_one`: the host's summed step read the raw `l1_ratio` for 'l2' and 'l1' (decay
  0.85 alpha against the devices' alpha); it now takes 0 / 1 like the device drivers.
- `python/mojolearn/_buffer.py` `_materialize`: object-dtype arrays of numbers convert like scikit-learn's
  check_array; a non-number leaf raises its TypeError (check_dtype_object).
- scikit-learn itself xfails check_sample_weight_equivalence_on_dense_data for SGDClassifier, Perceptron and
  SGDRegressor (`PER_ESTIMATOR_XFAIL_CHECKS`); the library's suite already declares the same, and
  PassiveAggressiveClassifier.fit takes no sample_weight (the check is skipped).

## Check (orchestrator, CPU box, built tree)

`python tools/sgd_quality_check.py <outdir>`: SGDQ lines (ours 'auto', ours per-sample, scikit-learn) and
SKCHECK lines for the four estimators; `SGDQ_SUMMARY ... rc=`.
