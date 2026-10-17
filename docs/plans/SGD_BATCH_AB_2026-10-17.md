# SGD-family batch_size A/B (2026-10-17)

Arm A `batch_size='auto'` (0.8.47 default: `n_samples // 32`, cap 4096 for SGD, 256 for Perceptron/PA) against arm B the
fixed default (4096 for SGDClassifier/SGDRegressor, 256 for Perceptron/PassiveAggressiveClassifier). Published 0.8.49 wheel,
NVIDIA L40S and AMD MI325X, one fit per arm, `tools/sgd_batch_ab.py`. Ratio = auto_ms / fixed_ms (above 1: 'auto' slower).
Quality: holdout accuracy (classifiers) or R2 (SGDRegressor); quality is identical on both vendors.

| Estimator | rows | NV auto ms | NV fixed ms | NV ratio | AMD auto ms | AMD fixed ms | AMD ratio | quality auto | quality fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGDClassifier | 1,000 | 6.379 | 1.583 | 4.03 | 12.358 | 3.410 | 3.62 | 0.940000 | 0.955000 |
| SGDClassifier | 10,000 | 21.914 | 1.837 | 11.93 | 33.891 | 3.534 | 9.59 | 0.968000 | 0.969500 |
| SGDClassifier | 100,000 | 17.728 | 9.357 | 1.89 | 30.263 | 17.352 | 1.74 | 0.977950 | 0.976350 |
| Perceptron | 1,000 | 4.625 | 1.670 | 2.77 | 9.144 | 2.795 | 3.27 | 0.940000 | 0.960000 |
| Perceptron | 10,000 | 4.884 | 6.057 | 0.81 | 7.726 | 9.407 | 0.82 | 0.958000 | 0.967000 |
| Perceptron | 100,000 | 52.776 | 52.743 | 1.00 | 81.609 | 81.732 | 1.00 | 0.970450 | 0.970450 |
| SGDRegressor | 1,000 | 17.576 | 66.297 | 0.27 | 34.866 | 144.659 | 0.24 | 0.995854 | 0.983048 |
| SGDRegressor | 10,000 | 41.653 | 68.089 | 0.61 | 66.929 | 136.529 | 0.49 | 0.994761 | 0.989420 |
| SGDRegressor | 100,000 | 102.989 | 94.920 | 1.08 | 216.176 | 193.672 | 1.12 | 0.995496 | 0.995294 |
| PassiveAggressiveClassifier | 1,000 | 17.622 | 12.618 | 1.40 | 34.306 | 20.000 | 1.72 | 0.945000 | 0.925000 |
| PassiveAggressiveClassifier | 10,000 | 38.962 | 38.586 | 1.01 | 62.159 | 59.869 | 1.04 | 0.971000 | 0.971500 |
| PassiveAggressiveClassifier | 100,000 | 210.225 | 210.217 | 1.00 | 328.000 | 327.748 | 1.00 | 0.979250 | 0.979250 |

## Verdict (rule: if 'auto' loses, the default goes back to fixed)

- SGDClassifier: 'auto' 3.6-4.0x slower at 1k rows, 9.6-11.9x at 10k, 1.7-1.9x at 100k; accuracy 0.940 vs 0.955 (1k), 0.968 vs 0.9695 (10k). LOSES: default back to `batch_size=4096`.
- Perceptron: 'auto' 2.8-3.3x slower at 1k, 0.8x at 10k, equal at 100k; accuracy 0.940 vs 0.960 (1k), 0.958 vs 0.967 (10k). LOSES: default back to `batch_size=256`.
- SGDRegressor: 'auto' 4x faster at 1k, 2x at 10k, 1.1x slower at 100k; R2 0.9959 vs 0.9830 (1k), 0.9948 vs 0.9894 (10k). KEEPS 'auto'.
- PassiveAggressiveClassifier: 'auto' 1.4-1.7x slower at 1k only; accuracy 0.945 vs 0.925 (1k), equal above. KEEPS 'auto' (quality first).

'auto' stays accepted on all four estimators. The change is in `python/mojolearn/_expansion_linear.py` (constructor defaults and docstrings only; no kernel change).

## Evidence

- `~/mojolearn-evidence/sgd-batch-ab-20261017/nvidia/summary.txt` (failures=0, rc=0)
- `~/mojolearn-evidence/sgd-batch-ab-20261017/amd/summary.txt` (failures=0, rc=0)

## Live test finding and final decision

The live test of the published 0.8.50 on Apple (M4, published 0.8.49 and 0.8.50 wheels; `breast_cancer`, 569 rows,
raw feature scales, 3-fold cross-validated accuracy, `random_state=0`) showed the fixed default collapsing on real
unscaled data. A fixed batch larger than the dataset is one full-batch step per epoch, which underfits or diverges on
raw feature scales. The A/B above used synthetic standard-normal data and could not show this.

| Estimator | fixed default (0.8.50) | 'auto' | 0 (per-sample) | scikit-learn | scaled data, all |
|---|---|---|---|---|---|
| SGDClassifier | 0.373 (4096) | 0.687 | 0.807 | 0.849 | 0.96-0.97 |
| Perceptron | 0.373 (256) | 0.759 | 0.907 | 0.791 | 0.95-0.97 |

Decision (orchestrator, 2026-10-17; quality before speed): 'auto' is the default again on all four estimators
(SGDClassifier, SGDRegressor, Perceptron, PassiveAggressiveClassifier), released as 0.8.51. The A/B verdict above for
SGDClassifier and Perceptron is superseded.

Open follow-up: even 'auto' trails per-sample and scikit-learn on unscaled small data (SGDClassifier 0.69 vs 0.81 vs
0.85), so the row-count rule itself (`n_samples // 32`, at least 32 steps an epoch) deserves a later lane: more steps per
epoch at small `n_samples`.

Evidence: `~/mojolearn-evidence/live-0850-apple/status.txt`, `~/mojolearn-evidence/live-0850-apple-fa/` (the venvs),
`~/mojolearn-evidence/briefs-2026-10-17/sgd-batch-auto-restore.md`.
