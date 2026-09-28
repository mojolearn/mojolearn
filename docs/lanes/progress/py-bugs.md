# lane/py-bugs progress

Brief: ~/mojolearn-evidence/py_work_brief.md; audit: ~/mojolearn-evidence/python_work_audit.md.
Base: lane/apple2-merged 0a11b50c7, merged forward to bd6af0c4b (the before column is bd6af0c4b). Scope: the Python front door's correctness bugs.

## Changes

1. Host libm and interpreter-dependent float math (DEVIATIONS 6900-6903, IDENTITY_PATHS rows 250-253):
   - `_portable_math`: `exp_array` (C loop `mojolearn_exp_f64` over the pinned scalar exp), `nsum`
     (CPython 3.12+ `sum` spelled out), `powi` (correctly rounded integer power), `powr` (libmpdec pow),
     `erfc`, `normal_cdf`, `normal_inv_cdf` (fdlibm erfc, statistics' AS241, on the pinned exp/log).
   - `x ** 2` -> one product: `_expansion_metrics` d2_brier / calinski / davies (`_sq`), `model_selection`
     StratifiedGroupKFold std and the search std; SVGP lengthscale; x_linear regressor score.
   - stdlib exp/log -> pinned: cluster BGMM / ext GMM predict_proba, log_det_chol_, bic; LogisticRegressionCV
     predict_proba; AutoARIMA bic; johnson_lindenstrauss_min_dim.
   - platform pow -> `powi` / `powr`: CNN Adam hyper table, CatBoost `_c_round`, LogisticRegressionCV Cs grid.
   - IterativeImputer sample_posterior + user estimator: NormalDist -> pinned normal cdf / inv_cdf.
   - Python `sum` over floats -> `nsum` where touched (StratifiedGroupKFold, CNN loss_curve_, LogCV softmax,
     IterativeImputer's host rounds, BGMM log_det_chol_).
   - SVC Platt + coupling: ledger row 253 (DEVIATION 6903), no code change.
2. Search folds: GridSearchCV / RandomizedSearchCV / validation_curve draw folds ONCE (scikit-learn).
3. learning_curve: one permutation per fold, nested prefixes per size (scikit-learn).
4. RNN / LSTM / GRU: the row order is sent as int32; the 2^24 cap on epochs x rows is gone (now < 2^31 - 1).
5. x_linear grid_dim=1: finding only (below).

## Before / after bit record

(pending: tools/py_bugs/job.sh on the shared NVIDIA pod; base tree = 0a11b50c7)

## Findings

(pending)
