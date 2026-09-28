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

NOT RUN. Job nvc1-0003 (shared pod nvc1, NVIDIA A40 + Xeon Gold 6342, started 19:39:34Z) was
cancelled at 20:06:11Z (exit 143) during the base column's binding builds, before any lane arm or
probe ran. Its trap re-applied the lane patch. The build outputs it made were deleted from the pod
tree. By Andrew's order no further job was submitted: lane py-consolidated runs the one global
check. The ready driver: tools/py_bugs/job.sh (one tree, `git apply -R` of
tools/py_bugs/lane_vs_base.patch for the before column), tools/py_bugs/check.py, tools/py_bugs/probe.py.
Regenerate the patch if lane code changes:
`git diff --binary <base> HEAD -- . ':!tools/py_bugs' ':!docs/lanes/progress/py-bugs.md' ':!IDENTITY_PATHS.md' ':!python/mojolearn/tests/test_py_bugs.py'`.

## Findings

- x_linear `grid_dim=1` (item 5) is REAL but narrower than the audit says. `x_linear/device.mojo`
  `fit_device` launches `fit_kernel` with `grid_dim=1`. On this base (and on the audited main
  6b9f6a5e8) `team_fit` names ALL 12 algorithms, so every fit runs on one block of LINEAR_TPB = 256
  threads, not on one thread. That is one SM of a 128-SM RTX 4090. Cost recorded by the linear lanes
  (docs/lanes/progress/linear-apple.md, 100k rows): the team fits run at about host speed
  (tweedie 0.118 s against a 0.122 s host, bayes-ridge 0.055 s against 0.053 s, lars 0.021 s against
  0.022 s); the old one-thread form took 5 to 6.6 s. The SGD family stays 20x to 35x slower than the
  host (sgd-clf 8.19 s against 0.24 s), because its pass is row-serial. linear-apple2's open item:
  "team fits whose gradient is one thread's chain per cell over all rows (Huber, LogisticRegressionCV,
  Poisson/Gamma, Quantile) remain at or slower than one host core at 100k rows". Moving past one block
  needs a multi-block fold schedule with the same chain order, which belongs to the linear lane.
  This lane did not change it.
- Measured on an M-series Mac: macOS `pow(10, -2.9333333333333336)` is one ulp below the correctly
  rounded value, so the old `LogisticRegressionCV(Cs=16)` grid, and every fit on it, differed between
  a Mac and a correctly rounded host. `powr` removes this.
- Measured on the Mac: the C `_statistics` accelerator's `inv_cdf` differs from statistics' own
  Python AS241 on 111,658 of 200,000 draws; the macOS erfc is up to 6 ulp off (fdlibm's is within 2).
- A pure-Python `sum()` over floats gives different bits on 3.10/3.11 than on 3.12+ (the wheel supports
  3.10+); only the touched sites were pinned (`nsum`). Others remain across the package.

## FINAL (2026-09-28)

Branch lane/py-bugs, pushed. What changed: items 1 to 4 above. DEVIATION rows: IDENTITY_PATHS registry
250-259 to py-bugs; rows 250 (6900, host libm), 251 (6901, `sum`), 252 (6902, NormalDist),
253 (6903, SVC Platt and coupling, the row that was owed); row 198 (6106) now forbids `**`/pow.

Unproven (nothing ran on any machine except the local pure-Python checks of the new helpers):
- every bit claim: expected SAME on every lane except where libm/pow was not correctly rounded
  (cluster predict_proba, LogCV predict_proba and Cs grid, AutoARIMA ic_, CNN loss_curve_ on 3.10/3.11,
  IterativeImputer posterior with a user estimator, JL near an integer). The unseeded search
  (item 2) and shuffled learning_curve (item 3) change output ON PURPOSE toward scikit-learn.
- `mojolearn_exp_f64` (new C symbol) was never compiled: the laptop builds nothing.
- sequence/pyapi.mojo (int32 order) was never compiled; the 17-epoch 1M-row RNN probe never ran.
- tests/test_py_bugs.py never ran under pytest (it needs the built package). Its AST scan ran locally
  and bites a planted `(a - b) ** 2`; nsum/powi/powr/erfc/inv_cdf were checked locally against the
  builtin, exact Fractions, mpmath and statistics' Python source.
- No timings before/after.
