# linear: progress

Design (pass 1): every fit is one sequential Mojo function in `x_linear/`
(`fit_dispatch`), called directly by the host binding and from a ONE-THREAD
kernel by the GPU binding, so CPU and GPU run the same operation sequence;
scoring (`x_linear_decision`) is one thread per (row, output). Parallel fit
schedules are pass 2 speed work. Sanity: `cd python && PYTHONPATH=/root/skl
python -m mojolearn.tests.test_x_linear_sanity <case>` on the pod (sklearn
installed with `pip --target /root/skl --no-deps scikit-learn scipy joblib
threadpoolctl cloudpickle narwhals`).

Skipped from the table: LinearSVC / LinearSVR are already public
(`mojolearn.svm`, lanes linear-svc*, linear-svr*).

| algorithm | commit | lanes | pod gate |
|---|---|---|---|
| SGDClassifier / SGDRegressor | 7f23b8d5c | x-sgd-clf, x-sgd-reg | sanity PASS; RESULT: PASS (AGREE on x-sgd-clf,x-sgd-reg), RTX 4090 |
| PoissonRegressor / GammaRegressor / TweedieRegressor | 273ecb094 | x-glm-poisson, x-glm-gamma, x-glm-tweedie | sanity PASS (coef within 2e-4 of sklearn); RESULT: PASS (AGREE on all three) |
| HuberRegressor | b81926452 | x-huber | sanity PASS (coef within 2e-5 of sklearn); RESULT: PASS (AGREE on x-huber) |
| BayesianRidge / ARDRegression | 762aa5ef9 | x-bayes-ridge, x-ard | sanity PASS (coef within 2e-6 of sklearn); RESULT: PASS (AGREE on both) |
| Lars / LassoLars | 90634cac4 | x-lars, x-lasso-lars | sanity PASS (coef within 2e-6 of sklearn); RESULT: PASS (AGREE on both) |
| QuantileRegressor | 73aa9bc9b | x-quantile | sanity PASS (objective within 3e-7 relative of sklearn's LP optimum); RESULT: PASS (AGREE) |
| Perceptron | 01e1dcdd0 | x-perceptron | sanity PASS (accuracy within 0.04 of sklearn, 2 and 3 classes); RESULT: PASS (AGREE) |
| PassiveAggressiveClassifier / PassiveAggressiveRegressor | 3d534315b | x-pa-clf, x-pa-reg | sanity PASS (accuracy within 0.02, R2 within 4e-4 of sklearn); RESULT: PASS (AGREE on both) |
| SGDOneClassSVM | feb2ad310 | x-sgd-ocsvm | sanity PASS (outlier fraction equal to sklearn's at nu 0.1 and 0.5); RESULT: PASS (AGREE) |
| RidgeClassifier | f372aaf47 | x-ridge-clf | sanity PASS (coef within 1e-6 of sklearn, predictions equal); RESULT: PASS (AGREE) |
| RidgeCV | e8dfdf91f | x-ridge-cv | sanity PASS (same alpha_, best_score_ within 4e-7 relative); RESULT: PASS (AGREE) |
| LassoCV | 0f6bad966 | x-lasso-cv | sanity PASS (same alpha_, mse_path_ within 1.2e-6 relative); RESULT: PASS (AGREE) |
| ElasticNetCV | 468117252 | x-enet-cv | sanity PASS (same alpha_ and l1_ratio_, mse_path_ within 1e-6 relative); RESULT: PASS (AGREE) |
| LogisticRegressionCV | 3b545850a | x-logistic-cv (and x-huber re-gated: lbfgs.mojo line search changed) | sanity PASS (StratifiedKFold ids equal, same C_, proba within 2e-4); RESULT: PASS (AGREE on x-logistic-cv, x-huber) |
| IsotonicRegression | 7e02a9bc4 | x-isotonic | sanity PASS (thresholds equal to sklearn, predict within 3e-7, same NaN mask); RESULT: PASS (AGREE) |

## Pass 2

Seam ledger: DEVIATIONS 5000-5009, IDENTITY_PATHS rows 100-109, x_linear/README.md.
Check driver x_linear/checks/seams_check.mojo (oracle seams_oracle.mojo), ten
arms in tools/identity_lanes/linear.checks. End-to-end sabotage for the
stewards: x_linear/checks/sabotage/e2e_device_fold.patch (every lane but
x-isotonic) and e2e_device_pava.patch (x-isotonic; the interp arm did not bite, replaced): device-only source edits.

| step | commit | result |
|---|---|---|
| seams + ledger | 0e7290fb4 (isotonic e2e arm swapped in 406335b92) | NVIDIA RTX 4090: `algos_lane_check.sh <all 21> --pass 2` RESULT: PASS, all 10 arms FAIL under their patch and PASS after reversal, 21 lanes AGREE |
| option: GLM sample_weight (Poisson/Gamma/Tweedie) | 651e359c8 | sanity PASS (weighted coef within 1.2e-4 of sklearn); lanes x-glm-* AGREE, x-glm-poisson-sw AGREE; opt_glm_sample_weight.patch (device drops weights) DISAGREE then AGREE; x-glm-poisson/gamma/tweedie hashes UNCHANGED vs the pass-2 run |
| option: sample_weight for HuberRegressor, QuantileRegressor, SGDClassifier/Regressor, Perceptron, PassiveAggressive*, SGDOneClassSVM, RidgeClassifier, RidgeCV, BayesianRidge, LogisticRegressionCV; class_weight for SGDClassifier, Perceptron, PassiveAggressiveClassifier, RidgeClassifier, LogisticRegressionCV | 289237990 | sanity PASS against sklearn for each; 8 new lanes (x-huber-sw, x-quantile-sw, x-sgd-clf-w, x-sgd-reg-sw, x-ridge-clf-w, x-ridge-cv-sw, x-bayes-ridge-sw, x-logistic-cv-w) AGREE; every device-only weights arm DISAGREE then AGREE; the 26 earlier lanes' hashes UNCHANGED |
| option: positive for LassoLars, LassoCV, ElasticNetCV | (the commit that adds this row) | sanity PASS (coef within 2e-3 of sklearn, all >= 0); lanes x-lasso-lars-pos, x-lasso-cv-pos AGREE; device-only arms DISAGREE then AGREE; earlier lanes UNCHANGED |

Directive 000 (session 3): the seam bites above were recorded before the
lane-check fix (3084ca09c). Re-run ONCE on the RTX 4090 pod with the fixed tool
(box tools/algos_lane_check.py md5 = origin/main's) at 7be5da1e3:
`algos_lane_check.sh x-isotonic --pass 2`. All ten arms (5000-5009, 5006 as
seam_5006_host_nan.patch) BUILD, RUN and FAIL under their patch, PASS after
reversal; no BROKEN arm; clean lane AGREE. Done; never repeat.

Stewards (the earlier m2pro FAILs were the 5006 arm that is null on Arm, fixed
in a3156303f): submitted at 7be5da1e3 to m2pro + do-amd:
- 1790535754001-linear-7be5da1e3d: the 31 lanes other than x-isotonic, sabotage e2e_device_fold.patch
- 1790535762888-linear-7be5da1e3d: x-isotonic, sabotage e2e_device_pava.patch

Phase now: (1) proof, waiting on those two verdicts
(`python3 tools/apple_steward.py status | grep linear-7be5da1e3d`). PASS on
m2pro and do-amd for both = proof phase done: merge, then STOP.
A FAIL: fix the cause, resubmit only the affected lanes.
Next phase: (2) option parity. Remaining NOT IMPLEMENTED rows in
x_linear/NOT_IMPLEMENTED.tsv (SGD early_stopping/average/warm_start/partial_fit,
GLM/Huber warm_start, Tweedie power ranges, Bayes compute_score/return_std,
Lars paths and LarsCV/LassoLarsCV/LassoLarsIC, RidgeClassifier solvers/positive,
RidgeCV k-fold/multi-target, CD CV splitters/selection=random/sample_weight,
LogisticRegressionCV l1/elasticnet) AND the existing public linear models
(python/mojolearn/linear_model.py: LinearRegression, Ridge, Lasso, ElasticNet,
LogisticRegression) against sklearn and cuML options.
