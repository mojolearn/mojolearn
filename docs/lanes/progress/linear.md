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
x-isotonic) and e2e_device_interp.patch (x-isotonic): device-only source edits.

| step | commit | result |
|---|---|---|
| seams + ledger | (the commit that adds this row) | NVIDIA RTX 4090: `algos_lane_check.sh <all 21> --pass 2` RESULT: PASS, all 10 arms FAIL under their patch and PASS after reversal, 21 lanes AGREE |

Next: AMD box (Hot Aisle fallback running), AGREE on AMD for all 21 lanes;
M2 Pro steward submissions; then option parity; then speed.
