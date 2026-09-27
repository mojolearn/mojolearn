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
| SGDOneClassSVM | (the commit that adds this row) | x-sgd-ocsvm | sanity PASS (outlier fraction equal to sklearn's at nu 0.1 and 0.5); RESULT: PASS (AGREE) |

Next: RidgeClassifier.
