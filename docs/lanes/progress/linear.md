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
| HuberRegressor | (the commit that adds this row) | x-huber | sanity PASS (coef within 2e-5 of sklearn); RESULT: PASS (AGREE on x-huber) |

Next: BayesianRidge / ARDRegression.
