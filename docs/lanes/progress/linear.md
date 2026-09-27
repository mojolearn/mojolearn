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
| SGDClassifier / SGDRegressor | (the commit that adds this row) | x-sgd-clf, x-sgd-reg | sanity PASS; RESULT: PASS (AGREE on x-sgd-clf,x-sgd-reg), RTX 4090 |

Next: PoissonRegressor / GammaRegressor / TweedieRegressor (x_linear/glm.mojo written).
