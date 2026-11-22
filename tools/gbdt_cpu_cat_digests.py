#!/usr/bin/env python3
"""Model digests of the GradientBoosting configurations the CPU binding used
to refuse with categorical columns (lane pv-gbdt-cpu-cat, 2026-11-22).

A testing tool, never product execution. One line per configuration:

    <name> digest=<model.digest_> text_sha256=<sha256 of model.model_> status=ok

or `status=error <message>` when the fit raises. Run it once with
`--device cpu` (the CPU box, a wheel or development build of the lane) and
once with `--device gpu` (the L40S); the two outputs must be equal line for
line. Every configuration is SymmetricTree, Plain boosting, explicit
learning rate and iteration count, so neither device resolves a different
default.

    python3 tools/gbdt_cpu_cat_digests.py --device cpu > cpu.txt
    python3 tools/gbdt_cpu_cat_digests.py --device gpu > gpu.txt
    diff cpu.txt gpu.txt

Parallel plan: the configurations are independent fits, split across worker
processes (one per allocated core on `--device cpu`, at most the number of
configurations). On `--device gpu` the default is one worker, because every
fit shares the one GPU and the measurement rules run cells serially within a
GPU; `--workers` overrides either.
"""
import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROWS = 3000
LOW = ["lo", "hi"]                          # 2 values: one-hot (one_hot_max_size 2)
MANY = ["c%02d" % i for i in range(40)]     # 40 values: CTR columns


def _data(binary):
    import numpy as np

    rng = np.random.default_rng(20261122)
    a = rng.integers(0, len(MANY), ROWS)
    b = rng.integers(0, len(MANY), ROWS)
    c = rng.integers(0, len(LOW), ROWS)
    num = rng.normal(size=(ROWS, 2))
    y = (0.05 * a - 0.03 * b + 0.7 * c + num[:, 0] - 0.5 * num[:, 1]
         + 0.1 * rng.normal(size=ROWS))
    if binary:
        y = (y > np.median(y)).astype(np.float32)
    else:
        y = y.astype(np.float32)
    rows = [[MANY[a[i]], MANY[b[i]], LOW[c[i]], float(num[i, 0]), float(num[i, 1])]
            for i in range(ROWS)]
    return rows, y


#: (name, loss, columns of the five above, cat_features within them, extra)
BASE = {"n_estimators": 20, "max_depth": 6, "learning_rate": 0.1,
        "boosting_type": "Plain", "random_state": 7}
CONFIGS = [
    # RMSE with categorical columns (was refused by name)
    ("rmse_onehot_default", "RMSE", [2, 3, 4], [0], {}),
    ("rmse_ctr_default", "RMSE", [0, 3, 4], [0], {}),
    ("rmse_ctr_onehot_default", "RMSE", [0, 1, 2, 3, 4], [0, 1, 2], {}),
    ("rmse_ctr_mcc2", "RMSE", [0, 1, 3, 4], [0, 1], {"max_ctr_complexity": 2}),
    ("rmse_ctr_bernoulli", "RMSE", [0, 3, 4], [0], {"bootstrap_type": "Bernoulli", "subsample": 0.7}),
    ("rmse_ctr_poisson", "RMSE", [0, 3, 4], [0], {"bootstrap_type": "Poisson"}),
    ("rmse_ctr_no_bootstrap_no_noise", "RMSE", [0, 3, 4], [0],
     {"bootstrap_type": "No", "random_strength": 0.0}),
    ("rmse_ctr_bfa_off", "RMSE", [0, 3, 4], [0], {"boost_from_average": False}),
    ("rmse_ctr_one_permutation", "RMSE", [0, 3, 4], [0], {"permutation_count": 1}),
    ("rmse_onehot_bernoulli", "RMSE", [2, 3, 4], [0], {"bootstrap_type": "Bernoulli"}),
    # Logloss with categorical columns under the bootstrap / score noise
    ("logloss_ctr_default", "Logloss", [0, 3, 4], [0], {}),
    ("logloss_ctr_bernoulli", "Logloss", [0, 3, 4], [0], {"bootstrap_type": "Bernoulli"}),
    ("logloss_ctr_poisson", "Logloss", [0, 3, 4], [0], {"bootstrap_type": "Poisson"}),
    ("logloss_ctr_noise_only", "Logloss", [0, 3, 4], [0], {"bootstrap_type": "No", "random_strength": 1.0}),
    ("logloss_ctr_mcc2", "Logloss", [0, 1, 3, 4], [0, 1], {"max_ctr_complexity": 2}),
    ("logloss_onehot_default", "Logloss", [2, 3, 4], [0], {}),
    ("logloss_onehot_bayesian_t2", "Logloss", [2, 3, 4], [0], {"bagging_temperature": 2.0}),
    # the numeric arms whose CPU bits this lane moves (the cursor update,
    # RMSE's noise scale): no categorical column
    ("rmse_numeric_default", "RMSE", [3, 4], None, {}),
    ("logloss_numeric_default", "Logloss", [3, 4], None, {}),
]


def _fit(job):
    name, loss, cols, cats, extra, device = job
    try:
        import mojolearn

        rows, y = _data(loss == "Logloss")
        X = [[r[c] for c in cols] for r in rows]
        if cats is None:
            import numpy as np

            X = np.asarray(X, dtype=np.float32)
        kw = dict(BASE)
        kw.update(extra)
        model = mojolearn.GradientBoosting(loss=loss, cat_features=cats, device=device, **kw)
        model.fit(X, y)
        text = model.model_
        if not isinstance(text, (bytes, bytearray)):
            text = str(text).encode()
        return "%s digest=%s text_sha256=%s status=ok" % (
            name, model.digest_, hashlib.sha256(text).hexdigest())
    except Exception as exc:  # report and keep going: every line is evidence
        msg = " ".join(str(exc).split())[:300]
        return "%s status=error %s: %s" % (name, type(exc).__name__, msg)


def _cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated configuration names")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    jobs = [(n, l, c, k, e, args.device) for (n, l, c, k, e) in CONFIGS if not only or n in only]
    workers = args.workers or (_cores() if args.device == "cpu" else 1)
    workers = max(1, min(workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(_fit, jobs):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
