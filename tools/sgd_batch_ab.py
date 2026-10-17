#!/usr/bin/env python3
"""A/B of the SGD-family default minibatch: batch_size='auto' (0.8.47+) against the old fixed default.

    python tools/sgd_batch_ab.py --vendor nvidia --manifest <out>/wheel-manifest.json --out <out>

Runs on a GPU box against the PUBLISHED wheel (tools/sgd_batch_ab.sh installs it with
tools/pypi_wheel_install.py), never a source build. For SGDClassifier, SGDRegressor, Perceptron and
PassiveAggressiveClassifier at 1k, 10k and 100k rows x 64 features (one fixed seed, data made once per size
with numpy outside the clock, 20% holdout): arm A batch_size='auto', arm B the old fixed default (4096 for
SGD, 256 for Perceptron/PA); every other parameter explicit and identical on both arms. One untimed warm-up
fit per estimator at 1k rows, then ONE timed fit per arm (one seed, one run). Quality (holdout accuracy or
R2, from the estimator's own score) is recorded and shown, never a gate; the speed verdict is the ratio.

Writes <out>/results.jsonl (one row per cell) and prints, per estimator x size:
    SGDAB <est> rows=<n> auto_ms=<a> fixed_ms=<b> ratio=<a/b> auto_q=<..> fixed_q=<..>
"""
import argparse
import json
import os
import sys
import time

SIZES = (1_000, 10_000, 100_000)
FEATURES = 64
SEED = 20261017
HOLDOUT = 0.2
WARMUP_ROWS = 1_000
#: x_linear/sgd.mojo SGD_AUTO_STEPS: 'auto' takes n // 256 rows a step, at least 1, at most the cap
#: (32 before lane sgd-quality, 2026-10-17; -D MOJOLEARN_SGD_AUTO_STEPS_256_OFF builds the old rule).
AUTO_STEPS = 256

#: The fixed default each estimator had before d0064d844 (0.8.47), which is also its 'auto' cap.
FIXED_DEFAULT = {
    "SGDClassifier": 4096,
    "SGDRegressor": 4096,
    "Perceptron": 256,
    "PassiveAggressiveClassifier": 256,
}
TASK = {
    "SGDClassifier": "classification",
    "SGDRegressor": "regression",
    "Perceptron": "classification",
    "PassiveAggressiveClassifier": "classification",
}
#: Every parameter but batch_size, set explicitly (the library defaults, written out) so both arms match.
COMMON_PARAMS = {
    "SGDClassifier": dict(loss="hinge", penalty="l2", alpha=1e-4, l1_ratio=0.15, fit_intercept=True,
                          max_iter=1000, tol=1e-3, shuffle=True, epsilon=0.1, random_state=0,
                          learning_rate="optimal", eta0=0.0, power_t=0.5, early_stopping=False,
                          n_iter_no_change=5, class_weight=None, average=False, warm_start=False),
    "SGDRegressor": dict(loss="squared_error", penalty="l2", alpha=1e-4, l1_ratio=0.15, fit_intercept=True,
                         max_iter=1000, tol=1e-3, shuffle=True, epsilon=0.1, random_state=0,
                         learning_rate="invscaling", eta0=0.01, power_t=0.25, early_stopping=False,
                         n_iter_no_change=5, average=False, warm_start=False),
    "Perceptron": dict(penalty=None, alpha=1e-4, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
                       shuffle=True, eta0=1.0, random_state=0, early_stopping=False, validation_fraction=0.1,
                       n_iter_no_change=5, class_weight=None, warm_start=False),
    "PassiveAggressiveClassifier": dict(C=1.0, fit_intercept=True, max_iter=1000, tol=1e-3, early_stopping=False,
                                        validation_fraction=0.1, n_iter_no_change=5, shuffle=True,
                                        loss="hinge", random_state=0, warm_start=False, class_weight=None,
                                        average=False),
}
ESTIMATORS = tuple(FIXED_DEFAULT)
ARMS = ("auto", "fixed")


def arm_params(est, arm):
    """The constructor kwargs of one arm: the common set plus batch_size."""
    if est not in COMMON_PARAMS:
        raise ValueError("unknown estimator %r" % (est,))
    if arm not in ARMS:
        raise ValueError("arm must be 'auto' or 'fixed', not %r" % (arm,))
    params = dict(COMMON_PARAMS[est])
    params["batch_size"] = "auto" if arm == "auto" else FIXED_DEFAULT[est]
    return params


def resolved_batch(est, arm, n_train):
    """The minibatch the binding uses (x_linear/sgd.mojo sgd_batch), for the record only."""
    cap = FIXED_DEFAULT[est]
    if arm == "fixed":
        return cap
    return max(1, min(cap, n_train // AUTO_STEPS))


def n_train_rows(rows):
    return rows - int(round(rows * HOLDOUT))


def _fmt(v, spec):
    return "nan" if v is None else format(v, spec)


def summary_line(est, rows, auto_ms, fixed_ms, auto_q, fixed_q):
    ratio = (auto_ms / fixed_ms) if (auto_ms is not None and fixed_ms) else None
    return ("SGDAB %s rows=%d auto_ms=%s fixed_ms=%s ratio=%s auto_q=%s fixed_q=%s"
            % (est, rows, _fmt(auto_ms, ".3f"), _fmt(fixed_ms, ".3f"), _fmt(ratio, ".4f"),
               _fmt(auto_q, ".6f"), _fmt(fixed_q, ".6f")))


def make_data(np, rows, task):
    """One fixed-seed synthetic set, split 80/20; made once per size, outside every clock."""
    rng = np.random.default_rng(SEED + rows)
    x = rng.standard_normal((rows, FEATURES)).astype(np.float32)
    w = rng.standard_normal(FEATURES).astype(np.float32)
    z = x @ w + np.float32(0.5) * rng.standard_normal(rows).astype(np.float32)
    y = (z > 0).astype(np.int64) if task == "classification" else z.astype(np.float32)
    n_tr = n_train_rows(rows)
    return (np.ascontiguousarray(x[:n_tr]), np.ascontiguousarray(y[:n_tr]),
            np.ascontiguousarray(x[n_tr:]), np.ascontiguousarray(y[n_tr:]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", required=True, choices=("nvidia", "amd"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--estimators", default=",".join(ESTIMATORS))
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    args = ap.parse_args(argv)
    if os.environ.get("MOJOLEARN_VENDOR", "").lower() == "cpu":
        raise SystemExit("sgd_batch_ab: MOJOLEARN_VENDOR=cpu is never timed; unset it")

    import numpy as np  # noqa: PLC0415
    import mojolearn as ml  # noqa: PLC0415
    import mojolearn.linear_model as lm  # noqa: PLC0415

    vendor_used = str(ml.vendor())
    print("SGDAB-HEADER vendor=%s vendor_used=%s mojolearn=%s manifest=%s"
          % (args.vendor, vendor_used, getattr(ml, "__version__", "?"), args.manifest), flush=True)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "results.jsonl")
    estimators = [e for e in args.estimators.split(",") if e]
    sizes = [int(s) for s in args.sizes.split(",") if s]
    data = {}
    for task in sorted(set(TASK[e] for e in estimators)):
        for rows in sorted(set(sizes) | {WARMUP_ROWS}):
            data[(task, rows)] = make_data(np, rows, task)

    failures = 0
    with open(out_path, "a") as fh:
        for est in estimators:
            cls = getattr(lm, est)
            task = TASK[est]
            metric = "accuracy" if task == "classification" else "r2"
            xw, yw, _, _ = data[(task, WARMUP_ROWS)]
            cls(**arm_params(est, "auto")).fit(xw, yw)  # untimed warm-up: compile/launch setup
            for rows in sizes:
                xtr, ytr, xte, yte = data[(task, rows)]
                cell = {}
                for arm in ARMS:
                    params = arm_params(est, arm)
                    row = dict(vendor=args.vendor, vendor_used=vendor_used, estimator=est, rows=rows,
                               train_rows=int(xtr.shape[0]), features=FEATURES, seed=SEED, arm=arm,
                               batch_size_param=params["batch_size"],
                               batch_size_resolved=resolved_batch(est, arm, int(xtr.shape[0])),
                               metric=metric, manifest=os.path.abspath(args.manifest))
                    try:
                        model = cls(**params)
                        t0 = time.perf_counter()
                        model.fit(xtr, ytr)
                        fit_ms = (time.perf_counter() - t0) * 1e3
                        q = float(model.score(xte, yte))
                        row.update(fit_ms=fit_ms, quality=q, n_iter=_scalar(getattr(model, "n_iter_", None)),
                                   status="ok")
                    except Exception as exc:  # recorded, the run goes on
                        failures += 1
                        fit_ms = q = None
                        row.update(fit_ms=None, quality=None, status="error", error=repr(exc)[:400])
                    cell[arm] = (fit_ms, q)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                print(summary_line(est, rows, cell["auto"][0], cell["fixed"][0], cell["auto"][1],
                                   cell["fixed"][1]), flush=True)
    print("SGDAB-DONE vendor=%s failures=%d results=%s" % (args.vendor, failures, out_path), flush=True)
    return 1 if failures else 0


def _scalar(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
