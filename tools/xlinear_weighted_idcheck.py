#!/usr/bin/env python3
"""Same-bits check of the x_linear fits the board's ID lines do not reach
(cgr-linear, 2026-10-03): weighted BayesianRidge, weighted IsotonicRegression,
RidgeCV(cv=5), plus weighted RidgeCV (leave-one-out), weighted
RidgeClassifier, ARDRegression, Lars, weighted QuantileRegressor and weighted
LogisticRegressionCV.

Each case fits a fixed seeded fixture twice, in separate subprocesses: once
with MOJOLEARN_VENDOR unset (the device column) and once with
MOJOLEARN_VENDOR=cpu (the host column), digests the fitted attributes'
float32 words, and prints

    IDCHECK <case> device=<digest> host=<digest> MATCH|DIFFER

The device digest is comparable across boxes (NVIDIA = AMD = Apple = host).
Usage: python tools/xlinear_weighted_idcheck.py [case ...]
"""
import hashlib
import json
import os
import subprocess
import sys

CASES = ["bayes-ridge-w", "isotonic-w", "isotonic-w-dec", "ridge-cv5", "ridge-cv-loo-w", "ridge-clf-w",
         "ard", "lars", "quantile-w", "logreg-cv-w"]


def _fixture(n=6000, d=12, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    y = (X @ w + 0.5 * rng.standard_normal(n)).astype(np.float32)
    sw = rng.uniform(0.2, 3.0, n).astype(np.float32)
    sw[rng.random(n) < 0.05] = 0.0  # some zero weights (isotonic drops them)
    return X, y, sw


def _digest(est, names):
    import numpy as np
    h = hashlib.sha256()
    for nm in names:
        v = getattr(est, nm)
        if isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(np.asarray(v[k], dtype=np.float32).tobytes())
        else:
            h.update(np.asarray(v, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _run_case(case):
    import numpy as np
    import mojolearn as ml
    X, y, sw = _fixture()
    if case == "bayes-ridge-w":
        e = ml.BayesianRidge().fit(X, y, sample_weight=sw)
        return _digest(e, ["coef_", "intercept_", "alpha_", "lambda_"])
    if case in ("isotonic-w", "isotonic-w-dec"):
        x1 = X[:, 0].copy()
        e = ml.IsotonicRegression(increasing=(case == "isotonic-w")).fit(x1, y, sample_weight=sw)
        return _digest(e, ["X_thresholds_", "y_thresholds_"])
    if case == "ridge-cv5":
        e = ml.RidgeCV(alphas=(0.1, 1.0, 10.0), cv=5).fit(X, y)
        return _digest(e, ["coef_", "intercept_", "alpha_"])
    if case == "ridge-cv-loo-w":
        e = ml.RidgeCV(alphas=(0.1, 1.0, 10.0)).fit(X, y, sample_weight=sw)
        return _digest(e, ["coef_", "intercept_", "alpha_", "best_score_"])
    if case == "ridge-clf-w":
        yc = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(np.int64)
        e = ml.RidgeClassifier(alpha=1.0).fit(X, yc, sample_weight=sw)
        return _digest(e, ["coef_", "intercept_"])
    if case == "ard":
        e = ml.ARDRegression().fit(X, y)
        return _digest(e, ["coef_", "intercept_", "alpha_", "lambda_"])
    if case == "lars":
        e = ml.Lars().fit(X, y)
        return _digest(e, ["coef_", "intercept_"])
    if case == "quantile-w":
        e = ml.QuantileRegressor(quantile=0.7, alpha=0.01).fit(X[:2000], y[:2000], sample_weight=sw[:2000])
        return _digest(e, ["coef_", "intercept_"])
    if case == "logreg-cv-w":
        yc = (y > np.median(y)).astype(np.int64)
        e = ml.LogisticRegressionCV(Cs=3, max_iter=100).fit(X, yc, sample_weight=sw)
        return _digest(e, ["coef_", "intercept_", "C_", "scores_"])
    raise ValueError(case)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        try:
            print(json.dumps({"digest": _run_case(sys.argv[2])}))
        except Exception as ex:  # reported as the digest, so a refusal shows
            print(json.dumps({"digest": "ERROR:" + type(ex).__name__ + ":" + str(ex)[:120]}))
        return 0
    cases = sys.argv[1:] or CASES
    bad = 0
    for case in cases:
        out = {}
        for col in ("device", "host"):
            env = dict(os.environ)
            env.pop("MOJOLEARN_VENDOR", None)
            if col == "host":
                env["MOJOLEARN_VENDOR"] = "cpu"
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", case],
                               env=env, capture_output=True, text=True, timeout=900)
            dig = "ERROR:rc=%d" % p.returncode
            for line in p.stdout.splitlines()[::-1]:
                if line.startswith("{"):
                    dig = json.loads(line)["digest"]
                    break
            if dig.startswith("ERROR"):
                sys.stderr.write(p.stderr[-2000:])
            out[col] = dig
        ok = out["device"] == out["host"] and not out["device"].startswith("ERROR")
        bad += 0 if ok else 1
        print(f"IDCHECK {case} device={out['device']} host={out['host']} {'MATCH' if ok else 'DIFFER'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
