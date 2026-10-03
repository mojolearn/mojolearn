#!/usr/bin/env python3
"""Same-bits check of lane cgr2-metrics-shap (2026-10-03): the curve scores
whose weighted prefix, collinear drop and AUC / AP sums moved to the device
(roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
the weighted percentile under median_absolute_error), and the model-agnostic
explainers now chunked on the device (KernelExplainer, PermutationExplainer).

Each case runs a fixed seeded fixture twice, in separate subprocesses: once
with MOJOLEARN_VENDOR unset (the device column) and once with
MOJOLEARN_VENDOR=cpu (the host column), digests the result's words, and
prints

    IDCHECK <case> device=<digest> host=<digest> MATCH|DIFFER

The device digest is comparable across boxes (NVIDIA = AMD = Apple = host).
No BLAS anywhere in the fixtures (elementwise float64 only).
Usage: python tools/metrics_shap_idcheck.py [case ...]
"""
import hashlib
import json
import os
import subprocess
import sys

CASES = ["auc", "auc-w", "auc-w-pfpr", "auc-pfpr", "auc-ovr-w", "ap", "ap-w", "ap-ovr-w", "roc-w", "roc-w-all",
         "pr-w", "mae-w", "kshap-full", "kshap-sampled-logit", "pshap"]


def _curve_fixture(n=60000, seed=11):
    import numpy as np
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(np.int64)
    s = np.round(rng.random(n) * 0.6 + 0.4 * y * rng.random(n), 3).astype(np.float32)  # ties
    w = rng.uniform(0.2, 3.0, n).astype(np.float32)
    w[rng.random(n) < 0.03] = 0.0
    return y, s, w


def _multi_fixture(n=30000, k=3, seed=12):
    import numpy as np
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n).astype(np.int64)
    raw = rng.random((n, k)) + 0.5 * (np.arange(k)[None, :] == y[:, None])
    p = raw / raw.sum(axis=1, keepdims=True)
    w = rng.uniform(0.2, 3.0, n).astype(np.float32)
    return y, p.astype(np.float32), w


def _model_fixture(d, n=24, nb=8, seed=13):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    bg = rng.standard_normal((nb, d)).astype(np.float32)
    c = rng.standard_normal(d)
    return X, bg, c


def _lin(X, c):
    import numpy as np
    acc = np.zeros(X.shape[0], dtype=np.float64)
    for j in range(X.shape[1]):
        acc += X[:, j].astype(np.float64) * float(c[j])
    acc += X[:, 0].astype(np.float64) * X[:, 1].astype(np.float64)
    return acc


def _np(v, dtype):
    """A numpy array of v (a mojolearn Array goes through tolist)."""
    import numpy as np
    if hasattr(v, "tolist") and not isinstance(v, np.ndarray):
        v = v.tolist()
    return np.asarray(v, dtype=dtype)


def _digest(*vals):
    h = hashlib.sha256()
    for v in vals:
        for x in (v if isinstance(v, tuple) else (v,)):
            h.update(_np(x, "float64").tobytes())
    return h.hexdigest()[:16]


def _metrics():
    try:
        import mojolearn.metrics as M
        if hasattr(M, "roc_auc_score"):
            return M
    except ImportError:
        pass
    from mojolearn import _metrics_impl as M
    return M


def _run_case(case):
    import numpy as np
    M = _metrics()
    if case.startswith(("auc", "ap", "roc", "pr")) and "ovr" not in case:
        y, s, w = _curve_fixture()
        sw = w if "-w" in case else None
        if case.startswith("auc"):
            mf = 0.3 if case.endswith("pfpr") else None
            return _digest(M.roc_auc_score(y, s, sample_weight=sw, max_fpr=mf))
        if case.startswith("ap"):
            return _digest(M.average_precision_score(y, s, sample_weight=sw))
        if case.startswith("roc"):
            return _digest(M.roc_curve(y, s, sample_weight=sw, drop_intermediate=(case != "roc-w-all")))
        return _digest(M.precision_recall_curve(y, s, sample_weight=sw))
    if case in ("auc-ovr-w", "ap-ovr-w"):
        y, p, w = _multi_fixture()
        if case == "auc-ovr-w":
            return _digest(M.roc_auc_score(y, p, sample_weight=w, multi_class="ovr", average=None))
        return _digest(M.average_precision_score(y, p, sample_weight=w, average=None))
    if case == "mae-w":
        y, s, w = _curve_fixture()
        return _digest(M.median_absolute_error(s, (s * 0.5 + 0.25 * y).astype(np.float32), sample_weight=w))
    from mojolearn._expansion_trees import KernelExplainer, PermutationExplainer
    if case == "kshap-full":
        X, bg, c = _model_fixture(6)
        f = lambda Z: _lin(_np(Z, np.float32), c).astype(np.float32)  # noqa: E731
        e = KernelExplainer(f, bg, random_state=3)
        return _digest(e.shap_values(X), e.expected_value)
    if case == "kshap-sampled-logit":
        X, bg, c = _model_fixture(12)

        def f(Z):
            t = _lin(_np(Z, np.float32), c)
            q = 0.5 + 0.4 * t / (1.0 + np.abs(t))
            return np.stack([q, 1.0 - q], axis=1).astype(np.float32)
        e = KernelExplainer(f, bg, link="logit", random_state=3)
        return _digest(e.shap_values(X, nsamples=200), e.expected_value)
    if case == "pshap":
        X, bg, c = _model_fixture(8)
        f = lambda Z: _lin(_np(Z, np.float32), c).astype(np.float32)  # noqa: E731
        e = PermutationExplainer(f, bg, random_state=3)
        return _digest(e.shap_values(X, npermutations=5), e.expected_value)
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
