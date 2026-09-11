#!/usr/bin/env python3
"""svm-speed lane probe (runs on the pod only).

  python3 svm_probe.py stage  ROOT DATA [datasets...]   staged fit per dataset (MOJOLEARN_STAGE_TIMES=1)
  python3 svm_probe.py time   ROOT DATA ROUNDS [datasets...]   wall ms per fit, no stage clock
  python3 svm_probe.py hash   ROOT DATA [datasets...]   fit hashes: n=400/600/2000 synthetic + dataset fits
"""
import hashlib
import os
import sys
import time

import numpy as np

mode, root, data = sys.argv[1], sys.argv[2], sys.argv[3]
rest = sys.argv[4:]
os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
sys.path.insert(0, root + "/python")
import mojolearn  # noqa: E402
from mojolearn import _svm_impl as m  # noqa: E402

print("ROOT", root, "ext", m._extension(None).__file__, flush=True)


def load(ds):
    with np.load(os.path.join(data, "svc-%s.npz" % ds)) as z:
        return {k: np.ascontiguousarray(z[k]) for k in z.files}


def est_for(X):
    return mojolearn.SVC(C=1.0, kernel="rbf", gamma=1.0 / X.shape[1], tol=1e-3)


def fit_hash(est, Xq):
    h = hashlib.sha256()
    for name in ("dual_coef_", "support_", "intercept_", "support_vectors_", "n_support_"):
        if hasattr(est, name):
            arr = np.ascontiguousarray(np.asarray(getattr(est, name)))
            h.update(name.encode())
            h.update(arr.tobytes())
    dec = np.ascontiguousarray(np.asarray(est.decision_function(Xq[:256]), dtype=np.float32))
    h.update(dec.tobytes())
    return h.hexdigest()[:16]


if mode == "stage":
    for ds in rest:
        d = load(ds)
        os.environ.pop("MOJOLEARN_STAGE_TIMES", None)
        est_for(d["X"]).fit(d["X"], d["y"])  # warm
        os.environ["MOJOLEARN_STAGE_TIMES"] = "1"
        print("STAGE-BEGIN", ds, d["X"].shape, flush=True)
        est = est_for(d["X"])
        est.fit(d["X"], d["y"])
        sys.stdout.flush()
        print("STAGE-END", ds, "n_support", est.n_support_, flush=True)
        os.environ.pop("MOJOLEARN_STAGE_TIMES", None)
elif mode == "time":
    rounds = int(rest[0])
    for ds in rest[1:]:
        d = load(ds)
        est_for(d["X"]).fit(d["X"], d["y"])  # warm
        ms = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            est = est_for(d["X"])
            est.fit(d["X"], d["y"])
            ms.append((time.perf_counter() - t0) * 1e3)
        acc = float(np.mean(np.asarray(est.predict(d["Xq"])) == d["yq"]))
        print("TIME %s median_ms=%.1f all=%s acc=%.4f nsv=%s" % (
            ds, float(np.median(ms)), ",".join("%.1f" % v for v in ms), acc, est.n_support_), flush=True)
elif mode == "hash":
    rng = np.random.default_rng(0)
    for n in (400, 600, 2000):
        X = rng.standard_normal((n, 220)).astype(np.float32)
        y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(np.float32)
        try:
            est = m.SVC()
            est.fit(X, y)
            h = hashlib.sha256()
            for name in ("dual_coef_", "support_", "intercept_", "support_vectors_", "n_support_"):
                if hasattr(est, name):
                    arr = np.ascontiguousarray(np.asarray(getattr(est, name)))
                    h.update(name.encode())
                    h.update(arr.tobytes())
            dec = np.ascontiguousarray(np.asarray(est.decision_function(X[:256]), dtype=np.float32))
            h.update(dec.tobytes())
            print("HASH n=%d %s nsv=%s" % (n, h.hexdigest()[:16], getattr(est, "n_support_", "?")), flush=True)
        except Exception as e:  # noqa: BLE001
            print("HASH n=%d FAIL %s" % (n, str(e)[-200:]), flush=True)
    for ds in rest:
        d = load(ds)
        est = est_for(d["X"])
        est.fit(d["X"], d["y"])
        print("HASH ds=%s %s nsv=%s" % (ds, fit_hash(est, d["Xq"]), est.n_support_), flush=True)
