#!/usr/bin/env python3
"""ab_arms.py: native entries vs their Python reference arms in ONE tree, same process data.
Prints SAME/DIFF per case. MOJOLEARN_VENDOR=cpu for the CPU column."""
import hashlib, os, time
import numpy as np
import mojolearn as ml

def h(*a):
    m = hashlib.sha256()
    for x in a:
        m.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return m.hexdigest()[:16]

def arm(env, f):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        t = time.perf_counter(); r = f(); return r, time.perf_counter() - t
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

r = np.random.default_rng(5)
X = (r.standard_normal((20000, 8)) @ r.standard_normal((8, 8))).astype(np.float32)
C = np.floor(r.random((3000, 200)) ** 4 * 4).astype(np.float32)
cases = {
 "mcd-400x6": ({"MOJOLEARN_XD_MCD_PYTHON": "1"}, lambda: (lambda m: h(m.location_, m.covariance_, m.dist_, np.int8(m.support_)))(ml.MinCovDet(random_state=0).fit(np.ascontiguousarray(X[:400, :6])))),
 "ee-400x6": ({"MOJOLEARN_XD_MCD_PYTHON": "1"}, lambda: (lambda e: h(e.decision_function(X[:400, :6]), np.float64(e.offset_)))(ml.EllipticEnvelope(contamination=0.05, random_state=1, support_fraction=0.7).fit(np.ascontiguousarray(X[:400, :6])))),
 "mcd-1200x5": ({"MOJOLEARN_XD_MCD_PYTHON": "1"}, lambda: (lambda m: h(m.location_, m.covariance_, m.dist_, np.int8(m.support_)))(ml.MinCovDet(random_state=3).fit(np.ascontiguousarray(X[:1200, :5])))),
 "mcd-5000x8": ({"MOJOLEARN_XD_MCD_PYTHON": "1"}, lambda: (lambda m: h(m.location_, m.covariance_, m.dist_, np.int8(m.support_)))(ml.MinCovDet(random_state=0).fit(np.ascontiguousarray(X[:5000])))),
 "mds-nm": ({"MOJOLEARN_XD_MDS_PYTHON": "1"}, lambda: h(ml.MDS(metric_mds=False, max_iter=15, n_init=1, random_state=0).fit_transform(X[:300, :4]))),
}
failures = []

col = "cpu" if os.environ.get("MOJOLEARN_VENDOR") == "cpu" else "gpu"
for name, (env, f) in cases.items():
    try:
        a, ta = arm({}, f); b, tb = arm(env, f)
        if a != b:
            failures.append(name)
        print(f"ARM {col} {name} {'SAME' if a == b else 'DIFF'} native={a} {ta:.2f}s python={b} {tb:.2f}s", flush=True)
    except Exception as e:
        failures.append(name)
        import traceback; traceback.print_exc()
        print(f"ARM {col} {name} ERROR {type(e).__name__}: {str(e)[:300]}", flush=True)

if failures:
    print("INCOMPLETE/FAILED cases:", ",".join(failures), flush=True)
    raise SystemExit(1)
