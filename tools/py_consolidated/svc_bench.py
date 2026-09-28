"""svc_bench.py: SVC probability=True fit, predict_proba, predict and
decision_function(ovr) wall times plus sha256 digests of every output, one
process, for the tree on PYTHONPATH. env: NB (binary rows), NM (multiclass
rows), K (classes), NQ (query rows), D (features)."""
import hashlib, os, sys, time
import numpy as np
import mojolearn as ml
D = int(os.environ.get("D", "28")); NB = int(os.environ.get("NB", "50000"))
NM = int(os.environ.get("NM", "20000")); K = int(os.environ.get("K", "10")); NQ = int(os.environ.get("NQ", "200000"))
col = "cpu" if os.environ.get("MOJOLEARN_VENDOR") == "cpu" else "gpu"
rng = np.random.default_rng(11)
W = rng.standard_normal((D, K)).astype(np.float32)
def data(n):
    X = rng.standard_normal((n, D)).astype(np.float32)
    s = X @ W + rng.standard_normal((n, K)).astype(np.float32) * 2
    return X, s
Q = rng.standard_normal((NQ, D)).astype(np.float32)
h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
def t(label, f):
    s = time.perf_counter(); r = f(); e = time.perf_counter() - s
    print(f"TIME {col} {label:28s} {e:9.3f} s  {h(r) if r is not None else ''}", flush=True)
    return r
Xb, sb = data(NB); yb = (sb[:, 0] > 0).astype(np.int64)
mb = t(f"binary fit prob n={NB}", lambda: ml.SVC(probability=True, random_state=0).fit(Xb, yb))
print("probA", h(mb.probA_), "probB", h(mb.probB_))
t(f"binary predict_proba q={NQ}", lambda: mb.predict_proba(Q))
t(f"binary predict q={NQ}", lambda: mb.predict(Q))
Xm, sm = data(NM); ym = np.argmax(sm, axis=1).astype(np.int64)
mm = t(f"K={K} fit prob n={NM}", lambda: ml.SVC(probability=True, random_state=0).fit(Xm, ym))
print("probA", h(mm.probA_), "probB", h(mm.probB_))
Qm = Q[: NQ // 4]
t(f"K={K} predict_proba q={len(Qm)}", lambda: mm.predict_proba(Qm))
t(f"K={K} predict q={len(Qm)}", lambda: mm.predict(Qm))
t(f"K={K} decision ovr q={len(Qm)}", lambda: mm.decision_function(Qm))
mm.break_ties = True
t(f"K={K} predict break_ties", lambda: mm.predict(Qm))
