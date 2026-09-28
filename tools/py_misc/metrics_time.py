"""lane py-misc-metrics: before/after seconds of each moved epilogue at 1M,
in one process: the Python reference route (MOJOLEARN_METRICS_EPILOGUE=python,
the before) and the native route (the after), the same call each, one warm
call then one timed call per arm, and the two results' bits compared.
Usage: metrics_time.py gpu|cpu (the column comes from MOJOLEARN_VENDOR)."""
import os, sys, time, struct, hashlib
import numpy as np
import mojolearn as ml
from mojolearn import metrics as mt

rng = np.random.default_rng(0)
N = 1_000_000


def digest(v):
    h = hashlib.sha256()
    if isinstance(v, tuple):
        for a in v:
            h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    else:
        h.update(struct.pack("<d", float(v)))
    return h.hexdigest()[:16]


def arm(route, f):
    if route == "python":
        os.environ["MOJOLEARN_METRICS_EPILOGUE"] = "python"
    else:
        os.environ.pop("MOJOLEARN_METRICS_EPILOGUE", None)
    f()
    t = time.perf_counter()
    r = f()
    return time.perf_counter() - t, digest(r)


yb = (rng.random(N) < 0.3).astype(np.int64)
sc = rng.random(N).astype(np.float32)
sw = (rng.random(N) + 0.5).astype(np.float32)
k = 5
yk = rng.integers(0, k, N)
P = rng.random((N, k)).astype(np.float32)
P /= P.sum(axis=1, keepdims=True)
P = P.astype(np.float32)
rel = rng.integers(0, 4, (N // 5, 5)).astype(np.float32)
rs = rng.random((N // 5, 5)).astype(np.float32)
xa = np.sort(rng.random(N))
ya = rng.random(N)
la = rng.integers(0, 1000, N)
lb = (la + rng.integers(0, 50, N)) % 1000
nX, kc, dc = 100_000, 300, 50
Xc = rng.random((nX, dc)).astype(np.float32)
lc = rng.integers(0, kc, nX)

cases = [
    ("precision_recall_curve w", "1M", lambda: tuple(mt.precision_recall_curve(yb, sc, sample_weight=sw, drop_intermediate=True))),
    ("det_curve", "1M", lambda: tuple(mt.det_curve(yb, sc))),
    ("roc_auc ovr weighted w", "1M x 5", lambda: mt.roc_auc_score(yk, P, multi_class="ovr", average="weighted", sample_weight=sw)),
    ("d2_log_loss_score w", "1M x 5", lambda: mt.d2_log_loss_score(yk, P, sample_weight=sw)),
    ("ndcg_score w", "200k x 5", lambda: mt.ndcg_score(rel, rs, sample_weight=sw[:N // 5])),
    ("auc", "1M points", lambda: mt.auc(xa, ya)),
    ("normalized_mutual_info", "1M, 1000 x 1000", lambda: mt.normalized_mutual_info_score(la, lb)),
    ("calinski_harabasz", "100k x 50, k 300", lambda: mt.calinski_harabasz_score(Xc, lc)),
    ("davies_bouldin", "100k x 50, k 300", lambda: mt.davies_bouldin_score(Xc, lc)),
]
col = sys.argv[1] if len(sys.argv) > 1 else "?"
print(f"column {col} vendor {os.environ.get('MOJOLEARN_VENDOR', 'gpu')}")
print("| case | size | python s | native s | speedup | bits |")
print("|---|---|---|---|---|---|")
bad = 0
for name, size, f in cases:
    try:
        tp, dp = arm("python", f)
        tn, dn = arm("native", f)
    except Exception as e:
        print(f"| {name} | {size} | ERROR {type(e).__name__}: {e} | | | |")
        bad += 1
        continue
    same = "equal" if dp == dn else f"DIFFER {dp} {dn}"
    bad += dp != dn
    print(f"| {name} | {size} | {tp:.3f} | {tn:.3f} | {tp / max(tn, 1e-9):.1f}x | {same} |")
print("TIMING BITS", "EQUAL" if not bad else f"NOT EQUAL ({bad})")

raise SystemExit(1 if bad else 0)
