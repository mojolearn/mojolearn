"""kern_bench.py: py-dn-kern before/after timings with output digests.
Run with PYTHONPATH=<tree>/python; MOJOLEARN_VENDOR=cpu for the CPU column.
env SCALE=cpu shrinks the query counts (the old host paths are serial)."""
import hashlib, os, sys, time
import numpy as np
import mojolearn as ml
from mojolearn._spectral_impl import SpectralEmbedding

failures = []

col = "cpu" if os.environ.get("MOJOLEARN_VENDOR") == "cpu" else "gpu"
small = os.environ.get("SCALE") == "cpu"
rng = np.random.default_rng(7)


def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def run(name, f):
    t = time.perf_counter()
    try:
        out = f()
        dt = time.perf_counter() - t
        print(f"BENCH {col} {name} {dt:.3f}s {h(out) if not isinstance(out, tuple) else ','.join(h(o) for o in out)}", flush=True)
    except Exception as e:
        failures.append(name)
        print(f"BENCH {col} {name} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)


nf, d = 5000, 16
Xf = rng.standard_normal((nf, d)).astype(np.float32)
nq = 20000 if small else 200000
Q = rng.standard_normal((nq, d)).astype(np.float32)
kp = ml.KernelPCA(n_components=8, kernel="rbf").fit(Xf)
run(f"kpca_transform_{nq}x{nf}", lambda: kp.transform(Q))
oc = ml.OneClassSVM(kernel="rbf", nu=0.2).fit(Xf[:3000])
run(f"ocsvm_score_{nq}", lambda: oc.score_samples(Q))
ns = 20000 if small else 200000
Xs = rng.standard_normal((ns, d)).astype(np.float32)
ys = rng.standard_normal(ns).astype(np.float32)
sv = ml.SVGP(n_inducing=64)
run(f"svgp_fit_{ns}", lambda: sv.fit(Xs, ys).q_mu_)
run(f"svgp_predict_{ns}", lambda: sv.predict_f(Xs))
n = 4000 if small else 10000
Xe = rng.standard_normal((n, 8)).astype(np.float32)
run(f"spectral_embedding_rbf_{n}", lambda: SpectralEmbedding(n_components=2, affinity="rbf", random_state=0).fit(Xe).embedding_)
D = np.abs(rng.standard_normal((n, n))).astype(np.float32)
D = (D + D.T) / 2
se = SpectralEmbedding(n_components=2, affinity="precomputed_nearest_neighbors", n_neighbors=10, random_state=0)
run(f"precomputed_knn_affinity_{n}", lambda: se._precomputed_knn_affinity(D).out())
run(f"spectral_embedding_pknn_{n}", lambda: SpectralEmbedding(n_components=2, affinity="precomputed_nearest_neighbors",
                                                                 n_neighbors=10, random_state=0).fit(D).embedding_)

if failures:
    print("INCOMPLETE/FAILED cases:", ",".join(failures), flush=True)
    raise SystemExit(1)
