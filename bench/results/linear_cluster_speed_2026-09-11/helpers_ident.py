# DEVIATION 2632 identity: the native host helpers' output bytes, one tree per process.
import os, sys, hashlib, time
os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
sys.path.insert(0, sys.argv[1])
import numpy as np
import mojolearn
from mojolearn import linear_model as lm
from mojolearn._array import Array
from mojolearn._buffer import as_f32_c
def h(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
d = np.load("/root/ctd-data/big-taxi.npz")
rng = np.random.default_rng(7)
cases = {
  "taxi": (np.ascontiguousarray(d["X"]), np.ascontiguousarray(d["y"])),
  "planted_order": (rng.standard_normal((1 << 20 | 3, 5)).astype(np.float32) * np.float32(1e7), None),
  "wide": (rng.standard_normal((20000, 220)).astype(np.float32) * np.logspace(-3, 6, 220).astype(np.float32), None),
  "tiny": (rng.standard_normal((5, 7)).astype(np.float32), None),
}
# planted: a column whose sequential float64 total differs from its correctly rounded total
p = cases["planted_order"][0]; p[:, 0] = np.float32(1.0); p[0, 0] = np.float32(3.0e38); p[1, 0] = np.float32(-3.0e38)
for name, (X, y) in cases.items():
    x, _ = as_f32_c(X, ndim=2, name="X")
    t0 = time.perf_counter(); mu = lm._column_means(x, None); t1 = time.perf_counter()
    c = lm._center(x, mu); t2 = time.perf_counter()
    w = np.abs(rng.standard_normal(X.shape[0])).astype(np.float32) + np.float32(0.5)
    s = lm._scale_rows(x, [float(v) for v in np.sqrt(w)]); t3 = time.perf_counter()
    mu64 = lm._column_means_f64(x, X.shape[0], X.shape[1])
    out = [name, "mu64", h(np.array(mu64)), "mu32", h(np.array(mu, dtype=np.float32)), "center", h(c), "scale", h(s)]
    if y is not None:
        t = lm._target_1d(y, X.shape[0], "a", "b"); out += ["ymean", repr(lm._vector_mean(t, None))]
    print(" ".join(out), "ms_means=%.1f ms_center=%.1f ms_scale=%.1f" % ((t1-t0)*1e3, (t2-t1)*1e3, (t3-t2)*1e3), flush=True)
