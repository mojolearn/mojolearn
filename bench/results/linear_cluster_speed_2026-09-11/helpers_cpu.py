# Wall vs process CPU time of the native helpers, warm, one tree per process.
import os, sys, time, resource
os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
sys.path.insert(0, sys.argv[1])
import numpy as np
import mojolearn
from mojolearn import linear_model as lm
from mojolearn._buffer import as_f32_c, addr, addr_ro, empty, _native
from mojolearn._array import Array
d = np.load("/root/ctd-data/big-taxi.npz")
X = np.ascontiguousarray(d["X"]); x, _ = as_f32_c(X, ndim=2, name="X")
rows, cols = X.shape
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
def bench(name, fn, n=6):
    fn()
    walls, cpus = [], []
    for _ in range(n):
        c0 = cpu(); t0 = time.perf_counter(); fn(); t1 = time.perf_counter(); c1 = cpu()
        walls.append((t1 - t0) * 1e3); cpus.append((c1 - c0) * 1e3)
    walls.sort(); cpus.sort()
    print("%-28s wall_ms_med=%.1f cpu_ms_med=%.1f" % (name, walls[n // 2], cpus[n // 2]), flush=True)
mfn = _native("column_mean_f64"); cfn = _native("center_columns_f32")
out8 = empty((cols,), "<f8")
bench("column_mean_f64", lambda: mfn(addr_ro(x, name="X"), rows, cols, addr(out8, name="o")))
mu = Array.from_list([float(v) for v in lm._column_means(x, None)], "<f8")
pre = empty(x.shape, "<f4"); cfn(addr_ro(x, name="X"), rows, cols, addr_ro(mu, name="m"), addr(pre, name="o"))
bench("center_into_touched_out", lambda: cfn(addr_ro(x, name="X"), rows, cols, addr_ro(mu, name="m"), addr(pre, name="o")))
bench("empty_alloc_only", lambda: empty(x.shape, "<f4"))
def fresh():
    o = empty(x.shape, "<f4"); cfn(addr_ro(x, name="X"), rows, cols, addr_ro(mu, name="m"), addr(o, name="o"))
bench("center_into_fresh_out", fresh)
e = mojolearn.LinearRegression()
bench("LinearRegression.fit", lambda: e.fit(X, d["y"]), n=4)
init = np.ascontiguousarray(d["init"])
bench("KMeans.fit", lambda: mojolearn.KMeans(n_clusters=64, init="array", n_init=1, max_iter=20, tol=1e-7, init_centroids=init).fit(X), n=4)
