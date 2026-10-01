import numpy as np
z = np.load("/root/datasets/gbm-bench/istella/istella_speed.npz")
xtr, rtr, xte, rte = z["x_train"], z["r_train"], z["x_test"], z["r_test"]
def stride(n, m):
    m = min(m, n); return (np.arange(m, dtype=np.int64) * n) // m
def clean(x):
    x = np.array(x, dtype=np.float32, order="C", copy=True)
    bad = ~np.isfinite(x) | (x >= np.finfo(np.float32).max); x[bad] = 0.0; return x
fi, ei = stride(xtr.shape[0], 1_000_000), stride(xte.shape[0], 100_000)
X, Xq = clean(xtr[fi]), clean(xte[ei])
f64 = X.astype(np.float64); mu = f64.mean(0); sd = f64.std(0); sd[sd == 0] = 1.0
X = np.ascontiguousarray(((f64 - mu) / sd).astype(np.float32))
Xq = np.ascontiguousarray(((Xq.astype(np.float64) - mu) / sd).astype(np.float32))
y = np.ascontiguousarray(np.asarray(rtr)[fi], dtype=np.float32)
yq = np.ascontiguousarray(np.asarray(rte)[ei], dtype=np.float32)
np.savez("/root/lars-quality/reg-istella.npz", X=X, y=y, Xq=Xq, yq=yq)
print(X.shape, Xq.shape, "const cols", int((f64.std(0) == 0).sum()), "y", y.mean(), y.std())
