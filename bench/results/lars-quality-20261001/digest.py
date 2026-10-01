"""Lars/LassoLars output digests: no-crossing cases (must not move) and crossing cases (DEVIATION 5010)."""
import hashlib, sys, numpy as np, mojolearn as ml
def h(m):
    b = np.asarray(m.coef_, np.float32).tobytes() + np.float32(m.intercept_).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]
def data(n=600, d=8, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32); w = rng.standard_normal(d).astype(np.float32)
    return X, (X @ w + noise * rng.standard_normal(n)).astype(np.float32)
def corr(n, d, seed):
    rng = np.random.default_rng(seed); Z = rng.standard_normal((n, 3))
    X = Z @ rng.standard_normal((3, d)) + 0.3 * rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d) + rng.standard_normal(n)
    return X.astype(np.float32), y.astype(np.float32)
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lars_inputs.npz")
if not os.path.exists(P):
    arrs = {}
    arrs["X"], arrs["y"] = data()
    for n, d, s in ((60, 5, 58), (100, 6, 44), (200, 8, 2)):
        arrs["X%d" % s], arrs["y%d" % s] = corr(n, d, s)
    np.savez(P, **arrs)
I = np.load(P)
print("inputs", hashlib.sha256(b"".join(I[k].tobytes() for k in sorted(I.files))).hexdigest()[:16])
X, y = I["X"], I["y"]
rows = []
for fi in (True, False):
    for nz in (3, 500):
        rows.append(("nocross lars fi=%s nz=%d" % (fi, nz), ml.Lars(fit_intercept=fi, n_nonzero_coefs=nz).fit(X, y)))
    for a in (0.5, 0.05, 0.001):
        rows.append(("nocross lassolars fi=%s a=%g" % (fi, a), ml.LassoLars(alpha=a, fit_intercept=fi).fit(X, y)))
for n, d, s in ((60, 5, 58), (100, 6, 44), (200, 8, 2)):
    Xc, yc = I["X%d" % s], I["y%d" % s]
    rows.append(("cross lars %dx%d s%d" % (n, d, s), ml.Lars(n_nonzero_coefs=500).fit(Xc, yc)))
    rows.append(("cross lassolars %dx%d s%d" % (n, d, s), ml.LassoLars(alpha=0.001).fit(Xc, yc)))
for tag, m in rows:
    print("%-34s %s n_iter=%s" % (tag, h(m), m.n_iter_))
