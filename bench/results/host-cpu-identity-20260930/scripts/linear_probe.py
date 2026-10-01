import sys, hashlib, numpy as np, mojolearn as ml
D=220
def inputs(n, ymode):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    nz = rng.standard_normal(n)
    if ymode == "matmul":
        xw = X @ w
    else:  # column loop: elementwise IEEE ops only, the same bits on every host
        acc = np.zeros(n, np.float64)
        for j in range(D):
            acc = acc + X[:, j].astype(np.float64) * float(w[j])
        xw = acc
    y = (xw + 0.1 * nz).astype(np.float32)
    return X, y
def h(*a): 
    m=hashlib.sha256()
    for x in a: m.update(np.ascontiguousarray(x).tobytes())
    return m.hexdigest()[:16]
def run(name, n, ymode, max_iter=100):
    X, y = inputs(n, ymode)
    if name=="sgd-reg":
        est = ml.SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4, max_iter=max_iter, tol=None, shuffle=True, random_state=7, learning_rate="constant", eta0=0.005, numeric_mode="identical")
    elif name=="lars":
        est = ml.Lars(n_nonzero_coefs=500, fit_intercept=True, random_state=7, numeric_mode="identical")
    else:
        print(name, n, ymode, "X", h(X), "y", h(y), flush=True); return
    est.fit(X,y)
    print(name, n, ymode, max_iter, "X", h(X), "y", h(y), "out", h(np.asarray(est.coef_), np.asarray(est.intercept_)), flush=True)
for a in sys.argv[1:]:
    p=a.split(":"); run(p[0], int(p[1]), p[2], int(p[3]) if len(p)>3 else 100)
