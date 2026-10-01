import os, sys, time, json, warnings
import numpy as np
which = sys.argv[1]
z = np.load("/root/lars-quality/reg-istella.npz"); X, y, Xq, yq = z["X"], z["y"], z["Xq"], z["yq"]
def r2(t, p):
    t = t.astype(np.float64); p = np.asarray(p, np.float64); return 1 - ((t-p)**2).sum()/((t-t.mean())**2).sum()
t0 = time.time()
if which == "ols":
    A = np.c_[X.astype(np.float64), np.ones(len(X))]
    w = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)[0]
    p, pt = Xq @ w[:-1] + w[-1], X @ w[:-1] + w[-1]; info = {}
elif which == "sk":
    from sklearn.linear_model import Lars
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        m = Lars(n_nonzero_coefs=500, fit_intercept=True, random_state=0, eps=2.220446049250313e-16).fit(X, y)
    p, pt = m.predict(Xq), m.predict(X)
    info = dict(n_iter=int(m.n_iter_), n_active=len(m.active_), alpha=float(m.alphas_[-1]), warns=len(W), coefnorm=float(np.abs(m.coef_).max()))
else:
    import mojolearn as ml
    m = ml.Lars(n_nonzero_coefs=500, fit_intercept=True, random_state=0).fit(X, y)
    c = np.asarray(m.coef_.tolist() if hasattr(m.coef_, "tolist") else m.coef_, np.float64)
    p, pt = Xq @ c + m.intercept_, X @ c + m.intercept_
    info = dict(n_iter=m.n_iter_, n_active=len(m.active_), alpha=m.alpha_, coefnorm=float(np.abs(c).max()), intercept=m.intercept_)
    np.save("/root/lars-quality/coef_%s.npy" % which, c)
print(which, "heldout_r2=%.4f train_r2=%.4f secs=%.1f" % (r2(yq, p), r2(y, pt), time.time()-t0), json.dumps(info))
