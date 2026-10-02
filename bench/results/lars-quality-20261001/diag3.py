import numpy as np, sys
sys.path.insert(0, "/root/lars-quality")
from port import *
z = np.load("/root/lars-quality/reg-istella.npz"); X, y, Xq, yq = z["X"], z["y"], z["Xq"], z["yq"]
n = X.shape[0]
def r2(t, p):
    t = t.astype(np.float64); return 1 - ((t-p)**2).sum()/((t-t.mean())**2).sum()
for name in ("gram64", "gram32chain"):
    g = np.load("/root/lars-quality/%s.npz" % name); G, xty, xm, ym = g["G"], g["xty"], g["xm"].astype(np.float64), float(g["ym"])
    for dt in (np.float32, np.float64):
        c, info = lars_path(G, xty, n, 500, dt, flip=False); c = c.astype(np.float64)
        b = ym - xm @ c
        print("%-12s %-8s heldout=%.4f train=%.4f %s" % (name, dt.__name__, r2(yq, Xq @ c + b), r2(y, X @ c + b), info), flush=True)
