import numpy as np, sys
sys.path.insert(0, "/root/lars-quality")
from port import *
z = np.load("/root/lars-quality/reg-istella.npz"); X, y, Xq, yq = z["X"], z["y"], z["Xq"], z["yq"]
g = np.load("/root/lars-quality/gram64.npz"); G64, xty64, xm, ym = g["G"], g["xty"], g["xm"], float(g["ym"])
n = X.shape[0]
def r2(t, p):
    t = t.astype(np.float64); return 1 - ((t-p)**2).sum()/((t-t.mean())**2).sum()
def score(c, tag, info):
    b = ym - xm @ c
    print("%-34s heldout=%.4f train=%.4f maxc=%.3g %s" % (tag, r2(yq, Xq @ c + b), r2(y, X @ c + b), np.abs(c).max(), info), flush=True)
for mi in (50, 500):
  for dt in (np.float64, np.float32):
    for flip in (True, False):
        c, info = lars_path(G64, xty64, n, mi, dt, flip=flip); score(c.astype(np.float64), "mi=%d %s flip=%s" % (mi, dt.__name__, flip), info)
