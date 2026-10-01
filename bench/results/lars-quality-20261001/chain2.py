import numpy as np, time, sys
sys.path.insert(0, "/root/lars-quality")
from port import lars_path
z = np.load("/root/lars-quality/reg-istella.npz"); X, y, Xq, yq = z["X"], z["y"], z["Xq"], z["yq"]
n, d = X.shape
g32 = np.load("/root/lars-quality/gram32chain.npz"); xm, ym = g32["xm"], g32["ym"]
g = np.load("/root/lars-quality/gram64.npz")
C = X - xm; yc = (y - ym).astype(np.float32)
def r2(t, p):
    t = t.astype(np.float64); return 1 - ((t-p)**2).sum()/((t-t.mean())**2).sum()
def report(tag, G, xty):
    rel = np.abs(G - g["G"]) / np.sqrt(np.outer(np.diag(g["G"]), np.diag(g["G"])) + 1e-30)
    c, info = lars_path(G, xty, n, 500, np.float32, flip=False); c = c.astype(np.float64)
    b = float(ym) - xm.astype(np.float64) @ c
    print("%-14s max rel err %.3g  heldout=%.4f train=%.4f k=%d skips=%d" % (tag, rel.max(), r2(yq, Xq @ c + b), r2(y, X @ c + b), info["k"], info["skips"]), flush=True)
# two-level: chains of B rows, then a chain over the block partials
B = 1024
G = np.zeros((d, d), np.float32); xty = np.zeros(d, np.float32)
for s in range(0, n, B):
    P = np.zeros((d, d), np.float32); p = np.zeros(d, np.float32)
    for i in range(s, min(s + B, n)):
        c = C[i]; P += np.multiply.outer(c, c); p += yc[i] * c
    G += P; xty += p
report("blocked1024", G, xty)
# Kahan (compensated) chain
G = np.zeros((d, d), np.float32); cG = np.zeros((d, d), np.float32); xty = np.zeros(d, np.float32); cx = np.zeros(d, np.float32)
for i in range(n):
    c = C[i]
    t = np.multiply.outer(c, c) - cG; s = G + t; cG = (s - G) - t; G = s
    t = yc[i] * c - cx; s = xty + t; cx = (s - xty) - t; xty = s
report("kahan", G, xty)
