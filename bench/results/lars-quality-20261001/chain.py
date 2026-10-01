import numpy as np, time
z = np.load("/root/lars-quality/reg-istella.npz"); X, y = z["X"], z["y"]
n, d = X.shape
t = time.time()
xm = np.zeros(d, np.float32); ys = np.float32(0)
for i in range(n): xm += X[i]
xm = (xm / np.float32(n)).astype(np.float32)
for i in range(n): ys = np.float32(ys + y[i])
ym = np.float32(ys / np.float32(n))
G = np.zeros((d, d), np.float32); xty = np.zeros(d, np.float32)
C = X - xm; yc = (y - ym).astype(np.float32)
for i in range(n):
    c = C[i]
    G += np.multiply.outer(c, c)
    xty += yc[i] * c
np.savez("/root/lars-quality/gram32chain.npz", G=G, xty=xty, xm=xm, ym=ym)
g = np.load("/root/lars-quality/gram64.npz")
rel = np.abs(G - g["G"]) / np.sqrt(np.outer(np.diag(g["G"]), np.diag(g["G"])) + 1e-30)
print("secs", time.time() - t, "max rel gram err", rel.max(), "median", np.median(rel), "xty abs err max", np.abs(xty - g["xty"]).max(), "xty max", np.abs(g["xty"]).max())
