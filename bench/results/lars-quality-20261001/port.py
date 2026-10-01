"""numpy port of x_linear/lars.mojo's path (lar method), dtype-parametric; for diagnosis."""
import numpy as np, sys, time

def chol(A):
    try:
        L = np.linalg.cholesky(A.astype(np.float64)).astype(A.dtype)
        return L
    except np.linalg.LinAlgError:
        return None

def lars_path(G, xty, n, max_iter, dt, rel_tol=None, abs_tol=1e-7, flip=True):
    d = G.shape[0]; G = G.astype(dt); xty = xty.astype(dt)
    coef = np.zeros(d, dt); state = np.zeros(d, int); act = []; sgn = []
    eq = dt(np.finfo(np.float32).eps); tiny = dt(np.finfo(np.float32).tiny)
    k = 0; n_iter = 0; drop = False; skips = 0; flips = 0
    while True:
        cov = xty - G @ coef
        ina = np.where(state == 0)[0]
        if len(ina):
            c_idx = ina[np.argmax(np.abs(cov[ina]))]; cbig = abs(cov[c_idx])
        else:
            c_idx = -1; cbig = dt(0)
        alpha = cbig / dt(n)
        if alpha <= eq: break
        if n_iter >= max_iter or k >= d: break
        if not drop:
            if c_idx < 0: break
            A = act + [c_idx]
            L = chol(G[np.ix_(A, A)])
            piv = None if L is None else L[-1, -1]
            bad = L is None or piv < abs_tol or (rel_tol is not None and piv * piv < rel_tol * G[c_idx, c_idx])
            if bad:
                state[c_idx] = 2; skips += 1; continue
            state[c_idx] = 1; act = A; sgn.append(np.sign(cov[c_idx])); k += 1
        sg = np.array(sgn, dt)
        Gaa = G[np.ix_(act, act)]
        ls = np.linalg.solve(Gaa.astype(np.float64), sg.astype(np.float64)).astype(dt)
        aa = dt(1) / np.sqrt(dt((ls * sg).sum())); ls = ls * aa
        gamma = cbig / aa
        ina = np.where(state == 0)[0]
        if len(ina):
            cj = G[np.ix_(ina, act)] @ ls; cv = cov[ina]
            g1 = (cbig - cv) / (aa - cj + tiny); g2 = (cbig + cv) / (aa + cj + tiny)
            for g in (g1, g2):
                g = g[(g > 0)]
                if len(g): gamma = min(gamma, g.min())
        drop = False
        z = -coef[act] / (ls + tiny)
        zp = z[z > 0]
        if flip and len(zp) and zp.min() < gamma:
            idx = np.where(z == zp.min())[0]
            for a in idx: sgn[a] = -sgn[a]
            drop = True; flips += 1
        n_iter += 1
        coef[act] = coef[act] + gamma * ls
    cov = xty - G @ coef
    return coef, dict(maxactcov=float(np.abs(cov[act]).max()/n) if act else 0, minactcov=float(np.abs(cov[act]).min()/n) if act else 0, n_iter=n_iter, k=k, skips=skips, flips=flips, alpha=float(alpha))

def chain_gram(X, xm, dt=np.float32, chunk=1):
    """sequential float32 chain per cell (rows ascending), as centered_gram."""
    d = X.shape[1]; G = np.zeros((d, d), dt)
    C = (X - xm.astype(dt)).astype(dt)
    for i in range(X.shape[0]):
        c = C[i]
        G = (G.astype(np.float64) + np.outer(c, c).astype(np.float64)).astype(dt)  # fmad: one rounding (approx)
    return G
