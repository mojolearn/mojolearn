# NumPy model (scratch, small n, one core): the RNN-round ward tree against the
# greedy Lance-Williams reference of x_cluster/agglo.mojo.
import sys
import numpy as np

def greedy(x):
    n = len(x)
    dm = ((x[:, None, :].astype(np.float32) - x[None, :, :].astype(np.float32)) ** 2).sum(-1).astype(np.float32)
    live = np.ones(n, bool); node = list(range(n)); size = np.ones(n)
    inf = np.float32(np.inf)
    children = []; dist = []
    for step in range(n - 1):
        best = None
        for i in range(n):
            if not live[i]: continue
            js = [j for j in range(i + 1, n) if live[j]]
            if not js: continue
            row = dm[i, js]
            q = int(np.argmin(row))
            if best is None or row[q] < best[0]:
                best = (row[q], i, js[q])
        dab, a, b = best
        lo, hi = sorted((node[a], node[b]))
        children.append((lo, hi)); dist.append(np.sqrt(dab))
        na, nb = size[a], size[b]
        for k in range(n):
            if not live[k] or k in (a, b): continue
            nk = size[k]
            v = ((na + nk) * float(dm[a, k]) + (nb + nk) * float(dm[b, k]) - nk * float(dab)) / (na + nb + nk)
            dm[a, k] = dm[k, a] = np.float32(v)
        live[b] = False; size[a] = na + nb; node[a] = n + step
    return children, np.array(dist)

def rounds(x):
    n, d = x.shape
    cen = x.astype(np.float64).copy(); sz = np.ones(n); node = list(range(n)); hgt = np.zeros(n, np.float32)
    ma = []; mb = []; mh = []
    L = n; nr = 0
    while L > 1:
        nr += 1
        c32 = cen[:L].astype(np.float32); s32 = sz[:L].astype(np.float32)
        diff = c32[:, None, :] - c32[None, :, :]
        acc = np.zeros((L, L), np.float32)
        for f in range(d):
            acc = acc + diff[:, :, f] * diff[:, :, f]
        coef = (np.float32(2) * s32[:, None] * s32[None, :]) / (s32[:, None] + s32[None, :])
        wd = (acc * coef).astype(np.float32)
        key = (wd.view(np.uint32).astype(np.uint64) << np.uint64(32)) | np.arange(L, dtype=np.uint64)[None, :]
        key[np.arange(L), np.arange(L)] = np.uint64(0xFFFFFFFFFFFFFFFF)
        kmin = key.min(1)
        nn = (kmin & np.uint64(0xFFFFFFFF)).astype(np.int64)
        md = (kmin >> np.uint64(32)).astype(np.uint32).view(np.float32)
        dead = np.zeros(L, bool); pairs = 0
        for p in range(L):
            q = int(nn[p])
            if q > p and int(nn[q]) == p:
                h = md[p]
                if h < hgt[p]: h = hgt[p]
                if h < hgt[q]: h = hgt[q]
                ma.append(node[p]); mb.append(node[q]); 
                node[p] = n + len(mh); mh.append(h)
                tot = sz[p] + sz[q]
                cen[p] = (sz[p] * cen[p] + sz[q] * cen[q]) / tot
                sz[p] = tot; hgt[p] = h; dead[q] = True; pairs += 1
        assert pairs > 0
        w = 0
        for p in range(L):
            if not dead[p]:
                if w != p:
                    cen[w] = cen[p]; sz[w] = sz[p]; node[w] = node[p]; hgt[w] = hgt[p]
                w += 1
        L = w
    mh = np.array(mh, np.float32)
    order = np.argsort(mh.view(np.uint32), kind="stable")
    rank = np.empty(len(mh), np.int64); rank[order] = np.arange(len(mh))
    fin = lambda t: t if t < n else n + int(rank[t - n])
    children = []
    for r, s in enumerate(order):
        a, b = fin(ma[s]), fin(mb[s])
        assert a < n + r and b < n + r, (a, b, r)
        children.append((min(a, b), max(a, b)))
    return children, np.sqrt(mh[order]), nr

def leaves_sets(children, n):
    sets = {i: frozenset([i]) for i in range(n)}
    out = []
    for t, (a, b) in enumerate(children):
        s = sets[a] | sets[b]; sets[n + t] = s; out.append(s)
    return out

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
for trial in range(6):
    n = 300; d = 8
    if trial % 2 == 0:
        x = rng.normal(size=(n, d)).astype(np.float32)
    else:
        cent = rng.normal(size=(6, d)) * 4
        x = (cent[rng.integers(0, 6, n)] + rng.normal(size=(n, d)) * 0.5).astype(np.float32)
    if trial == 4:
        x[:50] = x[50:100]  # planted duplicates (exact ties at height 0)
    cg, dg = greedy(x)
    cr, dr, nr = rounds(x)
    same_children = cg == cr
    sg, sr = leaves_sets(cg, n), leaves_sets(cr, n)
    same_tree = set(sg) == set(sr)
    top = sg[-7:] == sr[-7:] and cg[-7:] == cr[-7:]
    print(f"trial {trial} rounds={nr} children_equal={same_children} tree_equal={same_tree} top7_equal={top} "
          f"max_rel_height_diff={np.max(np.abs(dg - dr) / np.maximum(dg, 1e-30)):.2e}")
