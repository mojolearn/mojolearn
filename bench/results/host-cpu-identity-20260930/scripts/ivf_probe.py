import sys, hashlib, numpy as np, mojolearn as ml
D=220
def h(*a):
    m=hashlib.sha256()
    for x in a: m.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return m.hexdigest()[:12]
def run(n, iters=20, probes=32, dump=None):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    rng.standard_normal(n)  # the harness draws y's noise here
    q = rng.standard_normal((n // 100, D)).astype(np.float32)
    idx = ml.IVFIndex(n_lists=min(1024, n // 4), n_probes=probes, n_neighbors=10, kmeans_n_iters=iters,
                      metric="sqeuclidean", random_state=7, numeric_mode="identical")
    idx.fit(X); out = idx.search(q)
    dist, ind = out
    print(n, iters, probes, "X", h(X,q), "centers", h(idx.centers_), "norms", h(idx.center_norms_), "offs", h(idx.list_offsets_),
          "ids", h(idx.list_indices_), "data", h(idx.list_data_), "dist", h(dist), "ind", h(ind), "digest", h(dist, ind), flush=True)
    if dump:
        np.savez(dump, centers=np.asarray(idx.centers_), norms=np.asarray(idx.center_norms_), offs=np.asarray(idx.list_offsets_),
                 ids=np.asarray(idx.list_indices_), dist=np.asarray(dist), ind=np.asarray(ind))
for a in sys.argv[1:]:
    p=a.split(":"); run(int(p[0]), int(p[1]) if len(p)>1 else 20, int(p[2]) if len(p)>2 else 32, p[3] if len(p)>3 else None)
