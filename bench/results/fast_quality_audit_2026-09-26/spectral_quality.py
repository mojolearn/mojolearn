import sys, json, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]
def ari(a,b):
    from collections import Counter
    n=len(a); c=Counter(zip(a,b)); ca=Counter(a); cb=Counter(b)
    s=sum(v*(v-1)/2 for v in c.values()); sa=sum(v*(v-1)/2 for v in ca.values()); sb=sum(v*(v-1)/2 for v in cb.values())
    e=sa*sb/(n*(n-1)/2); return (s-e)/((sa+sb)/2-e)
def trust(X,E,k=15,m=3000):
    idx=np.random.default_rng(0).choice(len(X),m,replace=False); X=X[idx].astype(np.float64); E=E[idx].astype(np.float64)
    d2=lambda A:(A*A).sum(1)[:,None]+(A*A).sum(1)[None,:]-2*A@A.T
    DX=d2(X); np.fill_diagonal(DX,np.inf); DE=d2(E); np.fill_diagonal(DE,np.inf)
    rX=np.argsort(np.argsort(DX,1),1); nnE=np.argsort(DE,1)[:,:k]
    return float(1-2.0/(m*k*(2*m-3*k-1))*np.maximum(0,np.take_along_axis(rX,nnE,1)+1-k).sum())
for seed in range(5):
    rng=np.random.default_rng(100+seed); C=rng.standard_normal((8,10))*3; lab=rng.integers(0,8,20000)
    X=(C[lab]+rng.standard_normal((20000,10))).astype(np.float32)
    pl=np.asarray(M.SpectralClustering(n_clusters=8, random_state=seed).fit(X).labels_)
    print(json.dumps(dict(tag=tag,task="sc_blobs",seed=seed,ari=float(ari(lab.tolist(),pl.tolist()))))); sys.stdout.flush()
for ds in ("taxi","covtype"):
    d=spec.load_with_fallback(ds,"shipped",200000); Xa=np.asarray(d.X_train,dtype=np.float32)
    for seed in range(5):
        idx=np.random.default_rng(200+seed).choice(len(Xa),20000,replace=False); X=Xa[idx]; X=np.ascontiguousarray((X-X.mean(0))/(X.std(0)+1e-6))
        Em=np.asarray(M.SpectralEmbedding(n_components=2, random_state=seed).fit_transform(X))
        print(json.dumps(dict(tag=tag,task="se_"+ds,seed=seed,trust=trust(X,Em)))); sys.stdout.flush()
