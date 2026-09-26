"""UMAP init-tolerance quality, one build: per (dataset, seed) the embedding's
trustworthiness (k = 15, 3000-point subsample) and, where labels exist, 5-NN
accuracy in the embedding (80/20 split). Run once per build; pair by seed."""
import sys, json, time, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 30000
def trust(X, E, k=15, m=3000):
    idx=np.random.default_rng(0).choice(len(X), m, replace=False); X=X[idx].astype(np.float64); E=E[idx].astype(np.float64)
    d2=lambda A: (A*A).sum(1)[:,None]+(A*A).sum(1)[None,:]-2*A@A.T
    DX=d2(X); np.fill_diagonal(DX,np.inf); DE=d2(E); np.fill_diagonal(DE,np.inf)
    rX=np.argsort(np.argsort(DX,1),1); nnE=np.argsort(DE,1)[:,:k]
    pen=np.maximum(0,np.take_along_axis(rX,nnE,1)+1-k).sum()
    return 1-2.0/(m*k*(2*m-3*k-1))*pen
def knn_acc(E, y, k=5):
    n=len(E); rng=np.random.default_rng(1); p=rng.permutation(n); tr,te=p[:int(.8*n)],p[int(.8*n):]
    A=E[tr].astype(np.float64); B=E[te].astype(np.float64)
    d=(B*B).sum(1)[:,None]+(A*A).sum(1)[None,:]-2*B@A.T
    nn=np.argpartition(d,k,1)[:,:k]; votes=y[tr][nn]
    pred=np.array([np.bincount(v).argmax() for v in votes]); return float((pred==y[te]).mean())
for ds in ("taxi","covtype","istellareg"):
    try:
        d=spec.load_with_fallback(ds,"shipped",N)
    except Exception as e:
        print(json.dumps(dict(ds=ds,error=str(e)[:80]))); continue
    X=np.asarray(d.X_train[:N],dtype=np.float32); X=np.ascontiguousarray((X-X.mean(0))/(X.std(0)+1e-6))
    y=np.asarray(d.y_train[:N])
    for seed in range(8):
        t=time.time(); Em=np.asarray(M.UMAP(n_neighbors=15, random_state=seed).fit_transform(X)); dt=time.time()-t
        rec=dict(tag=tag,ds=ds,seed=seed,trust=float(trust(X,Em)),fit_s=round(dt,2))
        if ds!="istellareg": rec['knn5_acc']=knn_acc(Em, y.astype(np.int64))
        print(json.dumps(rec)); sys.stdout.flush()
