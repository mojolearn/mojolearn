import sys, json, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]
for ds in ("taxireg","year"):
    try: d=spec.load_with_fallback(ds,"shipped",1_000_000)
    except Exception as e: print(json.dumps(dict(ds=ds,error=str(e)[:60]))); continue
    Xa=np.asarray(d.X_train,dtype=np.float32); ya=np.asarray(d.y_train,dtype=np.float32); mu,sd=Xa.mean(0),Xa.std(0)+1e-6
    Xt=np.ascontiguousarray((np.asarray(d.X_test,dtype=np.float32)[:100000]-mu)/sd); yt=np.asarray(d.y_test,dtype=np.float64)[:100000]
    for seed in range(5):
        idx=np.sort(np.random.default_rng(9000+seed).choice(len(Xa), size=min(300000,len(Xa)), replace=False))
        X=np.ascontiguousarray((Xa[idx]-mu)/sd); y=ya[idx]
        for name,m in (("lasso",M.Lasso(alpha=0.01)),("enet",M.ElasticNet(alpha=0.01,l1_ratio=0.5))):
            p=np.asarray(m.fit(X,y).predict(Xt),dtype=np.float64); r2=float(1-((p-yt)**2).sum()/((yt-yt.mean())**2).sum())
            print(json.dumps(dict(tag=tag,ds=ds,est=name,seed=seed,r2=r2))); sys.stdout.flush()
