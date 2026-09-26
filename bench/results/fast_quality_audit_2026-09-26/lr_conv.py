import sys, json, time, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]
for ds in ("covtype",):
    d=spec.load_with_fallback(ds,"shipped",1_000_000)
    Xa=np.asarray(d.X_train,dtype=np.float32); ya=np.asarray(d.y_train).astype(np.int64); mu,sd=Xa.mean(0),Xa.std(0)+1e-6
    Xt=np.ascontiguousarray((np.asarray(d.X_test,dtype=np.float32)[:100000]-mu)/sd); yt=np.asarray(d.y_test).astype(np.int64)[:100000]
    for seed in range(5):
        idx=np.sort(np.random.default_rng(6000+seed).choice(len(Xa), size=200000, replace=False))
        m=M.LogisticRegression(max_iter=1000).fit(np.ascontiguousarray((Xa[idx]-mu)/sd), ya[idx])
        P=np.asarray(m.predict_proba(Xt),dtype=np.float64); cls=np.asarray(m.classes_)
        pos=np.searchsorted(cls, yt); ll=-np.mean(np.log(np.clip(P[np.arange(len(yt)),pos],1e-12,1)))
        acc=float((cls[P.argmax(1)]==yt).mean())
        print(json.dumps(dict(tag=tag,ds=ds,seed=seed,acc=acc,logloss=float(ll)))); sys.stdout.flush()
