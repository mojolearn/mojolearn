"""Lossguide quality, one build: for each seed (a different 300k training
subset, same test set), fit and report test metrics. Run once per build."""
import sys, time, json, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]; out=[]
for ds in ("taxi","istellareg"):
    d=spec.load_with_fallback(ds,"shipped",2_000_000)
    Xall=np.asarray(d.X_train,dtype=np.float32); yall=np.asarray(d.y_train)
    Xt=np.ascontiguousarray(np.asarray(d.X_test,dtype=np.float32)[:200000]); yt=np.asarray(d.y_test)[:200000]
    for seed in range(5):
        rng=np.random.default_rng(1000+seed); idx=np.sort(rng.choice(len(Xall), size=min(300000,len(Xall)), replace=False))
        X=np.ascontiguousarray(Xall[idx]); y=yall[idx]
        kw=dict(n_estimators=300, grow_policy='Lossguide', max_leaves=31, max_depth=10, random_state=seed)
        t=time.time()
        if ds=="taxi":
            m=M.GradientBoostingClassifier(**kw).fit(X,y.astype(np.int64)); p=np.asarray(m.predict_proba(Xt))[:,1].astype(np.float64)
            eps=1e-12; ll=-np.mean(yt*np.log(p+eps)+(1-yt)*np.log(1-p+eps)); acc=np.mean((p>0.5)==(yt==1))
            order=np.argsort(p); r=np.empty(len(p)); r[order]=np.arange(len(p)); pos=yt==1
            auc=(r[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum())
            rec=dict(ds=ds,seed=seed,logloss=float(ll),acc=float(acc),auc=float(auc))
        else:
            m=M.GradientBoostingRegressor(**kw).fit(X,y.astype(np.float32)); p=np.asarray(m.predict(Xt),dtype=np.float64)
            rec=dict(ds=ds,seed=seed,rmse=float(np.sqrt(np.mean((p-yt)**2))))
        rec['fit_s']=round(time.time()-t,2); rec['tag']=tag; print(json.dumps(rec)); sys.stdout.flush()
