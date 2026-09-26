import sys, json, time, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]
for ds in ("taxi","covtype"):
    d=spec.load_with_fallback(ds,"shipped",1_000_000)
    Xall=np.asarray(d.X_train,dtype=np.float32); yall=np.asarray(d.y_train).astype(np.int64)
    mu,sd=Xall.mean(0),Xall.std(0)+1e-6
    Xt_all=(np.asarray(d.X_test,dtype=np.float32)-mu)/sd; yt_all=np.asarray(d.y_test).astype(np.int64)
    for seed in range(5):
        rng=np.random.default_rng(5000+seed); idx=rng.choice(len(Xall), size=200000, replace=False); q=rng.choice(len(Xt_all), size=5000, replace=False)
        X=np.ascontiguousarray((Xall[idx]-mu)/sd); y=yall[idx]; Q=np.ascontiguousarray(Xt_all[q]); yq=yt_all[q]
        t=time.time(); acc=float((np.asarray(M.KNeighborsClassifier(n_neighbors=10).fit(X,y).predict(Q))==yq).mean()); ta=time.time()-t
        nn=M.NearestNeighbors(n_neighbors=10).fit(X); _,I=nn.kneighbors(Q[:500]); I=np.asarray(I)
        A=Q[:500].astype(np.float64); B=X.astype(np.float64)
        D=(A*A).sum(1)[:,None]+(B*B).sum(1)[None,:]-2*A@B.T; T=np.argsort(D,1,kind='stable')[:,:10]
        rec=float(np.mean([len(set(I[i].tolist())&set(T[i].tolist()))/10 for i in range(500)]))
        print(json.dumps(dict(tag=tag,ds=ds,seed=seed,acc=acc,recall10_vs_f64=rec,fit_pred_s=round(ta,2)))); sys.stdout.flush()
