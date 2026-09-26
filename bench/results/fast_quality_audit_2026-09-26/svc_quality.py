import sys, json, time, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]
d=spec.load_with_fallback("taxi","shipped",1_000_000)
Xall=np.asarray(d.X_train,dtype=np.float32); yall=np.asarray(d.y_train).astype(np.int64)
mu,sd=Xall.mean(0),Xall.std(0)+1e-6
Xt=np.ascontiguousarray((np.asarray(d.X_test,dtype=np.float32)[:20000]-mu)/sd); yt=np.asarray(d.y_test).astype(np.int64)[:20000]
for seed in range(5):
    rng=np.random.default_rng(3000+seed); idx=np.sort(rng.choice(len(Xall), size=50000, replace=False))
    X=np.ascontiguousarray((Xall[idx]-mu)/sd)
    t=time.time(); m=M.SVC(C=1.0).fit(X, yall[idx]); acc=float((np.asarray(m.predict(Xt))==yt).mean())
    print(json.dumps(dict(tag=tag,seed=seed,acc=acc,fit_s=round(time.time()-t,2)))); sys.stdout.flush()
dr=spec.load_with_fallback("taxireg","shipped",1_000_000)
XR=np.asarray(dr.X_train,dtype=np.float32); yR=np.asarray(dr.y_train,dtype=np.float32)
mr,sr=XR.mean(0),XR.std(0)+1e-6; ym,ys=float(yR.mean()),float(yR.std())
XRt=np.ascontiguousarray((np.asarray(dr.X_test,dtype=np.float32)[:10000]-mr)/sr); yRt=np.asarray(dr.y_test,dtype=np.float64)[:10000]
for seed in range(5):
    rng=np.random.default_rng(4000+seed); idx=np.sort(rng.choice(len(XR), size=20000, replace=False))
    t=time.time(); m=M.SVR(C=1.0).fit(np.ascontiguousarray((XR[idx]-mr)/sr), ((yR[idx]-ym)/ys).astype(np.float32))
    p=np.asarray(m.predict(XRt),dtype=np.float64)*ys+ym; r2=1-((p-yRt)**2).sum()/((yRt-yRt.mean())**2).sum()
    print(json.dumps(dict(tag=tag,est="SVR",seed=seed,r2=float(r2),fit_s=round(time.time()-t,2)))); sys.stdout.flush()
