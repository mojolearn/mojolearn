import sys, json, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]
d=spec.load_with_fallback("taxireg","shipped",1_000_000); c=spec.load_with_fallback("taxi","shipped",1_000_000)
def prep(d,n,seed,off):
    Xa=np.asarray(d.X_train,dtype=np.float32); mu,sd=Xa.mean(0),Xa.std(0)+1e-6
    idx=np.sort(np.random.default_rng(off+seed).choice(len(Xa), size=n, replace=False))
    return np.ascontiguousarray((Xa[idx]-mu)/sd), np.asarray(d.y_train)[idx], np.ascontiguousarray((np.asarray(d.X_test,dtype=np.float32)[:5000]-mu)/sd), np.asarray(d.y_test)[:5000]
r2=lambda p,y: float(1-((p-y)**2).sum()/((y-y.mean())**2).sum())
for seed in range(5):
    X,y,Xt,yt=prep(d,5000,seed,8000); ym,ys=y.mean(),y.std()
    p=np.asarray(M.KernelRidge(kernel="rbf").fit(X,((y-ym)/ys).astype(np.float32)).predict(Xt),dtype=np.float64)*ys+ym
    X2,y2,Xt2,yt2=prep(d,2000,seed,8100); ym2,ys2=y2.mean(),y2.std()
    p2=np.asarray(M.GaussianProcessRegressor().fit(X2,((y2-ym2)/ys2).astype(np.float32)).predict(Xt2),dtype=np.float64)*ys2+ym2
    X3,y3,Xt3,yt3=prep(c,2000,seed,8200)
    a3=float((np.asarray(M.GaussianProcessClassifier().fit(X3,y3.astype(np.int64)).predict(Xt3))==yt3.astype(np.int64)).mean())
    print(json.dumps(dict(tag=tag,seed=seed,krr_r2=r2(p,yt.astype(np.float64)),gpr_r2=r2(p2,yt2.astype(np.float64)),gpc_acc=a3))); sys.stdout.flush()
