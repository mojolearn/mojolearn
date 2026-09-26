import sys, json, time, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]
for ds in ("istellareg","year"):
    try: d=spec.load_with_fallback(ds,"shipped",1_000_000)
    except Exception as e: print(json.dumps(dict(ds=ds,error=str(e)[:80]))); continue
    Xall=np.asarray(d.X_train,dtype=np.float32); yall=np.asarray(d.y_train,dtype=np.float32)
    Xt=np.ascontiguousarray(np.asarray(d.X_test,dtype=np.float32)[:100000]); yt=np.asarray(d.y_test,dtype=np.float64)[:100000]
    for seed in range(5):
        rng=np.random.default_rng(2000+seed); idx=np.sort(rng.choice(len(Xall), size=min(300000,len(Xall)), replace=False))
        t=time.time(); m=M.ExtraTreesRegressor(n_estimators=50, random_state=seed).fit(np.ascontiguousarray(Xall[idx]), yall[idx])
        p=np.asarray(m.predict(Xt),dtype=np.float64)
        print(json.dumps(dict(tag=tag,ds=ds,seed=seed,rmse=float(np.sqrt(np.mean((p-yt)**2))),fit_s=round(time.time()-t,2)))); sys.stdout.flush()
