import sys, json, numpy as np
sys.path[:0]=['python','tools']
import speed_gbdt_arm as spec, mojolearn as M
tag=sys.argv[1]
for ds in ("covtype",):
    d=spec.load_with_fallback(ds,"shipped",1_000_000)
    Xa=np.asarray(d.X_train,dtype=np.float32); mu,sd=Xa.mean(0),Xa.std(0)+1e-6
    Xt=np.ascontiguousarray((np.asarray(d.X_test,dtype=np.float32)[:100000]-mu)/sd)
    for seed in range(5):
        idx=np.sort(np.random.default_rng(7000+seed).choice(len(Xa), size=200000, replace=False))
        g=M.GaussianMixture(n_components=8, random_state=seed, reg_covar=1e-3).fit(np.ascontiguousarray((Xa[idx]-mu)/sd))
        print(json.dumps(dict(tag=tag,ds=ds,seed=seed,test_ll=float(np.mean(np.asarray(g.score_samples(Xt),dtype=np.float64)))))); sys.stdout.flush()
