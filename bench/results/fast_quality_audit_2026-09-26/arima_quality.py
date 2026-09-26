import sys, json, time, numpy as np
sys.path[:0]=['python']
import mojolearn as M
tag=sys.argv[1]
def arma(n, phi, theta, d, seed):
    rng=np.random.default_rng(seed); e=rng.standard_normal(n+200); x=np.zeros(n+200)
    for t in range(len(x)):
        x[t]=e[t]+sum(phi[i]*x[t-1-i] for i in range(len(phi)) if t-1-i>=0)+sum(theta[j]*e[t-1-j] for j in range(len(theta)) if t-1-j>=0)
    x=x[200:]
    for _ in range(d): x=np.cumsum(x)
    return x
cases=[((1,0,1),[0.6],[0.3]),((2,0,1),[0.5,-0.2],[0.4]),((1,1,1),[0.4],[-0.3])]
for order,phi,theta in cases:
    for n in (2000, 20000):
        for seed in range(5):
            s=arma(n,phi,theta,order[1],seed*7+n)
            t=time.time(); m=M.ARIMA(order=order).fit(s); dt=time.time()-t
            ar=np.asarray(m.ar_).ravel(); ma=np.asarray(m.ma_).ravel()
            est=np.concatenate([ar,ma]) if ar.size+ma.size else np.asarray(getattr(m,'params_',[])).ravel()
            truth=np.array(phi+theta)
            perr=float(np.linalg.norm(est[:len(truth)]-truth)) if est.size>=len(truth) else None
            llf=float(np.asarray(m.llf_).ravel()[0])
            print(json.dumps(dict(tag=tag,order=list(order),n=n,seed=seed,llf=llf,perr=perr,fit_s=round(dt,2)))); sys.stdout.flush()
