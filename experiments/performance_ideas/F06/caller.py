#!/usr/bin/env python3
"""Bounded batched MCD: contamination, near singularity and fitted consumers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import MinCovDet,EllipticEnvelope
    from mojolearn import _mojolearn_x_decomp as binding
    cases={}
    for n,d,contamination in ((313,5,.1),(601,9,.2),(907,13,.1)):
        rng=np.random.default_rng(514)
        clean=rng.normal(size=(n,d)).astype('float32')
        clean[:,-1]=clean[:,0]+np.float32(.003)*clean[:,-1]
        data=clean.copy();data[:int(n*contamination)]+=np.float32(8)
        for kind in ('mcd','elliptic'):
            model=MinCovDet(random_state=7,numeric_mode='fast') if kind=='mcd' else EllipticEnvelope(random_state=7,contamination=contamination,numeric_mode='fast')
            before=int(binding.mcd_fast_batch_count())
            def fit():
                model.fit(data)
                return np.asarray(model.location_),np.asarray(model.covariance_),np.asarray(model.support_)
            _,fit_ms=consumed(fit)
            reached=int(binding.mcd_fast_batch_count())-before
            if args.arm=='B' and reached==0:raise AssertionError('bounded MCD batches never reached')
            loc=np.asarray(model.location_,float);cov=np.asarray(model.covariance_,float)
            oracle=np.cov(clean[int(n*contamination):].astype(float),rowvar=False)
            covariance_error=float(np.linalg.norm(cov-oracle)/np.linalg.norm(oracle))
            location_error=float(np.linalg.norm(loc))
            cases[f'{kind}-{n}-{d}']=dict(contract=dict(n=n,d=d,contamination=contamination,seed=7),
                metrics=dict(covariance=dict(value=covariance_error,rtol=.01,atol=1e-5),location=dict(value=location_error,rtol=.01,atol=1e-5)),
                fit_ms=fit_ms,batch_launches=reached,support_count=int(np.asarray(model.support_).sum()))
    return dict(binding=binding_check(binding,'x_decomp'),cases=cases)
if __name__=='__main__':capture_main(exercise)
