#!/usr/bin/env python3
# F05: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Actual softmax full optimizer, line-search reach, imbalance and conditioning."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import _mojolearn_estimators as binding
    from softmax_g2_quality import make
    cases={}
    for name in ('r2000-d8-c2','r5003-d37-c5','r12000-d97-c11','r3001-d700-c40'):
        x,y,q,qy=make(name)
        n,d=x.shape;c=int(y.max())+1
        # Same optimizer/stopping settings across arms. A broad condition spread
        # drives line search without changing the requested classification task.
        x[:,0]*=np.float32(64);q[:,0]*=np.float32(64)
        w=np.zeros((d+1,c),'float32');info=np.full(2,np.nan,'float32')
        binding.softmax_g2_reset()
        iterations,fit_ms=consumed(lambda: binding.qn_fit(x.ctypes.data,y.ctypes.data,w.ctypes.data,info.ctypes.data,
            [n,d,c,.01,0.,1e-4,1e-5,30,30,5,1,0,0,2]))
        reach=[int(binding.softmax_g2_count(i)) for i in range(3)]
        assert reach[1]>0 if args.arm=='B' else reach[1]==0
        scores=np.empty((len(q),c),'float32')
        def predict():
            binding.qn_decision_function(q.ctypes.data,w.ctypes.data,scores.ctypes.data,[len(q),d,1,c])
            return scores
        _,predict_ms=consumed(predict)
        assert np.isfinite(scores).all() and np.isfinite(w).all()
        z=scores.astype(float);z-=z.max(axis=1,keepdims=True)
        loss=float(np.mean(np.log(np.exp(z).sum(axis=1))-z[np.arange(len(q)),qy.astype(int)]))
        error=float(np.mean(np.argmax(scores,axis=1)!=qy))
        cases[name]=dict(contract=dict(shape=[n,d,c],stopping=[1e-4,1e-5,30],regularization=.01),
            metrics=dict(logloss=dict(value=loss,rtol=1e-3,atol=1e-6),misclassification=dict(value=error,rtol=0,atol=.002)),
            iterations=int(iterations),fit_ms=fit_ms,predict_ms=predict_ms,reach=reach)
    return dict(binding=binding_check(binding,'estimators'),cases=cases)
if __name__=='__main__':capture_main(exercise)
