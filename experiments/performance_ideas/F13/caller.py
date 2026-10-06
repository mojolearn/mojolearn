#!/usr/bin/env python3
"""Forest retained prediction with shallow/deep and skewed traversal fixtures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import RandomForestClassifier
    from mojolearn import _mojolearn_rf as binding
    cases={}
    for depth,classes,skew in ((4,2,False),(11,4,False),(9,3,True)):
        rng=np.random.default_rng(456);x=rng.normal(size=(1019,19)).astype('float32');q=rng.normal(size=(263,19)).astype('float32')
        if skew:x[:,0]=np.exp(x[:,0]);q[:,0]=np.exp(q[:,0])
        edges=np.quantile(x[:,0],np.arange(1,classes)/classes)
        y=np.searchsorted(edges,x[:,0]).astype('int32');qy=np.searchsorted(edges,q[:,0])
        model=RandomForestClassifier(n_estimators=37,max_depth=depth,random_state=7,inference_engine='parallel_groves',numeric_mode='fast')
        model.fit(x,y)
        cold,cold_ms=consumed(lambda:model.predict(q));repeat,repeated_ms=consumed(lambda:model.predict(q))
        error=float(np.mean(np.asarray(repeat).reshape(-1)!=qy))
        cases[f'depth{depth}-c{classes}-skew{skew}']=dict(contract=dict(depth=depth,classes=classes,skew=skew,trees=37,seed=7),
            metrics=dict(misclassification=dict(value=error,rtol=0,atol=.002)),cold_predict_ms=cold_ms,repeated_predict_ms=repeated_ms)
    return dict(binding=binding_check(binding,'rf'),cases=cases)
if __name__=='__main__':capture_main(exercise)
