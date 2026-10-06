#!/usr/bin/env python3
"""Real boosting partition reuse: fixed iterations, fitted state, full quality."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import GradientBoosting
    from mojolearn import _mojolearn_gbdt as binding
    cases={}
    for rows,d,depth in ((1009,11,4),(1031,17,6),(2017,23,5)):
        rng=np.random.default_rng(625)
        x=rng.normal(size=(rows,d)).astype('float32');q=rng.normal(size=(257,d)).astype('float32')
        y=np.asarray(np.sin(x[:,0])+x[:,1]*x[:,2]+.05*rng.normal(size=rows),'float32')
        qy=np.sin(q[:,0])+q[:,1]*q[:,2]
        model=GradientBoosting(loss='RMSE',n_estimators=24,max_depth=depth,learning_rate=.08,random_state=7)
        def fit():model.fit(x,y);return model.predict(q)
        output,fit_ms=consumed(fit)
        repeated,predict_ms=consumed(lambda:model.predict(q))
        error=float(np.sqrt(np.mean((np.asarray(output,float).reshape(-1)-qy)**2)))
        cases[f'{rows}-{d}-depth{depth}']=dict(contract=dict(rows=rows,d=d,depth=depth,iterations=24,seed=7,lr=.08),
            metrics=dict(rmse=dict(value=error,rtol=.001,atol=1e-5)),fit_and_predict_ms=fit_ms,predict_ms=predict_ms,
            repeated_prediction_error=float(np.max(np.abs(np.asarray(output)-np.asarray(repeated)))))
    return dict(binding=binding_check(binding,'gbdt'),cases=cases)
if __name__=='__main__':capture_main(exercise)
