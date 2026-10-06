#!/usr/bin/env python3
# F12: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Real boosting partition reuse: fixed iterations, fitted state, full quality."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    if args.variant in ("categorical","ranking","depthwise","lossguide"):
        return workloads(args)
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
def workloads(args):
    import numpy as np
    from mojolearn import GradientBoosting
    from mojolearn import _mojolearn_gbdt as binding
    cases={}
    for rows,d,depth in ((511,11,4),(1023,17,5)):
        rng=np.random.default_rng(625);x=rng.normal(size=(rows,d)).astype('float32');q=rng.normal(size=(257,d)).astype('float32')
        kind=args.variant
        params=dict(n_estimators=24,max_depth=depth,learning_rate=.08,random_state=7,bootstrap_type='No',random_strength=0.0)
        fit_args={}
        if kind=='categorical':
            x[:,0]=np.arange(rows)%5;q[:,0]=np.arange(257)%5
            x[:,1]=np.arange(rows)%3;q[:,1]=np.arange(257)%3
            y=np.asarray((x[:,0]==2)|((x[:,1]==1)&(x[:,2]>0)),'int32')
            target=(q[:,0]==2)|((q[:,1]==1)&(q[:,2]>0))
            params.update(loss='Logloss',cat_features=[0],one_hot_features=[1])
        elif kind=='ranking':
            y=np.asarray(x[:,0]>.4,'float32')+np.asarray(x[:,1]>.8,'float32')
            target=np.asarray(q[:,0]>.4,'float32')+np.asarray(q[:,1]>.8,'float32')
            groups=np.asarray(np.arange(rows)//17,'int32');fit_args['group_id']=groups
            params.update(loss='QueryRMSE')
        else:
            y=np.asarray(np.sin(x[:,0])+x[:,1]*x[:,2],'float32');target=np.sin(q[:,0])+q[:,1]*q[:,2]
            params.update(loss='RMSE',grow_policy='Depthwise' if kind=='depthwise' else 'Lossguide')
            if kind=='lossguide':params['max_leaves']=17
        model=GradientBoosting(**params)
        def fit():model.fit(x,y,**fit_args);return model.predict(q)
        predicted,cold=consumed(fit)
        repeated,reuse=consumed(lambda:model.predict(q))
        assert np.isfinite(np.asarray(model.loss_curve_)).all()
        if kind=='categorical':error=float(np.mean((np.asarray(predicted).reshape(-1)>.5)!=target))
        elif kind=='ranking':
            # Pair ordering inside fixed heldout groups measures ranking task.
            pred=np.asarray(predicted,float).reshape(-1);bad=total=0
            for start in range(0,len(pred),17):
                for i in range(start,min(start+17,len(pred))):
                    for j in range(i+1,min(start+17,len(pred))):
                        if target[i]!=target[j]:
                            total+=1;bad+=int((pred[i]-pred[j])*(target[i]-target[j])<=0)
            error=float(bad/max(total,1))
        else:error=float(np.sqrt(np.mean((np.asarray(predicted,float).reshape(-1)-target)**2)))
        cases[f'{kind}-{rows}-{d}']=dict(contract=dict(kind=kind,rows=rows,d=d,depth=depth,iterations=24,seed=7,lr=.08,bootstrap='No',score_noise=0),
            metrics=dict(task_error=dict(value=error,rtol=.001,atol=.002)),fit_and_first_predict_ms=cold,repeated_predict_ms=reuse,
            completed_iterations=len(model.loss_curve_),repeat_error=float(np.max(np.abs(np.asarray(predicted)-np.asarray(repeated)))))
    return dict(binding=binding_check(binding,'gbdt'),cases=cases)
if __name__=='__main__':capture_main(exercise)
