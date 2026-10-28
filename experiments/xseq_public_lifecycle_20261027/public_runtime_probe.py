#!/usr/bin/env python3
"""Root-only frozen public lifecycle probe; source authoring is not a pass."""
import argparse
import inspect
import json
from pathlib import Path
import traceback
import warnings


def run():
    import numpy as np
    import mojolearn
    from mojolearn import _sk_validate
    from mojolearn.exceptions import NotFittedError
    from mojolearn._verify_native_runtime import audit_native_call
    audits=[];cases=[]
    x=np.linspace(-1,1,24,dtype='f4').reshape(6,2,2)
    y=np.linspace(-.5,.5,6,dtype='f4');labels=np.array([0,1,0,1,0,1],dtype='i4')
    def call(fn,*args,**kw):
        record=audit_native_call(fn,*args,**kw)
        audits.append({'operation':fn.__qualname__,'audit':record.report()})
        return record.assert_native()
    def refuse(fn,kind,*args,**kw):
        try:fn(*args,**kw)
        except kind:return
        raise AssertionError((fn.__qualname__,'missing refusal',kind.__name__))
    # No old Python validator may participate, even through an inherited wrapper.
    names=('before_fit','before_call','after_fit','after_call','reraise')
    saved={name:getattr(_sk_validate,name) for name in names}
    def poison(*args,**kwargs):raise AssertionError('inherited Python lifecycle wrapper')
    try:
        for name in names:setattr(_sk_validate,name,poison)
        for family in ('RNN','LSTM','GRU','MLP'):
            for kind,target in [('Regressor',y),('Classifier',labels)]:
                cls=getattr(mojolearn,family+kind)
                options=dict(numeric_mode='identical',random_state=31)
                options.update(dict(hidden_layer_sizes=(2,),max_iter=1,batch_size=3) if family=='MLP' else dict(hidden_size=2,max_epochs=1,batch_size=3))
                model=cls(**options)
                source=x[:,0,:].copy() if family=='MLP' else x
                for method in ('fit','predict','score'):
                    assert Path(getattr(type(model),method).__code__.co_filename).name!='_mode.py',(family,kind,method)
                refuse(model.predict,NotFittedError,source)
                refuse(model.fit,ValueError,source,None)
                assert '_mojolearn_fitting' not in vars(model)
                assert call(model.fit,source,target) is model
                assert vars(model)['_mojolearn_fitted'] is True
                assert '_mojolearn_fitting' not in vars(model)
                result=call(model.predict,source)
                if kind=='Regressor':assert result.dtype==np.dtype('f8')
                else:assert call(model.predict_proba,source).shape[0]==len(source)
                assert np.isfinite(call(model.score,source,target))
                wrong=source[...,:1]
                refuse(model.predict,ValueError,wrong)
                if family=='MLP':
                    bad=source.copy();bad[0,0]=np.nan
                    refuse(model.fit,ValueError,bad,target)
                    refuse(model.predict,ValueError,bad)
                if kind=='Classifier':refuse(model.fit,ValueError,source,y)
                cases.append(family+kind)
        # Outer validation previously raveled a numeric column target and warned.
        model=mojolearn.MLPRegressor(hidden_layer_sizes=(2,),max_iter=1,batch_size=3,numeric_mode='identical',random_state=31)
        with warnings.catch_warnings(record=True) as found:
            warnings.simplefilter('always')
            call(model.fit,x[:,0,:].copy(),y[:,None])
        assert any('column-vector y' in str(item.message) for item in found)
        assert call(model.predict,x[:,0,:].copy()).shape==(6,)
        invalid=mojolearn.MLPRegressor(hidden_layer_sizes=(0,),max_iter=1,numeric_mode='identical')
        refuse(invalid.fit,ValueError,x[:,0,:].copy(),y)
        assert '_mojolearn_fitting' not in vars(invalid)
    finally:
        for name,fn in saved.items():setattr(_sk_validate,name,fn)
    return {'status':'passed','classes':cases,'audits':audits,'pending':['full metadata-once migration','other families','pandas feature-name cases']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    try:report=run()
    except Exception:
        args.output.write_text(json.dumps({'status':'failed','error':traceback.format_exc()},indent=2)+'\n');raise
    args.output.write_text(json.dumps(report,indent=2)+'\n')
