#!/usr/bin/env python3
# F13: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Forest retained prediction with shallow/deep and skewed traversal fixtures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    if args.variant=="shap":
        return shap(args)
    from mojolearn import RandomForestClassifier
    from mojolearn import _mojolearn_rf as binding
    layout=str(binding.forest_resident_layout())
    if args.variant=='packed-layout':
        assert layout==('separate_arrays' if args.arm=='A' else 'packed_siblings'),layout
    cases={}
    for depth,classes,skew in ((4,2,False),(11,4,False),(9,3,True)):
        rng=np.random.default_rng(456);x=rng.normal(size=(1019,19)).astype('float32');q=rng.normal(size=(263,19)).astype('float32')
        if skew:x[:,0]=np.exp(x[:,0]);q[:,0]=np.exp(q[:,0])
        edges=np.quantile(x[:,0],np.arange(1,classes)/classes)
        y=np.searchsorted(edges,x[:,0]).astype('int32');qy=np.searchsorted(edges,q[:,0])
        model=RandomForestClassifier(n_estimators=37,max_depth=depth,random_state=7,inference_engine='parallel_groves',numeric_mode='fast')
        model.fit(x,y)
        cold,cold_ms=consumed(lambda:model.predict(q))
        saved=np.asarray(cold).copy()
        repeat,repeated_ms=consumed(lambda:model.predict(q))
        assert np.array_equal(cold,saved),'cold output overwritten by retained prediction'
        assert np.array_equal(cold,repeat),'retained prediction differs'
        # Fitting invalidates the resident conversion; old outputs must survive.
        if args.variant=='packed-layout':
            refit,refit_ms=consumed(lambda:model.fit(x,y).predict(q))
            assert np.array_equal(cold,saved),'resident refit invalidated caller output'
            assert np.array_equal(refit,repeat),'fixed-seed refit changed prediction'
        else:refit_ms=None
        error=float(np.mean(np.asarray(repeat).reshape(-1)!=qy))
        cases[f'depth{depth}-c{classes}-skew{skew}']=dict(contract=dict(depth=depth,classes=classes,skew=skew,trees=37,seed=7),
            metrics=dict(misclassification=dict(value=error,rtol=0,atol=.002)),cold_predict_ms=cold_ms,repeated_predict_ms=repeated_ms,refit_and_conversion_ms=refit_ms,resident_layout=layout,retained_output_bytes=int(saved.nbytes))
    return dict(binding=binding_check(binding,'rf'),cases=cases)
def shap(args):
    import numpy as np
    import time
    from mojolearn import RandomForestClassifier,TreeExplainer
    from mojolearn import _mojolearn_x_trees as binding
    cases={}
    for depth,classes,skew in ((3,2,False),(6,3,False),(9,4,True)):
        rng=np.random.default_rng(456)
        x=rng.normal(size=(509,19)).astype('float32');q=rng.normal(size=(67,19)).astype('float32')
        if skew:x[:,0]=np.exp(x[:,0]);q[:,0]=np.exp(q[:,0])
        edges=np.quantile(x[:,0],np.arange(1,classes)/classes)
        y=np.searchsorted(edges,x[:,0]).astype('int32')
        model=RandomForestClassifier(n_estimators=17,max_depth=depth,random_state=7,numeric_mode='fast').fit(x,y)
        start=time.perf_counter_ns();explainer=TreeExplainer(model,data=x[:127])
        expected=np.asarray(explainer.expected_value,float);expected.tobytes()
        prepare_ms=(time.perf_counter_ns()-start)/1e6
        before=int(binding.x_trees_shap_pair_count())
        values,cold=consumed(lambda:explainer.shap_values(q))
        saved=np.asarray(values).copy()
        repeated,reuse=consumed(lambda:explainer.shap_values(q))
        reached=int(binding.x_trees_shap_pair_count())-before
        if args.arm=='B':assert reached>0,'paired SHAP row schedule not reached'
        probability,_=consumed(lambda:model.predict_proba(q))
        restored=np.asarray(values,float).sum(axis=1)+expected
        additivity=float(np.max(np.abs(restored-np.asarray(probability,float))))
        assert np.array_equal(values,saved),'SHAP output overwritten by reuse'
        repeat_error=float(np.max(np.abs(np.asarray(repeated,float)-np.asarray(values,float))))
        cases[f'depth{depth}-c{classes}-skew{skew}']=dict(contract=dict(depth=depth,classes=classes,skew=skew,trees=17,seed=7,background=127,queries=67),
            metrics=dict(additivity_error=dict(value=additivity,rtol=.1,atol=2e-5),repeat_error=dict(value=repeat_error,rtol=0,atol=2e-6)),
            preparation_ms=prepare_ms,cold_shap_ms=cold,repeated_shap_ms=reuse,paired_chunks=reached)
    return dict(binding=binding_check(binding,'x_trees'),cases=cases)
if __name__=='__main__':capture_main(exercise)
