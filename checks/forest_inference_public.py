#!/usr/bin/env python3
"""Public GPU parallel-grove inference: independent graph oracle and archives."""
import argparse, gc, hashlib, json, os, pickle, tempfile, time
from pathlib import Path
import numpy as np
from mojolearn import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from mojolearn import _backend


def ftz(a):
    b=np.array(a,dtype=np.float32,copy=True)
    u=b.view(np.uint32)
    mask=(u & 0x7f800000)==0
    u[mask] &= np.uint32(0x80000000)
    return b


def oracle(model, X):
    # Independent vector traversal and explicitly rounded fixed 32-grove graph.
    X=np.asarray(X,dtype=np.float32)
    if isinstance(model,(RandomForestClassifier,RandomForestRegressor)):
        X=ftz(X)
    groups=np.zeros((32,len(X),model._num_outputs),dtype=np.float32)
    for tree in range(model._n_trees):
        base=model._offsets[tree]
        node=np.full(len(X),base,dtype=np.int32)
        active=model._left_child[node]!=-1
        while np.any(active):
            rows=np.flatnonzero(active)
            n=node[rows]
            go_left=X[rows,model._colid[n]]<=model._quesval[n]
            node[rows]=base+model._left_child[n]+(~go_left).astype(np.int32)
            active=model._left_child[node]!=-1
        leaf=model._leaves.reshape(-1,model._num_outputs)[node]
        group=tree%32
        groups[group]=ftz(ftz(groups[group])+ftz(leaf))
    for stride in (16,8,4,2,1):
        groups[:stride]=ftz(ftz(groups[:stride])+ftz(groups[stride:2*stride]))
    return ftz((groups[0].astype(np.float64)/model._n_trees).astype(np.float32))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('fast','identical'), default='identical')
    parser.add_argument('--vendor', choices=('cuda','metal','hip'), default='cuda')
    args=parser.parse_args()
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE')==args.mode
    rng=np.random.default_rng(7)
    X=rng.normal(size=(513,5)).astype(np.float32)
    outputs=[]
    for cls in (RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor):
        classifier=cls._estimator_type=='classifier'
        y=(X[:,0]+X[:,1]>.25).astype(np.int32) if classifier else (X[:,0]*.7-X[:,1]*.2).astype(np.float32)
        m=cls(n_estimators=33,max_depth=5,random_state=7,numeric_mode=args.mode,inference_engine='parallel_groves').fit(X,y)
        native=m._bind()
        prefix='rf' if cls.__name__.startswith('Random') else 'trees'
        assert getattr(native,prefix+'_numeric_mode')()==(1 if args.mode=='identical' else 0)
        assert getattr(native,prefix+'_vendor')()==args.vendor
        pred=m.predict_proba(X) if classifier else m.predict(X)
        expected=oracle(m,X)
        if not classifier: expected=expected[:,0]
        np.testing.assert_array_equal(np.asarray(pred,dtype=np.float32).view(np.uint32),expected.view(np.uint32))
        resident=m._resident_forest
        again=m.predict_proba(X) if classifier else m.predict(X)
        assert m._resident_forest is resident
        np.testing.assert_array_equal(pred,again)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz';m.save(path);loaded=cls.load(path)
            restored=loaded.predict_proba(X) if classifier else loaded.predict(X)
            np.testing.assert_array_equal(pred,restored)
            assert loaded.inference_engine=='parallel_groves'
        restored=pickle.loads(pickle.dumps(m))
        assert not hasattr(restored, '_resident_forest')
        copy_pred=restored.predict_proba(X) if classifier else restored.predict(X)
        np.testing.assert_array_equal(pred,copy_pred)
        released_handle=restored._resident_forest.handle
        del restored
        gc.collect()
        try:
            native.forest_release_gpu(released_handle)
        except Exception as exc:
            assert 'released' in str(exc)
        else:
            raise AssertionError('destroyed estimator left a live GPU model')
        m.inference_engine='sequential'
        reference=m.predict_proba(X) if classifier else m.predict(X)
        np.testing.assert_allclose(pred,reference,rtol=2e-6,atol=2e-6)
        record=dict(estimator=cls.__name__,mode=args.mode,vendor=args.vendor,hash=hashlib.sha256(pred.tobytes()).hexdigest(),max_difference=float(np.max(np.abs(pred-reference))))
        print(json.dumps(record),flush=True);outputs.append(record)
    print('PASS public parallel-groves GPU inference, graph oracle, resident reuse/release, pickle, versioned archives')

if __name__=='__main__': main()
