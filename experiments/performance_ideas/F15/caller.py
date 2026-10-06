#!/usr/bin/env python3
"""Separate mini-batch label, stopping and retained scratch A/B fit fixtures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import MiniBatchKMeans
    from mojolearn import _mojolearn_x_cluster as binding
    cases={}
    for rows,d,k in ((2017,11,5),(2053,13,7),(4093,19,11)):
        rng=np.random.default_rng(319);centers=rng.normal(0,4,size=(k,d))
        labels=np.arange(rows)%k;x=np.asarray(centers[labels]+rng.normal(0,.3,size=(rows,d)),'float32')
        model=MiniBatchKMeans(n_clusters=k,batch_size=127,n_init=3,max_iter=30,random_state=17,numeric_mode='fast')
        def fit():model.fit(x);return model.cluster_centers_,model.labels_
        _,cold=consumed(fit);_,repeat=consumed(fit)
        assignments=np.asarray(model.labels_).astype(int);chosen=np.asarray(model.cluster_centers_,float)[assignments]
        inertia=float(np.sum((x.astype(float)-chosen)**2))
        assert int(model.n_iter_)<=30
        cases[f'{rows}-{d}-{k}']=dict(contract=dict(rows=rows,d=d,k=k,seed=17,n_init=3,max_iter=30,batch=127),
            metrics=dict(inertia=dict(value=inertia,rtol=.001,atol=1e-4)),cold_fit_ms=cold,repeated_fit_ms=repeat,
            iterations=int(model.n_iter_),centers=np.asarray(model.cluster_centers_).shape)
    return dict(binding=binding_check(binding,'x_cluster'),cases=cases)
if __name__=='__main__':capture_main(exercise)
