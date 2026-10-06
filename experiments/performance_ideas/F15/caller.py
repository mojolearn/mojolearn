#!/usr/bin/env python3
# F15: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Separate mini-batch label, stopping and retained scratch A/B fit fixtures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    if args.variant in ("dbscan-outputs","hdbscan-downloads","hdbscan-linkage","hdbscan-selection","hdbscan-core"):
        return graphs(args)
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
def graphs(args):
    import os
    import numpy as np
    from mojolearn import DBSCAN,HDBSCAN
    cases={};family='estimators' if args.variant=='dbscan-outputs' else 'hdbscan'
    binding=__import__('mojolearn._mojolearn_'+family,fromlist=['_'])
    for rows,d,k in ((509,7,3),(997,13,4),(1031,67,5)):
        rng=np.random.default_rng(319);centers=rng.normal(0,6,size=(k,d))
        planted=np.arange(rows)%k;x=np.asarray(centers[planted]+rng.normal(0,.12,size=(rows,d)),'float32')
        model=DBSCAN(eps=.8*np.sqrt(d/7),min_samples=5,algorithm='brute',max_mbytes_per_batch=1.0) if family=='estimators' else HDBSCAN(min_cluster_size=11,min_samples=5)
        def fit():model.fit(x);return model.labels_
        output,cold=consumed(fit)
        repeat,reuse=consumed(fit)
        labels=np.asarray(output).reshape(-1)
        actual=(labels[:,None]==labels[None,:])&(labels[:,None]>=0)&(labels[None,:]>=0)
        truth=planted[:,None]==planted[None,:]
        pair_error=float(np.mean(actual!=truth))
        assert pair_error<.1,'separated-cluster task failed'
        # Separate diagnostic fit. HDB stage instrumentation adds boundaries,
        # so these phase times never replace the uninstrumented scored fit.
        phase_var='MOJOLEARN_DBSCAN_PHASES' if family=='estimators' else 'MOJOLEARN_STAGE_TIMES'
        previous=os.environ.get(phase_var);os.environ[phase_var]='1'
        print('GRAPH_PHASE_BEGIN variant='+args.variant+' rows='+str(rows)+' d='+str(d),flush=True)
        try:consumed(fit)
        finally:
            if previous is None:os.environ.pop(phase_var,None)
            else:os.environ[phase_var]=previous
        print('GRAPH_PHASE_END variant='+args.variant+' rows='+str(rows),flush=True)
        cases[f'{rows}-{d}-{k}']=dict(contract=dict(rows=rows,d=d,k=k,seed=319,min_samples=5,task='separated-clusters'),
            metrics=dict(cluster_pair_error=dict(value=pair_error,rtol=0,atol=.001)),cold_fit_ms=cold,repeated_fit_ms=reuse,
            noise_fraction=float(np.mean(labels<0)),iterations=int(getattr(model,'n_boruvka_rounds_',0)),
            graph_phases='retained between GRAPH_PHASE_BEGIN/END in arm log; diagnostic rerun excluded from scored timings')
    return dict(binding=binding_check(binding,family),cases=cases)
if __name__=='__main__':capture_main(exercise)
