#!/usr/bin/env python3
# F14: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Exact streaming top-k grouping; independent exhaustive candidate certification."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import NearestNeighbors
    from mojolearn import _mojolearn as binding
    cases={}
    for rows,d,k in ((1031,7,5),(2017,15,11),(4093,23,13)):
        rng=np.random.default_rng(116);x=rng.normal(size=(rows,d)).astype('float32');q=rng.normal(size=(131,d)).astype('float32')
        model=NearestNeighbors(n_neighbors=k,metric='euclidean',algorithm='brute')
        model.fit(x)
        output,elapsed=consumed(lambda:model.kneighbors(q))
        distance,index=map(np.asarray,output)
        full=((q.astype(float)[:,None,:]-x.astype(float)[None,:,:])**2).sum(axis=2)
        reference=np.argsort(full,axis=1,kind='stable')[:,:k]
        # FAST different distances are allowed; exact APIs must return the
        # exhaustive nearest candidates. This is semantic exactness, not bits.
        missed=int(np.count_nonzero(index!=reference))
        assert missed==0,'exact API omitted candidates'
        reference_distance=np.sqrt(np.take_along_axis(full,reference,axis=1))
        error=float(np.max(np.abs(distance-reference_distance)))
        cases[f'{rows}-{d}-k{k}']=dict(contract=dict(rows=rows,d=d,k=k,seed=116,api='exact brute'),
            metrics=dict(missed_exact_candidates=dict(value=missed,rtol=0,atol=0),distance_error=dict(value=error,rtol=.1,atol=1e-5)),query_ms=elapsed)
    return dict(binding=binding_check(binding,'mojolearn'),cases=cases)
if __name__=='__main__':capture_main(exercise)
