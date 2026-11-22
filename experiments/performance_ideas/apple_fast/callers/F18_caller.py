#!/usr/bin/env python3
# F18: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Immutable resident KDE fit/score: source mutation, invalidation and outputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import KernelDensity
    from mojolearn import _mojolearn_estimators as binding
    cases={}
    for n,d in ((509,3),(521,7),(997,13)):
        rng=np.random.default_rng(918);x=rng.normal(size=(n,d)).astype('float32');original=x.copy();q=rng.normal(size=(37,d)).astype('float32')
        model=KernelDensity(bandwidth=.7).fit(x)
        output,cold=consumed(lambda:model.score_samples(q))
        saved=np.asarray(output).copy();times=[];errors=[];mutation_error=0.0
        for bandwidth in (.7,.8,.7):
            model.bandwidth=bandwidth
            values,elapsed=consumed(lambda:model.score_samples(q));times.append(elapsed)
            z=-np.sum((q.astype(float)[:,None,:]-original.astype(float)[None,:,:])**2,axis=2)/(2*bandwidth**2)
            maximum=z.max(axis=1);reference=maximum+np.log(np.exp(z-maximum[:,None]).sum(axis=1))-np.log(n)-d*np.log(bandwidth*np.sqrt(2*np.pi))
            errors.append(float(np.max(np.abs(np.asarray(values)-reference))))
        # Snapshot mode must retain fitted meaning when external source changes
        # and a mutable bandwidth forces a new native prepare.
        if args.variant=='immutable' and args.arm=='B':
            x[:]=np.float32(5);model.bandwidth=.8
            values,_=consumed(lambda:model.score_samples(q))
            assert not np.shares_memory(np.asarray(model._x),x)
            z=-np.sum((q.astype(float)[:,None,:]-original.astype(float)[None,:,:])**2,axis=2)/(2*.8**2)
            maximum=z.max(axis=1)
            reference=maximum+np.log(np.exp(z-maximum[:,None]).sum(axis=1))-np.log(n)-d*np.log(.8*np.sqrt(2*np.pi))
            mutation_error=float(np.max(np.abs(np.asarray(values)-reference)))
            assert mutation_error < 2e-5, 'immutable refit read externally mutated source'
        assert np.array_equal(output,saved),'later score overwrote returned output'
        peer=KernelDensity(bandwidth=.7).fit(original)
        consumed(lambda:peer.score_samples(q[:1]))
        try:model.score_samples(np.full((1,d),np.nan,'float32'))
        except Exception as error:
            # Mojo Error is exported as plain Exception; require this exact
            # expected refusal rather than accepting unrelated runtime failures.
            assert str(error)=='kde: query contains NaN at row 0, column 0 (DEVIATION 604)',str(error)
        else:raise AssertionError('nonfinite query accepted')
        model.fit(original);consumed(lambda:model.score_samples(q))
        cases[f'{n}-{d}']=dict(contract=dict(rows=n,d=d,queries=37,seed=918),
            metrics=dict(score_error=dict(value=max(errors),rtol=.1,atol=2e-5),mutation_error=dict(value=mutation_error,rtol=0,atol=2e-5)),cold_ms=cold,repeated_ms=times,refit_and_exception_recovery=True)
    return dict(binding=binding_check(binding,'estimators'),cases=cases)
if __name__=='__main__':capture_main(exercise)
