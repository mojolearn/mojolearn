#!/usr/bin/env python3
"""Real paired resample outputs: asymmetric sizes, ownership, refusal/recovery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import resample as draw, resample_indices
    from mojolearn import _mojolearn_resample as binding
    identity=binding_check(binding,'resample')
    assert int(binding.resample_gpu_gather_enabled())==1
    original=binding.resample_gather_gpu
    calls=[]
    def observed(addrs,params):
        result=original(addrs,params);calls.append(int(result));return result
    binding.resample_gather_gpu=observed
    cases={}
    try:
        for rows,widths,count in ((509,(1,37),337),(521,(3,129),509),(997,(1,7,67),701)):
            rng=np.random.default_rng(621)
            arrays=[rng.normal(size=(rows,width)).astype('float32') for width in widths]
            indices=np.asarray(resample_indices(rows,n_samples=count,random_state=109,numeric_mode='fast'))
            results,elapsed=consumed(lambda: tuple(draw(*arrays,n_samples=count,random_state=109,numeric_mode='fast')))
            assert calls[-1]==1, 'host fallback cannot qualify grouped gather'
            for source,result in zip(arrays,results):
                assert np.array_equal(np.asarray(result),source[indices]), 'sampling contract changed'
            # Old output must survive another operation and mutable source buffers.
            retained=[np.asarray(value).copy() for value in results]
            draw(*arrays,n_samples=count,random_state=113,numeric_mode='fast')
            assert all(np.array_equal(x,y) for x,y in zip(results,retained))
            try: draw(*arrays,n_samples=-1,random_state=109,numeric_mode='fast')
            except (ValueError,RuntimeError): pass
            else: raise AssertionError('negative count accepted')
            cases[str(widths)]=dict(contract=dict(rows=rows,widths=widths,count=count,seed=109),
                metrics=dict(sampling_errors=dict(value=0,rtol=0,atol=0)),public_ms=elapsed,
                successful_gpu_calls=sum(calls),output_bytes=count*sum(widths)*4)
    finally: binding.resample_gather_gpu=original
    return dict(binding=identity,cases=cases)
if __name__=='__main__': capture_main(exercise)
