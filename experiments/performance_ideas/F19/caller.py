#!/usr/bin/env python3
# F19: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Actual all-Mojo gather fixtures include wide tails and public refusals."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main, consumed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'F04'))
from caller import exercise as paired_gather

def exercise(args):
    result=paired_gather(args)
    import numpy as np
    from mojolearn.resample import resample
    # Unsupported weighted draws must retain the public refusal; we never
    # rescue them with host/NumPy sampling inside the candidate route.
    try:resample(np.zeros((17,3),'float32'),sample_weight=np.ones(17,'float32'),numeric_mode='fast')
    except ValueError as error:
        assert 'sample_weight' in str(error)
    else:raise AssertionError('unsupported weighted sampling silently admitted')
    result['weighted_mode']='blocked API prerequisite: sample_weight refused by name'
    if args.variant=='permutation':
        from mojolearn.resample import resample_indices
        from mojolearn import _mojolearn_resample as binding
        original=binding.resample_permutation_gather_gpu
        calls=[]
        def observed(addrs,params):
            status=int(original(addrs,params));calls.append(status);return status
        binding.resample_permutation_gather_gpu=observed
        try:
            for rows,width,count in ((509,3,337),(521,129,509),(997,67,701)):
                rng=np.random.default_rng(621);source=rng.normal(size=(rows,width)).astype('float32')
                indices=np.asarray(resample_indices(rows,n_samples=count,replace=False,random_state=109,numeric_mode='fast'))
                values,elapsed=consumed(lambda:resample(source,n_samples=count,replace=False,random_state=109,numeric_mode='fast'))
                if args.arm=='B':assert calls[-1]==1,'device permutation fell back to host'
                assert len(np.unique(indices))==count
                assert np.array_equal(np.asarray(values),source[indices]),'total key/position sampling order changed'
                result['cases'][f'permute-{rows}-{width}']=dict(contract=dict(rows=rows,width=width,count=count,seed=109,replace=False),
                    metrics=dict(sampling_errors=dict(value=0,rtol=0,atol=0)),public_ms=elapsed,device_calls=sum(calls),output_bytes=count*width*4)
            try:resample(source,n_samples=rows+1,replace=False,numeric_mode='fast')
            except Exception as error:
                # Native Mojo Error is exported as plain Python Exception.
                assert str(error)==f"resample: cannot sample {rows+1} out of arrays with dim {rows} when replace is False (scikit-learn's words)",str(error)
            else:raise AssertionError('permutation accepted oversampling')
        finally:binding.resample_permutation_gather_gpu=original
    result['without_replacement']='separate permutation variant: GPU keys, stable total-order device merges, tiled gather, caller-owned consumed output'
    return result
if __name__=='__main__':capture_main(exercise)
