#!/usr/bin/env python3
"""Actual all-Mojo gather fixtures include wide tails and public refusals."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'F04'))
from caller import exercise as paired_gather

def exercise(args):
    result=paired_gather(args)
    import numpy as np
    from mojolearn import resample
    # Unsupported weighted draws must retain the public refusal; we never
    # rescue them with host/NumPy sampling inside the candidate route.
    try:resample(np.zeros((17,3),'float32'),sample_weight=np.ones(17,'float32'),numeric_mode='fast')
    except ValueError as error:
        assert 'sample_weight' in str(error)
    else:raise AssertionError('unsupported weighted sampling silently admitted')
    result['weighted_mode']='blocked API prerequisite: sample_weight refused by name'
    result['without_replacement']='GPU gather integration remains owed; candidate route covers replacement only'
    return result
if __name__=='__main__':capture_main(exercise)
