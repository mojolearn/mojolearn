#!/usr/bin/env python3
"""Actual TSQR least-squares decomposition on rank and conditioning fixtures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    from mojolearn import lstsq
    from mojolearn import _mojolearn_x_decomp as binding
    cases={}
    for rows,columns,condition in ((509,17,1e2),(521,19,1e4),(997,31,1e6)):
        rng=np.random.default_rng(981)
        u,_=np.linalg.qr(rng.normal(size=(rows,columns)))
        v,_=np.linalg.qr(rng.normal(size=(columns,columns)))
        s=np.geomspace(1,1/condition,columns)
        a=np.asarray((u*s)@v.T,'float32');truth=rng.normal(size=(columns,3));b=np.asarray(a.astype(float)@truth,'float32')
        result,elapsed=consumed(lambda:lstsq(a,b,rcond=None,numeric_mode='fast'))
        answer=np.asarray(result[0],float)
        residual=float(np.linalg.norm(a.astype(float)@answer-b)/max(np.linalg.norm(b),1e-9))
        state_error=float(np.linalg.norm(answer-truth)/np.linalg.norm(truth))
        cases[f'{rows}-{columns}-{condition}']=dict(contract=dict(rows=rows,columns=columns,condition=condition,seed=981),
            metrics=dict(residual=dict(value=residual,rtol=.1,atol=2e-7),coefficient_error=dict(value=state_error,rtol=.01,atol=1e-5)),
            fit_ms=elapsed,rank=int(result[2]),singular_values=np.asarray(result[3]).tolist())
    return dict(binding=binding_check(binding,'x_decomp'),cases=cases)
if __name__=='__main__':capture_main(exercise)
