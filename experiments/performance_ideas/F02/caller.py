#!/usr/bin/env python3
"""Shared strided subtract qualified at actual LU factor and solve entrances."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apple_fast'))
from support import capture_main, binding_check, consumed

def exercise(args):
    import numpy as np
    from mojolearn import lu_factor, lu_solve
    from mojolearn import _mojolearn_x_decomp as binding
    cases = {}
    for n in (257, 319, 513):
        rng = np.random.default_rng(772)
        a = rng.normal(size=(n,n)).astype('float32')
        a += np.eye(n,dtype='float32') * np.float32(n)
        b = rng.normal(size=(n,3)).astype('float32')
        factor, factor_ms = consumed(lambda: lu_factor(a, numeric_mode='fast'))
        solution, solve_ms = consumed(lambda: lu_solve(factor, b, numeric_mode='fast'))
        residual = float(np.linalg.norm(a.astype(float) @ np.asarray(solution, float) - b) / np.linalg.norm(b))
        cases[str(n)] = dict(contract=dict(n=n, rhs=3, seed=772),
            metrics=dict(residual=dict(value=residual, rtol=.1, atol=2e-7)), factor_ms=factor_ms, solve_ms=solve_ms)
    return dict(binding=binding_check(binding,'x_decomp'), cases=cases)
if __name__ == '__main__': capture_main(exercise)
