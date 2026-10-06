#!/usr/bin/env python3
"""Shared strided subtract qualified at actual LU factor and solve entrances."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apple_fast'))
from support import capture_main, binding_check, consumed

def exercise(args):
    import numpy as np
    if args.variant == "cholesky":
        return cholesky(args)
    from mojolearn import lu_factor, lu_solve
    from mojolearn import _mojolearn_x_decomp as binding
    before=int(binding.x_decomp_shared_sub_count())
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
    reached=int(binding.x_decomp_shared_sub_count())-before
    if args.arm=='B': assert reached>0,'LU did not reach shared subtract'
    return dict(binding=binding_check(binding,'x_decomp'), cases=cases,shared_products=reached)
def cholesky(args):
    import numpy as np
    import time
    from mojolearn import Cholesky
    from mojolearn import _mojolearn_gp as binding
    before=int(binding.gp_shared_sub_count())
    cases = {}
    for n in (301, 1027, 2053):
        rng = np.random.default_rng(772)
        m = rng.normal(size=(n,n)).astype('float32')
        a = np.ascontiguousarray((m + m.T)*np.float32(.5))
        a[np.arange(n), np.arange(n)] += np.float32(2*np.sqrt(n))
        rhs = rng.normal(size=(n,3)).astype('float32')
        start = time.perf_counter_ns()
        model = Cholesky(jitter=0).fit(a)
        lower = np.asarray(model.L_)
        lower.tobytes()
        fit_ms = (time.perf_counter_ns()-start)/1e6
        assert model.info_ == 0 and np.isfinite(lower).all()
        assert not np.triu(lower,1).any()
        solved, solve_ms = consumed(lambda: model.solve(rhs))
        ld = lower.astype(float)
        factor_error = float(np.linalg.norm(ld @ ld.T-a)/np.linalg.norm(a))
        solve_error = float(np.linalg.norm(a.astype(float) @ solved-rhs)/np.linalg.norm(rhs))
        cases[str(n)] = dict(contract=dict(n=n,rhs=3,seed=772,operation='triangular-subtract'),
            metrics=dict(factor_error=dict(value=factor_error,rtol=.1,atol=5e-8),
                         solve_error=dict(value=solve_error,rtol=.1,atol=5e-8)),
            fit_ms=fit_ms,solve_ms=solve_ms,logdet=float(model.logdet_))
    reached=int(binding.gp_shared_sub_count())-before
    if args.arm=='B': assert reached>0,'Cholesky did not reach triangular shared subtract'
    return dict(binding=binding_check(binding,'gp'),cases=cases,shared_products=reached)
if __name__ == '__main__': capture_main(exercise)
