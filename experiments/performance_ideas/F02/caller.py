#!/usr/bin/env python3
# F02: qualification pending. Build every independent variant; device quality,
# complete-call speed, peak scratch and opponent admission remain separate gates.
# New experiment mechanisms remain opt-in; existing promoted defaults are retained.
"""Shared strided subtract qualified at actual LU factor and solve entrances."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apple_fast'))
from support import capture_main, binding_check, consumed

def exercise(args):
    import numpy as np
    if args.variant == "sdk":
        return sdk(args)
    if args.variant == "mcd":
        return mcd(args)
    if args.variant == "pca":
        return pca(args)
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
def mcd(args):
    import importlib.util
    from types import SimpleNamespace
    from mojolearn import _mojolearn_x_decomp as binding
    spec=importlib.util.spec_from_file_location('mcd_caller',Path(__file__).resolve().parents[1]/'F06/caller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before=int(binding.mcd_g1_gram_count(2))
    # This independent arm measures shared active batching, not F06 bounded
    # launch scheduling. Keep F06's exact seeds/task but its route assertion off.
    packet=module.exercise(SimpleNamespace(arm='A',variant='default'))
    reached=int(binding.mcd_g1_gram_count(2))-before
    if args.arm=='B':assert reached>0,'active batched shared MCD products never reached'
    packet['active_batched_products']=reached
    return packet

def pca(args):
    import importlib.util
    from types import SimpleNamespace
    from mojolearn import _mojolearn_estimators as binding
    spec=importlib.util.spec_from_file_location('pca_caller',Path(__file__).resolve().parents[1]/'F11/caller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before=int(binding.scoped_gemm_count(2,2))
    # Reuse fixed PCA quality fixtures, but assert this adapter's own counter.
    packet=module.pca(SimpleNamespace(arm='A',variant='pca'))
    reached=int(binding.scoped_gemm_count(2,2))-before
    if args.arm=='B':assert reached>0,'PCA split adapter never reached'
    packet['split_products']=reached
    return packet

def sdk(args):
    import numpy as np
    from mojolearn import _mojolearn_scoped_gemm_probe as binding
    cases={}
    # SDK NN/NT and true alias-Gram; preserve n1/GEMV and tails controls.
    for entrance,ta,tb,alias,m,n,k in ((1,0,1,0,509,37,129),(1,0,1,1,137,137,521),
        (2,0,1,0,521,31,133),(2,0,0,0,509,41,131),(2,0,1,0,17,1,67)):
        rng=np.random.default_rng(904)
        a=rng.normal(size=(m,k)).astype('float32')
        b=a if alias else rng.normal(size=(n,k) if tb else (k,n)).astype('float32')
        c=np.full((m,n),np.nan,'float32')
        before=[int(binding.route_count(r,t)) for r in range(3) for t in range(3)]
        def launch():binding.gemm(a.ctypes.data,b.ctypes.data,c.ctypes.data,[m,n,k,ta,tb,alias,entrance]);return c
        result,elapsed=consumed(launch)
        delta=[int(binding.route_count(r,t))-before[r*3+t] for r in range(3) for t in range(3)]
        if args.arm=='B' and n>1:assert sum(delta[1::3])+sum(delta[2::3])>0,'unfused SDK adapter missed'
        oracle=a.astype(float)@(b.astype(float).T if tb else b.astype(float))
        error=float(np.linalg.norm(np.asarray(result,float)-oracle)/np.linalg.norm(oracle))
        cases[f'{entrance}-{m}-{n}-{k}-{alias}']=dict(contract=dict(entrance=entrance,m=m,n=n,k=k,ta=ta,tb=tb,alias=alias,seed=904),
            metrics=dict(product_error=dict(value=error,rtol=.1,atol=2e-7)),call_ms=elapsed,products=delta)
    return dict(binding=binding_check(binding,'scoped_gemm_probe'),cases=cases)

if __name__ == '__main__': capture_main(exercise)
