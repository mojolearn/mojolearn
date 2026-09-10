#!/usr/bin/env python3
"""Bounded n_train=20,000 GP prediction comparison; no GP fit or opponent run.

Both extensions must first pass their small surface/output gates. Uses a
synthetic preconstructed model and measures native complete prediction calls.
"""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name+'._mojolearn_gp', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    import numpy as np
    modules = {name: load(path, name) for name, path in [('before',args.before),('after',args.after)]}
    metadata = {name: {'mode': int(mod.gp_numeric_mode()), 'vendor': str(mod.gp_vendor())}
                for name, mod in modules.items()}
    assert all(row['mode'] == 1 and row['vendor'] == 'cuda' for row in metadata.values()), metadata
    n = 20000
    x = (np.arange(n, dtype=np.float32) % 127).reshape(n,1) * np.float32(.03125)
    factor = np.eye(n, dtype=np.float32)
    dual = ((np.arange(n, dtype=np.float32) % 17) - np.float32(8)) * np.float32(.015625)
    queries = np.array([[.25],[.5],[.75],[1]], dtype=np.float32)
    kinds = np.array([2],np.int32)  # GP_K_RBF
    params = np.array([0],np.float32)
    lengths = np.array([1],np.int32)
    scales = np.array([1],np.float32)
    inputs = [x,factor,dual,queries,kinds,params,lengths,scales]
    outputs = {name: [np.full(4,np.nan,np.float32) for _ in range(3)]+[np.full(4,-1,np.int32)]
               for name in modules}
    samples={name:[] for name in modules}; comparisons=[]
    start=time.monotonic()
    for pair in range(6):
        for name in (('before','after') if pair%2==0 else ('after','before')):
            arrays=inputs+outputs[name]
            addresses=[int(value.ctypes.data) for value in arrays]
            tick=time.perf_counter()
            count=modules[name].gpr_predict(addresses,[n,1,4,1,1,0,0])
            elapsed=time.perf_counter()-tick
            assert count == 0
            if pair: samples[name].append(elapsed)
        match=outputs['before'][0].tobytes()==outputs['after'][0].tobytes()
        assert match and np.isfinite(outputs['after'][0]).all()
        comparisons.append({'pair':pair,'bits_match':match})
    minima={name:min(values) for name,values in samples.items()}
    baseline=samples['before']
    spread=(max(baseline)-min(baseline))/min(baseline)
    result={'shape':{'n_train':n,'n_features':1,'n_star':4,'return_std':False},
            'factor_bytes':factor.nbytes,'synthetic_preconstructed_model':True,
            'metadata':metadata,'samples_seconds':samples,'minimum_seconds':minima,
            'median_seconds':{k:statistics.median(v) for k,v in samples.items()},
            'baseline_range_over_min':spread,
            'baseline_first_to_last_relative':abs(baseline[-1]-baseline[0])/baseline[0],
            'timing_window_valid':spread<=.2,'comparisons':comparisons,
            'elapsed_seconds':time.monotonic()-start,
            'qualification':'native complete-prediction calls; excludes Python estimator preparation; no opponent'}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
