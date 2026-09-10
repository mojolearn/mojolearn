#!/usr/bin/env python3
"""Bounded complete native classification calls at 400k rows / 4k queries."""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path


def load(path, arm):
    spec = importlib.util.spec_from_file_location(arm+'._mojolearn', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.mojolearn_numeric_mode() == 1
    assert module.mojolearn_vendor() == 'cuda'
    return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('before',type=Path);ap.add_argument('after',type=Path)
    ap.add_argument('out',type=Path);args=ap.parse_args()
    import numpy as np
    modules={arm:load(path,arm) for arm,path in [('before',args.before),('after',args.after)]}
    rng=np.random.default_rng(2487)
    x=rng.normal(size=(400000,32)).astype(np.float32)
    q=rng.normal(size=(4000,32)).astype(np.float32)
    q[:16]=x[:16]  # exercise exact matches in the weighted arm
    labels=(np.arange(len(x))%7).astype(np.int32)
    rows=[]
    for k in (10,15):
        for weights in (0,1):
            output={arm:[np.zeros(1,np.int32),np.full((len(q),7),np.nan,np.float32),np.zeros(7,np.int32)] for arm in modules}
            samples={arm:[] for arm in modules}
            for pair in range(6):
                for arm in (('before','after') if pair%2==0 else ('after','before')):
                    arrays=[x,q,labels]+output[arm]
                    start=time.perf_counter()
                    tile=modules[arm].knn_classify(*[int(a.ctypes.data) for a in arrays],
                        [len(x),len(q),32,k,0,1,1,7],[-1,2.0,weights])
                    elapsed=time.perf_counter()-start
                    assert tile>0
                    if pair:samples[arm].append(elapsed)
                for a,b in zip(output['before'][1:],output['after'][1:]):
                    assert a.tobytes()==b.tobytes()
                assert np.isfinite(output['after'][1]).all()
            spread=(max(samples['before'])-min(samples['before']))/min(samples['before'])
            rows.append({'n_index':len(x),'n_queries':len(q),'n_features':32,'k':k,'weights':weights,
                         'bits_match':True,'pairs_including_warmup':6,'samples_seconds':samples,
                         'minimum_seconds':{a:min(v) for a,v in samples.items()},
                         'median_seconds':{a:statistics.median(v) for a,v in samples.items()},
                         'baseline_range_over_min':spread,'timing_window_valid':spread<=.2})
            args.out.write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':main()
