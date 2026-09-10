#!/usr/bin/env python3
"""WP0: interleaved instrumented/uninstrumented ET fits; NumPy is data oracle only."""
import argparse
import hashlib
import json
import os
from time import perf_counter
import numpy as np
import mojolearn as ml

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', required=True, help='HIGGS.f32.npy, label in column zero')
p.add_argument('--rows', type=int, default=1_000_000)
p.add_argument('--trees', type=int, default=100)
p.add_argument('--depth', type=int, default=16)
p.add_argument('--pairs', type=int, default=2)
p.add_argument('--mode', default='fast', choices=('fast','identical','deterministic'))
a = p.parse_args()
if a.rows < 1_000_000:
    print('REMINDER: small inputs are diagnostic only; tree timing target is >=1M rows', flush=True)
raw = np.load(a.data, mmap_mode='r')[:a.rows]
assert raw.shape == (a.rows, 29), raw.shape
x = np.asfortranarray(raw[:, 1:], dtype=np.float32)
y = np.ascontiguousarray(raw[:, 0], dtype=np.float32)
print(json.dumps(dict(rows=a.rows, features=28, trees=a.trees, depth=a.depth,
                     mode=a.mode, vendor=ml.vendor(), data_sha256=hashlib.sha256(x.tobytes(order='F')+y.tobytes()).hexdigest(),
                     scope='stage attribution; instrumented fits are not certified performance')), flush=True)
reference = None
for pair in range(-1, a.pairs):
    for timed in ((False,) if pair == -1 else ((False, True) if pair % 2 == 0 else (True, False))):
        os.environ['MOJOLEARN_STAGE_TIMES'] = '1' if timed else '0'
        model = ml.ExtraTreesClassifier(n_estimators=a.trees,max_depth=a.depth,random_state=0,numeric_mode=a.mode)
        start = perf_counter()
        model.fit(x, y)
        elapsed = (perf_counter() - start) * 1000
        digest = hashlib.sha256(b''.join(getattr(model,k).tobytes() for k in ('_offsets','_colid','_quesval','_left_child','_leaves'))).hexdigest()
        if reference is None: reference = digest
        assert digest == reference, 'instrumentation or repeated fit changed model bytes'
        print('BOUNDARY_FIT',json.dumps(dict(pair=pair,timed=timed,total_ms=elapsed,nodes=model._colid.size,model_sha256=digest)),flush=True)
        del model
