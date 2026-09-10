#!/usr/bin/env python3
"""WP2a same-fit large-forest export A/B; training excluded from these timings."""
import argparse
import hashlib
import json
import os
from time import perf_counter
import numpy as np
from mojolearn import ExtraTreesClassifier
from mojolearn._buffer import empty
from mojolearn._arrays import _addr as addr, _addr_ro as addr_ro
from mojolearn._forest_protocol import _forest_fit_arrays
from mojolearn.extratrees import _fit_params

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',required=True)
p.add_argument('--mode',choices=('fast','identical','deterministic'),default='fast')
p.add_argument('--pairs',type=int,default=5)
a=p.parse_args()
os.environ['MOJOLEARN_STAGE_TIMES']='0'
raw=np.load(a.data,mmap_mode='r')[:1_000_000]
x=np.asfortranarray(raw[:,1:],dtype=np.float32);y=np.ascontiguousarray(raw[:,0],dtype=np.float32)
m=ExtraTreesClassifier(n_estimators=100,max_depth=16,random_state=0,numeric_mode=a.mode)
native=m._bind()
params=_fit_params(len(y),x.shape[1],2,m._cfg,m.device,m._criterion_code)
t0=perf_counter();descriptor=native.et_classifier_fit_export(addr_ro(x),addr_ro(y),params)
training_ms=(perf_counter()-t0)*1000
handle,trees,nodes,outputs,meta=descriptor
rows=[];expected=None
try:
 for pair in range(-1,a.pairs):
  for arm in (('legacy','into') if pair%2 else ('into','legacy')):
   t0=perf_counter()
   if arm=='legacy':
    result=_forest_fit_arrays(native.forest_export_legacy(handle))
   else:
    sizes=(trees+1,nodes,nodes,nodes,nodes*outputs);dtypes=('<i4','<i4','<f4','<i4','<f4')
    arrays=tuple(empty(n,d) for n,d in zip(sizes,dtypes))
    native.forest_export(handle,*(addr(v) for v in arrays),[trees,nodes,outputs])
    result=(*arrays,meta)
   elapsed=(perf_counter()-t0)*1000
   digest=hashlib.sha256(b''.join(v.tobytes() for v in result[:5])).hexdigest()
   if expected is None:expected=digest
   assert digest==expected,'export changed model bytes'
   if pair>=0:rows.append(dict(pair=pair,arm=arm,ms=elapsed))
   del result
   if arm=='into':del arrays
 baseline=[r['ms'] for r in rows if r['arm']=='legacy'];candidate=[r['ms'] for r in rows if r['arm']=='into']
 spread=abs(baseline[-1]-baseline[0])/baseline[0]
 print(json.dumps(dict(scope='same-native-fit export and Array allocation only; no training speed claim',rows=len(y),features=x.shape[1],trees=trees,nodes=nodes,mode=a.mode,vendor=native.trees_vendor(),training_ms=training_ms,samples=rows,model_sha256=expected,baseline_min_ms=min(baseline),candidate_min_ms=min(candidate),baseline_endpoint_spread=spread,window_valid=spread<=.2),indent=2),flush=True)
 if spread>.2:raise SystemExit('VOID: baseline endpoint drift exceeds20%')
finally:native.forest_export_release(handle)
