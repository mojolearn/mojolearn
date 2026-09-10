#!/usr/bin/env python3
"""WP8: same-process whole-fit A/B of scalar vs vector staging comparisons."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from mojolearn import ExtraTreesClassifier
from mojolearn.extratrees import _ExtraTreesBase

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',required=True)
p.add_argument('--scalar-so',required=True)
p.add_argument('--vector-so',required=True)
p.add_argument('--pairs',type=int,default=5)
a=p.parse_args()
os.environ['MOJOLEARN_STAGE_TIMES']='0'
os.environ['MOJOLEARN_FOREST_EXPORT']='into'
modules={};hashes={}
for key,path in [('scalar_reference',a.scalar_so),('vector_bytes',a.vector_so)]:
 path=str(Path(path).resolve())
 spec=importlib.util.spec_from_file_location('_mojolearn_trees',path)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 assert mod.trees_stage_copy_policy()==key
 assert mod.trees_numeric_mode()==0 and mod.trees_vendor()=='metal'
 modules[key]=mod;hashes[key]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
raw=np.load(a.data,mmap_mode='r')[:1_000_000]
assert raw.shape==(1_000_000,29)
x=np.asfortranarray(raw[:,1:],dtype=np.float32);y=np.ascontiguousarray(raw[:,0],dtype=np.float32)
original=_ExtraTreesBase._bind
selected=None
_ExtraTreesBase._bind=lambda self,name=None:selected
results=[];expected=None
try:
 for pair in range(-1,a.pairs):
  order=('scalar_reference','vector_bytes') if pair%2 else ('vector_bytes','scalar_reference')
  for arm in order:
   selected=modules[arm]
   model=ExtraTreesClassifier(n_estimators=100,max_depth=16,random_state=0,numeric_mode='fast')
   start=perf_counter();model.fit(x,y);elapsed=(perf_counter()-start)*1000
   digest=hashlib.sha256(b''.join(getattr(model,k).tobytes() for k in ('_offsets','_colid','_quesval','_left_child','_leaves'))).hexdigest()
   if expected is None:expected=digest
   assert digest==expected,'staging changed model bytes'
   row=dict(pair=pair,arm=selected.trees_stage_copy_policy(),ms=elapsed,nodes=model._colid.size,model_sha256=digest)
   print('PAIR',json.dumps(row),flush=True)
   if pair>=0:results.append(row)
   del model
 scalar=[r['ms'] for r in results if r['arm']=='scalar_reference'];vector=[r['ms'] for r in results if r['arm']=='vector_bytes']
 spread=abs(scalar[-1]-scalar[0])/scalar[0]
 print('RESULT',json.dumps(dict(scope='whole-fit only WP8 differs; both use borrowed-X and native export',rows=len(y),features=28,trees=100,depth=16,vendor='metal',mode='fast',binary_sha256=hashes,model_sha256=expected,scalar_min_ms=min(scalar),vector_min_ms=min(vector),baseline_endpoint_spread=spread,window_valid=spread<=.2,samples=results)),flush=True)
 if spread>.2:raise SystemExit('VOID: baseline endpoint drift exceeds20%')
finally:_ExtraTreesBase._bind=original
