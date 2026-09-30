#!/usr/bin/env python3
"""Targeted compact label-graph checks; never runs the full benchmark board."""
import argparse, hashlib, importlib.util, json, os, resource, subprocess, sys, time
from pathlib import Path
os.environ.setdefault('MOJOLEARN_NUMERIC_MODE','identical')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--host',required=True);p.add_argument('--out',required=True);a=p.parse_args()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
gpu=load('_mojolearn_x_neighbors',a.gpu);host=load('_mojolearn_x_neighbors_host',a.host)
root=Path(__file__).resolve().parents[1]
import mojolearn
mod=load('mojolearn._label_graph_candidate',root/'python/mojolearn/_expansion_neighbors.py')
records=[]
def call(b,name,arrays,ints,fl=()):getattr(b,'xn_'+name)([x.ctypes.data for x in arrays],ints,list(fl))
def same(x,y):np.testing.assert_array_equal(x.view(np.uint32),y.view(np.uint32))
rng=np.random.default_rng(7)
for n,m,k in [(17,17,1),(33,33,7),(9,9,9),(11,23,7)]:
 idx=rng.integers(0,m,(n,k),dtype=np.int32);idx[0]=-1 # duplicates, missing, self and isolated vertices
 for variant in ([0,1,2] if n==m else [2]):
  dense=np.zeros((n,m),np.float32);call(gpu,'knn_graph',[idx,dense],[n,m,k])
  want=np.empty_like(dense)
  if variant==0:call(gpu,'row_normalize',[dense,want],[n,m])
  elif variant==1:
   deg=np.empty(n,np.float32);call(gpu,'col_degree',[dense,deg],[n]);call(gpu,'ls_laplacian_deg',[dense,deg,want],[n])
  else:want=dense
  pairs=[]
  for b in [gpu,host]:
   cols=np.empty_like(idx);vals=np.empty(idx.shape,np.float32);call(b,'lp_knn_graph',[idx,cols,vals],[n,m,k,variant]);pairs.append((cols,vals))
   restored=np.zeros((n,m),np.float32)
   for i in range(n):
    for j,v in zip(cols[i],vals[i]):
     if j>=0:restored[i,j]=v
   same(restored,want)
  np.testing.assert_array_equal(pairs[0][0],pairs[1][0]);same(pairs[0][1],pairs[1][1])
  for nonfinite in [False,True]:
   x=rng.standard_normal((m,3)).astype(np.float32)
   if nonfinite:x[0,0]=np.inf;x[-1,1]=np.nan
   expected=np.empty((n,3),np.float32);call(gpu,'matmul',[want,x,expected],[n,m,3])
   for b,(cols,vals) in zip([gpu,host],pairs):
    actual=np.empty_like(expected);call(b,'lp_knn_product',[cols,vals,x,actual],[n,m,k,3]);same(actual,expected)
  print(f'PASS graph/product n={n} m={m} k={k} variant={variant}',flush=True)
  records.append({'check':'graph-and-product','n':n,'m':m,'k':k,'variant':variant})

def cls(base,dense=False):
 class Model(base):
  def _bind(self,name=None):return gpu
  def _compact_graph(self,idx,n_reference,variant):
   if not dense:return super()._compact_graph(idx,n_reference,variant)
   n,k=idx.shape;G=mod.empty((n,n_reference),'<f4');self._op('knn_graph',[(idx,0),(G,1)],(n,n_reference,k))
   if variant==2:return G
   out=mod.empty(G.shape,'<f4')
   if variant==0:self._op('row_normalize',[(G,0),(out,1)],G.shape)
   else:
    deg=mod.empty((n,),'<f4');self._op('col_degree',[(G,0),(deg,1)],(n,));self._op('ls_laplacian_deg',[(G,0),(deg,0),(out,1)],(n,))
   return out
 return Model

for base in [mod.LabelPropagation,mod.LabelSpreading]:
 for n,k,iters in [(17,1,7),(33,7,30),(96,7,100),(5,9,0)]:
  X=rng.standard_normal((n,4)).astype(np.float32);X[1]=X[0]
  y=np.where(np.arange(n)%3==0,np.arange(n)%2,-1);Q=X[::3].copy()
  sparse=cls(base)(kernel='knn',n_neighbors=k,max_iter=iters).fit(X,y)
  dense=cls(base,True)(kernel='knn',n_neighbors=k,max_iter=iters).fit(X,y)
  for field in ['label_distributions_','transduction_']:same(np.asarray(getattr(sparse,field)),np.asarray(getattr(dense,field)))
  same(np.asarray(sparse.predict_proba(Q)),np.asarray(dense.predict_proba(Q)))
  assert sparse.n_iter_==dense.n_iter_
  print(f'PASS dense parity {base.__name__} n={n} k={k}',flush=True)
  records.append({'check':'public-dense-parity','algorithm':base.__name__,'n':n,'k':k,'max_iter':iters,'n_iter':sparse.n_iter_,
    'digest':hashlib.sha256(np.asarray(sparse.label_distributions_).tobytes()).hexdigest()})

# The failure was graph storage, not neighbor search. Supply a fixed 7-neighbor
# ring to exercise public fit/predict at the failing size without a 40B-pair race.
n=200000;k=7;max_cells=0
old_empty,old_out=mod.empty,mod._empty_out
def bounded(fn):
 def alloc(shape,*args,**kw):
  global max_cells
  cells=int(np.prod(shape));max_cells=max(max_cells,cells)
  if cells>n*k:raise MemoryError(('unexpected dense allocation',shape))
  return fn(shape,*args,**kw)
 return alloc
mod.empty=bounded(old_empty);mod._empty_out=bounded(old_out)
def large_class(base,dense=False):
 class Large(cls(base,dense)):
  def _knn_sq(self,Q,R,k,exclude_self):
   ids=((np.arange(len(Q),dtype=np.int64)[:,None]+np.arange(k))%len(R)).astype(np.int32)
   return None,mod.Array.from_buffer(ids)
 return Large
for base in [mod.LabelPropagation,mod.LabelSpreading]:
 Large=large_class(base)
 t=time.perf_counter();X=np.zeros((n,1),np.float32);y=np.where(np.arange(n)%10==0,np.arange(n)%3,-1)
 try:
  large_class(base,True)(kernel='knn',n_neighbors=k,max_iter=2).fit(X,y)
 except MemoryError as exc:
  assert exc.args[0][1]==(n,n),exc
 else:raise AssertionError('allocation guard did not reject legacy dense graph')
 max_cells=0
 model=Large(kernel='knn',n_neighbors=k,max_iter=2).fit(X,y);prediction=np.asarray(model.predict_proba(X[:20000]))
 assert np.isfinite(prediction).all()
 print(f'PASS memory scaling {base.__name__} n={n} k={k}',flush=True)
 records.append({'check':'public-memory-scaling','algorithm':base.__name__,'rows':n,'neighbors':k,
   'legacy_dense_allocation_rejected':True,'largest_python_allocation_cells':max_cells,
   'distribution_digest':hashlib.sha256(np.asarray(model.label_distributions_).tobytes()).hexdigest(),
   'prediction_digest':hashlib.sha256(prediction.tobytes()).hexdigest(),'wall_s':time.perf_counter()-t,'neighbor_search':'fixed ring fixture; not a full neighbor-search benchmark'})
r={'passed':True,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'vendor':gpu.x_neighbors_vendor(),'numeric_mode':gpu.x_neighbors_numeric_mode(),'checks':records,
 'peak_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_units':'bytes' if sys.platform=='darwin' else 'KiB'}
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
