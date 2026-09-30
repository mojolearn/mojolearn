#!/usr/bin/env python3
"""Focused ClassicalMDS quality, identity and 5,000-row regression check.

Runs candidate Python control flow over installed wheel bindings; does not
replace that wheel or count these development checks as board measurements.
"""
import argparse, hashlib, importlib.util, json, os, time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ['MOJOLEARN_NUMERIC_MODE'] = 'identical'
import numpy as np
from mojolearn import _backend
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--large-data')
a = p.parse_args()
path = Path(__file__).resolve().parents[1] / 'python/mojolearn/_expansion_decomp.py'
spec = importlib.util.spec_from_file_location('mojolearn._mds_candidate', path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
gpu = m._Kit('identical')
assert m._kit_vendor(gpu) in ('cuda','hip','metal'), m._kit_vendor(gpu)
host = m._Kit('identical', _backend.load_host_module('_mojolearn_x_decomp_host'))
records = []
def model(kit, metric):
 class Candidate(m.ClassicalMDS):
  def _kit(self):return kit
 return Candidate(n_components=2, metric=metric, numeric_mode='identical')
def digest(x):return hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()
def check(X, metric, cpu=True, reference=True):
 outputs=[]; timings=[]
 for kit in ([gpu,gpu,host] if cpu else [gpu,gpu]):
  t=time.perf_counter(); est=model(kit,metric).fit(X);timings.append(time.perf_counter()-t)
  Y=np.asarray(est.embedding_).copy();w=np.asarray(est.eigenvalues_).copy()
  assert np.isfinite(Y).all() and np.isfinite(w).all() and (w>=0).all()
  outputs.append((Y,w))
 for Y,w in outputs[1:]:
  np.testing.assert_array_equal(outputs[0][0].view('u4'),Y.view('u4'))
  np.testing.assert_array_equal(outputs[0][1].view('u4'),w.view('u4'))
 r={'rows':len(X),'metric':metric,'seconds':timings,'embedding_sha256':digest(outputs[0][0]),
    'eigenvalues_sha256':digest(outputs[0][1]),'input_sha256':digest(X),'cpu_gpu_identical':cpu,'repeat_identical':True}
 if reference:
  from scipy.linalg import eigh
  D=np.asarray(est.dissimilarity_matrix_,dtype=np.float64)
  B=-.5*D*D;B=B-B.mean(0)[None,:]-B.mean(1)[:,None]+B.mean()
  rw,rv=eigh(B,subset_by_index=[len(X)-2,len(X)-1]);rw=rw[::-1];rv=rv[:,::-1]
  Y,w=outputs[0];err=float(np.max(abs(w-rw))/max(np.max(abs(rw)),1e-30))
  r['reference_eigenvalue_relative_error']=err
  assert err<5e-5,(r,w,rw)
  if min(rw)>max(rw)*1e-5:
   V=Y/np.sqrt(w);res=float(np.linalg.norm(B@V-V*w)/max(np.max(abs(w)),1e-30))
   cos=np.linalg.svd(rv.T@np.linalg.qr(V.astype('f8'))[0],compute_uv=False)
   r.update(relative_residual=res,subspace_cosines=cos.tolist())
   assert res<5e-5 and min(cos)>.9999,r
 records.append(r);print(json.dumps(r),flush=True)
 Path(a.out).write_text(json.dumps({'vendor':m._kit_vendor(gpu),'records':records,'passed':True},indent=2)+'\n')
# Exact dyadic inputs, reproducible on every machine, with separated eigenvalues.
def data(n,d):
 z=np.arange(n*d,dtype=np.int64).reshape(n,d)
 return (((z*1103515245+12345)%65536-32768)/32768).astype('f4')*np.arange(d,0,-1,dtype='f4')
for n in (17,257):
 X=data(n,7);check(X,'euclidean')
 D=np.abs(np.arange(n,dtype='f4')[:,None]-np.arange(n,dtype='f4')[None,:])
 # A non-Euclidean distance example with multiple positive axes.
 D=np.minimum(D,n-D);check(D,'precomputed')
check(np.zeros((17,3),'f4'),'euclidean')
X=np.load(a.large_data) if a.large_data else data(5000,14)
X=np.ascontiguousarray(X[np.arange(min(len(X),5000))*len(X)//min(len(X),5000)],dtype='f4')
check(X,'euclidean',cpu=False,reference=False)
