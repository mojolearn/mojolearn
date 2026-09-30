#!/usr/bin/env python3
"""Bounded GPU top-k eigensolve experiment; no shipped estimator changes.

Uses the current neighbor binding for kernel formation/centering and the
existing decomp Lanczos implementation for the eight requested eigenpairs.
An independent float64 SciPy check is correctness-only, at <=1000 rows.
Larger inputs use the actual Ritz residual and orthogonality, not timing
alone, to accept the result. Never falls back to the large host eigensolve.
"""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MOJOLEARN_NUMERIC_MODE','identical')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from mojolearn import KernelPCA
from mojolearn._expansion_neighbors import _f32, _empty_out
from mojolearn._expansion_decomp import _Kit, _M, _lanczos_top, _kit_vendor

p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--rows',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
Xall=np.load(a.data);idx=np.linspace(0,len(Xall)-1,min(a.rows,len(Xall)),dtype=int)
X=np.ascontiguousarray(Xall[idx],dtype=np.float32);n,d=X.shape;nc=8
k=_Kit('identical');vendor=_kit_vendor(k)
if vendor not in ['cuda','metal','hip']:raise RuntimeError('GPU required: '+vendor)
est=KernelPCA(n_components=nc,kernel='rbf',random_state=7)
x=_f32(X);est._gamma=np.float32(1.0/d).item()
t=time.perf_counter();K=est._k(x,x)
cols=est._scale_div(est._colsum(K),float(n));total=est._scale_div(est._colsum(cols.reshape((1,n))),float(n))
Kc=_empty_out((n,n),'<f4');est._op('kpca_center',[(K,0),(cols,0),(cols,0),(total,0),(Kc,1)],(n,n))
host=np.asarray(Kc,dtype=np.float32)
kernel_ms=(time.perf_counter()-t)*1000
store=array.array('f');store.frombytes(host.tobytes());A=_M(store,n,n)
t=time.perf_counter();got=_lanczos_top(k,A,nc)
if got is None:raise RuntimeError('Lanczos did not converge within its bounded basis; no dense fallback')
w,V=got;V=V.neg_cols(k.absmax_flags(V,True))
values=np.asarray(w.s,dtype=np.float32);vectors=np.asarray(V.s,dtype=np.float32).reshape(n,nc).copy()
solver_ms=(time.perf_counter()-t)*1000
# Residual uses our GPU GEMM; no large CPU eigensolve or CPU timing race.
Av=np.asarray(k.mm(A,V).s,dtype=np.float32).reshape(n,nc)
res=np.linalg.norm(Av-vectors*values,axis=0)/max(float(np.max(np.abs(values))),1e-30)
orth=float(np.max(np.abs(vectors.T@vectors-np.eye(nc))))
r={'rows':n,'features':d,'components':nc,'vendor':vendor,'numeric_operations':'identical; NEW iterative algorithm, not certified equivalent to old output bits',
   'input_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'kernel_and_center_ms':kernel_ms,'solver_ms':solver_ms,
   'eigenvalues':values.tolist(),'relative_residuals':res.tolist(),'orthogonality_max_abs':orth,
   'passed':bool(max(res)<1e-5 and orth<1e-4)}
if n<=1000:
 from scipy.linalg import eigh
 refw,refv=eigh(host.astype(np.float64),subset_by_index=[n-nc,n-1]);refw=refw[::-1];refv=refv[:,::-1]
 cos=np.linalg.svd(refv.T@vectors,compute_uv=False)
 err=float(np.max(np.abs(values-refw))/max(abs(refw)))
 r.update(reference='SciPy float64, correctness only',reference_relative_eigenvalue_error=err,subspace_cosines=cos.tolist())
 r['passed']=r['passed'] and err<1e-5 and float(min(cos))>0.9999
Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if not r['passed']:raise SystemExit(2)
