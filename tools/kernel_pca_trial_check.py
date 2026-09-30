#!/usr/bin/env python3
"""Quality/repeatability check for the opt-in public KernelPCA candidate."""
import argparse, array, hashlib, json, os, time
from pathlib import Path
os.environ['MOJOLEARN_XN_KPCA_LANCZOS']='1'
os.environ['MOJOLEARN_NUMERIC_MODE']='identical'
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from mojolearn import KernelPCA
from mojolearn._expansion_decomp import _Kit, _M, _kit_vendor
from mojolearn._expansion_neighbors import _empty_out
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--rows',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
src=np.load(a.data);X=np.ascontiguousarray(src[(np.arange(a.rows)*len(src))//a.rows]);n,d=X.shape
times=[];outputs=[]
for repeat in range(2):
 est=KernelPCA(n_components=8,kernel='rbf',random_state=7,numeric_mode='identical')
 t=time.perf_counter();Y=np.asarray(est.fit_transform(X)).copy();times.append((time.perf_counter()-t)*1000)
 outputs.append(hashlib.sha256(Y.tobytes()).hexdigest())
V=np.asarray(est.eigenvectors_);w=np.asarray(est.eigenvalues_)
K=est._k(est._fit_X,est._fit_X);Kc=_empty_out((n,n),'<f4')
est._op('kpca_center',[(K,0),(est._fit_cols,0),(est._fit_cols,0),(est._fit_all,0),(Kc,1)],(n,n))
kit=_Kit('identical')
def mat(a,r,c):
 s=array.array('f');s.frombytes(a.tobytes());return _M(s,r,c)
AV=np.asarray(kit.mm(mat(Kc,n,n),mat(V,n,8)).s).reshape(n,8)
res=float(max(np.linalg.norm(AV-V*w,axis=0))/max(abs(w)))
orth=float(np.max(abs(V.T@V-np.eye(8))))
Z=np.asarray(est.transform(X));transform_error=float(np.max(abs(Z-Y))/max(np.max(abs(Y)),1e-30))
r={'rows':n,'features':d,'fit_transform_ms':times,'output_sha256':outputs,'repeat_identical':len(set(outputs))==1,
 'vendor':_kit_vendor(kit),'input_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'relative_residual':res,
 'orthogonality_max_abs':orth,'fit_transform_vs_transform_relative_max':transform_error,
 'passed':res<1e-5 and orth<1e-4 and transform_error<1e-4 and len(set(outputs))==1,
 'scope':'Experimental opt-in. Same-device repeatability only; Apple/AMD identity NOT certified.'}
if n<=1000:
 from scipy.linalg import eigh
 H=np.asarray(Kc).astype(np.float64);rw,rv=eigh(H,subset_by_index=[n-8,n-1]);rw=rw[::-1];rv=rv[:,::-1]
 Q=np.linalg.qr(V.astype(np.float64),mode='reduced')[0]
 cos=np.linalg.svd(rv.T@Q,compute_uv=False)
 err=float(max(abs(w-rw))/max(abs(rw)))
 r.update(reference_relative_eigenvalue_error=err,subspace_cosines=cos.tolist())
 r['passed']=r['passed'] and err<1e-5 and min(cos)>0.9999
r['passed']=bool(r['passed'])
Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if not r['passed']:raise SystemExit(2)
