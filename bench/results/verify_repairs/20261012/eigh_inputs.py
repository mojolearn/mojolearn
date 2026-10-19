import json, pathlib, numpy as np
import mojolearn as ml
from mojolearn import _identity_break as h
ref=json.load(open('/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/identity_break/2026-10-10-0837d/apple-m4/apple-m4.identical.json'))
out=[]
for f in h.FIXTURES:
 g=np.ascontiguousarray(h.fixture(f)[0][:64,:5],dtype=np.float32)
 serial=np.zeros((5,5),np.float32)
 for row in g: serial += row[:,None]*row[None,:]
 double=(g.astype(np.float64).T@g.astype(np.float64)).astype(np.float32)
 methods={'blas':g.T@g,'serial':serial,'double':double,'einsum':np.einsum('ri,rj->ij',g,g,optimize=False),'fortran':np.asfortranarray(g).T@np.asfortranarray(g)}
 expected=ref['cells']['linalg-eigh/'+f]['parts'][0]
 for name,m in methods.items():
  m=np.ascontiguousarray((m+m.T)*np.float32(.5));w,v=ml.linalg.eigh(m)
  wn,vn=np.asarray(w),np.asarray(v)
  resid=float(np.linalg.norm(m.astype(float)@vn-vn*wn)/(np.linalg.norm(m.astype(float)) or 1))
  orth=float(np.linalg.norm(vn.astype(float).T@vn-np.eye(5)))
  out.append(dict(fixture=f,method=name,input=h._h(m),w=h._h(w),v=h._h(v),w_matches=h._h(w)==expected['gram_w'],v_matches=h._h(v)==expected['gram_v'],residual=resid,orthogonality=orth))
pathlib.Path('/Users/andrewhendel/CascadeProjects/mojolearn/bench/results/verify_repairs/20261012/eigh-inputs.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out:
 if x['fixture'] in ('hashed','wide','odd','negative'):print(json.dumps(x))
