#!/usr/bin/env python3
"""Actual qn_fit multiclass quality gate. M3 only; no timing/build fallback."""
import argparse, hashlib, importlib.util, json, os, subprocess
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
if not __debug__: raise RuntimeError('Assertions required')
# Generic spread (rows 2k..200k, features 8..1500, classes 2..64), no shape
# window: the route has none since 2026-10-04 (old window cases removed).
CASES = {
 'r2000-d8-c2':(2000,8,2), 'r5003-d37-c5':(5003,37,5),
 'r12000-d97-c11':(12000,97,11), 'r20011-d300-c24':(20011,300,24),
 'r8000-d1500-c3':(8000,1500,3), 'r3001-d700-c40':(3001,700,40),
 'r60000-d150-c64':(60000,150,64), 'r200000-d64-c9':(200000,64,9),
}
POLICY='softmax-g2-real-fit-zero-regression-v2-no-window'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    with Path(p).open('x') as f: json.dump(data,f,indent=2,allow_nan=False)
def make(case):
    import numpy as np
    n,d,c=CASES[case];rng=np.random.default_rng(2026100411)
    teacher=rng.normal(size=(d,c))/np.sqrt(d)
    x=rng.normal(size=(n,d)).astype(np.float32)
    q=rng.normal(size=(4097,d)).astype(np.float32)
    y=np.argmax(x.astype(float)@teacher+.2*rng.normal(size=(n,c)),axis=1).astype(np.float32)
    qy=np.argmax(q.astype(float)@teacher+.2*rng.normal(size=(len(q),c)),axis=1).astype(np.float32)
    assert len(np.unique(y))==c
    return x,y,q,qy

def capture(a):
    import numpy as np
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE')=='fast'
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert source==a.source and len(source)==40
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
    binary=Path(a.binding).resolve(strict=True);assert sha(binary)==a.binding_sha256
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists() and not Path(str(output)+'.json').exists()
    write(str(output)+'.started.json',dict(source=source,case=a.case,arm=a.arm,status='RUNNING'))
    spec=importlib.util.spec_from_file_location('_mojolearn_estimators',binary)
    binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    assert int(binding.estimators_numeric_mode())==0 and str(binding.estimators_vendor())=='metal'
    assert int(binding.softmax_g2_state())==a.arm
    n,d,c=CASES[a.case];x,y,q,qy=make(a.case)
    w=np.zeros((d+1,c),np.float32);info=np.full(2,np.nan,np.float32)
    phases={}
    def counts(name):
        phases[name]=dict(counts=[int(binding.softmax_g2_count(i)) for i in range(3)],
                          last=[int(binding.softmax_g2_last(i)) for i in range(5)])
    binding.softmax_g2_reset()
    iterations=int(binding.qn_fit(x.ctypes.data,y.ctypes.data,w.ctypes.data,info.ctypes.data,
        [n,d,c,0.,0.,1e-4,1e-5,30,30,5,1,0,0,2]))
    counts('fit')
    arrays=dict(x=x,y=y,q=q,qy=qy,coef=w,info=info,iterations=np.array(iterations))
    for name,data in [('train',x),('query',q)]:
        scores=np.full((len(data),c),np.nan,np.float32);binding.softmax_g2_reset()
        binding.qn_decision_function(data.ctypes.data,w.ctypes.data,scores.ctypes.data,[len(data),d,1,c])
        counts(name);arrays[name+'_scores']=scores
    assert all(np.isfinite(value).all() for value in arrays.values())
    for phase,rows in [('fit',n),('train',n),('query',len(q))]:
        expected=rows>=1 and c>=1 and d>=1  # kernel limits only (no window)
        reached=phases[phase]['counts']
        assert reached[1]+reached[2]>0, 'NO_REACH '+phase
        assert (reached[0]>0) if expected else (reached[0]==0)
        assert reached[1]==(reached[0] if a.arm else 0)
        assert phases[phase]['last']==[rows,c,d,int(expected),int(expected and a.arm==1)]
    with output.open('xb') as f: np.savez(f,**arrays)
    write(str(output)+'.json',dict(policy=POLICY,source=source,case=a.case,arm=a.arm,
        binding=str(binary),binding_sha256=a.binding_sha256,capture_sha256=sha(output),
        input_sha256=hashlib.sha256(x.tobytes()+y.tobytes()+q.tobytes()+qy.tobytes()).hexdigest(),
        phases=phases,scored_timings=0,promotion_authorized=False))

def compare(a):
    import numpy as np
    A,B=[np.load(p,allow_pickle=False) for p in (a.a,a.b)]
    ma,mb=[json.loads(Path(str(p)+'.json').read_text()) for p in (a.a,a.b)]
    for key in ('policy','source','case','input_sha256'): assert ma[key]==mb[key]
    assert ma['policy']==POLICY and ma['arm']==0 and mb['arm']==1
    assert ma['capture_sha256']==sha(a.a) and mb['capture_sha256']==sha(a.b)
    assert all(np.array_equal(A[k],B[k]) for k in ('x','y','q','qy'))
    assert all(np.isfinite(v).all() for arm in (A,B) for v in arm.values())
    metrics={}
    def le(name,av,bv):
        metrics[name]=dict(A=float(av),B=float(bv),ok=bool(bv<=av))
    def softmax(z):
        zz=z-z.max(axis=1,keepdims=True);e=np.exp(zz)
        return e/e.sum(axis=1,keepdims=True),zz
    def loss(z,y):
        _,zz=softmax(z)
        return np.mean(np.log(np.exp(zz).sum(axis=1))-zz[np.arange(len(y)),y.astype(int)])
    details=[]
    for arm in (A,B):
        x=arm['x'].astype(float);w=arm['coef'].astype(float)
        z=x@w[:-1]+w[-1];qz=arm['q'].astype(float)@w[:-1]+w[-1]
        prob,_=softmax(z);prob[np.arange(len(x)),arm['y'].astype(int)]-=1
        grad=np.vstack((x.T@prob/len(x),prob.mean(axis=0)))
        details.append(dict(train=loss(z,arm['y']),holdout=loss(qz,arm['qy']),
            gradient=np.linalg.norm(grad),classification=np.mean(qz.argmax(1)!=arm['qy']),
            scores=np.max(np.abs(arm['query_scores'].astype(float)-qz)),
            train_scores=np.max(np.abs(arm['train_scores'].astype(float)-z)),
            objective_error=abs(float(arm['info'][0])-loss(z,arm['y']))))
    for key in details[0]: le(key,details[0][key],details[1][key])
    status_equal=bool(A['info'][1]==B['info'][1])
    good=all(m['ok'] for m in metrics.values()) and status_equal
    write(a.output,dict(policy=POLICY,source=ma['source'],case=ma['case'],
        status='PASS' if good else 'HOLD',metrics=metrics,retcode_equal=status_equal,
        phases_A=ma['phases'],phases_B=mb['phases'],degradation_allowance=0,
        scored_timings=0,promotion_authorized=False))
    return 0 if good else 1

def main():
    assert subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()=='Apple M3 Ultra'
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    c=s.add_parser('capture')
    for key in ('source','binding','binding-sha256','output'):c.add_argument('--'+key,required=True)
    c.add_argument('--case',choices=CASES,required=True);c.add_argument('--arm',type=int,choices=(0,1),required=True)
    c=s.add_parser('compare')
    for key in ('a','b','output'):c.add_argument('--'+key,required=True)
    a=p.parse_args();return capture(a) if a.mode=='capture' else compare(a)
if __name__=='__main__':raise SystemExit(main())
