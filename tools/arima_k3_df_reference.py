#!/usr/bin/env python3
"""M3 reference-only float-float K3 proposal versus saved actual-main errors.
No GPU execution, timers, optimizer changes or admission from this experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import arima_scalar_quality as fixture
import numpy as np

REPORT_SHA = '0bb5fa599ca808fd5d51c3dcc854c83edcc411f527e64dbf3744a53a7cf0e4ec'
LENGTHS = (1,2,255,256,257,1391,1392,1393,4095,4096,4097,65535,65536)
F = np.float32


def df(x):
    x=np.asarray(x,np.float32)
    return x,np.zeros_like(x)


def two_sum(a,b):
    s=F(a+b);v=F(s-a)
    return s,F(F(a-F(s-v))+F(b-v))


def quick(a,b):
    s=F(a+b)
    return s,F(b-F(s-a))


def add(a,b):
    s=two_sum(a[0],b[0]);t=two_sum(a[1],b[1])
    u=quick(s[0],F(s[1]+t[0]))
    return quick(u[0],F(t[1]+u[1]))


def neg(a): return -a[0],-a[1]
def sub(a,b): return add(a,neg(b))


def mul(a,b):
    p=F(a[0]*b[0])
    # Float64 here models only FMA's exact f32 product residual; not a
    # float64 candidate dot product, likelihood or production compute path.
    e=F(a[0].astype(np.float64)*b[0].astype(np.float64)-p.astype(np.float64))
    e=F(a[0].astype(np.float64)*b[1].astype(np.float64)+e.astype(np.float64))
    e=F(a[1].astype(np.float64)*b[0].astype(np.float64)+e.astype(np.float64))
    return quick(p,e)


def div(a,b):
    q1=F(a[0]/b[0]);r=sub(a,mul(b,df(q1)))
    q2=F(r[0]/b[0]);r=sub(r,mul(b,df(q2)))
    q3=F(r[0]/b[0])
    return add(quick(q1,q2),df(q3))


def rounded(a): return F(a[0]+a[1])


def log_positive(x):
    # Exact power-of-two range reduction; 0.7071 <= m < 1.4143.
    # log(m)=2*(z+z^3/3+...+z^25/25), |z|<=0.171573.
    # Series remainder < 2e-22; arithmetic error remains measured separately.
    mant,expo=np.frexp(np.asarray(x,np.float32))
    adjust=mant<F(0.7071067811865476)
    mant=np.where(adjust,F(mant*F(2)),mant).astype(np.float32)
    expo=expo-adjust.astype(expo.dtype)
    z=div(sub(df(mant),df(1)),add(df(mant),df(1)))
    z2=mul(z,z);term=z;total=z
    for denominator in range(3,26,2):
        term=mul(term,z2)
        total=add(total,div(term,df(denominator)))
    ln2=(np.asarray(F(0.6931471805599453)),np.asarray(F(-1.904654299957768e-9)))
    return add(mul(total,df(2)),mul(df(expo),ln2))


def reduce256(values):
    hi,lo=values;batch,n=hi.shape;chunks=(n+255)//256
    def level(h,l):
        h=h.copy();l=l.copy()
        width=128
        while width:
            v=add((h[...,:width],l[...,:width]),(h[...,width:2*width],l[...,width:2*width]))
            h[...,:width],l[...,:width]=v
            width//=2
        return h[...,0],l[...,0]
    h=np.pad(hi,((0,0),(0,chunks*256-n))).reshape(batch,chunks,256)
    l=np.pad(lo,((0,0),(0,chunks*256-n))).reshape(batch,chunks,256)
    h,l=level(h,l)
    return level(np.pad(h,((0,0),(0,256-chunks))),np.pad(l,((0,0),(0,256-chunks))))


def candidate(states,y,intercept):
    T,Q,P,a,mu=states
    pred=mul(df(T[:,None]),df(y[:,:-1]))
    if intercept: pred=add(pred,df(mu[:,None]))
    pred=(np.concatenate((a[:,None],pred[0]),axis=1),np.concatenate((np.zeros_like(a[:,None]),pred[1]),axis=1))
    variance=np.broadcast_to(Q[:,None],y.shape).copy();variance[:,0]=P
    residual=sub(df(y),pred)
    bad=(variance<=0)|~np.isfinite(variance)|~np.isfinite(rounded(residual))
    info=np.where(bad.any(axis=1),bad.argmax(axis=1)+1,0).astype(np.int32)
    if np.any(info): return None,None,info
    logs=log_positive(variance)
    squares=div(mul(residual,residual),df(variance))
    sl,ss=reduce256(logs),reduce256(squares)
    # Same fixed likelihood objective: no analytic derivative substitution.
    log2pi=(np.asarray(F(1.8378770664093453)),np.asarray(F(3.1268354230284965e-8)))
    n=df(y.shape[1])
    ll=mul(df(-.5),add(mul(n,add(div(ss,n),log2pi)),sl))
    return (rounded(pred),rounded(residual),variance,rounded(ll),Q.copy()),ll,info


def inputs(n):
    states,ys,labels=[],[],[];rng=np.random.default_rng(20261004+n)
    for phi in (0.,.65,.985,-.985,.999,-.999):
        for scale in (1e-3,1.,1e3):
            raw=np.array([2*np.arctanh(phi),.08*scale,scale*scale],np.float32)
            state=fixture.model(raw);noise=rng.normal(size=n+256)*scale
            signal=np.empty(n+256,np.float64);value=float(state[3])
            for i,eps in enumerate(noise):
                value=float(state[0])*value+float(state[4])+eps;signal[i]=value
            y=signal[256:].astype(np.float32)
            for parameter in (-1,0,1,2):
                perturbed=raw.copy()
                if parameter>=0:perturbed[parameter]=F(perturbed[parameter]+fixture.H)
                states.append(fixture.model(perturbed));ys.append(y)
            labels.append(dict(phi=phi,scale=scale,raw=raw.tolist()))
    return np.stack(states,axis=1),np.stack(ys),labels


def compare(error,baseline):
    baseline=np.asarray(baseline,np.float64)
    return dict(ok=bool(np.all(error<=baseline)),truly_worse=int(np.sum(error>baseline)),
                improved=int(np.sum(error<baseline)),equal=int(np.sum(error==baseline)),
                candidate_error=error.tolist(),baseline_error=baseline.tolist())


def evaluate(saved):
    n=saved['n'];states,y,labels=inputs(n)
    assert labels==saved['labels']
    assert hashlib.sha256(y.tobytes()+states.tobytes()).hexdigest()==saved['input_sha256']
    ref,llref,P=fixture.reference(states,y,1)
    out,ll,info=candidate(states,y,1);assert not np.any(info)
    refs=(*ref,llref,P);checks={}
    for name,value,target in zip(('prediction','innovation','variance','loglike','final_covariance'),out,refs):
        error=np.abs(value.astype(np.float64)-target)
        if error.ndim>1:error=error.max(axis=1)
        checks[name]=compare(error,saved['checks'][name]['baseline_error'])
    gradient=None
    if n>1:
        # Keep LL low words through exactly the existing fixed-h difference,
        # then divide by h and n-1 and round only the returned gradient.
        h,l=(x.reshape(-1,4) for x in ll)
        difference=sub((h[:,1:],l[:,1:]),(h[:,:1],l[:,:1]))
        g=rounded(div(neg(div(difference,df(fixture.H))),df(n-1)))
        fref=(-llref/(n-1)).reshape(-1,4)
        gr=(fref[:,1:]-fref[:,:1])/float(fixture.H)
        assert np.array_equal(gr,np.asarray(saved['gradient']['reference']))
        gradient=compare(np.abs(g.astype(np.float64)-gr),saved['gradient']['baseline_error'])
        gradient.update(candidate=g.tolist(),reference=gr.tolist())
    return dict(n=n,checks=checks,gradient=gradient,
                status='PASS' if all(x['ok'] for x in checks.values()) and (gradient is None or gradient['ok']) else 'HOLD')


def controls(saved):
    results=[]
    states=np.array([[.5,.5,.5],[1.,0.,1.],[0.,1.,1.],[0.,0.,0.],[0.,0.,0.]],np.float32)
    _,_,info=candidate(states,np.zeros((3,257),np.float32),0)
    results.append(dict(name='nonpositive-variance',ok=bool(np.array_equal(info,[1,2,0])),info=info.tolist()))
    for original in saved[1:]:
        name=original['name'];n=1393
        states=fixture.model(np.array([2*np.arctanh(.985),0.,1.],np.float32))[:,None]
        y=np.zeros((1,n),np.float32)
        if name=='constant':y.fill(1e3)
        elif name=='impulse':y[0,n//2]=1e3
        elif name=='alternating':y[0]=np.where(np.arange(n)%2,1e3,-1e3)
        else:raise ValueError(name)
        ref,llref,P=fixture.reference(states,y,0);out,ll,info=candidate(states,y,0)
        checks={}
        for key,v,t in zip(('0','1','2','ll','P'),out,(*ref,llref,P)):
            e=np.abs(v.astype(np.float64)-t)
            if e.ndim>1:e=e.max(axis=1)
            checks[key]=compare(e,original['checks'][key]['baseline_error'])
        results.append(dict(name=name,ok=not np.any(info) and all(c['ok'] for c in checks.values()),checks=checks))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-report',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    assert subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()=='Apple M3 Ultra'
    data=Path(args.baseline_report).expanduser().read_bytes();assert hashlib.sha256(data).hexdigest()==REPORT_SHA
    saved=json.loads(data);assert saved['fixture']==fixture.FIXTURE and saved['degradation_allowance']==0
    assert tuple(c['n'] for c in saved['cases'])==LENGTHS
    output=Path(args.output).expanduser();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(dict(status='RUNNING'),stream)
    records=[]
    with np.errstate(all='raise',under='ignore'):
        for case in saved['cases']:
            row=evaluate(case);records.append(row);print('K3-DF-REFERENCE n='+str(row['n'])+' '+row['status'],flush=True)
        checks=controls(saved['controls'])
    report=dict(status='PASS' if all(r['status']=='PASS' for r in records) and all(c['ok'] for c in checks) else 'HOLD',
                cases=records,controls=checks,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                fixture='k3-df-reference-v1',baseline_report_sha256=REPORT_SHA,
                baseline='saved actual-main GPU errors from actual-tail quality',
                candidate='NumPy float-float proposal emulation, NOT GPU bits',
                initialization='exact same hash-verified supplied-state fixture; actual device Jones/initializer not certified',
                degradation_allowance=0,gradient_step=float(fixture.H),scored_timings=0,promotion_authorized=False)
    temp=output.with_suffix(output.suffix+'.next')
    with temp.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    temp.replace(output)


if __name__=='__main__':main()
