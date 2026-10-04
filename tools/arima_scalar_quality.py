#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""M3 numerical gate for the actual main RD1 and opt-in K3 GPU kernels.

The candidate-only private probe launches both kernels on identical input
states. NumPy is an independent reference, never a product CPU route.
No timers/opponents. This kernel gate does NOT replace full AutoARIMA fit
and holdout forecast quality checks against independently built main.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np


H = np.float32(2.0**-10)
FIXTURE = "arima-scalar-k3-v1"


def model(raw):
    """Common supplied scalar state; initializer rounding is not certified.

    raw=(unconstrained AR, intercept, sigma2). The actual product continues
    to use its unchanged initializer. This probe isolates filtering/reduction.
    """
    phi = np.clip(np.tanh(raw[0]*np.float32(.5)),np.float32(-.9999),np.float32(.9999))
    sigma = np.maximum(raw[2],np.float32(1e-6))
    P0 = np.float32(float(sigma)/(1-float(phi)**2))
    denominator = 1-float(phi)
    if abs(denominator)<1e-3:
        denominator = np.copysign(1e-3,denominator)
    alpha0 = np.float32(float(raw[1])/denominator)
    return np.array([phi,sigma,P0,alpha0,raw[1]],np.float32)


def reference(states, y, intercept):
    T,Q,P,a,mu = states.astype(np.float64)
    pred = np.empty(y.shape,np.float64)
    pred[:,0] = a
    pred[:,1:] = T[:,None]*y[:,:-1].astype(np.float64)+(mu[:,None] if intercept else 0)
    F = np.broadcast_to(Q[:,None],y.shape).copy()
    F[:,0] = P
    residual = y.astype(np.float64)-pred
    nn = y.shape[1]
    ll = -.5*(np.log(F).sum(axis=1)+nn*((residual**2/F).sum(axis=1)/nn+np.log(2*np.pi)))
    return np.stack((pred,residual,F)),ll,Q


def probe(binding,states,y,intercept):
    y,states = np.ascontiguousarray(y,np.float32),np.ascontiguousarray(states,np.float32)
    batch,n = y.shape
    stages = np.empty((2,3,batch,n),np.float32)
    stats = np.empty((4,batch),np.float32)
    info = np.empty((2,batch),np.int32)
    reached = binding._arima_scalar_ll_probe(y.ctypes.data,states.ctypes.data,
        stages.ctypes.data,stats.ctypes.data,info.ctypes.data,[batch,n,intercept])
    if int(reached)!=1:
        raise AssertionError("K3 private probe did not reach candidate")
    return stages,stats,info


def compare_field(a,b,ref):
    a,b,ref = np.asarray(a,np.float64),np.asarray(b,np.float64),np.asarray(ref,np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(ok=False,reason="nonfinite output")
    da,db = np.abs(a-ref),np.abs(b-ref)
    # Compare worst error within this field, independently for each model.
    # No post-hoc noise allowance or unrelated model averaging.
    axes = tuple(range(1,da.ndim))
    ea = da.max(axis=axes) if axes else da
    eb = db.max(axis=axes) if axes else db
    return dict(ok=bool(np.all(eb<=ea)),baseline_error=ea.tolist(),candidate_error=eb.tolist())


def fixture(binding,n):
    states,ys = [],[]
    rng = np.random.default_rng(20261004+n)
    labels = []
    for phi in (0.,.65,.985,-.985,.999,-.999):
        for scale in (1e-3,1.,1e3):
            raw = np.array([2*np.arctanh(phi),.08*scale,scale*scale],np.float32)
            base = model(raw)
            noise = rng.normal(size=n+256)*scale
            signal = np.empty(n+256,np.float64)
            value = float(base[3])
            for i,eps in enumerate(noise):
                value = float(base[0])*value+float(base[4])+eps
                signal[i] = value
            y = signal[256:].astype(np.float32)
            # Each group: base then three actual h-sized raw perturbations.
            for parameter in (-1,0,1,2):
                perturbed = raw.copy()
                if parameter>=0:
                    perturbed[parameter] = np.float32(perturbed[parameter]+H)
                states.append(model(perturbed)); ys.append(y)
            labels.append(dict(phi=phi,scale=scale,raw=raw.tolist()))
    states,y = np.stack(states,axis=1),np.stack(ys)
    actual,stats,info = probe(binding,states,y,1)
    ref,ll,finalP = reference(states,y,1)
    checks = {}
    for field,name in enumerate(("prediction","innovation","variance")):
        checks[name] = compare_field(actual[0,field],actual[1,field],ref[field])
    checks["loglike"] = compare_field(stats[0],stats[1],ll)
    checks["final_covariance"] = compare_field(stats[2],stats[3],finalP)
    grad = None
    if n>1:
        fa = np.float32(-stats[0]/np.float32(n-1)).reshape(-1,4)
        fb = np.float32(-stats[1]/np.float32(n-1)).reshape(-1,4)
        fr = (-ll/(n-1)).reshape(-1,4)
        ga = np.float32((fa[:,1:]-fa[:,:1])/H)
        gb = np.float32((fb[:,1:]-fb[:,:1])/H)
        gr = (fr[:,1:]-fr[:,:1])/float(H)
        # Strict PER-COMPONENT comparison; a large improvement in one raw
        # parameter cannot hide a degradation in another parameter's gradient.
        ea,eb = np.abs(ga.astype(np.float64)-gr),np.abs(gb.astype(np.float64)-gr)
        grad = dict(ok=bool(np.all(eb<=ea)),baseline=ga.tolist(),candidate=gb.tolist(),
                    reference=gr.tolist(),baseline_error=ea.tolist(),candidate_error=eb.tolist())
    finite = bool(np.isfinite(stats).all() and np.isfinite(actual).all())
    valid = finite and bool(np.all(info==0)) and all(c["ok"] for c in checks.values()) and (grad is None or grad["ok"])
    return dict(n=n,status="PASS" if valid else "HOLD",labels=labels,checks=checks,gradient=grad,
        finite=finite,info=info.tolist(),stats=stats.tolist(),input_sha256=hashlib.sha256(y.tobytes()+states.tobytes()).hexdigest(),
        stage_sha256=hashlib.sha256(actual.tobytes()).hexdigest())


def controls(binding):
    # First-step and later-step nonpositive variance refusal; no gradient.
    states = np.array([[.5,.5,.5],[1.,0.,1.],[0.,1.,1.],[0.,0.,0.],[0.,0.,0.]],np.float32)
    y = np.zeros((3,257),np.float32)
    _,_,info = probe(binding,states,y,0)
    expected = np.array([[1,2,0],[1,2,0]],np.int32)
    records = [dict(name="nonpositive-variance",ok=bool(np.array_equal(info,expected)),info=info.tolist())]
    for name in ("constant","impulse","alternating"):
        n=1393
        states = model(np.array([2*np.arctanh(.985),0.,1.],np.float32))[:,None]
        y=np.zeros((1,n),np.float32)
        if name=="constant": y.fill(1e3)
        elif name=="impulse": y[0,n//2]=1e3
        else: y[0]=np.where(np.arange(n)%2,1e3,-1e3)
        actual,stats,info = probe(binding,states,y,0)
        ref,ll,P = reference(states,y,0)
        checks={str(i):compare_field(actual[0,i],actual[1,i],ref[i]) for i in range(3)}
        checks["ll"]=compare_field(stats[0],stats[1],ll)
        checks["P"]=compare_field(stats[2],stats[3],P)
        records.append(dict(name=name,ok=bool(np.all(info==0)) and all(c["ok"] for c in checks.values()),checks=checks))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    machine=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip()
    if "Apple M3 Ultra" not in machine:
        raise SystemExit("K3 quality runs only in the M3 serial queue")
    if os.environ.get("MOJOLEARN_NUMERIC_MODE")!="fast":
        raise SystemExit("MOJOLEARN_NUMERIC_MODE=fast required")
    from mojolearn import _backend
    binding=_backend.binding("_mojolearn_arima","fast")
    if str(binding.arima_vendor())!="metal" or int(binding.arima_numeric_mode())!=0 or not bool(binding.arima_scalar_ll_enabled()):
        raise SystemExit("Wrong binding: expected opt-in FAST Metal K3")
    out=Path(args.output).expanduser()
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as stream: json.dump(dict(status="RUNNING",fixture=FIXTURE),stream)
    records=[]
    for n in (1,2,255,256,257,1391,1392,1393,4095,4096,4097,65535,65536):
        records.append(fixture(binding,n))
    checks=controls(binding)
    good=all(r["status"]=="PASS" for r in records) and all(c["ok"] for c in checks)
    report=dict(fixture=FIXTURE,status="PASS" if good else "HOLD",cases=records,controls=checks,
        source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        binding=str(binding.__file__),binding_sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
        baseline="actual existing batched_kalman_loop_kernel[1,False]",candidate="actual K3 kernels",
        initialization="common supplied scalar state, not device initializer certification",
        gradient_step=float(H),degradation_allowance=0,scored_timings=0,promotion_authorized=False)
    temp=out.with_suffix(out.suffix+".next")
    with temp.open("x") as stream: json.dump(report,stream,indent=2,allow_nan=False)
    os.replace(temp,out)
    print("ARIMA-K3-QUALITY "+json.dumps(dict(status=report["status"],groups=len(records),output=str(out),scored_timings=0)))
    return 0 if good else 1


if __name__=="__main__":
    raise SystemExit(main())
