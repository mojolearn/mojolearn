#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""M3 reference-only K1 Gaussian scan with FULL conditional covariance.

Fresh hypothesis after arima-assoc-oracle-v1 HOLD. This retains the same
rounded Q and conditional C rather than assuming rank-one C=0. It checks
the K1 source equations at 9ab2d3d3fb770498ef025db08f595a0149792bb7,
experiments/apple_fast_path/kalman/gaussian_scan.metal, against the same
rounded-model serial64 reference. NumPy solves/matmul are NOT a shader
bit-replay, main binary, actual fit, or forecast-quality certificate.

No timing, builds, GPU calls, opponent execution, or product CPU dispatch.
Uses the entire original v1 corpus and fixed thresholds without deleting
hard cases. K2 and K3 are separate candidates; this receipt covers neither.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import arima_assoc_scan_oracle as base
import numpy as np


FIXTURE = "arima-gaussian-k1-v1"
METAL_SOURCE = "9ab2d3d3fb770498ef025db08f595a0149792bb7"


def leaves(inputs,y,dtype):
    T,R,sigma,Q,c,P,a = (np.asarray(x,dtype=dtype).copy() for x in inputs)
    n,rd = len(y),len(a)
    cov = np.broadcast_to(Q,(n,rd,rd)).copy()
    cov[0] = P
    transition = np.broadcast_to(T,(n,rd,rd)).copy()
    transition[0] = 0
    prior = np.broadcast_to(c,(n,rd)).copy()
    prior[0] = a
    # K1 source uses pz*pz^T: preserve that exact symmetry convention for
    # this first audit; do not silently symmetrize rounded Q or set C=0.
    pz = cov[:,:,0]
    s = cov[:,0,0]
    if np.any(s<=0) or not np.isfinite(s).all():
        raise FloatingPointError("K1 leaf variance nonpositive/nonfinite")
    residual = y.astype(dtype)-prior[:,0]
    gain = pz/s[:,None]
    ht = transition[:,0,:]
    A = transition-gain[:,:,None]*ht[:,None,:]
    b = prior+gain*residual[:,None]
    C = cov-gain[:,:,None]*pz[:,None,:]
    J = ht[:,:,None]*ht[:,None,:]/s[:,None,None]
    eta = ht*residual[:,None]/s[:,None]
    return A,b,C,J,eta


def compose(left,right):
    Ai,bi,Ci,Ji,ei = left
    Aj,bj,Cj,Jj,ej = right
    rd = Ai.shape[-1]
    identity = np.broadcast_to(np.eye(rd,dtype=Ai.dtype),Ai.shape)
    # K1 uses two independent pivoted inverses, deliberately not assuming
    # rounded Ci/Jj symmetry. NumPy solve is the independent reference for
    # the same matrices; source's 1e-20 pivot heuristic is not certified.
    G = np.linalg.solve(identity+Ci@Jj,identity)
    H = np.linalg.solve(identity+Jj@Ci,identity)
    rg = Aj@G
    AiT = np.swapaxes(Ai,-1,-2)
    A = rg@Ai
    b = (rg@(bi+(Ci@ej[...,None])[...,0])[...,None])[...,0]+bj
    C = rg@Ci@np.swapaxes(Aj,-1,-2)+Cj
    eta = ei+(AiT@H@(ej-(Jj@bi[...,None])[...,0])[...,None])[...,0]
    J = Ji+AiT@H@Jj@Ai
    return A,b,C,J,eta


def scan(inputs,y,dtype):
    T,R,sigma,Q,c,P,a = (np.asarray(x,dtype=dtype).copy() for x in inputs)
    factors = leaves(inputs,y,dtype)
    step=1
    while step<len(y):
        merged = compose(tuple(x[:-step] for x in factors),tuple(x[step:] for x in factors))
        factors = tuple(np.concatenate((old[:step],new)) for old,new in zip(factors,merged))
        step*=2
    A,b,C,J,eta = factors
    # The t0 leaf incorporates the actual prior, so inclusive prefixes'
    # b/C directly give filtered state. Recover next predictive convention.
    prior_a = (T[None,:,:]@b[...,None])[...,0]+c
    prior_P = T[None,:,:]@C@T.T[None,:,:]+Q
    pred = np.concatenate((a[:1],prior_a[:-1,0]))
    F = np.concatenate((P[0,:1],prior_P[:-1,0,0]))
    # K1 source retains ascending likelihood reduction. Keep that control
    # separate from the parallel reduction that any promotion must add.
    return base.finish(pred,F,y,prior_a[-1],prior_P[-1],dtype,tree=False)


def evaluate(raw,p,q,y):
    model=base.state(raw,p,q)
    reference=base.serial(model,y,np.float64)
    outputs,exceptions={},{}
    for name,fn,dtype in (("math_scan",scan,np.float64),("baseline",base.serial,np.float32),
                          ("candidate",scan,np.float32)):
        try:
            outputs[name]=fn(model,y,dtype)
        except (FloatingPointError,np.linalg.LinAlgError) as exc:
            exceptions[name]=type(exc).__name__+": "+str(exc)
    er64=base.errors(outputs["math_scan"],reference) if "math_scan" in outputs else None
    erA=base.errors(outputs["baseline"],reference) if "baseline" in outputs else None
    erB=base.errors(outputs["candidate"],reference) if "candidate" in outputs else None
    math_ok=er64 is not None and all(v["max_abs"]<=base.MATH_ATOL+base.MATH_RTOL*v["ref_scale"] for v in er64.values())
    float_ok=erA is not None and erB is not None and all(
        erB[k]["max_abs"]<=erA[k]["max_abs"]+base.F32_DEGRADATION_ALLOWANCE for k in erA)
    f=-float(reference["loglike"])/(len(y)-1)
    values=dict(reference=f,math_reference=f)
    for name in ("math_scan","baseline","candidate"):
        values[name]=(None if name not in outputs else
            (-float(outputs[name]["loglike"])/(len(y)-1) if name=="math_scan" else
             float(np.float32(-outputs[name]["loglike"]/np.float32(len(y)-1)))))
    return dict(math_ok=math_ok,float32_ok=float_ok,errors64=er64,
        baseline_errors=erA,candidate_errors=erB,objectives=values,exceptions=exceptions)


def boundary_checks():
    """Zero-pass/odd scan lengths beyond the fixed v1 regression corpus.

    The n=1 case has no optimizer-gradient normalization (n-1=0); compare
    kernel outputs directly instead of manufacturing an objective for it.
    """
    out=[]
    raw=base.raw_from_roots((.65,),(.4,),1.)
    model=base.state(raw,1,1)
    for n in (1,2,3,7,8,9,15,16,17,31,32,33):
        y=base.observations(model,n,1.,"model",20261010+n)
        record=dict(n=n)
        try:
            ref=base.serial(model,y,np.float64)
            er=base.errors(scan(model,y,np.float64),ref)
            record.update(errors64=er,math_ok=all(v["max_abs"]<=base.MATH_ATOL+base.MATH_RTOL*v["ref_scale"] for v in er.values()))
        except Exception as exc:
            record.update(math_ok=False,exception=type(exc).__name__+": "+str(exc))
        out.append(record)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    brand=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip()
    if "Apple M3 Ultra" not in brand:
        raise SystemExit("Reference verification requires the M3 serial queue")
    output=Path(args.output).expanduser()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(dict(status="RUNNING",fixture=FIXTURE),stream)
    # Reuse the unchanged fixture, perturbation and comparison logic. This
    # module runs in its own process: injection only replaces its evaluator,
    # never edits/replays the original v1 harness or its saved report.
    base.evaluate=evaluate
    with np.errstate(all="raise",under="ignore"):
        cases=[base.check_fixture(spec,i) for i,spec in enumerate(base.fixtures())]
        boundaries=boundary_checks()
    counts={key:sum(not c[key] for c in cases) for key in ("math_ok","float32_ok","gradient_ok")}
    counts["boundary_math_ok"]=sum(not c["math_ok"] for c in boundaries)
    source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    report=dict(fixture=FIXTURE,source=source,metal_source=METAL_SOURCE,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_harness_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        machine=brand,host=platform.node(),numpy=np.__version__,python=sys.version,
        status="PASS" if not any(counts.values()) else "HOLD",failures=counts,cases=cases,boundaries=boundaries,
        model="same rounded Q32/P0/alpha0, full conditional C retained; no rank-one projection",
        baseline="NumPy float32 serial emulation, NOT actual main binary",
        candidate="K1 full Gaussian equations; NumPy solve/matmul NOT shader bit replay",
        likelihood_reduction="ascending reference control; serial GPU reduction NOT eligible for promotion",
        math_atol=base.MATH_ATOL,math_rtol=base.MATH_RTOL,
        float32_degradation_allowance=base.F32_DEGRADATION_ALLOWANCE,fd_step=base.H,
        scored_timings=0,promotion_authorized=False)
    temp=output.with_suffix(output.suffix+".next")
    with temp.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    os.replace(temp,output)
    print("ARIMA-K1-ORACLE "+json.dumps(dict(status=report["status"],cases=len(cases),
        failures=counts,output=str(output),source=source,scored_timings=0),sort_keys=True))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
