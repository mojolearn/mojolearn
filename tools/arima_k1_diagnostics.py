#!/usr/bin/env python3
"""M3 reference-only fixed K1 model/conditioning diagnostics; no admission gate.

Original K1 report stays HOLD. Controls change model assumptions explicitly,
never the existing quality reference or production route. No scored timings.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import arima_assoc_scan_oracle as base
import arima_gaussian_scan_oracle as k1

# Six failed groups, including base plus relevant failed perturbations;
# four representative passing-math groups. None selected as an admission subset.
CASES = (
    ('grid-p0-q3', (None, 4)), ('grid-p1-q3', (None, 5)),
    ('grid-p3-q3', (None, 7)), ('ma-near-pair-n1393', (None, 0, 1, 2, 3, 4)),
    ('ma-repeated-n1393', (None, 0)), ('ma-order3-n1393', (None, 7)),
    ('grid-p0-q0', (None,)), ('grid-p1-q1', (None,)),
    ('ma-near-positive-n1393', (None, 0)), ('constant', (None, 2)),
)


def matrix_audit(x):
    x = np.asarray(x, np.float64)
    return dict(max_asymmetry=float(np.max(np.abs(x-np.swapaxes(x,-1,-2)))),
                min_symmetric_eigenvalue=float(np.min(np.linalg.eigvalsh((x+np.swapaxes(x,-1,-2))*.5))),
                max_abs=float(np.max(np.abs(x))))


def serial(model, y, repair):
    T,R,sigma,Q,c,P,a = (np.asarray(x,np.float64).copy() for x in model)
    pred,F = np.empty(len(y)),np.empty(len(y))
    stats=dict(negative_diagonal_steps=0, max_abs_diagonal_correction=0.,
               max_symmetry_correction=0., first_negative_diagonal=None)
    for t,obs in enumerate(y.astype(np.float64)):
        pred[t],F[t] = a[0],P[0,0]
        if F[t]<=0 or not np.isfinite(F[t]):
            raise FloatingPointError('invalid serial F at '+str(t))
        TP=T@P; gain=TP[:,0]/F[t]
        a=T@a+gain*(obs-pred[t])+c
        L=T.copy(); L[:,0]-=gain
        P=TP@L.T+Q
        sym=.5*(P+P.T)
        stats['max_symmetry_correction']=max(stats['max_symmetry_correction'],float(np.max(np.abs(sym-P))))
        diag=np.diag(sym)
        correction=float(np.max(np.abs(np.abs(diag)-diag)))
        stats['max_abs_diagonal_correction']=max(stats['max_abs_diagonal_correction'],correction)
        if np.any(diag<0):
            stats['negative_diagonal_steps']+=1
            if stats['first_negative_diagonal'] is None: stats['first_negative_diagonal']=t
        if repair:
            P=sym; P[np.diag_indices(len(P))]=np.abs(np.diag(P))
    return base.finish(pred,F,y,a,P,np.float64,tree=False),stats


def scan_audited(model,y):
    levels=[]
    original=k1.compose
    def compose(left,right):
        Ci,Jj=left[2],right[3]
        eye=np.broadcast_to(np.eye(Ci.shape[-1]),Ci.shape)
        M,N=eye+Ci@Jj,eye+Jj@Ci
        G,H=np.linalg.solve(M,eye),np.linalg.solve(N,eye)
        def residual(A,X):
            num=np.linalg.norm(A@X-eye,axis=(-2,-1))
            den=np.linalg.norm(A,axis=(-2,-1))*np.linalg.norm(X,axis=(-2,-1))+np.linalg.norm(eye,axis=(-2,-1))
            return float(np.max(num/den))
        out=original(left,right)
        levels.append(dict(level=len(levels), max_cond_G=float(np.max(np.linalg.cond(M))),
                           max_cond_H=float(np.max(np.linalg.cond(N))),
                           backward_residual_G=residual(M,G), backward_residual_H=residual(N,H),
                           C=matrix_audit(out[2]), J=matrix_audit(out[3])))
        return out
    k1.compose=compose
    try:
        result=k1.scan(model,y,np.float64)
    finally:
        k1.compose=original
    return result,levels


def control(model,y):
    result=dict(Q=matrix_audit(model[3]), P0=matrix_audit(model[5]))
    outputs={}
    for label,repair in (('serial_repaired',True),('serial_unrepaired',False)):
        try:
            outputs[label],result[label+'_repairs']=serial(model,y,repair)
        except (FloatingPointError,np.linalg.LinAlgError) as e:
            result[label+'_exception']=str(e)
    try:
        factors=k1.leaves(model,y,np.float64)
        result['leaf_C']=matrix_audit(factors[2])
        outputs['scan'],result['scan_levels']=scan_audited(model,y)
    except (FloatingPointError,np.linalg.LinAlgError) as e:
        result['scan_exception']=str(e)
    for a,b in (('scan','serial_repaired'),('scan','serial_unrepaired'),('serial_unrepaired','serial_repaired')):
        if a in outputs and b in outputs:
            errors=base.errors(outputs[a],outputs[b])
            for field,metric in errors.items():
                delta=np.abs(np.asarray(outputs[a][field])-np.asarray(outputs[b][field])).reshape(-1)
                exceeded=np.flatnonzero(delta>base.MATH_ATOL+base.MATH_RTOL*metric['ref_scale'])
                metric['first_flat_index_exceeding_original_math_bound']=int(exceeded[0]) if len(exceeded) else None
            result[a+'_versus_'+b]=errors
    return result


def json_safe(value):
    # Nonfinite conditioning is evidence, not an invalid JSON artifact.
    if isinstance(value,float) and not math.isfinite(value): return str(value)
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list): return [json_safe(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    brand=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    assert brand=='Apple M3 Ultra'
    output=Path(args.output).expanduser();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(dict(status='RUNNING'),stream)
    specs={s[0]:(i,s) for i,s in enumerate(base.fixtures())}
    records=[]
    with np.errstate(all='raise',under='ignore'):
        for name,parameters in CASES:
            i,spec=specs[name]
            _,ar,ma,n,scale,pattern,_=spec
            p,q=len(ar),len(ma);raw=base.raw_from_roots(ar,ma,scale)
            y=base.observations(base.state(raw,p,q),n,scale,pattern,20261004+i)
            for parameter in parameters:
                rp=raw.copy()
                if parameter is not None: rp[parameter]=np.float32(rp[parameter]+np.float32(base.H))
                rounded=[np.asarray(x,np.float64).copy() for x in base.state(rp,p,q)]
                symmetric=[x.copy() for x in rounded]; symmetric[3]=.5*(rounded[3]+rounded[3].T)
                exact=[x.copy() for x in rounded];exact[3]=exact[2]*np.outer(exact[1],exact[1])
                T=exact[0];rd=len(T)
                exact[5]=np.linalg.solve(np.eye(rd*rd)-np.kron(T,T),exact[3].reshape(-1)).reshape(rd,rd)
                exact[5]=.5*(exact[5]+exact[5].T)
                row=dict(name=name, parameter=parameter,
                         effective_step=0. if parameter is None else float(rp[parameter]-raw[parameter]),
                         input_sha256=hashlib.sha256(y.tobytes()).hexdigest(),controls={})
                for label,model in (('rounded_Q_shared_rounded_prior',rounded),
                                    ('symmetrized_Q_shared_rounded_prior',symmetric),
                                    ('intended_rank_one_Q_recomputed_prior64',exact)):
                    try: row['controls'][label]=control(model,y)
                    except (FloatingPointError,np.linalg.LinAlgError) as e: row['controls'][label]=dict(exception=str(e))
                records.append(row)
                print('K1-DIAGNOSTIC '+name+' parameter='+str(parameter),flush=True)
    report=dict(status='DIAGNOSTIC_ONLY_ORIGINAL_HOLD_UNCHANGED',source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),machine=brand,
                original_report='arima-k1-oracle-v1: HOLD math6 float32_48 gradient14',
                controls_change_model=True, actual_GPU=False, scored_timings=0,promotion_authorized=False,
                thresholds_changed=False,records=records)
    temp=output.with_suffix(output.suffix+'.next')
    with temp.open('x') as stream:json.dump(json_safe(report),stream,indent=2,allow_nan=False)
    temp.replace(output)


if __name__=='__main__': main()
