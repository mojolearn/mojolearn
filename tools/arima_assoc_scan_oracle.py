#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""M3 serial-queue REFERENCE verification, never a product implementation.

No timing, opponents, GPU imports, or production dispatch changes. NumPy
implements an independent serial state-space reference and the proposed
rank-one associative filter. Separate gates distinguish float64 algebra
from float32 numerical viability and finite-difference gradient viability.

The float32 serial arm is a NumPy EMULATION, NOT the measured main binary:
BLAS/FMA, Lyapunov initialization and transcendental rounding may differ.
A PASS permits further device implementation/review, never promotion. A
failure is evidence against this proposed arithmetic, not a main regression.

Run only as a serial M3 queue job, with a fresh --output JSON path. All
fixtures and thresholds are fixed below before execution. Report every
failure; never silently omit difficult/nonfinite cases or loosen thresholds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

# Reference verification must not acquire a machine-wide BLAS thread pool.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


FIXTURE = "arima-assoc-rankone-v1"
H = 2.0 ** -10
# Absolute+relative float64 algebra gate, per field's max reference value.
MATH_ATOL = 1e-9
MATH_RTOL = 1e-8
# No positive tolerance is added to float32 error or gradient comparisons.
F32_DEGRADATION_ALLOWANCE = 0.0


def jones(raw, is_ar):
    """Source transform's mathematical layout, NumPy float32 arithmetic.

    See arima/impl/timeSeries/jones_transform.mojo. This deliberately does
    not claim to emulate device tanhf/FMA bit-for-bit.
    """
    out = np.tanh(np.asarray(raw, np.float32) * np.float32(.5))
    sign = np.float32(-1 if is_ar else 1)
    for j in range(1, len(out)):
        old = out.copy()
        out[:j] = old[:j] + sign * (old[j] * old[:j][::-1])
    return np.clip(out, np.float32(-.9999), np.float32(.9999))


def inverse_jones(coefs, is_ar):
    out = np.asarray(coefs, np.float64).copy()
    sign = 1.0 if is_ar else -1.0
    for j in range(len(out) - 1, 0, -1):
        old = out.copy()
        out[:j] = (old[:j] + sign * old[j] * old[:j][::-1]) / (1 - old[j] ** 2)
    if not np.all(np.abs(out) < 1):
        raise ValueError("Fixture is outside Jones inverse domain")
    return (2 * np.arctanh(out)).astype(np.float32)


def state(raw, p, q):
    """Common rounded state inputs for all three filters.

    The initializer solves in float64 then rounds once to float32. Its
    accuracy is intentionally NOT being tested: the device candidate would
    retain main's initializer. Every filter receives identical P0/alpha0.
    """
    raw = np.asarray(raw, np.float32)
    ar, ma = jones(raw[1:1+p], True), jones(raw[1+p:1+p+q], False)
    rd = max(p, q + 1)
    T = np.zeros((rd, rd), np.float32)
    T[:p, 0] = ar
    if rd > 1:
        T[np.arange(rd-1), np.arange(1, rd)] = 1
    # Main's explicit rd=2, AR(2) singularity guard.
    if rd == 2 and p == 2 and abs(float(T[1, 0]) + 1) < .01:
        T[1, 0] = np.float32(-.99)
    R = np.zeros(rd, np.float32)
    R[0], R[1:q+1] = 1, ma
    sigma = np.maximum(raw[-1], np.float32(1e-6))
    Q = (R * sigma)[:, None] * R[None, :]
    c = np.zeros(rd, np.float32)
    c[0] = raw[0]
    td, qd = T.astype(np.float64), Q.astype(np.float64)
    P = np.linalg.solve(np.eye(rd*rd)-np.kron(td, td), qd.reshape(-1)).reshape(rd, rd)
    P = ((P + P.T) * .5).astype(np.float32)
    imt = np.eye(rd) - td
    if rd == 1 and abs(imt[0, 0]) < 1e-3:
        imt[0, 0] = np.copysign(1e-3, imt[0, 0])
    alpha = np.linalg.solve(imt, c.astype(np.float64)).astype(np.float32)
    return T, R, sigma, Q, c, P, alpha


def reduce_tree(values):
    values = np.asarray(values).copy()
    while len(values) > 1:
        pairs = len(values) // 2
        sums = values[:2*pairs:2] + values[1:2*pairs:2]
        values = np.concatenate((sums, values[-1:])) if len(values) % 2 else sums
    return values[0]


def finish(pred, F, y, alpha, P, dtype, tree):
    v = y.astype(dtype) - pred
    if not np.isfinite(F).all() or np.any(F <= 0):
        raise FloatingPointError("Nonpositive/nonfinite innovation variance")
    logs = np.log(F)
    squares = (v * v) / F
    if tree:
        sl, ss = reduce_tree(logs), reduce_tree(squares)
    else:
        sl, ss = dtype(0), dtype(0)
        for a, b in zip(logs, squares):
            sl, ss = dtype(sl + a), dtype(ss + b)
    nn = dtype(len(y))
    inner = dtype(dtype(ss / nn) + dtype(np.log(2*np.pi)))
    # Source divides v²/F by n before its final multiply-add. A float64
    # intermediate approximates the final float32 FMA; this remains emulation.
    total = dtype(float(nn) * float(inner) + float(sl))
    ll = dtype(-.5) * total
    out = dict(pred=pred, innovation=v, variance=F,
               alpha=alpha, covariance=P, loglike=np.asarray(ll))
    if not all(np.isfinite(a).all() for a in out.values()):
        raise FloatingPointError("Nonfinite filter output")
    return out


def serial(inputs, y, dtype):
    """Main recurrence layout, including per-step symmetrize/abs diagonal.

    Float64 is the independent time-varying reference. Float32 uses NumPy
    matrix arithmetic and is labelled an emulation throughout the report.
    """
    T, R, sigma, Q, c, P, a = (np.asarray(x, dtype=dtype).copy() for x in inputs)
    pred, F = np.empty(len(y), dtype), np.empty(len(y), dtype)
    for t, obs in enumerate(y.astype(dtype)):
        pred[t], F[t] = a[0], P[0, 0]
        if not F[t] > 0 or not np.isfinite(F[t]):
            raise FloatingPointError("Serial invalid variance at %d" % t)
        TP = T @ P
        gain = TP[:, 0] / F[t]
        a = T @ a + gain * (obs - pred[t]) + c
        L = T.copy()
        L[:, 0] -= gain
        P = TP @ L.T + Q
        P = dtype(.5) * (P + P.T)
        P[np.diag_indices(len(P))] = np.abs(np.diag(P))
    return finish(pred, F, y, a, P, dtype, tree=False)


def compose(left, right):
    """Chronological left then right, batched over independent prefixes."""
    Ai, bi, Ji, ei = left
    Aj, bj, Jj, ej = right
    AiT = np.swapaxes(Ai, -1, -2)
    A = Aj @ Ai
    b = (Aj @ bi[..., None])[..., 0] + bj
    J = Ji + AiT @ Jj @ Ai
    e = ei + (AiT @ (ej - (Jj @ bi[..., None])[..., 0])[..., None])[..., 0]
    return A, b, J, e


def scan(inputs, y, dtype):
    """Proposed Hillis-Steele scan, with complete time-varying covariance.

    It exploits exact model rank one, not a converged-gain approximation.
    Initial P is singular after observing y0: only I+C0 J is solved.
    """
    T, R, sigma, Q, c, P, a = (np.asarray(x, dtype=dtype).copy() for x in inputs)
    n, rd = len(y), len(a)
    y = y.astype(dtype)
    F0 = P[0, 0]
    if not F0 > 0 or not np.isfinite(F0):
        raise FloatingPointError("Scan invalid initial variance")
    k0 = P[:, 0] / F0
    m0 = a + k0 * (y[0] - a[0])
    C0 = P - np.outer(k0, P[0, :])
    C0 = dtype(.5) * (C0 + C0.T)
    # First row/column are mathematically zero for a noiseless observation.
    # Do not invert C0, perturb it with jitter, or apply abs to its diagonal.
    C0[0, :], C0[:, 0] = 0, 0
    u = T[0, :]
    Aleaf = T - np.outer(R, u)
    Jleaf = np.outer(u, u) / sigma
    count = n - 1
    A = np.broadcast_to(Aleaf, (count, rd, rd)).copy()
    b = c[None, :] + (y[1:] - c[0])[:, None] * R[None, :]
    J = np.broadcast_to(Jleaf, (count, rd, rd)).copy()
    e = (y[1:] - c[0])[:, None] * u[None, :] / sigma
    step = 1
    while step < count:
        # Both operands are from the previous level, never overwritten reads.
        merged = compose(tuple(x[:-step] for x in (A,b,J,e)),
                         tuple(x[step:] for x in (A,b,J,e)))
        A,b,J,e = tuple(np.concatenate((x[:step], z))
                         for x,z in zip((A,b,J,e), merged))
        step *= 2
    W = np.linalg.solve(np.eye(rd, dtype=dtype)[None,:,:] + C0[None,:,:] @ J,
                        np.broadcast_to(C0, (count,rd,rd)))
    m_initial = m0[None,:] + (W @ (e - (J @ m0[:,None])[...,0])[...,None])[...,0]
    m = (A @ m_initial[...,None])[...,0] + b
    C = A @ W @ np.swapaxes(A,-1,-2)
    C = dtype(.5) * (C + np.swapaxes(C,-1,-2))
    # Every observation's prior is recovered from the preceding posterior.
    post_m = np.concatenate((m0[None,:], m))
    post_C = np.concatenate((C0[None,:,:], C))
    priors = (T[None,:,:] @ post_m[...,None])[...,0] + c
    prior_P = T[None,:,:] @ post_C @ T.T[None,:,:] + Q
    pred = np.concatenate((a[:1], priors[:-1,0]))
    F = np.concatenate((np.asarray([F0], dtype=dtype), prior_P[:-1,0,0]))
    return finish(pred, F, y, priors[-1], prior_P[-1], dtype, tree=True)


def errors(got, ref):
    result = {}
    for name, value in ref.items():
        ref64, got64 = np.asarray(value, np.float64), np.asarray(got[name], np.float64)
        result[name] = dict(max_abs=float(np.max(np.abs(got64-ref64))),
                            ref_scale=float(np.max(np.abs(ref64))))
    return result


def raw_from_roots(ar_roots, ma_roots, scale):
    ar = -np.poly(ar_roots)[1:] if ar_roots else np.empty(0)
    ma = np.poly(ma_roots)[1:] if ma_roots else np.empty(0)
    return np.concatenate(([.08*scale], inverse_jones(ar,True),
                           inverse_jones(ma,False), [scale*scale])).astype(np.float32)


def fixtures():
    # Full board order grid, and independent near-unit/repeated-root stresses.
    ar_grid = ((), (.65,), (.49,.49), (.6,.2,-.1))
    ma_grid = ((), (-.4,), (.49,.49), (.7,-.3,.2))
    records = []
    for p, ar in enumerate(ar_grid):
        for q, ma in enumerate(ma_grid):
            records.append(("grid-p%d-q%d"%(p,q), ar,ma,1392,1.0,"model",True))
    stress = [
        ("ar-near-positive",(.985,),(.4,)),
        ("ar-near-negative",(-.985,),(-.4,)),
        ("ma-near-positive",(.55,),(.995,)),
        ("ma-near-negative",(.55,),(-.995,)),
        ("ma-near-pair",(.6,),(.985,-.985)),
        ("ma-repeated",(.49,.49),(.49,.49)),
        ("ma-order3",(.6,.2,-.1),(.99,-.6,.5)),
    ]
    for i,(name,ar,ma) in enumerate(stress):
        for length in (127,255,256,257,1391,1393,1401):
            scale = (1e-3,1.0,1e3)[(length+i)%3]
            records.append((name+"-n%d"%length,ar,ma,length,scale,"model",length==1393))
    for pattern in ("constant","impulse","alternating"):
        records.append((pattern, (.985,), (.995,), 1393, 1e3, pattern, True))
    return records


def observations(inputs, n, scale, pattern, seed):
    if pattern == "constant":
        return np.full(n,3*scale,np.float32)
    if pattern == "impulse":
        y = np.zeros(n,np.float32); y[n//2] = 100*scale
        return y
    if pattern == "alternating":
        return (scale*np.where(np.arange(n)%2,1,-1)).astype(np.float32)
    T,R,sigma,Q,c,P,a = (np.asarray(x,np.float64).copy() for x in inputs)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n+512)*np.sqrt(sigma)
    out = np.empty(n+512)
    for i,eps in enumerate(noise):
        a = T@a+c+R*eps
        out[i] = a[0]
    return out[512:].astype(np.float32)


def evaluate(raw, p, q, y):
    model = state(raw,p,q)
    reference = serial(model,y,np.float64)
    # Pure algebra must use truly rank-one Q in float64. The device-style
    # elementwise-rounded Q32 is not exactly rank one when lifted to float64.
    # Keep that difference in the FLOAT32 viability gate, not the algebra gate.
    exact_model = [np.asarray(x,np.float64).copy() for x in model]
    exact_model[3] = (exact_model[1]*exact_model[2])[:,None] * exact_model[1][None,:]
    math_reference = serial(exact_model,y,np.float64)
    outputs,exceptions = {},{}
    for key,fn,args in (("math_scan",scan,(exact_model,y,np.float64)),
                        ("baseline",serial,(model,y,np.float32)),
                        ("candidate",scan,(model,y,np.float32))):
        try:
            outputs[key] = fn(*args)
        except (FloatingPointError,np.linalg.LinAlgError) as exc:
            exceptions[key] = type(exc).__name__+": "+str(exc)
    er64 = errors(outputs["math_scan"],math_reference) if "math_scan" in outputs else None
    erA = errors(outputs["baseline"],reference) if "baseline" in outputs else None
    erB = errors(outputs["candidate"],reference) if "candidate" in outputs else None
    math_ok = er64 is not None and all(v["max_abs"] <= MATH_ATOL+MATH_RTOL*v["ref_scale"] for v in er64.values())
    f32_ok = erA is not None and erB is not None and all(
        erB[k]["max_abs"] <= erA[k]["max_abs"]+F32_DEGRADATION_ALLOWANCE for k in erA)
    # Match production's objective normalization; numerator's reductions differ.
    values = dict(reference=-float(reference["loglike"])/(len(y)-1),
                  math_reference=-float(math_reference["loglike"])/(len(y)-1))
    for key in ("math_scan","baseline","candidate"):
        values[key] = (None if key not in outputs else
            (-float(outputs[key]["loglike"])/(len(y)-1) if key=="math_scan" else
             float(np.float32(-outputs[key]["loglike"]/np.float32(len(y)-1)))))
    return dict(math_ok=math_ok,float32_ok=f32_ok,errors64=er64,
                baseline_errors=erA,candidate_errors=erB,objectives=values,exceptions=exceptions)


def check_fixture(spec, index):
    name,ar,ma,n,scale,pattern,check_gradient = spec
    p,q = len(ar),len(ma)
    raw = raw_from_roots(ar,ma,scale)
    y = observations(state(raw,p,q),n,scale,pattern,20261004+index)
    record = dict(name=name,p=p,q=q,rd=max(p,q+1),n=n,scale=scale,
                  pattern=pattern,gradient_checked=check_gradient,
                  raw=raw.tolist(),input_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    try:
        base = evaluate(raw,p,q,y)
        record["filter"] = base
        record["math_ok"],record["float32_ok"] = base["math_ok"],base["float32_ok"]
        record["gradient_ok"] = True
        if check_gradient:
            gradients = {k:[] for k in base["objectives"]}
            math_bounds = []
            record["perturbations"] = []
            for j in range(len(raw)):
                perturbed = raw.copy()
                perturbed[j] = np.float32(perturbed[j]+np.float32(H))
                result = evaluate(perturbed,p,q,y)
                record["math_ok"] &= result["math_ok"]
                record["float32_ok"] &= result["float32_ok"]
                record["perturbations"].append(dict(parameter=j,
                    effective_step=float(perturbed[j]-raw[j]),result=result))
                for k,f in result["objectives"].items():
                    if f is None or base["objectives"][k] is None:
                        gradients[k].append(None)
                        continue
                    delta = f-base["objectives"][k]
                    # Production gradient subtraction is also float32.
                    gradients[k].append(float(np.float32(np.float32(f)-np.float32(base["objectives"][k]))/np.float32(H))
                                        if k in ("baseline","candidate") else delta/H)
                f0,fp = base["objectives"]["math_reference"],result["objectives"]["math_reference"]
                math_bounds.append((2*MATH_ATOL+MATH_RTOL*(abs(f0)+abs(fp)))/H)
            gr = np.asarray(gradients["reference"])
            numeric_complete = all(v is not None for k in ("baseline","candidate") for v in gradients[k])
            ea = np.abs(np.asarray(gradients["baseline"])-gr) if numeric_complete else None
            eb = np.abs(np.asarray(gradients["candidate"])-gr) if numeric_complete else None
            e64 = (np.abs(np.asarray(gradients["math_scan"])-np.asarray(gradients["math_reference"]))
                   if all(v is not None for v in gradients["math_scan"]) else None)
            # Float64 gradient algebra tolerance follows the stated objective
            # comparison bound divided by the fixed finite-difference step.
            record["math_ok"] &= e64 is not None and bool(np.all(e64 <= math_bounds))
            record["gradient_ok"] = numeric_complete and bool(np.all(eb <= ea+F32_DEGRADATION_ALLOWANCE))
            record["gradients"] = dict(values=gradients,baseline_abs_error=None if ea is None else ea.tolist(),
                candidate_abs_error=None if eb is None else eb.tolist(),
                math_abs_error=None if e64 is None else e64.tolist(),math_bound=math_bounds)
    except Exception as exc:
        record.update(math_ok=False,float32_ok=False,gradient_ok=False,
                      exception=type(exc).__name__+": "+str(exc))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    brand = subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip()
    if "Apple M3 Ultra" not in brand:
        raise SystemExit("Reference job requires the authorized M3 Ultra serial queue")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True,exist_ok=True)
    # Claim before any work, so partial jobs cannot be silently overwritten.
    with output.open("x") as stream:
        json.dump(dict(status="RUNNING",fixture=FIXTURE),stream)
    source = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    report = dict(fixture=FIXTURE,source=source,host=platform.node(),machine=brand,
        numpy=np.__version__,python=sys.version,role="REFERENCE_ONLY_NOT_PRODUCT",
        baseline="numpy_float32_emulation_NOT_main_binary",
        math_atol=MATH_ATOL,math_rtol=MATH_RTOL,
        float32_degradation_allowance=F32_DEGRADATION_ALLOWANCE,fd_step=H,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=[])
    with np.errstate(all="raise",under="ignore"):
        for index,spec in enumerate(fixtures()):
            report["cases"].append(check_fixture(spec,index))
    cases = report["cases"]
    counts = {key:sum(not c[key] for c in cases) for key in ("math_ok","float32_ok","gradient_ok")}
    report.update(failures=counts,status="PASS" if not any(counts.values()) else "HOLD",
                  scored_timings=0,promotion_authorized=False)
    temporary = output.with_suffix(output.suffix+".next")
    with temporary.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    os.replace(temporary,output)
    print("ARIMA-ASSOC-ORACLE "+json.dumps(dict(status=report["status"],cases=len(cases),
          failures=counts,output=str(output),source=source,scored_timings=0),sort_keys=True))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
