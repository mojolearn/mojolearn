#!/usr/bin/env python3
"""Actual compensated K3 GPU kernel gate; no timers, no product admission.

Full hash-pinned reference9723 corpus versus saved actual main GPU errors.
No new allowance. LL low words never leave GPU before gradient computation.
Run only in the M3 serial queue with an independently verified M2 artifact.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import arima_k3_df_reference as proposal
import arima_scalar_quality as fixture

FIXTURE = 'arima-k3-df-gpu-v1'


def probe(binding, states, y, intercept, gradient=False):
    states = np.ascontiguousarray(states, np.float32)
    y = np.ascontiguousarray(y, np.float32)
    batch, n = y.shape
    assert states.shape == (5, batch)
    stages = np.full((3, batch, n), np.nan, np.float32)
    stats = np.full((4, batch), np.nan, np.float32)
    info = np.full(batch, -777, np.int32)
    gradients = np.full((batch // 4, 3), np.nan, np.float32)
    reached = binding._arima_k3_df_probe(y.ctypes.data, states.ctypes.data,
        stages.ctypes.data, stats.ctypes.data, info.ctypes.data,
        gradients.ctypes.data, [batch, n, intercept, int(gradient)])
    assert int(reached) == 3, 'Wrong/private K3 DF ABI'
    return stages, stats, info, gradients


def errors(value, target):
    value, target = np.asarray(value, np.float64), np.asarray(target, np.float64)
    if not np.isfinite(value).all():
        raise AssertionError('Nonfinite GPU output: infrastructure/numerical failure, never a PASS')
    error = np.abs(value - target)
    if error.ndim > 1:
        error = error.max(axis=1)
    return error


def evaluate(binding, saved):
    n = saved['n']
    states, y, labels = proposal.inputs(n)
    assert labels == saved['labels']
    digest = hashlib.sha256(y.tobytes() + states.tobytes()).hexdigest()
    assert digest == saved['input_sha256'], 'Corpus drift'
    ref, llref, finalP = fixture.reference(states, y, 1)
    actual, stats, info, gradients = probe(binding, states, y, 1, gradient=n > 1)
    assert np.all(info == 0) and np.isfinite(stats).all()
    checks = {}
    outputs = (*actual, stats[0], stats[1])
    targets = (*ref, llref, finalP)
    for name, value, target in zip(('prediction', 'innovation', 'variance', 'loglike', 'final_covariance'), outputs, targets):
        checks[name] = proposal.compare(errors(value, target), saved['checks'][name]['baseline_error'])
    gradient = None
    if n > 1:
        assert np.isfinite(gradients).all()
        fr = (-llref / (n - 1)).reshape(-1, 4)
        gr = (fr[:, 1:] - fr[:, :1]) / float(fixture.H)
        assert np.array_equal(gr, np.asarray(saved['gradient']['reference'])), 'Oracle drift'
        gradient = proposal.compare(np.abs(gradients.astype(np.float64) - gr), saved['gradient']['baseline_error'])
        gradient.update(candidate=gradients.tolist(), reference=gr.tolist(),
                        source='actual GPU retained-low-word k3_df_gradient_kernel')
    passed = all(c['ok'] for c in checks.values()) and (gradient is None or gradient['ok'])
    return dict(n=n, status='PASS' if passed else 'HOLD', checks=checks, gradient=gradient,
                input_sha256=digest, labels=labels, info=info.tolist(),
                stage_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), stats=stats.tolist())


def controls(binding, saved):
    assert [c['name'] for c in saved] == ['nonpositive-variance', 'constant', 'impulse', 'alternating']
    records = []
    states = np.array([[.5,.5,.5],[1.,0.,1.],[0.,1.,1.],[0.,0.,0.],[0.,0.,0.]], np.float32)
    _, _, info, _ = probe(binding, states, np.zeros((3,257), np.float32), 0)
    records.append(dict(name='nonpositive-variance', ok=bool(np.array_equal(info,[1,2,0])), info=info.tolist()))
    for original in saved[1:]:
        name, n = original['name'], 1393
        states = fixture.model(np.array([2*np.arctanh(.985),0.,1.], np.float32))[:,None]
        y = np.zeros((1,n), np.float32)
        if name == 'constant': y.fill(1e3)
        elif name == 'impulse': y[0,n//2] = 1e3
        elif name == 'alternating': y[0] = np.where(np.arange(n)%2,1e3,-1e3)
        else: raise AssertionError('Unexpected control')
        ref, llref, finalP = fixture.reference(states, y, 0)
        actual, stats, info, _ = probe(binding, states, y, 0)
        assert np.isfinite(stats).all()
        checks = {}
        for name_, value, target in zip(('0','1','2','ll','P'), (*actual,stats[0],stats[1]), (*ref,llref,finalP)):
            checks[name_] = proposal.compare(errors(value,target), original['checks'][name_]['baseline_error'])
        records.append(dict(name=name, ok=bool(np.all(info==0)) and all(c['ok'] for c in checks.values()), checks=checks))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-report', required=True)
    p.add_argument('--binding', required=True)
    p.add_argument('--binding-sha256', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'], text=True).strip() == 'Apple M3 Ultra'
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    source = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    assert source == args.source and len(source) == 40, 'Wrong pinned source'
    binary = Path(args.binding).expanduser().resolve(strict=True)
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == args.binding_sha256
    data = Path(args.baseline_report).expanduser().read_bytes()
    assert hashlib.sha256(data).hexdigest() == proposal.REPORT_SHA
    saved = json.loads(data)
    assert saved['fixture'] == fixture.FIXTURE and saved['degradation_allowance'] == 0
    assert tuple(c['n'] for c in saved['cases']) == proposal.LENGTHS
    spec = importlib.util.spec_from_file_location('_mojolearn_arima', binary)
    assert spec and spec.loader
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    assert str(binding.arima_vendor()) == 'metal' and int(binding.arima_numeric_mode()) == 0
    assert hasattr(binding, '_arima_k3_df_probe'), 'Missing opt-in GPU probe export'
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(status='RUNNING', fixture=FIXTURE, source=source), stream)
    records = []
    with np.errstate(all='raise', under='ignore'):
        for saved_case in saved['cases']:
            result = evaluate(binding, saved_case)
            records.append(result)
            print('K3-DF-GPU n='+str(result['n'])+' '+result['status'], flush=True)
        checks = controls(binding, saved['controls'])
    good = all(r['status']=='PASS' for r in records) and all(c['ok'] for c in checks)
    report = dict(status='PASS' if good else 'HOLD', fixture=FIXTURE, cases=records, controls=checks,
        source=source, binding=str(binary), binding_sha256=args.binding_sha256,
        baseline_report_sha256=proposal.REPORT_SHA, degradation_allowance=0,
        gradient_step=float(fixture.H), scored_timings=0, promotion_authorized=False,
        baseline='saved actual-main GPU errors; no replay of completed main arms',
        candidate='actual private compensated K3 GPU kernels and retained-low-word gradient',
        initialization='exact hash-verified supplied-state fixture; device Jones/initializer and optimizer NOT certified',
        product_dispatch=False)
    temporary = output.with_suffix(output.suffix+'.next')
    with temporary.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    temporary.replace(output)
    print('K3-DF-GPU '+report['status']+' output='+str(output), flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
