#!/usr/bin/env python3
"""Untimed public GramCD/Prophet source-column comparison with per-output raw bits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def fixtures(gate):
    import numpy as np
    if gate == 'gram-cd':
        n, d = 4096, 16
        i = np.arange(n, dtype=np.uint32)
        x = np.empty((n, d), dtype=np.float32, order='F')
        for j in range(d):
            bits = i & np.uint32(j + 1)
            parity = np.zeros(n, dtype=np.uint32)
            for b in range(12):
                parity ^= (bits >> np.uint32(b)) & np.uint32(1)
            x[:, j] = 1 - 2 * parity.astype(np.int32)
        beta = np.zeros(d, dtype=np.float32)
        beta[[0, 1, 5]] = [0.8, -0.6, 0.2]
        return {'x': x, 'y': x @ beta}
    n = 200
    i = np.arange(n)
    holidays = (np.arange(n + 14) % 50 == 7).astype(np.float32)
    y = (10 + np.where(i < 100, 0.05 * i, 5 - 0.02 * (i - 100))
         + np.sin(2 * np.pi * i / 7) + 3 * holidays[:n]
         + 0.1 * np.random.default_rng(0).standard_normal(n)).astype(np.float32)
    return {'t': np.arange(n, dtype=np.float64) + 18000.0, 'y': y,
            'future': np.arange(n, n + 14, dtype=np.float64) + 18000.0,
            'holidays': holidays}


def child(root, out, vendor, gate):
    sys.path.insert(0, str(root / 'python'))
    from identical_wave_worker import source_provenance
    import numpy as np
    import mojolearn as ml
    inputs = fixtures(gate)
    arrays = {}
    def save(name, value):
        arrays[name] = np.array(value, copy=True, order='C')
    def learned(prefix, estimator):
        # Include all public fitted numerical state, not just selected scores.
        for name, value in vars(estimator).items():
            if name.endswith('_') and not name.startswith('_'):
                array = np.asarray(value)
                if array.dtype.kind in 'fiu':
                    save(prefix + '-' + name, array)
    if gate == 'gram-cd':
        assert not os.environ.get('MOJOLEARN_IDENTITY_TRACE')
        for ratio in (1.0, 0.5):
            for intercept in (False, True):
                prefix = f'ElasticNet-ratio{ratio}-intercept{intercept}'
                estimator = ml.ElasticNet(alpha=0.05, l1_ratio=ratio,
                    fit_intercept=intercept, max_iter=1000, tol=1e-7).fit(inputs['x'], inputs['y'])
                learned(prefix, estimator)
                save(prefix + '-prediction', estimator.predict(inputs['x'][:257]))
    else:
        for mode in ('additive', 'multiplicative'):
            for batch in (False, True):
                prefix = f'Prophet-{mode}-batch{batch}'
                y = inputs['y']
                if batch:
                    y = np.stack([y, y * np.float32(2.0)])
                estimator = ml.ProphetForecaster(seasonality_mode=mode).fit(
                    inputs['t'], y, holidays=inputs['holidays'][:200, None])
                learned(prefix, estimator)
                save(prefix + '-fit', estimator.predict(inputs['t'], holidays=inputs['holidays'][:200, None]))
                save(prefix + '-fit-trend', estimator.trend_)
                save(prefix + '-forecast', estimator.predict(inputs['future'], holidays=inputs['holidays'][200:, None]))
                save(prefix + '-forecast-trend', estimator.trend_)
    input_info = {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
        'sha256': hashlib.sha256(v.tobytes(order='C')).hexdigest()} for k, v in inputs.items()}
    (out / (vendor + '-inputs.json')).write_text(json.dumps(input_info, indent=2))
    np.savez(out / (vendor + '.npz'), **arrays)
    (out / (vendor + '-provenance.json')).write_text(json.dumps(source_provenance(root, vendor), indent=2))
    print('FAMILY_IDENTITY_CHILD', gate, vendor, 'outputs=' + str(len(arrays)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vendor', choices=('cuda', 'hip', 'cpu'), required=True)
    p.add_argument('--child', action='store_true')
    p.add_argument('--gate', choices=('gram-cd', 'prophet'), required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        p.error('execute only on the authorized Linux boxes')
    root = a.source.resolve()
    if a.child:
        child(root, a.out, a.vendor, a.gate)
        return 0
    if a.vendor == 'cpu':
        p.error('parent requires cuda or hip')
    a.out.mkdir(parents=True, exist_ok=False)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    report = {'source_sha': sha, 'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gate': a.gate, 'purpose': 'untimed raw-bit diagnostics', 'columns': {}, 'outputs': []}
    for vendor in (a.vendor, 'cpu'):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MODULAR_MOJO_'))}
        env.update(MOJOLEARN_VENDOR=vendor, MOJOLEARN_NUMERIC_MODE='identical',
                   PYTHONPATH=str(root / 'python'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                   LD_LIBRARY_PATH=str(root / 'python/mojolearn/.libs') + ':' + env.get('LD_LIBRARY_PATH', ''))
        with (a.out / (vendor + '.log')).open('w') as log:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child',
                '--source', str(root), '--out', str(a.out), '--vendor', vendor, '--gate', a.gate],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        report['columns'][vendor] = {'rc': proc.returncode}
        if proc.returncode:
            report['status'] = 'ERROR'
            (a.out / 'comparison.json').write_text(json.dumps(report, indent=2))
            return 1
        report['columns'][vendor]['provenance'] = json.loads((a.out / (vendor + '-provenance.json')).read_text())
    report['inputs'] = json.loads((a.out / (a.vendor + '-inputs.json')).read_text())
    assert report['inputs'] == json.loads((a.out / 'cpu-inputs.json').read_text()), 'input bits differ'
    import numpy as np
    with np.load(a.out / (a.vendor + '.npz'), allow_pickle=False) as device, np.load(a.out / 'cpu.npz', allow_pickle=False) as host:
        assert set(device.files) == set(host.files)
        for key in device.files:
            left, right = device[key], host[key]
            row = {'output': key, 'device_shape': list(left.shape), 'host_shape': list(right.shape),
                   'device_dtype': str(left.dtype), 'host_dtype': str(right.dtype),
                   'device_sha256': hashlib.sha256(left.tobytes()).hexdigest(),
                   'host_sha256': hashlib.sha256(right.tobytes()).hexdigest()}
            if left.shape != right.shape or left.dtype != right.dtype:
                row['status'] = 'SHAPE_OR_DTYPE_DIFFER'
            else:
                word = np.dtype('u' + str(left.dtype.itemsize))
                lb, rb = left.view(word).ravel(), right.view(word).ravel()
                indices = np.flatnonzero(lb != rb)
                row.update(status='PASS' if not len(indices) else 'DIFFER', changed_words=int(len(indices)),
                           total_words=int(left.size), first_differences=[{'index': int(i),
                               'device_bits': hex(int(lb[i])), 'host_bits': hex(int(rb[i])),
                               'device_value': float(left.ravel()[i]), 'host_value': float(right.ravel()[i])}
                               for i in indices[:4]])
            report['outputs'].append(row)
    report['status'] = 'PASS' if all(row['status'] == 'PASS' for row in report['outputs']) else 'DIFFER'
    (a.out / 'comparison.json').write_text(json.dumps(report, indent=2))
    print('FAMILY_IDENTITY', report['status'], 'outputs=' + str(len(report['outputs'])),
          'differing=' + str(sum(row['status'] != 'PASS' for row in report['outputs'])), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
