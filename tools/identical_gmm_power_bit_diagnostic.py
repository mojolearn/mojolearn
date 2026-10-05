#!/usr/bin/env python3
"""Untimed public GMM/PowerTransformer source-column comparison with per-output raw bits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def child(root, out, vendor, gate):
    sys.path.insert(0, str(root / 'python'))
    from identical_wave_worker import source_provenance
    import numpy as np
    import mojolearn as ml
    inputs, arrays, routes = {}, {}, []
    def save(name, value):
        arrays[name] = np.array(value, copy=True, order='C')
    def learned(prefix, estimator):
        for name, value in vars(estimator).items():
            if (name.endswith('_') and not name.startswith('_')) or name in ('_mean', '_scale'):
                array = np.asarray(value)
                if array.dtype.kind in 'fibu':
                    save(prefix + '-' + name, array)
    if gate == 'gmm':
        from mojolearn.mixture import GaussianMixture
        assert not os.environ.get('MOJOLEARN_IDENTITY_TRACE')
        for n, d in ((257, 7), (513, 33)):
            rng = np.random.default_rng(7700 + d)
            x = (rng.standard_normal((n, d)) * 0.4
                 + (np.arange(n) % 3)[:, None] * 2).astype(np.float32)
            inputs[f'x-{n}x{d}'] = x
            estimator = GaussianMixture(n_components=3, init_params='random',
                random_state=11, max_iter=5, reg_covar=0.01, tol=1e-5).fit(x)
            prefix = f'GMM-{n}x{d}'
            learned(prefix, estimator)
            for method in ('predict', 'predict_proba', 'score_samples'):
                save(prefix + '-' + method, getattr(estimator, method)(x[:129]))
            routes.append({'shape': [n, d], 'path': 'public full-covariance fit',
                'guard': 'IDENTICAL default, GMM_SAB_NONE, d<=GMM_IDN_CHOL_MAX_D=256',
                'kernel': 'idn_precision_cholesky_kernel',
                'source': 'mixture/checks/mstep.mojo::gmm_precision_cholesky',
                'guard_file_sha256': hashlib.sha256((root / 'mixture/chol_order.mojo').read_bytes()).hexdigest()})
    else:
        from mojolearn._expansion_prep import PowerTransformer
        from mojolearn import _expansion_prep as prep
        from collections import Counter
        counts = Counter()
        original_stage = prep._Prog.stage
        def counted_stage(self, op, total, *params):
            counts[op] += 1
            return original_stage(self, op, total, *params)
        prep._Prog.stage = counted_stage
        flags = prep._idn_fam('identical')
        fast_flags = prep._ptimpute_flags('identical')
        assert prep._blocked() and flags & 1 and flags & 8
        assert not fast_flags & 32, 'Apple FAST anchor is outside IDENTICAL'
        n, d = 4099, 4
        index = np.arange(n * d, dtype=np.int64).reshape(n, d)
        base = (((index * 73) % 1009).astype(np.float32) / np.float32(113.0) + np.float32(0.25))
        for method in ('yeo-johnson', 'box-cox'):
            x = base.copy()
            if method == 'yeo-johnson':
                x -= np.float32(4.0)
            x[17, 1] = np.float32(np.nan)
            inputs[method] = x
            for standardize in (False, True):
                prefix = f'Power-{method}-standardize{standardize}'
                estimator = PowerTransformer(method=method, standardize=standardize).fit(x)
                learned(prefix, estimator)
                transformed = estimator.transform(x)
                save(prefix + '-transform', transformed)
                save(prefix + '-inverse', estimator.inverse_transform(transformed))
                assert estimator._pt_anchor is None and estimator._pt_anchor_kind is None
        prep._Prog.stage = original_stage
        for op in ('colb_part', 'colb_fold', 'colb_ss', 'colb_var', 'ptb_part1', 'ptb_mean', 'ptb_part2', 'ptb_fin'):
            assert counts[op] > 0, 'missing route: ' + op
        routes.append({'shape': [n, d], 'idn_family_flags': flags,
            'fast_flags': fast_flags, 'staged_operations': dict(counts),
            'anchor': 'merged argument slots present; FAST Apple-only anchor disabled in IDENTICAL',
            'block_rows': prep._XB, 'blocks': (n + prep._XB - 1) // prep._XB})
    input_info = {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
        'sha256': hashlib.sha256(v.tobytes(order='C')).hexdigest()} for k, v in inputs.items()}
    (out / (vendor + '-inputs.json')).write_text(json.dumps(input_info, indent=2))
    (out / (vendor + '-routes.json')).write_text(json.dumps(routes, indent=2))
    np.savez(out / (vendor + '.npz'), **arrays)
    (out / (vendor + '-provenance.json')).write_text(json.dumps(source_provenance(root, vendor), indent=2))
    print('FAMILY_IDENTITY_CHILD', gate, vendor, 'outputs=' + str(len(arrays)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vendor', choices=('cuda', 'hip', 'cpu'), required=True)
    p.add_argument('--child', action='store_true')
    p.add_argument('--gate', choices=('gmm', 'power'), required=True)
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
