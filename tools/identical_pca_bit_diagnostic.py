#!/usr/bin/env python3
"""Untimed public PCA/TSVD source-column comparison with per-output raw bits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def child(root, out, vendor):
    sys.path.insert(0, str(root / 'python'))
    from identical_wave_worker import source_provenance
    import numpy as np
    from mojolearn.decomposition import PCA, TruncatedSVD
    arrays = {}
    for n, d, k in ((4096, 220, 16), (1500, 33, 33), (900, 7, 3)):
        i = np.arange(n * d, dtype=np.int64).reshape(n, d)
        x = ((((i * 2654435761) % 1000003) - 500001).astype(np.float32) / np.float32(977.0)
             * (1.0 + (np.arange(d) % 11)).astype(np.float32))
        x = np.ascontiguousarray(x, dtype=np.float32)
        for estimator in (PCA(n_components=k), TruncatedSVD(n_components=min(k, d - 1))):
            estimator.fit(x)
            prefix = f'{type(estimator).__name__}-{n}x{d}'
            for name in ('components_', 'explained_variance_', 'explained_variance_ratio_',
                         'singular_values_', 'mean_'):
                value = getattr(estimator, name, None)
                if value is not None:
                    arrays[prefix + '-' + name] = np.ascontiguousarray(value)
            arrays[prefix + '-transform'] = np.ascontiguousarray(estimator.transform(x[:64]))
    np.savez(out / (vendor + '.npz'), **arrays)
    (out / (vendor + '-provenance.json')).write_text(json.dumps(source_provenance(root, vendor), indent=2))
    print('PCA_DIAGNOSTIC_CHILD', vendor, 'outputs=' + str(len(arrays)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vendor', choices=('cuda', 'hip', 'cpu'), required=True)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    if sys.platform != 'linux':
        p.error('execute only on the authorized Linux boxes')
    root = a.source.resolve()
    if a.child:
        child(root, a.out, a.vendor)
        return 0
    if a.vendor == 'cpu':
        p.error('parent requires cuda or hip')
    a.out.mkdir(parents=True, exist_ok=False)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    report = {'source_sha': sha, 'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'purpose': 'untimed raw-bit diagnostics', 'columns': {}, 'outputs': []}
    for vendor in (a.vendor, 'cpu'):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MODULAR_MOJO_'))}
        env.update(MOJOLEARN_VENDOR=vendor, MOJOLEARN_NUMERIC_MODE='identical',
                   PYTHONPATH=str(root / 'python'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                   LD_LIBRARY_PATH=str(root / 'python/mojolearn/.libs') + ':' + env.get('LD_LIBRARY_PATH', ''))
        with (a.out / (vendor + '.log')).open('w') as log:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child',
                '--source', str(root), '--out', str(a.out), '--vendor', vendor],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        report['columns'][vendor] = {'rc': proc.returncode}
        if proc.returncode:
            report['status'] = 'ERROR'
            (a.out / 'comparison.json').write_text(json.dumps(report, indent=2))
            return 1
        report['columns'][vendor]['provenance'] = json.loads((a.out / (vendor + '-provenance.json')).read_text())
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
    print('PCA_DIAGNOSTIC', report['status'], 'outputs=' + str(len(report['outputs'])),
          'differing=' + str(sum(row['status'] != 'PASS' for row in report['outputs'])), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
