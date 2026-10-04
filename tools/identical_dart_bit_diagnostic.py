#!/usr/bin/env python3
"""Untimed DART repeatability, tree-state and source-column raw-bit comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def child(root, out, vendor, full=False, repeats=2):
    sys.path.insert(0, str(root / 'python'))
    from identical_wave_worker import source_provenance
    import numpy as np
    import mojolearn as ml
    arrays = {}
    inputs = {}
    params = dict(n_estimators=24, num_leaves=31, max_depth=5, min_child_samples=8,
                  max_bin=63, random_state=7, drop_seed=7, bagging_seed=7,
                  feature_fraction_seed=7, drop_rate=0.2, skip_drop=0.25)
    data=Path('/root/board-0833/cache/algos-data')/('rows-full' if full else 'rows-small')
    reference=json.loads(Path(__file__).with_name('identical_wave_dart_reference.json').read_text())
    for kind, cls in (('clf', ml.DARTClassifier), ('reg', ml.DARTRegressor)):
        block='cls' if kind=='clf' else 'reg'
        with np.load(data / (block + '-taxi.npz')) as z:
            n=len(z['X']) if full else 16384
            nq=len(z['Xq']) if full else 257
            x=np.ascontiguousarray(z['X'][:n]); y=np.ascontiguousarray(z['y'][:n])
            xq=np.ascontiguousarray(z['Xq'][:nq])
        if full:
            lane='dart' if kind=='clf' else 'dart-reg'
            params=next(r['params'] for r in reference['rows'] if r['lane']==lane)
        inputs[kind]={n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in [('X',x),('y',y),('Xq',xq)]}
        for repeat in range(repeats):
            model=cls(**params).fit(x,y)
            prefix=f'{kind}-repeat{repeat}'
            arrays[prefix+'-prediction']=np.ascontiguousarray(model.predict(xq))
            if kind=='clf': arrays[prefix+'-probability']=np.ascontiguousarray(model.predict_proba(xq))
            for name in ('init_score_','tree_coefs_','tree_weights_'):
                arrays[prefix+'-'+name]=np.ascontiguousarray(getattr(model,name))
            for i,tree in enumerate(model.trees_):
                for name in ('_offsets','_colid','_quesval','_left_child'):
                    arrays[f'{prefix}-tree{i:03d}-{name}']=np.ascontiguousarray(getattr(tree,name))
                arrays[f'{prefix}-tree{i:03d}-values']=np.ascontiguousarray(model.tree_values_[i])
    np.savez(out / (vendor + '.npz'), **arrays)
    (out / (vendor + '-provenance.json')).write_text(json.dumps(source_provenance(root, vendor), indent=2))
    (out / (vendor + '-inputs.json')).write_text(json.dumps(inputs,indent=2))
    repeated=[]
    for key,value in arrays.items():
        if repeats>1 and '-repeat0-' in key:
            other=arrays[key.replace('-repeat0-','-repeat1-')]
            if value.shape!=other.shape or value.dtype!=other.dtype or value.tobytes()!=other.tobytes():
                repeated.append(key)
    (out / (vendor + '-repeat.json')).write_text(json.dumps({
        'status': 'NOT_RUN' if repeats < 2 else ('PASS' if not repeated else 'DIFFER'),
        'repeats': repeats, 'differing_outputs': repeated}, indent=2))
    print('DART_DIAGNOSTIC_CHILD', vendor, 'outputs=' + str(len(arrays)), 'repeat_differences='+str(len(repeated)),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vendor', choices=('cuda', 'hip', 'cpu'), required=True)
    p.add_argument('--child', action='store_true')
    p.add_argument('--full', action='store_true', help='Exact stored full-row DART parameters and data')
    p.add_argument('--repeats', type=int, choices=(1,2), default=2)
    a = p.parse_args()
    if sys.platform != 'linux':
        p.error('execute only on the authorized Linux boxes')
    root = a.source.resolve()
    if a.child:
        child(root, a.out, a.vendor, a.full, a.repeats)
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
                '--source', str(root), '--out', str(a.out), '--vendor', vendor,
                '--repeats',str(a.repeats), *(['--full'] if a.full else [])],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        report['columns'][vendor] = {'rc': proc.returncode}
        if proc.returncode:
            report['status'] = 'ERROR'
            (a.out / 'comparison.json').write_text(json.dumps(report, indent=2))
            return 1
        report['columns'][vendor]['provenance'] = json.loads((a.out / (vendor + '-provenance.json')).read_text())
    assert (a.out / (a.vendor + '-inputs.json')).read_text() == (a.out / 'cpu-inputs.json').read_text(), 'input bytes differ'
    report['repeatability'] = {v:json.loads((a.out / (v+'-repeat.json')).read_text()) for v in (a.vendor,'cpu')}
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
    report['status'] = 'PASS' if all(row['status'] == 'PASS' for row in report['outputs']) and all(r['status']=='PASS' for r in report['repeatability'].values()) else 'DIFFER'
    (a.out / 'comparison.json').write_text(json.dumps(report, indent=2))
    print('DART_DIAGNOSTIC', report['status'], 'outputs=' + str(len(report['outputs'])),
          'differing=' + str(sum(row['status'] != 'PASS' for row in report['outputs'])), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
