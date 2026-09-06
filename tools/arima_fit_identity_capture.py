#!/usr/bin/env python3
"""Retain named ARIMA fit/predict/forecast bytes; compare without tolerance.

Main lane only. Requires a freshly built IDENTICAL ARIMA binding. This
extends observation beyond the Kalman card, not to all ARIMA configurations.
Run the full ARIMA surface accuracy gate separately before admitting results.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[1]
CASES = {'ar1': [1, 0, 0], 'ma1': [0, 0, 1], 'arma11': [1, 0, 1]}
SCHEMA = 'arima.fit.bytes.v2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_digest():
    files = set()
    # Include the imported GLM/TSA helpers as well as the ARIMA implementation.
    for name in ('arima', 'core', 'checks', 'glm', 'tsa'):
        directory = ROOT / name
        if not directory.is_dir():
            raise ValueError('Missing source dependency: ' + name)
        files.update(directory.rglob('*.mojo'))
    for name in ('bindings/_mojolearn_arima.mojo', 'bindings/build_arima.sh',
                 'python/mojolearn/_arima_impl.py', 'python/mojolearn/_backend.py',
                 'python/mojolearn/tests/test_arima_surface.py', 'pixi.toml', 'pixi.lock',
                 'tools/arima_fit_identity_capture.py'):
        files.add(ROOT / name)
    inventory = [(p.relative_to(ROOT).as_posix(), sha(p.read_bytes())) for p in sorted(files)]
    return sha(json.dumps(inventory, separators=(',', ':')).encode())


def tensor_contract(case):
    width = sum(CASES[case]) + 1
    contract = {name: ([6, width], '<f4') for name in ('params_', 'x_', 'x0_')}
    contract.update({name: ([6], '<i4') for name in ('n_iter_', 'retcode_')})
    contract.update(input=([6, 512], '<f4'), llf_=([6], '<f8'), fx_=([6], '<f4'),
                    prediction=([6, 512], '<f4'), forecast=([6, 16], '<f4'))
    return contract


def check_arrays(case, arrays):
    import numpy as np
    contract = tensor_contract(case)
    if set(arrays) != set(contract):
        raise ValueError('Missing tensor: ' + case)
    for name, (shape, dtype) in contract.items():
        value = arrays[name]
        if list(value.shape) != shape or value.dtype.str != dtype or not np.isfinite(value).all():
            raise ValueError('Bad shape/dtype/nonfinite tensor: ' + case + '/' + name)
    if np.any(arrays['retcode_'] != 0):
        raise ValueError('Unconverged optimizer: ' + case)
    if np.any(arrays['n_iter_'] < 0) or np.any(arrays['n_iter_'] > 1000):
        raise ValueError('Invalid optimizer iteration count: ' + case)
    if np.any(arrays['params_'][:, -1] <= 0):
        raise ValueError('Nonpositive fitted variance: ' + case)


def capture(out, commit):
    import numpy as np
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        raise ValueError('Set MOJOLEARN_NUMERIC_MODE=identical before import')
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Require a full source commit SHA')
    import mojolearn
    fixture = runpy.run_path(str(ROOT / 'python/mojolearn/tests/test_arima_surface.py'))
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema=SCHEMA, source_commit=commit, source_sha256=source_digest(),
                  numpy=np.__version__, mode='identical', package=mojolearn.__file__,
                  repeats=2, cases={}, status='INCOMPLETE')
    try:
        for case in fixture['PLANTED']:
            x = fixture['planted_batch'](case, 512)
            original_input = x.tobytes()
            baseline = None
            for repeat in range(2):
                # Independent model instances; retain copied arrays so an in-place
                # update cannot make both sides of the repeat check change together.
                model = mojolearn.ARIMA(order=case['order'], seasonal_order=case['seasonal'],
                                        trend='n', method='ml', maxiter=1000)
                model.numeric_mode = 'identical'
                binding = model._extension()
                if int(binding.arima_numeric_mode()) != 1:
                    raise ValueError('ARIMA binding is not compiled IDENTICAL')
                vendor = str(binding.arima_vendor())
                if vendor not in ('cuda', 'hip', 'metal'):
                    raise ValueError('No compiled GPU vendor')
                record['binding'] = str(Path(binding.__file__).resolve())
                record['binding_sha256'] = sha(Path(binding.__file__).read_bytes())
                record['compiled_vendor'] = vendor
                model.fit(x)
                current = {name: np.array(getattr(model, name), copy=True, order='C') for name in
                           ('params_', 'x_', 'x0_', 'n_iter_', 'retcode_', 'llf_', 'fx_')}
                current['prediction'] = np.array(model.predict(0, 512), copy=True, order='C')
                current['forecast'] = np.array(model.forecast(16), copy=True, order='C')
                current['input'] = np.array(x, copy=True, order='C')
                if x.tobytes() != original_input:
                    raise ValueError('Fit/predict changed input bytes: ' + case['name'])
                check_arrays(case['name'], current)
                if baseline is not None and any(current[k].tobytes() != baseline[k].tobytes() for k in current):
                    raise ValueError('Repeated fit changed bytes: ' + case['name'])
                baseline = current
            tensors = {}
            for name, value in baseline.items():
                filename = case['name'] + '.' + name + '.bin'
                raw = value.tobytes()
                (out / filename).write_bytes(raw)
                tensors[name] = dict(file=filename, shape=list(value.shape), dtype=value.dtype.str,
                                     bytes=len(raw), sha256=sha(raw))
            record['cases'][case['name']] = dict(order=list(case['order']), seasonal=list(case['seasonal']),
                                                trend='n', method='ml', maxiter=1000, tensors=tensors)
        if source_digest() != record['source_sha256']:
            raise ValueError('Sources changed during capture')
        record['status'] = 'CAPTURED_REPEATABLE'
    except Exception as exc:
        record['status'] = 'FAILED'
        record['reason'] = repr(exc)
        raise
    finally:
        (out / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print('ARIMA fit capture: three six-series/512-observation cases, repeated bytes stable')


def validate(root):
    import numpy as np
    record = json.loads((root / 'manifest.json').read_text())
    if (record.get('schema') != SCHEMA or record.get('status') != 'CAPTURED_REPEATABLE'
            or record.get('mode') != 'identical' or record.get('repeats') != 2):
        raise ValueError('Incomplete or wrong-mode capture: ' + str(root))
    if record.get('compiled_vendor') not in ('cuda', 'hip', 'metal'):
        raise ValueError('Invalid GPU vendor')
    for key, length in (('source_commit', 40), ('source_sha256', 64), ('binding_sha256', 64)):
        if not re.fullmatch('[0-9a-f]{' + str(length) + '}', record.get(key, '')):
            raise ValueError('Missing source/binding provenance: ' + key)
    if set(record['cases']) != set(CASES):
        raise ValueError('Missing named fixture')
    for name, case in record['cases'].items():
        if (case['order'] != CASES[name] or case['seasonal'] != [0, 0, 0, 0]
                or case['trend'] != 'n' or case.get('method') != 'ml' or case.get('maxiter') != 1000):
            raise ValueError('Wrong named fixture configuration')
        contract = tensor_contract(name)
        if set(case['tensors']) != set(contract):
            raise ValueError('Missing tensor')
        arrays = {}
        for tensor, entry in case['tensors'].items():
            shape, dtype = contract[tensor]
            if (entry['file'] != name + '.' + tensor + '.bin'
                    or entry['shape'] != shape or entry['dtype'] != dtype
                    or entry['bytes'] != math.prod(shape) * np.dtype(dtype).itemsize):
                raise ValueError('Bad tensor schema: ' + tensor)
            raw = (root / entry['file']).read_bytes()
            if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
                raise ValueError('Retained bytes changed')
            arrays[tensor] = np.frombuffer(raw, dtype=dtype).reshape(shape)
        check_arrays(name, arrays)
    return record


def compare(left, right):
    records = [validate(root) for root in (left, right)]
    for key in ('schema', 'source_commit', 'source_sha256', 'mode', 'repeats', 'cases'):
        if records[0][key] != records[1][key]:
            raise ValueError('ARIMA fit captures differ: ' + key)
    vendors = [r['compiled_vendor'] for r in records]
    scope = 'cross-vendor' if vendors[0] != vendors[1] else 'same-vendor'
    print('BITWISE PASS (' + scope + ', ' + '/'.join(vendors) +
          '): three named ARIMA fits, optimizer outputs, predictions and forecasts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('out', type=Path)
    cap.add_argument('--source-commit', required=True)
    diff = sub.add_parser('compare')
    diff.add_argument('left', type=Path)
    diff.add_argument('right', type=Path)
    args = parser.parse_args()
    if args.command == 'capture':
        capture(args.out, args.source_commit)
    else:
        compare(args.left, args.right)
