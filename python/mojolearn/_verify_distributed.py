#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded installed/source GPU capture for forecasting, GPC and IVF.

Run with the target interpreter and package already selected (installed wheel:
use an external cwd and do not set PYTHONPATH). No source import-path injection.
One/two/reversed-device results are checkpointed after each case. Physical
kernel traces are deliberately not inferred from subprocess PID receipts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import re
import base64
from contextlib import contextmanager

__all__ = ["main", "compare", "validate_receipt"]
PROTOCOL = "distributed-classical-v2"
PROFILE_FILES = frozenset(('__init__.py', 'parallel_forecasting.py', 'parallel_gaussian_process.py',
                           'parallel_ivf.py', '_verify_distributed.py', '_parallel_pool.py',
                           '_parallel_worker.py', '_gpu_witness.py'))
CASES = ("arima", "holtwinters", "gpc", "ivf", "gpc_fit")
NUMERICAL_OPERATIONS = frozenset(("forecast_predict", "gpc_class_fit", "gpc_class_predict", "ivf_search_stored"))


def digest(value):
    import numpy as np
    x = np.asarray(value)
    return dict(shape=list(x.shape), dtype=x.dtype.str,
                sha256=hashlib.sha256(x.tobytes()).hexdigest())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def _require_used_groups(receipt, group_ids, case):
    from ._gpu_witness import require_distinct_workers
    if not group_ids:
        raise ValueError('numerical cell used no inventoried workers')
    operation = {'arima':'forecast_predict', 'holtwinters':'forecast_predict',
                 'gpc':'gpc_class_predict', 'gpc_fit':'gpc_class_fit', 'ivf':'ivf_search_stored'}[case]
    for group_id in group_ids:
        group = receipt['worker_groups'][group_id]
        require_distinct_workers(group['inventory'], receipt['vendor'], len(group['devices']))
        expected = {entry['pid'] for entry in group['inventory']}
        actual = {call['pid'] for call in receipt['worker_calls']
                  if call['group'] == group_id and call['operation'] == operation}
        if expected != actual:
            raise ValueError('an inventoried GPU worker received no numerical operation')


@contextmanager
def transport_fault(pool_class, fault, control):
    """Perturb one actual numerical result batch; never report a native fault."""
    if fault not in ('drop_result', 'reverse_results'):
        raise ValueError('unknown transport fault')
    original = pool_class.map
    def altered(pool, requests):
        requests = list(requests)
        values = original(pool, requests)
        if (not control['triggered'] and len(values) > 1 and requests
                and requests[0][0] in NUMERICAL_OPERATIONS):
            control['triggered'] = True
            return values[:-1] if fault == 'drop_result' else list(reversed(values))
        return values
    pool_class.map = altered
    try:
        yield
    finally:
        pool_class.map = original


def _valid_digests(values):
    return (isinstance(values, list) and bool(values) and all(
        isinstance(v, dict) and set(v) == {'shape', 'dtype', 'sha256'}
        and isinstance(v['shape'], list) and all(type(n) is int and n > 0 for n in v['shape'])
        and isinstance(v['dtype'], str) and bool(v['dtype'])
        and isinstance(v['sha256'], str) and re.fullmatch('[0-9a-f]{64}', v['sha256'])
        for v in values))


def validate_receipt(receipt):
    """Validate numerical, transport and placement coverage; no execution claim."""
    if (receipt.get('protocol') != PROTOCOL
            or receipt.get('status') != 'NUMERICAL_MATCH_EXECUTION_TRACE_OWED'
            or receipt.get('repeats') != 2 or receipt.get('vendor') not in ('cuda', 'hip')):
        raise ValueError('incomplete or incompatible distributed capture')
    devices = receipt.get('devices', [])
    if len(devices) != 2 or len(set(devices)) != 2 or any(type(d) is not int or d < 0 for d in devices):
        raise ValueError('capture requires two distinct device indices')
    layouts = [[devices[0]], devices, list(reversed(devices))]
    expected = {(case, layout, repeat) for case in CASES for layout in range(3) for repeat in range(2)}
    seen, canonical = set(), {}
    for cell in receipt.get('cells', []):
        key = (cell['case'], cell['layout'], cell['repeat'])
        if key in seen or key not in expected or cell['devices'] != layouts[cell['layout']]:
            raise ValueError('duplicate, unexpected or misordered capture cell')
        seen.add(key)
        if (cell.get('match') is not True or not _valid_digests(cell['actual'])
                or cell['actual'] != cell['expected']):
            raise ValueError('numerical mismatch or malformed digest')
        baseline = canonical.setdefault(cell['case'], cell['expected'])
        if cell['expected'] != baseline:
            raise ValueError('canonical reference changed across repetitions')
        for index in cell['worker_groups']:
            if receipt['worker_groups'][index]['devices'] != cell['devices']:
                raise ValueError('cell placement differs from requested devices')
        _require_used_groups(receipt, cell['worker_groups'], cell['case'])
    if seen != expected:
        raise ValueError('missing requested distributed cells')
    controls = receipt.get('controls', [])
    keys = [(c['case'], c['fault']) for c in controls]
    if (len(keys) != 10 or set(keys) != {(case, fault) for case in CASES
                                     for fault in ('drop_result', 'reverse_results')}
            or any(c.get('kind') != 'transport' or c.get('triggered') is not True
                   or c.get('detected') is not True for c in controls)):
        raise ValueError('missing or unobserved transport controls')
    if (set(receipt.get('inputs', {})) != {'arima', 'holtwinters', 'gpc_x', 'gpc_y', 'ivf'}
            or not all(_valid_digests([v]) for v in receipt['inputs'].values())):
        raise ValueError('missing or invalid fixture input witnesses')
    required_bindings = {'_mojolearn_arima', '_mojolearn_tsa', '_mojolearn_gp', '_mojolearn_ivf'}
    if set(receipt.get('bindings', {})) != required_bindings or set(receipt.get('source_files', {})) != PROFILE_FILES:
        raise ValueError('missing package or native binding provenance')
    if any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h)
           for h in receipt['source_files'].values()):
        raise ValueError('invalid source profile hashes')
    for binding in receipt['bindings'].values():
        if (binding.get('vendor') != receipt['vendor'] or
                not re.fullmatch('[0-9a-f]{64}', binding.get('sha256', ''))):
            raise ValueError('binding vendor or digest differs from capture')
    return True


def compare(left, right):
    """Compare complete captures from compatible source, including cross-vendor."""
    validate_receipt(left)
    validate_receipt(right)
    if left.get('inputs') != right.get('inputs'):
        raise ValueError('distributed fixture input bytes differ')
    if left['source_files'] != right['source_files']:
        raise ValueError('distributed source profiles differ')
    def parts(receipt):
        return {(c['case'], c['layout'], c['repeat']): c['actual'] for c in receipt['cells']}
    return parts(left) == parts(right)


def verify_distribution_records(dist, package, source_files, bindings):
    """Require actual source/native bytes to match this installed wheel RECORD."""
    records = {str(path): path for path in (dist.files or ())}
    def check(relative, sha):
        record = records.get(relative)
        expected = base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip('=')
        if record is None or record.hash is None or record.hash.mode != 'sha256' or record.hash.value != expected:
            raise ValueError('installed file differs from wheel RECORD: ' + relative)
    for filename, sha in source_files.items():
        check('mojolearn/' + filename, sha)
    for binding in bindings.values():
        native_path = Path(binding['path'])
        if not native_path.is_relative_to(package):
            raise ValueError('native binding is outside installed package: ' + str(native_path))
        check(str(native_path.relative_to(package.parent)), binding['sha256'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices', help='two distinct GPU indices, e.g. 0,1')
    ap.add_argument('--out', type=Path, help='new checkpoint JSON path')
    ap.add_argument('--require-installed', action='store_true')
    ap.add_argument('--compare', type=Path, nargs=2, metavar=('LEFT', 'RIGHT'))
    args = ap.parse_args(argv)
    if args.compare:
        if args.devices or args.out or args.require_installed:
            ap.error('--compare cannot be combined with capture options')
        matched = compare(*(json.loads(path.read_text()) for path in args.compare))
        print(json.dumps({'status': 'MATCH_EXECUTION_TRACE_OWED' if matched else 'NUMERICAL_MISMATCH'}))
        return 0 if matched else 1
    if not args.devices or args.out is None:
        ap.error('capture requires --devices and --out')
    try:
        devices = tuple(int(x) for x in args.devices.split(','))
    except ValueError:
        ap.error('devices must be comma-separated integer indices')
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        ap.error('exactly two distinct nonnegative GPU indices are required')
    if args.out.exists():
        ap.error('out already exists; preserve previous capture')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['MOJOLEARN_NUMERIC_MODE'] = 'identical'
    import numpy as np
    import mojolearn as ml
    from mojolearn import _backend
    from mojolearn._parallel_pool import DevicePool
    from mojolearn import parallel_forecasting as pf, parallel_gaussian_process as pg
    from mojolearn.parallel_ivf import DistributedIVFIndex
    if _backend.numeric_mode() != 'identical':
        ap.error('distributed capture requires identical numeric mode in a fresh process')
    vendor = _backend.vendor()
    if vendor not in ('cuda', 'hip'):
        ap.error('physical qualification capture requires CUDA or HIP')
    package = Path(ml.__file__).resolve().parent
    distribution = None
    try:
        import importlib.metadata as metadata
        dist = metadata.distribution('mojolearn')
        distribution = {'version': dist.version, 'root': str(Path(dist.locate_file('')).resolve())}
        expected = Path(dist.locate_file('mojolearn/__init__.py')).resolve()
        distribution['import_matches_distribution'] = expected == Path(ml.__file__).resolve()
        direct = dist.read_text('direct_url.json')
        distribution['editable'] = bool(direct and json.loads(direct).get('dir_info', {}).get('editable'))
    except metadata.PackageNotFoundError:
        pass
    if args.require_installed and (not distribution or not distribution['import_matches_distribution'] or distribution['editable']):
        ap.error('import does not match an installed mojolearn distribution')
    from mojolearn._verify import environment
    receipt = {'protocol': PROTOCOL, 'environment': environment(), 'repeats': 2, 'status': 'RUNNING',
               'physical_execution_trace': 'OWED', 'native_fault_controls': 'OWED',
               'vendor': vendor,
               'package': str(package), 'distribution': distribution,
               'devices': list(devices), 'cells': [], 'worker_calls': [],
               'bindings': {}, 'source_files': {}, 'worker_groups': [], 'controls': []}
    from mojolearn import _parallel_pool, _parallel_worker, _gpu_witness
    for module in (ml, pf, pg, sys.modules[DistributedIVFIndex.__module__],
                   sys.modules[__name__], _parallel_pool, _parallel_worker, _gpu_witness):
        p = Path(module.__file__)
        receipt['source_files'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('_mojolearn_arima', '_mojolearn_tsa', '_mojolearn_gp', '_mojolearn_ivf'):
        native = _backend.binding(name, 'identical')
        p = Path(native.__file__).resolve()
        receipt['bindings'][name] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                          vendor=_backend.read_vendor(native))
    from mojolearn._verify import _git_commit
    receipt['source_commit'] = _git_commit(package)
    guarded = os.environ.get('MOJOLEARN_COMMIT')
    if guarded is not None and not re.fullmatch('[0-9a-f]{40}', guarded):
        ap.error('MOJOLEARN_COMMIT must be a full lowercase SHA when provided')
    receipt['guarded_source_commit'] = guarded
    if args.require_installed:
        try:
            verify_distribution_records(dist, package, receipt['source_files'], receipt['bindings'])
        except ValueError as exc:
            ap.error(str(exc))
        receipt['distribution']['source_record_hashes_match'] = True
        receipt['distribution']['native_record_hashes_match'] = True
    from mojolearn._gpu_witness import require_distinct_workers
    original_call, original_start = DevicePool._call, DevicePool._start
    worker_groups = {}
    def start_pool(pool):
        started = bool(pool._workers)
        original_start(pool)
        if started:
            return
        inventory = [original_call(w, ('device_inventory', None, ())) for w in pool._workers]
        require_distinct_workers(inventory, vendor, len(pool.devices))
        group = len(receipt['worker_groups'])
        receipt['worker_groups'].append(dict(group=group, devices=list(pool.devices), inventory=inventory))
        for w, witness in zip(pool._workers, inventory):
            if witness['pid'] != w.pid:
                raise RuntimeError('worker inventory PID differs from launched process')
            worker_groups[id(w)] = group
    def observe(worker, request):
        answer = original_call(worker, request)
        receipt['worker_calls'].append({'pid': worker.pid, 'operation': request[0], 'group': worker_groups[id(worker)]})
        return answer
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, receipt)
    start = time.monotonic()
    DevicePool._start = start_pool
    DevicePool._call = staticmethod(observe)
    try:
        y = np.random.default_rng(42).normal(size=(5, 32)).astype(np.float32)
        arima = ml.ARIMA(order=(1, 1, 0), maxiter=3, numeric_mode='identical').fit(y)
        hw_y = np.asarray([[4 + s + .05 * t + (t % 4) * .1 for t in range(24)] for s in range(5)], np.float32)
        hw = ml.ExponentialSmoothing(hw_y, ts_num=5, seasonal_periods=4).fit()
        x = np.random.default_rng(75).normal(size=(9, 3)).astype(np.float32)
        labels = [i % 3 for i in range(9)]
        gpc = ml.GaussianProcessClassifier(max_iter_predict=3, numeric_mode='identical').fit(x, labels)
        ivf_x = np.random.default_rng(97).normal(size=(65, 17)).astype(np.float32)
        ivf_x[1] = ivf_x[0]
        ivf = ml.IVFIndex(4, 4, 5, kmeans_n_iters=3, metric='euclidean', numeric_mode='identical').fit(ivf_x)
        receipt['inputs'] = {name: digest(value) for name, value in
                             [('arima', y), ('holtwinters', hw_y), ('gpc_x', x),
                              ('gpc_y', np.asarray(labels, dtype=np.int32)), ('ivf', ivf_x)]}
        expected = {'arima': [digest(arima.predict(0, 36))],
                    'holtwinters': [digest(hw.predict(26, 31))],
                    'gpc': [digest(gpc.predict_proba(x[:5])), digest(gpc.predict(x[:5]))],
                    'ivf': [digest(v) for v in ivf.search(ivf_x[:7])]}
        expected['gpc_fit'] = [digest(getattr(e, name)) for e in gpc.estimators_
                               for name in ('y_train_', 'L_', 'pi_', 'W_sr_')]
        def run_case(case, selected):
            if case == 'arima':
                return [pf.predict_arima(arima, 0, 36, devices=selected, series_per_shard=2)]
            if case == 'holtwinters':
                return [pf.predict_exponential_smoothing(hw, 26, 31, devices=selected, series_per_shard=2)]
            if case == 'gpc':
                return [pg.predict_gaussian_process_classifier(gpc, x[:5], devices=selected, method=m)
                        for m in ('predict_proba', 'predict')]
            if case == 'gpc_fit':
                fitted = pg.fit_gaussian_process_classifier(
                    ml.GaussianProcessClassifier(max_iter_predict=3, numeric_mode='identical'),
                    x, labels, devices=selected)
                return [getattr(e, name) for e in fitted.estimators_
                        for name in ('y_train_', 'L_', 'pi_', 'W_sr_')]
            with DistributedIVFIndex.from_index(ivf, devices=selected) as distributed:
                return list(distributed.search(ivf_x[:7]))
        for layout, selected in enumerate(((devices[0],), devices, tuple(reversed(devices)))):
            for repeat in range(2):
                for case in CASES:
                    first_group = len(receipt['worker_groups'])
                    got = [digest(v) for v in run_case(case, selected)]
                    groups = list(range(first_group, len(receipt['worker_groups'])))
                    _require_used_groups(receipt, groups, case)
                    cell = dict(case=case, devices=list(selected), layout=layout, repeat=repeat,
                                worker_groups=groups, expected=expected[case], actual=got,
                                match=got == expected[case])
                    receipt['cells'].append(cell)
                    write(args.out, receipt)
                    if not cell['match']:
                        raise AssertionError(f'{case} differs on devices={selected}, repeat={repeat}')
        # These perturb actual transport result batches, not native arithmetic.
        for case in CASES:
            for fault in ('drop_result', 'reverse_results'):
                control = dict(case=case, fault=fault, kind='transport', triggered=False, detected=False)
                try:
                    with transport_fault(DevicePool, fault, control):
                        got = [digest(v) for v in run_case(case, devices)]
                    control['detected'] = got != expected[case]
                except (ValueError, RuntimeError, IndexError) as exc:
                    control['detected'] = True
                    control['error'] = repr(exc)
                receipt['controls'].append(control)
                write(args.out, receipt)
                if not control['triggered'] or not control['detected']:
                    raise AssertionError(f'{case} did not detect {fault}')
        receipt['status'] = 'NUMERICAL_MATCH_EXECUTION_TRACE_OWED'
        validate_receipt(receipt)
    except BaseException as exc:
        receipt['status'] = 'FAILED'
        receipt['error'] = repr(exc)
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - start
        write(args.out, receipt)
        DevicePool._call = staticmethod(original_call)
        DevicePool._start = original_start
    print(json.dumps({'status': receipt['status'], 'cells': len(receipt['cells']), 'out': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
