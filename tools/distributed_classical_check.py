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


def digest(value):
    import numpy as np
    x = np.asarray(value)
    return dict(shape=list(x.shape), dtype=x.dtype.str,
                sha256=hashlib.sha256(x.tobytes()).hexdigest())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--devices', required=True, help='two distinct GPU indices, e.g. 0,1')
    ap.add_argument('--out', type=Path, required=True, help='new checkpoint JSON path')
    ap.add_argument('--require-installed', action='store_true')
    args = ap.parse_args(argv)
    devices = tuple(int(x) for x in args.devices.split(','))
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
    except metadata.PackageNotFoundError:
        pass
    if args.require_installed and (not distribution or not distribution['import_matches_distribution']):
        ap.error('import does not match an installed mojolearn distribution')
    receipt = {'protocol': 'distributed-classical-v1', 'status': 'RUNNING',
               'physical_execution_trace': 'OWED', 'native_fault_controls': 'OWED',
               'vendor': vendor,
               'package': str(package), 'distribution': distribution,
               'devices': list(devices), 'cells': [], 'worker_calls': [],
               'bindings': {}, 'source_files': {}}
    for module in (pf, pg, sys.modules[DistributedIVFIndex.__module__]):
        p = Path(module.__file__)
        receipt['source_files'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('_mojolearn_arima', '_mojolearn_tsa', '_mojolearn_gp', '_mojolearn_ivf'):
        native = _backend.binding(name, 'identical')
        p = Path(native.__file__).resolve()
        receipt['bindings'][name] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                          vendor=_backend.read_vendor(native))
    from mojolearn._verify import _git_commit
    receipt['source_commit'] = _git_commit()
    original_call = DevicePool._call
    def observe(worker, request):
        answer = original_call(worker, request)
        receipt['worker_calls'].append({'pid': worker.pid, 'operation': request[0]})
        return answer
    DevicePool._call = staticmethod(observe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, receipt)
    start = time.monotonic()
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
        expected = {'arima': [digest(arima.predict(0, 36))],
                    'holtwinters': [digest(hw.predict(26, 31))],
                    'gpc': [digest(gpc.predict_proba(x[:5])), digest(gpc.predict(x[:5]))],
                    'ivf': [digest(v) for v in ivf.search(ivf_x[:7])]}
        expected['gpc_fit'] = [digest(getattr(e, name)) for e in gpc.estimators_
                               for name in ('y_train_', 'L_', 'pi_', 'W_sr_')]
        for selected in ((devices[0],), devices, tuple(reversed(devices))):
            for repeat in range(2):
                for case in expected:
                    if case == 'arima':
                        values = [pf.predict_arima(arima, 0, 36, devices=selected, series_per_shard=2)]
                    elif case == 'holtwinters':
                        values = [pf.predict_exponential_smoothing(hw, 26, 31, devices=selected, series_per_shard=2)]
                    elif case == 'gpc':
                        values = [pg.predict_gaussian_process_classifier(gpc, x[:5], devices=selected, method=m)
                                  for m in ('predict_proba', 'predict')]
                    elif case == 'gpc_fit':
                        fitted = pg.fit_gaussian_process_classifier(ml.GaussianProcessClassifier(max_iter_predict=3, numeric_mode='identical'), x, labels, devices=selected)
                        values = [getattr(e, name) for e in fitted.estimators_
                                  for name in ('y_train_', 'L_', 'pi_', 'W_sr_')]
                    else:
                        with DistributedIVFIndex.from_index(ivf, devices=selected) as distributed:
                            values = list(distributed.search(ivf_x[:7]))
                    got = [digest(v) for v in values]
                    cell = dict(case=case, devices=list(selected), repeat=repeat,
                                expected=expected[case], actual=got, match=got == expected[case])
                    receipt['cells'].append(cell)
                    write(args.out, receipt)
                    if not cell['match']:
                        raise AssertionError(f'{case} differs on devices={selected}, repeat={repeat}')
        receipt['status'] = 'NUMERICAL_MATCH_EXECUTION_TRACE_OWED'
    except BaseException as exc:
        receipt['status'] = 'FAILED'
        receipt['error'] = repr(exc)
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - start
        write(args.out, receipt)
        DevicePool._call = staticmethod(original_call)
    print(json.dumps({'status': receipt['status'], 'cells': len(receipt['cells']), 'out': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
