# SPDX-License-Identifier: Apache-2.0
"""Native constructor wire and authenticated worker inventory contracts."""
import array
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mojolearn import Array, GaussianNB, LinearRegression, model_selection as ms, _backend
from mojolearn._native_process import _native_process
from mojolearn._verify_native_runtime import audit_native_call


def entry(operation, args=()):
    return ms._NATIVE(None, operation, args, ms._NATIVE_CONTEXT)


def clean(call, *args, error=None, **kwargs):
    observed = audit_native_call(call, *args, **kwargs)
    report = observed.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                           'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(observed.exception, error), report
        return observed.exception
    return observed.assert_native()


@pytest.mark.parametrize('order', ['C', 'F'])
def test_native_wire_preserves_independent_constructor_buffers_and_dtype(order):
    dtype = np.dtype({'names': ['x', 'y'], 'formats': ['<i4', '<f8'], 'offsets': [0, 8], 'itemsize': 24},
                     align=True, metadata={'unit': 'test'})
    structured = np.zeros((2, 3), dtype=dtype, order=order)
    structured['x'] = np.arange(6).reshape(2, 3)
    structured['y'] = .25
    numeric = np.array([[1., 2.], [3., 4.]], dtype='>f8', order=order)
    values = dict(numeric=numeric, structured=structured, scalar=np.float32(.25),
                  integers=array.array('i', [2, 3]), mutable=bytearray(b'abc'),
                  nested=[(complex(2, 3), range(1, 8, 2), {2, 3}, frozenset({4, 5}))])
    result = clean(entry, 'family_msel_parallel_wire', (values,))
    for name in ('numeric', 'structured'):
        assert result[name] is not values[name]
        assert result[name].dtype == values[name].dtype
        assert result[name].dtype.metadata == values[name].dtype.metadata
        assert result[name].flags.f_contiguous == values[name].flags.f_contiguous
        assert result[name].tobytes(order='A') == values[name].tobytes(order='A')
    assert type(result['scalar']) is np.float32
    assert result['integers'] == values['integers'] and result['integers'] is not values['integers']
    assert result['mutable'] == values['mutable'] and result['mutable'] is not values['mutable']
    assert result['nested'] == values['nested'] and result['nested'] is not values['nested']


def test_numpy_constructor_metadata_restored_in_fresh_native_worker(tmp_path, monkeypatch):
    # The independent test bootstrap audits the entire real worker-loop call,
    # including decoding, estimator reconstruction, fit, score and outputs.
    module = tmp_path / 'native_cv_audited_worker.py'
    module.write_text('''
import json, os, sys
from pathlib import Path
if os.environ.get('MOJOLEARN_NATIVE_CV_NUMPY') == '1':
    import numpy
else:
    # Model an environment without the optional dependency before package import.
    import importlib.abc
    class BlockNumPy(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "numpy" or fullname.startswith("numpy."):
                raise ModuleNotFoundError("NumPy is absent in this fixture", name="numpy")
    sys.meta_path.insert(0, BlockNumPy())
from mojolearn import model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call
active = [False]
events = []
def hook(event, args):
    if active[0]: events.append(event)
sys.addaudithook(hook)
mode = os.environ['NATIVE_CV_AUDIT_MODE']
if mode == 'profile':
    result = audit_native_call(ms._NATIVE, None, 'family_msel_parallel_worker', (sys.argv,), ms._NATIVE_CONTEXT)
    report = result.report()
    report['exception'] = None if result.exception is None else repr(result.exception)
else:
    active[0] = True
    try:
        ms._NATIVE(None, 'family_msel_parallel_worker', (sys.argv,), ms._NATIVE_CONTEXT)
    finally:
        active[0] = False
    report = {'exception': None}
report['audit_mode'] = mode
report['numpy_loaded'] = 'numpy' in sys.modules
report['audit_events'] = events
Path(os.environ['NATIVE_CV_AUDIT_DIR'], str(os.getpid()) + '.json').write_text(json.dumps(report, indent=2))
if mode == 'profile': result.assert_native()
''')
    monkeypatch.setenv('NATIVE_CV_AUDIT_DIR', str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    x = Array.from_list([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 0.], [0., 2.]], '<f4')
    y = Array.from_list([0, 1, 0, 1, 1, 0], '<i4')
    reports = []
    for use_numpy, audit_mode in ((False, 'profile'), (False, 'hooks'), (True, 'profile'), (True, 'hooks')):
        monkeypatch.setenv('NATIVE_CV_AUDIT_MODE', audit_mode)
        priors = np.array([.5, .5], dtype='f8') if use_numpy else [.5, .5]
        model = clean(entry, 'family_msel_parallel_model_pack', (GaussianNB(priors=priors),))
        pool = clean(_native_process, 'create', ((0,), False, module.stem))
        pool['environment'] = {'MOJOLEARN_NATIVE_CV_NUMPY': '1' if use_numpy else '0'}
        try:
            identities = clean(_native_process, 'map', (pool, [('identity',)]))
            pid = pool['workers'][0]['pid']
            clean(entry, 'family_msel_parallel_require_inventory', (identities, _backend.vendor(), [pid], os.getpid()))
            result = clean(_native_process, 'map', (pool, [('fold', 0, model, None, x, y, x, y, False)]))
            assert result[0][0] == 0 and type(result[0][1]) is float and result[0][2] is None
        finally:
            clean(_native_process, 'close', (pool,))
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        report = json.loads((tmp_path / (str(pid) + '.json')).read_text())
        assert report['exception'] is None, report
        assert report['numpy_loaded'] is use_numpy, report
        for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
            assert not report.get(key, []), report
        assert report['audit_events'] == [], report
        reports.append(result[0][1])
    assert len(set(reports)) == 1


def test_dependency_free_public_cv_stays_numpy_free_in_parent_and_child(tmp_path):
    script = '''
import json, sys
from mojolearn import Array, LinearRegression, model_selection as ms
from mojolearn.parallel_model_selection import cross_val_score
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules
x = Array.from_list([[float(i), float(i % 3)] for i in range(15)], '<f4')
y = Array.from_list([float(i * 2 + (i % 3)) for i in range(15)], '<f4')
result = audit_native_call(cross_val_score, LinearRegression(), x, y, devices=(0,), cv=3)
result.assert_native()
assert 'numpy' not in sys.modules
trace = ms._NATIVE(None, 'family_msel_parallel_last_run', (), ms._NATIVE_CONTEXT)
assert trace['closed'] and trace['status'] == 'COMPLETE'
print(json.dumps({'scores': list(result.result), 'trace': trace}))
'''
    # The native worker is a fresh interpreter.  A fixture-owned startup module
    # makes NumPy unavailable in both the parent and its inherited child env.
    (tmp_path / 'sitecustomize.py').write_text('import sys\n# Model an environment without the optional dependency before package import.\nimport importlib.abc\nclass BlockNumPy(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname == "numpy" or fullname.startswith("numpy."):\n            raise ModuleNotFoundError("NumPy is absent in this fixture", name="numpy")\nsys.meta_path.insert(0, BlockNumPy())\n')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(tmp_path) + os.pathsep + env.get('PYTHONPATH', '')
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=120, env=env)
    (tmp_path / 'numpy-free.stdout').write_text(completed.stdout)
    (tmp_path / 'numpy-free.stderr').write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr
    assert len(json.loads(completed.stdout)['scores']) == 3


def record(vendor, pid, gpu=1):
    result = dict(kind='visible-device-inventory' if vendor in ('cuda', 'hip') else
                       ('worker-process-identity' if vendor == 'cpu' else 'single-metal-worker'),
                  vendor=vendor, pid=pid, ppid=os.getpid())
    if vendor in ('cuda', 'hip'):
        result['devices'] = [dict(ordinal=0, uuid=f'{gpu:032x}', pci_bus_id=f'0000:{gpu:02x}:00.0')]
    return result


@pytest.mark.parametrize('vendor', ['cpu', 'metal'])
def test_cpu_metal_inventory_uses_native_pid_without_driver_or_audit_hooks(monkeypatch, vendor):
    meta = ms._NATIVE_CONTEXT['family_msel_parallel']
    monkeypatch.setitem(meta, 'vendor', vendor)
    # No visibility dictionary is needed on these routes; a driver attempt
    # would fail this test independently of the host's installed vendor.
    monkeypatch.setitem(meta, 'environ', None)
    actual = clean(entry, 'family_msel_parallel_inventory')
    assert actual == dict(kind='worker-process-identity' if vendor == 'cpu' else 'single-metal-worker',
                          vendor=vendor, pid=os.getpid(), ppid=os.getppid())


@pytest.mark.parametrize('vendor', ['cpu', 'metal', 'cuda', 'hip'])
@pytest.mark.parametrize('fault', ['pid', 'parent', 'kind', 'bool-pid', 'hostile-key'])
def test_inventory_binds_route_parent_and_authenticated_pid(vendor, fault):
    item = record(vendor, 501)
    calls = []
    if fault == 'pid': item['pid'] = 502
    elif fault == 'parent': item['ppid'] = 1 if os.getpid() != 1 else 2
    elif fault == 'kind': item['kind'] = 'wrong'
    elif fault == 'bool-pid': item['pid'] = True
    else:
        class Key:
            def __hash__(self): return hash('vendor')
            def __eq__(self, other): calls.append('equal'); return False
        item[Key()] = 'foreign'
        calls.clear()
    clean(entry, 'family_msel_parallel_require_inventory', ([item], vendor, [501], os.getpid()), error=ValueError)
    assert calls == []


@pytest.mark.parametrize('fault', ['uuid', 'pci', 'pid', 'ordinal', 'count'])
def test_repeated_or_unisolated_gpu_inventory_refused_without_driver_calls(fault):
    items = [record('cuda', 501, 1), record('cuda', 502, 2)]
    expected = [501, 502]
    if fault == 'uuid': items[1]['devices'][0]['uuid'] = items[0]['devices'][0]['uuid']
    elif fault == 'pci': items[1]['devices'][0]['pci_bus_id'] = items[0]['devices'][0]['pci_bus_id']
    elif fault == 'pid': items[1]['pid'] = expected[1] = 501
    elif fault == 'ordinal': items[1]['devices'][0]['ordinal'] = 1
    else: items[1]['devices'].append(copy.deepcopy(items[0]['devices'][0]))
    clean(entry, 'family_msel_parallel_require_inventory', (items, 'cuda', expected, os.getpid()), error=ValueError)
