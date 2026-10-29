# SPDX-License-Identifier: Apache-2.0
"""Actual installed native exception ownership and saved-host CPU prediction.

The subprocess guards retain a keeper reference and exit immediately on a
reference-count mismatch: an old broken extension must fail this regression,
not corrupt the pytest controller while it tears down a dangling exception.
"""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest


@pytest.mark.parametrize('route', ['generic', 'mixin'])
def test_missing_attribute_keeps_owner_and_original_exception(tmp_path, route):
    code = r'''
import gc, os, sys, types, weakref
import mojolearn
from mojolearn.tokenizer import _binding
from mojolearn._mode import PublicOutputsMixin
class Plain:
    pass
class Public(PublicOutputsMixin):
    pass
model = Plain() if ROUTE == 'generic' else Public()
keeper = [model]
reference = weakref.ref(model)
context = {'attributes': None}
entry = _binding()._module.public_attribute_generic_native_call
baseline = sys.getrefcount(model)
for iteration in range(32):
    if ROUTE == 'generic':
        try:
            entry(model, 'absent', context)
        except AttributeError as error:
            if error.obj is not model or error.name != 'absent':
                print('original AttributeError metadata changed', flush=True)
                os._exit(72)
        else:
            os._exit(73)
    else:
        assert getattr(model, 'absent', None) is None
        assert not hasattr(model, 'absent')
    if sys.getrefcount(model) != baseline:
        print('lost owner reference', ROUTE, iteration, baseline,
              sys.getrefcount(model), flush=True)
        os._exit(74)
del model
keeper.clear()
gc.collect()
assert reference() is None
print('32 missing lookups, stable references, released owner', flush=True)
'''
    result = subprocess.run([sys.executable, '-c', 'ROUTE = ' + repr(route) + '\n' + textwrap.dedent(code)],
                            cwd=tmp_path, env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('name', ['DBSCAN', 'AgglomerativeClustering', 'SpectralClustering'])
def test_saved_host_public_predict_uses_connected_cpu_owner(tmp_path, name):
    import mojolearn as ml
    from mojolearn import _native_public_attributes as attributes
    from mojolearn._native_family_bridge import shared_context
    from mojolearn.tests.test_host_archive_native import _archive

    archive = _archive(tmp_path, name)
    model = ml.host_model(archive)
    assert type(model) in attributes._CONTEXT['plans']
    rows = [row for row in shared_context()['family_host_binding_connections'] if row[0] is type(model)]
    assert len(rows) == 1
    module = rows[0][2]
    assert Path(vars(module)['__file__']).is_file()
    vendor_reader = rows[0][1].removeprefix('_mojolearn_') + '_vendor'
    assert vars(module)[vendor_reader]() == 'cpu'
    query = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    actual = np.asarray(model.predict(query))
    explicit = np.asarray(ml.host_predict(archive, query))
    np.testing.assert_array_equal(actual, explicit)
    fields = object.__getattribute__(model, '__dict__')
    held = fields.get('_resident_holder', {}).get('index')
    if held is not None:
        assert held[2] is module, 'saved host prediction selected a non-CPU resident owner'
    # Repeated missing accesses used by the crosscheck must not corrupt the
    # restored model or change later ordinary prediction.
    baseline = sys.getrefcount(model)
    for _ in range(8):
        assert getattr(model, '_binding', None) is None
    assert sys.getrefcount(model) == baseline
    np.testing.assert_array_equal(np.asarray(model.predict(query)), explicit)


def test_saved_host_binding_replacement_refused_without_callback(tmp_path, monkeypatch):
    import mojolearn as ml
    from mojolearn._native_family_bridge import shared_context
    from mojolearn.tests.test_host_archive_native import _archive

    model = ml.host_model(_archive(tmp_path, 'AgglomerativeClustering'))
    row = next(row for row in shared_context()['family_host_binding_connections'] if row[0] is type(model))
    module, exports = row[2], row[3]
    calls = []
    name = exports[0][0]
    monkeypatch.setattr(module, name, lambda *a: calls.append(a))
    with pytest.raises(TypeError, match='export changed'):
        model.predict(np.array([[1, 2]], dtype=np.float32))
    assert calls == []


def test_host_availability_does_not_admit_replaced_descriptor(tmp_path, monkeypatch):
    import mojolearn as ml
    from mojolearn import _classical_host as host
    from mojolearn import _native_public_attributes as attributes
    from mojolearn.tests.test_host_archive_native import _archive

    model = ml.host_model(_archive(tmp_path, 'AgglomerativeClustering'))
    calls = []
    monkeypatch.setattr(host.HostAgglomerativeClustering, 'predict', property(lambda self: calls.append(self)))
    with pytest.raises(TypeError):
        _ = model.predict
    assert calls == []
    missing = object()
    base_rules = attributes._hierarchy_availability(ml)
    inherited = attributes._inherited_host_availability(ml, base_rules, missing)
    assert 'predict' not in inherited.get(host.HostAgglomerativeClustering, {})


@pytest.mark.parametrize('name,entry_name', [
    ('DBSCAN', 'labeled_reference_predict'),
    ('AgglomerativeClustering', 'labeled_reference_predict'),
    ('SpectralClustering', 'spectral_predict'),
])
def test_expected_cpu_execution_entry_is_reached(tmp_path, monkeypatch, name, entry_name):
    """Native-only sabotage distinguishes executing CPU from GPU-vs-GPU equality.

    Reconnect only this fixture's CPU export to a different compiled zero-arg
    entry. Both prediction entries take two arguments. Its native arity refusal
    must be observed by the public call; no Python callback computes or runs.
    """
    import mojolearn as ml
    from mojolearn._native_family_bridge import shared_context
    from mojolearn.tests.test_host_archive_native import _archive

    model = ml.host_model(_archive(tmp_path, name))
    context = shared_context()
    rows = context['family_host_binding_connections']
    row = next(row for row in rows if row[0] is type(model))
    module = row[2]
    replacement = vars(module)[row[4]]
    # Mojo's generated native arity guard currently raises plain Exception
    # containing the TypeError label, rather than CPython TypeError. Retained
    # installed evidence: release-green-20261029/attempt1/focused.log. Capture
    # that exact native refusal and require the public path to propagate it.
    with pytest.raises(Exception, match=r'^TypeError: .*takes 0 positional arguments but 2 were given') as expected:
        replacement([], [])
    assert type(expected.value) is Exception
    monkeypatch.setattr(module, entry_name, replacement)
    # This test deliberately reconnects the native export, so the expected
    # failure occurs at execution rather than the export-identity guard above.
    exports = tuple((key, replacement if key == entry_name else value) for key, value in row[3])
    replacement_row = row[:3] + (exports, row[4])
    monkeypatch.setitem(context, 'family_host_binding_connections',
                        tuple(replacement_row if value is row else value for value in rows))
    # DBSCAN and Spectral preserve the generated pending exception;
    # Agglomerative translates its explicit label to CPython TypeError.
    # Compare the exact arity diagnostic after that existing translation.
    error_type = TypeError if name == 'AgglomerativeClustering' else Exception
    with pytest.raises(error_type) as actual:
        model.predict(np.array([[1, 2]], dtype=np.float32))
    assert type(actual.value) is error_type
    expected_message = str(expected.value)
    if error_type is TypeError:
        expected_message = expected_message.removeprefix('TypeError: ')
    assert actual.value.args == (expected_message,)


def test_public_fit_save_host_load_and_prediction_roundtrip(tmp_path):
    import mojolearn as ml
    values = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                       [8, 8], [8, 9], [9, 8], [9, 9]], dtype=np.float32)
    original = ml.AgglomerativeClustering(n_clusters=2, prediction_data=True)
    original.fit(values)
    expected = np.asarray(original.predict(values))
    path = tmp_path / 'public-agglomerative.npz'
    original.save(path)
    restored = ml.host_model(path)
    assert getattr(restored, '_binding', None) is None
    np.testing.assert_array_equal(np.asarray(restored.predict(values)), expected)
    np.testing.assert_array_equal(np.asarray(ml.host_predict(path, values)), expected)
