"""Legacy aliases retain the one compiled classical and resident-tree owner."""
from types import BuiltinFunctionType

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend, host_surface
from mojolearn._verify_native_runtime import audit_native_call


def owners():
    legacy = _backend.load_host_module('_mojolearn_tokenizer_host')
    native = _backend.load_host_module('_mojolearn_runtime_classical_host')
    return legacy, native


def test_every_legacy_classical_name_is_the_exact_new_native_entry():
    legacy, native = owners()
    spec, = host_surface.HOST_IMPORT_ALIASES['_mojolearn_tokenizer_host']
    assert len(spec['functions']) == 29
    assert legacy.tokenizer_host_runtime_classical_abi() == 1
    for name in spec['functions']:
        entry = getattr(legacy, name)
        assert type(entry) is BuiltinFunctionType
        assert entry is getattr(native, name)
    assert legacy._TreesResident is native._TreesResident
    assert legacy._mojolearn_missing_import_owners == ()
    assert _backend.load_host_module('_mojolearn_tokenizer_host') is legacy
    assert _backend.load_host_module('_mojolearn_runtime_classical_host') is native


def test_public_classical_fit_predict_and_archive_use_native_entries(tmp_path):
    axis = np.linspace(-1, 1, 12, dtype=np.float32)
    x = np.stack((axis, axis * axis), axis=1)
    y = (x[:, 0] * np.float32(2) - x[:, 1] * np.float32(.5) + np.float32(3)).copy()
    model = audit_native_call(ml.LinearRegression, numeric_mode='identical').assert_native()
    audit_native_call(model.fit, x, y).assert_native()
    expected = np.asarray(audit_native_call(model.predict, x).assert_native()).copy()
    np.testing.assert_allclose(expected.reshape(-1), y, rtol=0, atol=0.001)
    archive = str(tmp_path / 'linear.npz')
    audit_native_call(model.save, archive).assert_native()
    restored = audit_native_call(ml.LinearRegression.load, archive).assert_native()
    got = np.asarray(audit_native_call(restored.predict, x).assert_native())
    np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))


def test_tree_resident_type_comes_from_the_single_classical_owner():
    legacy, native = owners()
    x = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = np.arange(20, dtype=np.int32) % 2
    model = audit_native_call(ml.RandomForestClassifier, n_estimators=3, max_depth=3,
                              random_state=13, numeric_mode='identical').assert_native()
    audit_native_call(model.fit, x, y).assert_native()
    audit_native_call(model.predict, x).assert_native()
    fields = object.__getattribute__(model, '__dict__')
    holder = fields.get('_resident_holder', {}).get('native')
    if holder is None:
        pytest.skip('this selected backend exposes no resident forest snapshot; GPU lifecycle proof remains owed')
    assert type(holder[0]) is native._TreesResident
    assert type(holder[0]) is legacy._TreesResident


def test_classical_public_input_protocol_is_refused_before_callback():
    calls = []

    class Input:
        def __array__(self, *args, **kwargs):
            calls.append('__array__')
            raise AssertionError('input protocol must never run')

    model = ml.LinearRegression(numeric_mode='identical')
    audit = audit_native_call(model.fit, Input(), [1., 2.])
    assert isinstance(audit.exception, TypeError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
                                           'unresolved_native_calls', 'wrapper_work',
                                           'unobserved_existing_threads')), report
    assert calls == []
