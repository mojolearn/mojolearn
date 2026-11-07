"""Legacy aliases retain the compiled classical owners and the one resident-tree owner."""
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


def test_classical_doors_are_connected_from_their_own_owners():
    legacy, _ = owners()
    from mojolearn._native_classical_bridge import _state_context
    context = _state_context()
    trees = _backend.load_host_module('_mojolearn_runtime_trees_host')
    family = _backend.load_host_module('_mojolearn_runtime_family_host')
    assert context['classical_trees_door'] is trees.trees_native_call is legacy.trees_native_call
    assert context['classical_family_door'] is family.family_native_call is legacy.family_native_call


def test_every_legacy_classical_name_is_the_exact_new_native_entry():
    legacy, native = owners()
    specs = host_surface.HOST_IMPORT_ALIASES['_mojolearn_tokenizer_host']
    # 2026-11-07 host split (docs/build/HOT_GROUPS.md section 6): one spec per
    # owner, the classical owner (which registers the types) first.
    assert specs[0]['binding'] == '_mojolearn_runtime_classical_host'
    assert len(specs) == 11
    assert legacy.tokenizer_host_runtime_classical_abi() == 2
    for spec in specs:
        owner = _backend.load_host_module(spec['binding'])
        for name in spec['functions']:
            entry = getattr(legacy, name)
            assert type(entry) is BuiltinFunctionType
            assert entry is getattr(owner, name)
    assert legacy._TreesResident is native._TreesResident
    assert legacy._TreesResidentDeath is native._TreesResidentDeath
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
    # The snapshot lives in the classical owner's native resident table
    # (bindings/trees_resident_table.mojo), never in the model's __dict__.
    from mojolearn import _native_trees_bridge
    entry, context = _native_trees_bridge.connect()
    kept_before = entry(None, 'trees_resident_stats', (), context)[0]
    audit_native_call(model.predict, x).assert_native()
    fields = object.__getattribute__(model, '__dict__')
    assert not fields.get('_resident_holder')
    assert entry(None, 'trees_resident_stats', (), context)[0] == kept_before + 1
    assert native._TreesResidentDeath is legacy._TreesResidentDeath


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
