"""Public manifold operations and legacy forwarding across native owners."""
from types import BuiltinFunctionType

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend, _native_family_bridge
from mojolearn import _expansion_ann as ann, _expansion_decomp as decomp, _umap_impl as umap


def _arguments(model, x):
    return (_backend.binding(decomp._BINDING, model.numeric_mode), _backend.binding(decomp._BINDING), x)


def test_manifold_public_modules_share_the_new_native_owner():
    module = _backend.load_host_module('_mojolearn_runtime_manifold_host')
    entry = module.manifold_native_call
    assert isinstance(entry, BuiltinFunctionType)
    for source in (ann, decomp, umap):
        assert source._NATIVE is entry
        assert source._NATIVE_CONTEXT['family_mani_door'] is entry
    assert decomp._NATIVE_CONTEXT is umap._NATIVE_CONTEXT is ann._NATIVE_CONTEXT


def test_public_classical_mds_and_generic_native_forwarding_preserve_bits_and_state():
    x = np.random.default_rng(19).normal(size=(18, 4)).astype(np.float32)
    original = x.copy()
    public = ml.ClassicalMDS(n_components=2, numeric_mode='identical')
    direct = np.asarray(public.fit_transform(x))
    legacy = ml.ClassicalMDS(n_components=2, numeric_mode='identical')
    entry, context = _native_family_bridge.connect()
    forwarded = np.asarray(entry(legacy, 'family_mani_cmds_fit_transform', _arguments(legacy, x), context))
    assert direct.shape == forwarded.shape == (18, 2)
    assert np.isfinite(direct).all()
    np.testing.assert_array_equal(direct.view(np.uint32), forwarded.view(np.uint32))
    np.testing.assert_array_equal(np.asarray(public.embedding_), direct)
    np.testing.assert_array_equal(np.asarray(legacy.embedding_), forwarded)
    np.testing.assert_array_equal(x, original)


def test_native_manifold_error_class_message_and_subsequent_call_are_preserved():
    x = np.random.default_rng(23).normal(size=(12, 3)).astype(np.float32)
    entry, context = _native_family_bridge.connect()
    direct = ml.ClassicalMDS(metric='minkowski', metric_params={'p': 0}, numeric_mode='identical')
    forwarded = ml.ClassicalMDS(metric='minkowski', metric_params={'p': 0}, numeric_mode='identical')
    with pytest.raises(ValueError) as first:
        direct.fit_transform(x)
    with pytest.raises(ValueError) as second:
        entry(forwarded, 'family_mani_cmds_fit_transform', _arguments(forwarded, x), context)
    assert type(first.value) is type(second.value)
    assert str(first.value) == str(second.value)
    assert np.isfinite(np.asarray(ml.ClassicalMDS(n_components=2, numeric_mode='identical').fit_transform(x))).all()


def test_legacy_forwarding_rejects_python_door_before_invoking_it():
    entry, context = _native_family_bridge.connect()
    calls = []

    def callback(*args):
        calls.append(args)
        return None

    invalid = dict(context, family_mani_door=callback)
    with pytest.raises(TypeError, match='compiled native entry'):
        entry(None, 'family_mani_cmds_fit_transform', (), invalid)
    assert calls == []
