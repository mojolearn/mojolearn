"""Public model-selection calls across the independent native owner."""
import itertools

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend, _native_family_bridge, model_selection as ms


def test_public_and_lazy_splitters_share_the_model_selection_owner():
    module = _backend.load_host_module("_mojolearn_runtime_model_selection_host")
    assert ms._NATIVE is module.model_selection_native_call
    assert ms._NATIVE_CONTEXT["family_msel_door"] is ms._NATIVE
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    splitter = ms.LeavePOut(2)
    splits = splitter.split(x)
    expected = list(itertools.combinations(range(len(x)), 2))
    actual = [tuple(np.asarray(test).tolist()) for _, test in splits]
    assert actual == expected
    assert splitter.get_n_splits(x) == len(expected)


def test_cross_validation_calls_the_estimator_owner_and_restores_native_errors():
    x = np.random.default_rng(7).normal(size=(18, 3)).astype(np.float32)
    y = 3 * x[:, 0] - 2 * x[:, 1] + 0.5 * x[:, 2] + 1
    scores = ms.cross_val_score(ml.LinearRegression(), x, y, cv=3, scoring="r2")
    np.testing.assert_allclose(np.asarray(scores), 1.0, rtol=1e-4, atol=1e-4)
    invalid = x.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        ms.cross_val_score(ml.LinearRegression(), invalid, y, cv=3, error_score="raise")


def test_splitter_error_class_is_preserved_at_the_new_native_boundary():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    with pytest.raises(ValueError, match="n_splits"):
        list(ms.KFold(n_splits=7).split(x))


def test_legacy_generic_entry_forwards_to_the_native_owner_and_refuses_callbacks():
    entry, context = _native_family_bridge.connect()
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    splitter = ms.KFold(n_splits=3)
    args = (x, None, None)
    assert entry(splitter, "family_msel_n_splits", args, context) == splitter.get_n_splits(x)
    called = []

    def callback(*args):
        called.append(args)
        return 3

    refused = dict(context, family_msel_door=callback)
    with pytest.raises(TypeError, match="compiled native entry"):
        entry(splitter, "family_msel_n_splits", args, refused)
    assert not called
