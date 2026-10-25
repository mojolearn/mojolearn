# SPDX-License-Identifier: Apache-2.0
"""Import metadata selects each tree owner's actual public lifecycle."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def bridge(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "python/mojolearn/_native_trees_bridge.py"
    spec = importlib.util.spec_from_file_location("_tree_registration_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = dict(trees_classes={}, trees_defaults={}, trees_constructors={})
    monkeypatch.setattr(module, "connect", lambda: (None, context))
    return module, context


@pytest.mark.parametrize("kind", ["gbdt", "gbdt_adapter", "iforest"])
def test_other_tree_schemas_keep_their_registration_without_forest_hooks(bridge, monkeypatch, kind):
    module, context = bridge
    marker = object()

    class Estimator:
        def __init__(self, *, criterion=marker):
            raise AssertionError("registration must not construct an estimator")

        def fit(self, x, y=None):
            raise AssertionError("registration must not execute a public method")

    original_fit = Estimator.fit
    original_getattribute = Estimator.__getattribute__

    def wrong_owner(*args):
        raise AssertionError("non-forest schema was connected to the forest lifecycle")

    monkeypatch.setattr(module, "_connect_public_forest", wrong_owner)
    table = {"criterion": marker}
    module.register(Estimator, kind, tree_specific_table=table)
    assert context["trees_classes"][Estimator] == kind
    assert context["trees_defaults"][Estimator] == {"criterion": marker}
    assert context["tree_specific_table"] is table
    assert context["trees_constructors"] == {}
    assert Estimator.fit is original_fit
    assert Estimator.__getattribute__ is original_getattribute
    assert "_mojolearn_native_public" not in vars(Estimator)


@pytest.mark.parametrize("kind", ["rf_classifier", "rf_regressor", "et_classifier",
                                  "et_regressor", "dt_classifier", "dt_regressor"])
def test_public_forest_schemas_connect_after_constructor_metadata(bridge, monkeypatch, kind):
    module, context = bridge

    class Forest:
        def __init__(self, *, numeric_mode=None):
            raise AssertionError("registration must not construct an estimator")

    metadata = ((), object(), ())
    context["trees_constructors"][Forest] = metadata
    connected = []

    def connect_public(cls, received):
        assert received["trees_constructors"][cls] is metadata
        assert received["trees_classes"][cls] == kind
        connected.append(cls)

    monkeypatch.setattr(module, "_connect_public_forest", connect_public)
    module.register(Forest, kind)
    assert connected == [Forest]
    assert context["trees_defaults"][Forest] == {"numeric_mode": None}


def test_missing_forest_constructor_metadata_refuses_before_partial_registration(bridge, monkeypatch):
    module, context = bridge

    class Forest:
        pass

    def wrong_order(*args):
        raise AssertionError("descriptor connection preceded constructor metadata")

    monkeypatch.setattr(module, "_connect_public_forest", wrong_order)
    with pytest.raises(TypeError, match="constructor metadata before registration"):
        module.register(Forest, "rf_classifier")
    assert context == dict(trees_classes={}, trees_defaults={}, trees_constructors={})
    assert "_mojolearn_native_public" not in vars(Forest)


def test_meta_registration_removes_only_public_lifecycle_decorators(bridge):
    import functools
    module, context = bridge
    context['trees_meta_classes'] = {}
    def original(self, value):
        raise AssertionError('registration must not execute a method')
    @functools.wraps(original)
    def decorated(self, value):
        raise AssertionError('registration must not execute a decorator')
    class Meta:
        fit = predict = score = decorated
        __init__ = set_params = decorated
    module.register_meta(Meta, 'voting_regressor')
    assert context['trees_meta_classes'][Meta] == 'voting_regressor'
    assert Meta.fit is Meta.predict is Meta.score is original
    assert Meta.__init__ is Meta.set_params is decorated
    assert Meta._mojolearn_native_public is True


def test_unavailable_meta_owner_does_not_partially_unwrap(bridge):
    module, context = bridge
    context['trees_unavailable'] = True
    class Meta:
        def fit(self, value):
            raise AssertionError('unavailable registration must not call fit')
    before = Meta.fit
    module.register_meta(Meta, 'voting_regressor')
    assert Meta.fit is before
    assert '_mojolearn_native_public' not in vars(Meta)
