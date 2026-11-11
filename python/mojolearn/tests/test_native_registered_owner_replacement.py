# SPDX-License-Identifier: Apache-2.0
"""Public replacement admission for exact native owners and their retained pins.

These are Python boundary seams, not numerical or performance fixtures. GPU
resident-only cases skip explicitly on images without that capability. The
test controller schedules independent parametrized cases normally.
"""
import numpy as np
import pytest

import mojolearn as ml


def _resident_model(family):
    if family == "kmeans":
        x = np.array([[0, 0], [1, 0], [2, 0], [10, 0], [11, 0], [12, 0]], dtype=np.float32)
        model = ml.KMeans(
            n_clusters=2, init="array", init_centroids=np.array([[0, 0], [10, 0]], dtype=np.float32),
            n_init=1, max_iter=10, random_state=7, numeric_mode="identical",
        ).fit(x)
        model.predict(x)
        field, parameter, replacement = "_resident_holder", "max_iter", 5
    elif family == "kde":
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        model = ml.KernelDensity(bandwidth=0.75, numeric_mode="identical").fit(x)
        model.score_samples(x)
        field, parameter, replacement = "_resident", "bandwidth", 1.25
    else:
        x = np.array([[0, 0], [0.1, 0], [2, 2], [2.1, 2], [10, 10]], dtype=np.float32)
        model = ml.DBSCAN(
            eps=0.25, min_samples=2, algorithm="brute", prediction_data=True,
            numeric_mode="identical",
        ).fit(x)
        model.predict(x)
        field, parameter, replacement = "_resident_holder", "eps", 0.5
    owner = vars(model).get(field)
    if owner is None:
        pytest.skip(f"{family} resident owner requires a resident-capable installed image")
    assert not isinstance(owner, (dict, tuple, list))
    return model, owner, field, parameter, replacement


@pytest.mark.parametrize("family", ["kmeans", "kde", "dbscan"])
def test_full_reset_accepts_registered_native_field_owner(family):
    model, owner, field, parameter, replacement = _resident_model(family)
    assert model.set_params(**{parameter: replacement}) is model
    assert getattr(model, parameter) == replacement
    assert vars(model).get(field) is not owner
    assert vars(model).get("n_features_in_") is None


@pytest.mark.parametrize("family", ["kmeans", "kde", "dbscan"])
def test_linear_reset_traverses_nested_registered_native_owner(family):
    model, owner, _, _, _ = _resident_model(family)
    linear = ml.Ridge(alpha=1.0)
    # coef_ is one of Ridge's declared fitted fields. An unrelated arbitrary
    # user attribute would intentionally be retained by this setter.
    linear.coef_ = {"pins": [owner]}
    assert linear.set_params(alpha=2.0) is linear
    assert linear.alpha == 2.0
    assert "coef_" not in vars(linear)
    assert model is not None  # Keep the original field owner alive throughout.


@pytest.mark.parametrize("name", ["_NeuralDescriptorProvider", "_KMeansResidentOwner"])
@pytest.mark.parametrize("has_finalizer", [False, True])
def test_native_looking_python_type_is_not_registered(name, has_finalizer):
    calls = []

    def finalized(self):
        calls.append("finalized")

    namespace = {"__slots__": ()}
    if has_finalizer:
        namespace["__del__"] = finalized
    fake = type(name, (object,), namespace)()
    linear = ml.Ridge(alpha=1.0)
    linear.coef_ = fake
    try:
        with pytest.raises(TypeError, match="native.*(owner|parameter|release)"):
            linear.set_params(alpha=2.0)
        assert linear.alpha == 1.0
        assert linear.coef_ is fake
        assert calls == []
    finally:
        del linear.coef_


def _hidden_array_pin():
    model, owner, field, parameter, replacement = _resident_model("kmeans")
    from mojolearn._array import Array

    centers = model.cluster_centers_
    assert type(centers) is Array
    # Only the native owner's private centers pin now reaches this Array
    # during reset. The new public value is independently owned/admissible.
    model.cluster_centers_ = np.asarray(centers).copy()
    return model, owner, field, parameter, replacement, centers


def test_native_owner_does_not_hide_an_unsafe_retained_array_pin():
    model, owner, field, parameter, replacement, centers = _hidden_array_pin()
    calls = []

    class Finalized:
        __slots__ = ()

        def __del__(self):
            calls.append("finalized")

    unsafe = Finalized()
    old_pin, previous = centers._pin, getattr(model, parameter)
    centers._pin = unsafe
    try:
        with pytest.raises(TypeError, match="native.*(owner|parameter|release)"):
            model.set_params(**{parameter: replacement})
        assert getattr(model, parameter) == previous
        assert vars(model)[field] is owner
        assert calls == []
    finally:
        centers._pin = old_pin


def test_native_owner_pin_cycle_keeps_existing_recursion_bound():
    model, owner, field, parameter, replacement, centers = _hidden_array_pin()
    old_pin, previous = centers._pin, getattr(model, parameter)
    centers._pin = owner
    try:
        with pytest.raises(TypeError, match="owner nesting exceeds the native bound"):
            model.set_params(**{parameter: replacement})
        assert getattr(model, parameter) == previous
        assert vars(model)[field] is owner
    finally:
        centers._pin = old_pin
