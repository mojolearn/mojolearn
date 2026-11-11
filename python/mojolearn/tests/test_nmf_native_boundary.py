# SPDX-License-Identifier: Apache-2.0
"""NMF public storage, descriptor, option and publication boundary seams."""
import numpy as np
import pytest
import mojolearn as ml


def _data():
    return np.array([[1, 2, 1], [3, 1, 2], [1, 1, 4], [2, 3, 1], [1, 2, 2], [4, 1, 1]], dtype=np.float32)


def _model(**settings):
    options = dict(n_components=2, init="random", solver="cd", max_iter=3, random_state=7, numeric_mode="identical")
    options.update(settings)
    return ml.NMF(**options)


@pytest.mark.parametrize("method", ["fit", "fit_transform", "transform", "inverse_transform"])
@pytest.mark.parametrize("subclass", [False, True])
def test_nmf_fit_depth_is_admitted_before_numeric_hooks(method, subclass):
    called = []

    class Depth:
        def __int__(self):
            called.append("int")
            raise AssertionError("Python fitting-depth integer hook executed")

        def __index__(self):
            called.append("index")
            raise AssertionError("Python fitting-depth index hook executed")

        def __bool__(self):
            called.append("bool")
            raise AssertionError("Python fitting-depth truth hook executed")

    class NativeSubclass(ml.NMF):
        pass

    model = NativeSubclass(n_components=2) if subclass else _model()
    model._mojolearn_fitting = Depth()
    # The exact-int guard precedes fitted-state and source checks on every
    # method. No fit is needed to exercise this boundary refusal.
    with pytest.raises(TypeError, match="native fitting depth requires an exact integer"):
        getattr(model, method)(_data())
    assert called == []


@pytest.mark.parametrize("init", ["random", "nndsvd", "nndsvda", "nndsvdar", "custom"])
@pytest.mark.parametrize("solver", ["cd", "mu"])
def test_nmf_public_methods_do_not_call_legacy_python_entries(init, solver, monkeypatch):
    from mojolearn import _backend

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy NMF Python call envelope executed")

    binding = _backend.binding("_mojolearn_x_decomp", "identical")
    monkeypatch.setattr(binding, "x_decomp_nmf_fit", forbidden)
    monkeypatch.setattr(binding, "x_decomp_nmf_inverse", forbidden)
    model = _model(init=init, solver=solver)
    kwargs = {"W": np.ones((6, 2), dtype=np.float32), "H": np.ones((2, 3), dtype=np.float32)} if init == "custom" else {}
    scores = model.fit_transform(_data(), **kwargs)
    assert np.asarray(scores).shape == (6, 2)
    assert np.asarray(model.transform(_data())).shape == (6, 2)
    assert np.asarray(model.inverse_transform(scores)).shape == (6, 3)
    assert isinstance(model.n_iter_, int)
    assert isinstance(model.reconstruction_err_, float)
    assert calls == []


@pytest.mark.parametrize("field", ["beta_loss", "solver", "init", "n_components", "alpha_W", "alpha_H", "l1_ratio", "max_iter", "shuffle", "tol", "random_state"])
@pytest.mark.parametrize("kind", ["object", "int_subclass"])
def test_rebound_nmf_options_refuse_hooks_before_invocation(field, kind):
    calls = []

    class Hooks:
        def __int__(self):
            calls.append("int")
            return 2

        def __float__(self):
            calls.append("float")
            return 2.0

        def __bool__(self):
            calls.append("bool")
            return True

        def __eq__(self, other):
            calls.append("eq")
            return True

        def __hash__(self):
            calls.append("hash")
            return 2

        def __repr__(self):
            calls.append("repr")
            return "hooks"

        @property
        def __class__(self):
            calls.append("class")
            return int

    class Integer(int):
        __int__ = Hooks.__int__
        __float__ = Hooks.__float__
        __bool__ = Hooks.__bool__
        __eq__ = Hooks.__eq__
        __hash__ = Hooks.__hash__
        __repr__ = Hooks.__repr__

    model = _model()
    vars(model)[field] = Hooks() if kind == "object" else Integer(2)
    with pytest.raises(TypeError, match="native values|integer seed"):
        model.fit(_data())
    assert calls == []


@pytest.mark.parametrize("operation", ["fit", "fit_transform", "transform", "inverse_transform"])
def test_replaced_nmf_descriptor_cannot_call_python(operation, monkeypatch):
    from mojolearn._expansion_decomp import _DECOMP_CONTEXT

    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python NMF factory executed")

    model = _model().fit(_data())
    scores = model.transform(_data())
    monkeypatch.setitem(_DECOMP_CONTEXT["nmf_bindings"], "identical", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        getattr(model, operation)(scores if operation == "inverse_transform" else _data())
    assert calls == []


def test_custom_initialization_ignores_requested_count_and_keeps_dimension_refusal_before_seed():
    calls = []

    class Integer:
        def __int__(self):
            calls.append(True)
            return 2

    model = _model(init="custom")
    vars(model)["n_components"] = Integer()
    result = model.fit_transform(_data(), W=np.ones((6, 2), dtype=np.float32), H=np.ones((2, 3), dtype=np.float32))
    assert np.asarray(result).shape == (6, 2)
    assert model.n_components_ == 2
    vars(model)["random_state"] = Integer()
    with pytest.raises(ValueError, match="W must be"):
        model.fit(_data(), W=np.ones((5, 2), dtype=np.float32), H=np.ones((2, 3), dtype=np.float32))
    assert calls == []


def test_noncustom_initialization_does_not_inspect_ignored_w_and_h():
    calls = []

    class Buffer:
        def __array__(self, *args, **kwargs):
            calls.append(True)
            raise AssertionError("ignored custom input inspected")

    model = _model()
    assert model.fit(_data(), W=Buffer(), H=Buffer()) is model
    assert calls == []


@pytest.mark.parametrize("dtype", ["U", "S"])
@pytest.mark.parametrize("field,text", [("n_components", "2"), ("max_iter", "2"), ("tol", "0.01"), ("alpha_W", "0.01"), ("alpha_H", "0.01"), ("l1_ratio", "0.5")])
def test_native_zero_dimensional_text_scalar_conversions_remain_supported(dtype, field, text):
    model = _model()
    vars(model)[field] = np.array(text, dtype=dtype)
    assert model.fit(_data()) is model
    assert model.n_components_ == 2


def test_native_array_beta_preserves_unhashable_refusal():
    model = _model(beta_loss=np.array(2.0))
    with pytest.raises(TypeError, match="unhashable"):
        model.fit(_data())


def test_missing_provider_precedes_regularization_conversion(monkeypatch):
    from mojolearn._expansion_decomp import _DECOMP_CONTEXT

    calls = []

    class Float:
        def __float__(self):
            calls.append(True)
            return 0.0

    model = _model()
    vars(model)["alpha_W"] = Float()
    monkeypatch.setitem(_DECOMP_CONTEXT["nmf_bindings"], "identical", None)
    with pytest.raises(NotImplementedError, match="x_decomp_nmf_fit"):
        model.fit(_data())
    assert calls == []


@pytest.mark.parametrize("invalid", [np.nan, -1.0])
def test_failed_input_keeps_mode_but_does_not_publish_fitted_outputs(invalid):
    model = _model()
    source = _data()
    source[0, 0] = invalid
    with pytest.raises(ValueError, match="NaN|finite|Negative"):
        model.fit(source)
    assert model.numeric_mode_ == "identical"
    for name in ("components_", "n_components_", "n_iter_", "n_features_in_", "reconstruction_err_"):
        assert name not in vars(model)


def test_fixed_h_transform_preserves_fitted_public_state():
    model = _model().fit(_data())
    before = (model.n_iter_, model.reconstruction_err_, np.asarray(model.components_).copy())
    model.transform(_data())
    assert model.n_iter_ == before[0]
    assert model.reconstruction_err_ == before[1]
    np.testing.assert_array_equal(np.asarray(model.components_), before[2])


def test_inverse_observes_rebound_writable_components_in_place():
    model = _model().fit(_data())
    components = np.asarray(model.components_).copy()
    model.components_ = components
    source = np.ones((2, 2), dtype=np.float32)
    model.inverse_transform(source)
    components[0, 0] += np.float32(0.5)
    np.testing.assert_allclose(np.asarray(model.inverse_transform(source)), source @ components, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("field", ["solver", "init"])
def test_invalid_native_array_options_do_not_call_numpy_formatters(field):
    calls = []

    def forbidden(value):
        calls.append(True)
        raise AssertionError("NumPy Python formatter executed")

    model = _model()
    vars(model)[field] = np.array("unsupported")
    with np.printoptions(formatter={"all": forbidden}):
        with pytest.raises(ValueError, match="native scalar array"):
            model.fit(_data())
    assert calls == []


def test_beta_is_compared_before_extended_precision_float_conversion():
    value = np.nextafter(np.longdouble(2), np.longdouble(3))
    if float(value) != 2.0:
        pytest.skip("this platform has no wider-than-float64 long double")
    with pytest.raises(ValueError, match="general float beta"):
        _model(beta_loss=value).fit(_data())
