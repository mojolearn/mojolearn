# SPDX-License-Identifier: Apache-2.0
"""KernelPCA public admission, descriptor, storage and publication seams.

Numerical kernels, centering and eigen identities live in standalone Mojo
fixtures. These cases retain only the actual Python API boundary contracts.
"""
import gc
import weakref

import numpy as np
import pytest
import mojolearn as ml


def _data():
    return np.array([[1, 2, 1], [3, 1, 2], [1, 1, 4], [2, 3, 1], [1, 2, 2], [4, 1, 1]], dtype=np.float32)


def _model(**options):
    settings = dict(n_components=2, kernel="linear", numeric_mode="identical")
    settings.update(options)
    return ml.KernelPCA(**settings)


def _hook(calls, kind="object"):
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
            return hash("linear")

        def __repr__(self):
            calls.append("repr")
            return "linear"

        @property
        def __class__(self):
            calls.append("class")
            return str

    class Integer(int):
        __int__ = Hooks.__int__
        __float__ = Hooks.__float__
        __bool__ = Hooks.__bool__
        __eq__ = Hooks.__eq__
        __hash__ = Hooks.__hash__
        __repr__ = Hooks.__repr__

    class Text(str):
        __int__ = Hooks.__int__
        __float__ = Hooks.__float__
        __bool__ = Hooks.__bool__
        __eq__ = Hooks.__eq__
        __hash__ = Hooks.__hash__
        __repr__ = Hooks.__repr__

    return Hooks() if kind == "object" else Integer(2) if kind == "int" else Text("linear")


@pytest.mark.parametrize("old_items", [False, True])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "precomputed"])
def test_kpca_public_methods_do_not_dispatch_legacy_python_entries(old_items, kernel, monkeypatch):
    from mojolearn import _backend, _expansion_neighbors, _native_decomp_bridge

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy KernelPCA Python dispatch called")

    near = _backend.binding("_mojolearn_x_neighbors", "identical")
    decomp = _backend.binding("_mojolearn_x_decomp", "identical")
    for name in ("x_neighbors_vendor", "x_neighbors_numeric_mode", "x_neighbors_kpca_resident", "xn_kernel",
                 "xn_colsum", "xn_rowsum", "xn_scale_div", "xn_kpca_center", "xn_kpca_alpha_scale",
                 "xn_kpca_transform", "xn_matmul"):
        if hasattr(near, name):
            monkeypatch.setattr(near, name, forbidden)
    monkeypatch.setattr(decomp, "x_decomp_kpca_eig", forbidden, raising=False)
    _, context = _native_decomp_bridge.connect(
        {"KernelPCA": ml.KernelPCA}, neighbors=True,
        metadata={"kpca_kernels": dict(_expansion_neighbors._KERNELS), "kpca_old_items": old_items})
    monkeypatch.setattr(_expansion_neighbors, "_KPCA_CONTEXT", context)
    x = _data()
    source = x @ x.T if kernel == "precomputed" else x
    model = _model(kernel=kernel)
    assert np.asarray(model.fit_transform(source)).shape == (6, 2)
    assert np.asarray(model.transform(source[:2])).shape == (2, 2)
    assert model.fit(source) is model
    assert calls == []


@pytest.mark.parametrize("operation,table", [
    ("fit", "kpca_neighbors_bindings"), ("fit", "kpca_info_bindings"), ("fit", "kpca_decomp_bindings"),
    ("fit_transform", "kpca_neighbors_bindings"), ("fit_transform", "kpca_info_bindings"),
    ("fit_transform", "kpca_decomp_bindings"), ("transform", "kpca_neighbors_bindings"),
    ("transform", "kpca_info_bindings"),
])
def test_kpca_descriptor_replacement_refuses_python_factory(operation, table, monkeypatch):
    from mojolearn._expansion_neighbors import _KPCA_CONTEXT

    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python descriptor factory called")

    model = _model().fit(_data())
    monkeypatch.setitem(_KPCA_CONTEXT[table], "identical", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        getattr(model, operation)(_data())
    assert calls == []


@pytest.mark.parametrize("field", ["kernel", "gamma", "n_components", "eigen_solver", "degree", "coef0",
                                   "fit_inverse_transform", "kernel_params", "remove_zero_eig"])
@pytest.mark.parametrize("kind", ["object", "int", "str"])
def test_kpca_rebound_options_refuse_hooks_before_execution(field, kind):
    calls = []
    model = _model()
    setattr(model, field, _hook(calls, kind))
    with pytest.raises(TypeError, match="native|scalar|hook|protocol|subclass"):
        model.fit(_data())
    assert calls == []


def test_kpca_callable_kernel_refusal_does_not_format_or_invoke_it():
    calls = []

    class Kernel:
        def __call__(self, *args):
            calls.append("call")
            raise AssertionError("Python kernel called")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("Python callable repr called")

    with pytest.raises(NotImplementedError, match="callable|native kernel"):
        _model(kernel=Kernel()).fit(_data())
    assert calls == []


@pytest.mark.parametrize("dtype", ["U", "S"])
def test_kpca_exact_native_text_scalars_keep_numeric_conversion(dtype):
    model = _model(n_components=np.array("2", dtype=dtype), gamma=np.array("0.5", dtype=dtype),
                   degree=np.array("3", dtype=dtype), coef0=np.array("1.0", dtype=dtype),
                   kernel=np.str_("linear"), eigen_solver=np.array("auto"))
    assert np.asarray(model.fit_transform(_data())).shape == (6, 2)
    assert model._gamma == 0.5
    assert np.asarray(model.transform(_data()[:2])).shape == (2, 2)


@pytest.mark.parametrize("value", [False, None, 0, "", [], (), {}, set(), np.array([0]), np.array([""])])
def test_kpca_native_false_options_keep_container_truth(value):
    model = _model(fit_inverse_transform=value, kernel_params=value, remove_zero_eig=value)
    assert model.fit(_data()) is model


@pytest.mark.parametrize("field", ["degree", "coef0"])
def test_kpca_precomputed_fit_preserves_ignored_options_but_query_admits_them(field):
    calls = []
    x = _data()
    gram = x @ x.T
    model = _model(kernel="precomputed")
    setattr(model, field, _hook(calls))
    assert model.fit(gram) is model
    assert calls == []
    with pytest.raises(TypeError, match="native|scalar|hook"):
        model.transform(gram[:2])
    assert calls == []


@pytest.mark.parametrize("field", ["gamma", "n_components", "degree", "coef0"])
def test_kpca_failed_numeric_conversion_keeps_publication_order(field):
    model = _model(gamma=0.5)
    setattr(model, field, "bad-number")
    with pytest.raises(ValueError):
        model.fit(_data())
    assert hasattr(model, "_gamma") == (field != "gamma")
    assert not hasattr(model, "eigenvalues_")
    assert not hasattr(model, "_fit_X")


@pytest.mark.parametrize("field,shape", [("eigenvalues_", (0,)), ("_fit_cols", (0,)), ("_fit_all", (0,)),
                                        ("eigenvectors_", (5, 2))])
def test_kpca_rebound_short_storage_is_refused_before_kernel_reads(field, shape):
    model = _model().fit(_data())
    setattr(model, field, np.ones(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="shorter|cover the fitted row"):
        model.transform(_data()[:2])


@pytest.mark.parametrize("old_items", [False, True])
def test_kpca_rebound_storage_and_borrowed_fit_source_remain_live(old_items, monkeypatch):
    from mojolearn._expansion_neighbors import _KPCA_CONTEXT

    monkeypatch.setitem(_KPCA_CONTEXT, "kpca_old_items", old_items)
    x = _data()
    reference = weakref.ref(x)
    model = _model().fit(x)
    del x
    gc.collect()
    assert reference() is not None
    assert np.shares_memory(np.asarray(model._fit_X), reference())
    query = _data()[:2]
    before = np.asarray(model.transform(query)).copy()
    vectors = np.asarray(model.eigenvectors_).copy()
    values = np.asarray(model.eigenvalues_).copy()
    model.eigenvectors_ = vectors
    model.eigenvalues_ = values
    np.testing.assert_array_equal(np.asarray(model.transform(query)), before)
    vectors *= 2
    # Metamorphic public-storage seam: mutation must reach the next query.
    np.testing.assert_allclose(np.asarray(model.transform(query)), before * 2, rtol=2e-5, atol=2e-5)


def test_kpca_kernel_table_admits_keys_before_hash_collision_comparison(monkeypatch):
    from mojolearn._expansion_neighbors import _KPCA_CONTEXT

    calls = []

    class Key:
        def __hash__(self):
            return hash("linear")

        def __eq__(self, other):
            calls.append("eq")
            raise AssertionError("Python kernel table key compared")

    # Build an independent malformed table; never poison the shared original.
    monkeypatch.setitem(_KPCA_CONTEXT, "kpca_kernels", {Key(): 0})
    with pytest.raises(TypeError, match="exact string state keys"):
        _model().fit(_data())
    assert calls == []


def test_kpca_query_does_not_acquire_unused_eigensolver(monkeypatch):
    from mojolearn._expansion_neighbors import _KPCA_CONTEXT

    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("query acquired eigensolver")

    model = _model().fit(_data())
    monkeypatch.setitem(_KPCA_CONTEXT["kpca_decomp_bindings"], "identical", forbidden)
    assert np.asarray(model.transform(_data()[:2])).shape == (2, 2)
    assert calls == []


def test_kpca_neighbor_metadata_and_execution_require_same_image(monkeypatch):
    from mojolearn._expansion_neighbors import _KPCA_CONTEXT

    # Both factories expose the same KPCA metadata ABI, but only the
    # neighboring execution image may supply its route flags and vendor.
    decomp = _KPCA_CONTEXT["kpca_decomp_bindings"]["identical"]
    monkeypatch.setitem(_KPCA_CONTEXT["kpca_info_bindings"], "identical", decomp)
    with pytest.raises(RuntimeError, match="same provider image"):
        _model().fit(_data())
