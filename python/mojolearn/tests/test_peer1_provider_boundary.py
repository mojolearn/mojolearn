# SPDX-License-Identifier: Apache-2.0
"""Public neighbor/kernel controls for the 2026-10-27 provider migration.

Run against the integrated compiled bindings. These tests call the installed
public operations without unwrapping decorators or bypassing fitted checks.
The audit lane syntax-checks this file only; no runtime pass is implied.
Hook counters catch callbacks regardless of profiler visibility. The numerical
fixtures are boundary regressions, not cross-vendor identity evidence.
"""

import numpy as np
import pytest


# The test represents an application even if collected as mojolearn.tests.
__mojolearn_user_boundary__ = True


_FIT_CASES = (
    "NearestNeighbors", "KNeighborsClassifier", "KNeighborsRegressor",
    "KernelDensity", "LocalOutlierFactor", "NearestCentroid", "IVFIndex",
    "KernelRidge", "Nystroem", "RBFSampler", "PolynomialCountSketch",
)


def _data():
    # No random generators or estimator-dependent input preparation inside
    # an operation. All arrays and independent oracles belong to the test.
    x = np.array([[0., 0.], [1., 0.], [0., 2.], [3., 1.], [4., 3.], [6., 2.]],
                 dtype=np.float32)
    return x, np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)


def _model(name):
    import mojolearn

    options = {
        "NearestNeighbors": {"n_neighbors": 2},
        "KNeighborsClassifier": {"n_neighbors": 2},
        "KNeighborsRegressor": {"n_neighbors": 2},
        "KernelDensity": {"bandwidth": 1.0},
        "LocalOutlierFactor": {"n_neighbors": 2, "novelty": True},
        "NearestCentroid": {},
        "IVFIndex": {"n_lists": 2, "n_probes": 2, "n_neighbors": 2, "kmeans_n_iters": 2},
        "KernelRidge": {"alpha": 1.0, "kernel": "rbf", "gamma": 0.5},
        "Nystroem": {"n_components": 2, "random_state": 0},
        "RBFSampler": {"n_components": 2, "random_state": 0},
        "PolynomialCountSketch": {"n_components": 2, "random_state": 0},
    }
    return getattr(mojolearn, name)(numeric_mode="identical", **options[name])


def _fit(model, x):
    # Even ignored y is supplied consistently; supervised estimators need it.
    return model.fit(x, _data()[1])


@pytest.mark.parametrize("name", _FIT_CASES)
def test_fit_does_not_call_mutable_python_binding_selector(name):
    model = _model(name)
    operation = model.fit
    fields = vars(model)
    calls = []

    def replacement(*args, **kwargs):
        calls.append("_bind")
        raise AssertionError("product operation called Python _bind")

    fields["_bind"] = replacement
    try:
        try:
            result = operation(_data()[0], _data()[1])
        except TypeError:
            # Refusing changed private dispatch state also preserves the
            # no-callback contract; accepting it must return the fitted owner.
            pass
        else:
            assert result is model
        assert calls == []
    finally:
        fields.pop("_bind", None)


@pytest.mark.parametrize("name", _FIT_CASES)
@pytest.mark.parametrize("protocol", ("ndarray_shape", "array_protocol", "list_length"))
def test_fit_refuses_input_protocols_before_entering_hooks(name, protocol):
    model = _model(name)
    calls = []
    x, _ = _data()

    class ArrayHook(np.ndarray):
        @property
        def shape(self):
            calls.append("shape")
            return np.ndarray.shape.__get__(self)

    class ArrayProtocol:
        def __array__(self, *args, **kwargs):
            calls.append("array")
            return x

        @property
        def __array_interface__(self):
            calls.append("array_interface")
            return x.__array_interface__

    class ListHook(list):
        def __len__(self):
            calls.append("len")
            return list.__len__(self)

        def __iter__(self):
            calls.append("iter")
            return list.__iter__(self)

    value = {
        "ndarray_shape": lambda: x.view(ArrayHook),
        "array_protocol": ArrayProtocol,
        "list_length": lambda: ListHook(x.tolist()),
    }[protocol]()
    with pytest.raises(TypeError):
        _fit(model, value)
    assert calls == []


@pytest.mark.parametrize("name,field,kind", (
    ("NearestNeighbors", "n_neighbors", "int"),
    ("KNeighborsClassifier", "p", "int"),
    ("KernelDensity", "bandwidth", "float"),
    ("LocalOutlierFactor", "n_neighbors", "int"),
    ("IVFIndex", "n_lists", "int"),
    ("KernelRidge", "degree", "int"),
    ("KernelRidge", "gamma", "float"),
    ("KernelRidge", "kernel", "str"),
    ("Nystroem", "gamma", "float"),
    ("RBFSampler", "gamma", "float"),
    ("PolynomialCountSketch", "degree", "int"),
))
def test_fit_refuses_mutated_scalar_subclasses_without_coercion(name, field, kind):
    model = _model(name)
    operation = model.fit
    calls = []

    class IntHook(int):
        def __int__(self):
            calls.append("int")
            return 2

        def __index__(self):
            calls.append("index")
            return 2

        def __eq__(self, other):
            calls.append("eq")
            return int.__eq__(self, other)

    class FloatHook(float):
        def __float__(self):
            calls.append("float")
            return 0.5

    class StringHook(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            calls.append("eq")
            return str.__eq__(self, other)

        def __str__(self):
            calls.append("str")
            return "rbf"

    value = {"int": IntHook(2), "float": FloatHook(0.5), "str": StringHook("rbf")}[kind]
    fields = vars(model)
    previous = fields[field]
    fields[field] = value
    try:
        with pytest.raises(TypeError):
            operation(_data()[0], _data()[1])
        assert calls == []
    finally:
        fields[field] = previous


@pytest.mark.parametrize("name,method,params", (
    ("KernelRidge", "predict", "_kernel_params"),
    ("Nystroem", "transform", "_kernel_params"),
    ("RBFSampler", "transform", "_params"),
))
@pytest.mark.parametrize("mutation", ("parameter_tuple", "feature_count"))
def test_kernel_inference_refuses_mutated_fitted_metadata(name, method, params, mutation):
    model = _model(name)
    x, _ = _data()
    _fit(model, x)
    operation = getattr(model, method)
    calls = []

    class TupleHook(tuple):
        def __getitem__(self, index):
            calls.append("getitem")
            return tuple.__getitem__(self, index)

        def __len__(self):
            calls.append("len")
            return tuple.__len__(self)

    class IntHook(int):
        def __int__(self):
            calls.append("int")
            return 2

        def __eq__(self, other):
            calls.append("eq")
            return int.__eq__(self, other)

    fields = vars(model)
    field = params if mutation == "parameter_tuple" else "n_features_in_"
    previous = fields[field]
    fields[field] = TupleHook(previous) if mutation == "parameter_tuple" else IntHook(previous)
    try:
        with pytest.raises(TypeError):
            operation(x[:2])
        assert calls == []
    finally:
        fields[field] = previous


def test_neighbors_return_completed_distance_label_and_probability_results():
    import mojolearn

    x = np.array([[0.], [2.], [5.]], dtype=np.float32)
    query = np.array([[0.5], [4.5]], dtype=np.float32)
    neighbors = mojolearn.NearestNeighbors(n_neighbors=2, metric="euclidean", numeric_mode="identical").fit(x)
    distances, indices = neighbors.kneighbors(query)
    np.testing.assert_array_equal(np.asarray(indices), [[0, 1], [2, 1]])
    np.testing.assert_allclose(np.asarray(distances), [[0.5, 1.5], [0.5, 2.5]], rtol=0, atol=1e-6)
    classifier = mojolearn.KNeighborsClassifier(n_neighbors=1, numeric_mode="identical")
    assert classifier.fit(x, np.array(["cold", "warm", "hot"])) is classifier
    np.testing.assert_array_equal(np.asarray(classifier.predict(query)), ["cold", "hot"])
    np.testing.assert_array_equal(np.asarray(classifier.predict_proba(query)), [[1., 0., 0.], [0., 1., 0.]])
    assert classifier.score(query, ["cold", "hot"]) == 1.0


def test_precomputed_kernel_ridge_returns_multioutput_result():
    from mojolearn.kernel_ridge import KernelRidge

    gram = np.diag(np.array([1., 2., 3.], dtype=np.float32))
    targets = np.array([[2., 4.], [3., 6.], [4., 8.]], dtype=np.float32)
    model = KernelRidge(alpha=1.0, kernel="precomputed", numeric_mode="identical")
    assert model.fit(gram, targets) is model
    result = np.asarray(model.predict(gram))
    # Public regression predictions preserve the existing float64 interface.
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [[1., 2.], [2., 4.], [3., 6.]], rtol=2e-6, atol=2e-6)


def test_lof_public_score_samples_and_decision_have_matching_offset():
    from mojolearn.neighbors import LocalOutlierFactor

    x, _ = _data()
    model = LocalOutlierFactor(n_neighbors=2, novelty=True, numeric_mode="identical").fit(x)
    samples = np.asarray(model.score_samples(x[:2]))
    decision = np.asarray(model.decision_function(x[:2]))
    assert samples.shape == decision.shape == (2,)
    assert samples.dtype == decision.dtype == np.float64
    assert np.isfinite(samples).all()
    np.testing.assert_allclose(decision, samples - model.offset_, rtol=2e-6, atol=2e-6)


def test_centroid_public_log_probabilities_are_finished_results():
    from mojolearn.neighbors import NearestCentroid

    x, y = _data()
    model = NearestCentroid(numeric_mode="identical").fit(x, y)
    proba = np.asarray(model.predict_proba(x[:2]))
    log_proba = np.asarray(model.predict_log_proba(x[:2]))
    assert proba.shape == log_proba.shape == (2, 2)
    np.testing.assert_allclose(np.exp(log_proba), proba, rtol=3e-6, atol=1e-7)


def test_kernel_aliases_resolve_to_the_same_public_classes():
    import mojolearn
    from mojolearn import kernel_approximation, kernel_methods, kernel_ridge

    assert kernel_ridge.KernelRidge is kernel_methods.KernelRidge is mojolearn.KernelRidge
    assert kernel_approximation.Nystroem is kernel_methods.Nystroem is mojolearn.Nystroem
    assert kernel_approximation.RBFSampler is kernel_methods.RBFSampler is mojolearn.RBFSampler


@pytest.mark.parametrize("name,method", (
    ("LocalOutlierFactor", "score_samples"),
    ("NearestCentroid", "predict_log_proba"),
))
def test_new_public_results_do_not_reenter_python_compatibility(name, method, monkeypatch):
    from mojolearn import _mode, _sk_validate

    x, _ = _data()
    model = _model(name)
    _fit(model, x)
    operation = getattr(model, method)

    def forbidden(*args, **kwargs):
        raise AssertionError("public operation entered Python validation/output compatibility")

    with monkeypatch.context() as patch:
        patch.setattr(_sk_validate, "before_call", forbidden)
        patch.setattr(_sk_validate, "after_call", forbidden)
        patch.setattr(_mode, "_public_result", forbidden)
        patch.setattr(_mode, "_ndarray", forbidden)
        result = operation(x[:2])
    assert type(result) is np.ndarray
    assert result.shape[0] == 2


def test_ivf_extend_accepts_strided_rows_and_searches_appended_ids():
    x, _ = _data()
    model = _model("IVFIndex")
    assert _fit(model, x) is model
    backing = np.array([[9., -100., 1., -100.], [11., -100., 4., -100.]], dtype=np.float64)
    appended = backing[:, ::2]
    assert not appended.flags.c_contiguous
    assert model.extend(appended) is model
    assert model.n_rows_ == x.shape[0] + appended.shape[0]
    distances, indices = model.search(appended)
    np.testing.assert_array_equal(np.asarray(indices)[:, 0], [6, 7])
    np.testing.assert_allclose(np.asarray(distances)[:, 0], [0., 0.], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(model.n_candidates_), [8, 8])
    assert np.asarray(model.extend_labels_).shape == (2,)


@pytest.mark.parametrize("mutation", ("wrong_dimensions", "fitted_array", "fitted_scalar"))
def test_ivf_extend_admits_before_mutation_and_leaves_fit_unchanged_on_refusal(mutation):
    model = _model("IVFIndex")
    x, _ = _data()
    _fit(model, x)
    # Materialize the lazy stored data before the operation being tested.
    # This makes fitted-storage identity observable on resident and host paths.
    _ = model.list_data_
    fields = vars(model)
    names = ("centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_")
    previous = {name: fields[name] for name in names}
    copies = {name: np.asarray(value).copy() for name, value in previous.items()}
    n_rows = fields["n_rows_"]
    operation = model.extend
    calls = []

    class ArrayHook(np.ndarray):
        @property
        def shape(self):
            calls.append("shape")
            return np.ndarray.shape.__get__(self)

    class IntHook(int):
        def __int__(self):
            calls.append("int")
            return n_rows

        def __index__(self):
            calls.append("index")
            return n_rows

    value = x[:1]
    if mutation == "wrong_dimensions":
        value = np.ones((1, 3), dtype=np.float32)
    elif mutation == "fitted_array":
        fields["centers_"] = copies["centers_"].view(ArrayHook)
    else:
        fields["n_rows_"] = IntHook(n_rows)
    error = ValueError if mutation == "wrong_dimensions" else TypeError
    try:
        with pytest.raises(error):
            operation(value)
        assert calls == []
        for name in names:
            if mutation == "fitted_array" and name == "centers_":
                continue
            assert fields[name] is previous[name]
            np.testing.assert_array_equal(np.asarray(fields[name]), copies[name])
        if mutation != "fitted_scalar":
            assert fields["n_rows_"] == n_rows
    finally:
        fields.update(previous)
        fields["n_rows_"] = n_rows


def test_tsne_source_leaves_both_binding_routes_for_native_admission():
    # Parent-requested source-only guard for the two TSNE wrappers in the
    # shared ANN Python file. It does not import or execute TSNE and does not
    # qualify its native implementation or installed wrapper lifecycle.
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).parents[1] / "_expansion_ann.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TSNE")
    for name in ("fit", "fit_transform"):
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        statements = method.body[1:] if ast.get_docstring(method) else method.body
        assert len(statements) == 1 and isinstance(statements[0], ast.Return)
        call = statements[0].value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_NATIVE"
        arguments = call.args[2]
        assert isinstance(arguments, ast.Tuple) and len(arguments.elts) == 3
        assert all(isinstance(arg, ast.Constant) and arg.value is None for arg in arguments.elts[:2])
        assert isinstance(arguments.elts[2], ast.Name) and arguments.elts[2].id == "X"


@pytest.mark.parametrize("name", ("Nystroem", "RBFSampler"))
@pytest.mark.parametrize("bad_y", ("length", "nonfinite"))
def test_kernel_feature_fit_keeps_ignored_target_validation_and_cleans_fit_depth(name, bad_y):
    model = _model(name)
    x, y = _data()
    target = y[:2] if bad_y == "length" else y.astype(np.float64)
    if bad_y == "nonfinite":
        target[0] = np.nan
    with pytest.raises(ValueError):
        model.fit(x, target)
    assert not vars(model).get("_mojolearn_fitting", 0)
    assert model.fit(x, y) is model
    assert vars(model).get("_mojolearn_fitted") is True
    transformed = model.transform(x[:2])
    assert type(transformed) is np.ndarray
    assert transformed.shape == (2, 2)
    assert transformed.dtype == np.float32


def test_kernel_ridge_column_target_warns_and_returns_flat_public_prediction():
    model = _model("KernelRidge")
    x, y = _data()
    with pytest.warns(UserWarning, match="column-vector y"):
        assert model.fit(x, y.astype(np.float32)[:, None]) is model
    prediction = model.predict(x[:2])
    assert type(prediction) is np.ndarray
    assert prediction.shape == (2,)
    assert prediction.dtype == np.float64
    assert not vars(model).get("_mojolearn_fitting", 0)


@pytest.mark.parametrize("name", ("IVFPQIndex", "IVFSQIndex", "IVFRaBitQIndex", "CagraIndex"))
def test_ann_search_returns_stable_completed_results_with_strided_filters(name):
    import mojolearn

    # Small ordinary shape; five features exercises PQ padding and RaBitQ's
    # power-of-two rotation width. This is a correctness fixture, never a race.
    x = np.random.default_rng(108).normal(size=(12, 5)).astype(np.float32)
    options = {"n_lists": 2, "n_probes": 2, "n_neighbors": 2,
               "kmeans_n_iters": 2, "random_state": 0}
    if name == "IVFPQIndex":
        options.update(pq_dim=2, pq_bits=1, pq_kmeans_n_iters=2)
    elif name == "CagraIndex":
        options = {"graph_degree": 2, "intermediate_graph_degree": 4,
                   "n_neighbors": 2, "itopk_size": 4, "search_width": 1,
                   "random_state": 0}
    model = getattr(mojolearn, name)(numeric_mode="identical", **options)
    assert model.fit(x) is model
    query = x[::3]
    first = model.search(query)
    repeated = model.search(query)
    for a, b in zip(first, repeated):
        assert type(a) is type(b) is np.ndarray
        assert a.shape == b.shape == (4, 2)
        np.testing.assert_array_equal(a, b)
    assert first[0].dtype == np.float32 and first[1].dtype == np.int32
    backing = np.ones(24, dtype=np.bool_)
    backing[::4] = False
    mask = backing[::2]
    assert not mask.flags.c_contiguous
    distances, indices = model.search(query, filter=mask)
    assert type(distances) is type(indices) is np.ndarray
    valid = indices >= 0
    assert np.all(indices[valid] < x.shape[0])
    assert np.all(mask[indices[valid]])
    assert np.isinf(distances[~valid]).all()
    if name != "CagraIndex":
        np.testing.assert_array_equal(np.asarray(model.n_candidates_), np.full(4, 6))
    if name == "IVFPQIndex":
        # Exercise the public lazy export after searches reused the resident
        # fit. Materializing codes must retain the search result.
        assert np.asarray(model.codes_).shape == (12, 2)
        exported = model.search(query)
        for a, b in zip(first, exported):
            np.testing.assert_array_equal(a, b)
