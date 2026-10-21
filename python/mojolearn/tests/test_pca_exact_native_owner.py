# SPDX-License-Identifier: Apache-2.0
"""Private exact PCA owner: complete operations and independent SVD oracles."""
import io

import numpy as np
import pytest

from mojolearn import PCA
from mojolearn._native_pca_bridge import connect
from mojolearn._verify_native_runtime import audit_native_call

ENTRY, CONTEXT = connect(PCA)


def native_fit(model, x, y=None):
    return ENTRY(model, "fit", (x, y), CONTEXT)


def native_fit_transform(model, x, y=None):
    return ENTRY(model, "fit_transform", (x, y), CONTEXT)


def native_transform(model, x):
    return ENTRY(model, "transform", (x,), CONTEXT)


def native_inverse(model, x):
    return ENTRY(model, "inverse_transform", (x,), CONTEXT)


def native_initialize(model, n_components=None, copy=True, whiten=False, svd_solver="auto", tol=0.0,
                      iterated_power="auto", n_oversamples=10, power_iteration_normalizer="auto",
                      random_state=None, numeric_mode=None):
    return ENTRY(model, "initialize", (n_components, copy, whiten, svd_solver, tol, iterated_power,
                                       n_oversamples, power_iteration_normalizer, random_state,
                                       numeric_mode), CONTEXT)


def native_get(model, deep=True):
    return ENTRY(model, "get_params", (deep,), CONTEXT)


def native_set(model, **params):
    return ENTRY(model, "set_params", (params,), CONTEXT)


def native_fitted(model):
    return ENTRY(model, "is_fitted", (), CONTEXT)


def native_save(model, path):
    return ENTRY(model, "save", (path,), CONTEXT)


def native_load(path):
    return ENTRY(PCA, "load", (path,), CONTEXT)


def clean(audit, error=None):
    report = audit.report()
    assert not report["python_frames"], report["python_frames"]
    assert not report["forbidden_native_calls"], report["forbidden_native_calls"]
    assert not report["wrapper_work"], report["wrapper_work"]
    assert not report["unobserved_existing_threads"]
    if error is None:
        return audit.assert_native()
    assert isinstance(audit.exception, error), repr(audit.exception)


def fixture(rows=21, cols=4):
    rng = np.random.default_rng(9021)
    return (rng.normal(size=(rows, cols)) * np.arange(1, cols + 1) +
            np.arange(cols) / 3).astype("f4")


def bits(value):
    return np.asarray(value, dtype="f4").view("u4")


def assert_state_equal(actual, expected):
    for name in ("components_", "mean_", "singular_values_", "explained_variance_",
                 "explained_variance_ratio_"):
        np.testing.assert_array_equal(bits(getattr(actual, name)), bits(getattr(expected, name)))
    assert np.float64(actual.noise_variance_).view("u8") == np.float64(expected.noise_variance_).view("u8")
    for name in ("n_components_", "n_features_in_", "n_samples_"):
        assert getattr(actual, name) == getattr(expected, name)


@pytest.mark.parametrize("solver,rank,rows,cols", [
    ("auto", 2, 21, 4), ("covariance_eigh", None, 21, 4),
    ("jacobi", 2, 21, 4), ("full", 2, 21, 4), ("full", 2, 4, 7),
    ("full", 1, 21, 1), ("covariance_eigh", 0.8, 21, 4),
    ("full", 0.8, 21, 4), ("covariance_eigh", "mle", 21, 4),
])
@pytest.mark.parametrize("whiten", [False, True])
def test_complete_exact_owner_matches_existing_bits_and_independent_subspace(solver, rank, rows, cols, whiten):
    x = fixture(rows, cols)
    old = PCA(rank, svd_solver=solver, whiten=whiten, numeric_mode="identical").fit(x)
    expected = np.asarray(old.transform(x)).copy()
    expected_inverse = np.asarray(old.inverse_transform(expected)).copy()
    model = PCA(rank, svd_solver=solver, whiten=whiten, numeric_mode="identical")
    actual = clean(audit_native_call(native_fit_transform, model, x))
    assert_state_equal(model, old)
    np.testing.assert_array_equal(bits(actual), bits(expected))
    np.testing.assert_array_equal(bits(clean(audit_native_call(native_inverse, model, actual))), bits(expected_inverse))
    # Independent float64 SVD validates the complete learned subspace and
    # spectrum, without assuming a particular eigenvector sign convention.
    xd = x.astype("f8")
    centered = xd - xd.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    nc = model.n_components_
    components = np.asarray(model.components_, dtype="f8")
    np.testing.assert_allclose(components.T @ components, vt[:nc].T @ vt[:nc], rtol=3e-3, atol=3e-4)
    np.testing.assert_allclose(model.mean_, xd.mean(axis=0), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(model.singular_values_, singular[:nc], rtol=4e-3, atol=4e-4)
    np.testing.assert_allclose(model.explained_variance_, singular[:nc] ** 2 / (rows - 1), rtol=5e-3, atol=5e-4)
    if whiten and nc:
        np.testing.assert_allclose(np.asarray(actual, dtype="f8").T @ np.asarray(actual, dtype="f8") / (rows - 1),
                                   np.eye(nc), rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize("layout", ["float64", "float16", "fortran", "strided", "readonly", "list", "memoryview"])
def test_exact_input_preparation_and_consumed_output_bypass_python_helpers(monkeypatch, layout):
    base = fixture()
    x = {"float64": lambda: base.astype("f8"), "float16": lambda: base.astype("f2"),
         "fortran": lambda: np.asfortranarray(base), "strided": lambda: base[::-1],
         "readonly": lambda: base.view(), "list": lambda: base.tolist(),
         "memoryview": lambda: memoryview(base)}[layout]()
    if layout == "readonly":
        x.flags.writeable = False
    old = PCA(2).fit(x)
    expected = np.asarray(old.transform(x)).copy()
    copied = old.input_copied_
    model = PCA(2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python PCA helper reached")

    from mojolearn import _backend, _buffer, decomposition
    with monkeypatch.context() as patch:
        patch.setattr(_backend, "binding", forbidden)
        patch.setattr(_buffer, "as_f32_c", forbidden)
        patch.setattr(decomposition, "as_f32_c", forbidden)
        patch.setattr(np, "empty", forbidden)
        patch.setattr(np.linalg, "svd", forbidden)
        assert clean(audit_native_call(native_fit, model, x)) is model
        actual = clean(audit_native_call(native_transform, model, x))
    assert model.input_copied_ == copied
    assert_state_equal(model, old)
    np.testing.assert_array_equal(bits(actual), bits(expected))


def test_exact_constructor_metadata_and_transactional_reset():
    model = object.__new__(PCA)
    assert clean(audit_native_call(native_initialize, model)) is None
    assert clean(audit_native_call(native_get, model)) == PCA().get_params()
    assert clean(audit_native_call(native_fitted, model)) is False
    clean(audit_native_call(native_fit, model, fixture()))
    assert clean(audit_native_call(native_fitted, model)) is True
    original = vars(model).copy()
    clean(audit_native_call(native_set, model, invalid=2), ValueError)
    assert vars(model).keys() == original.keys()
    assert vars(model)["components_"] is original["components_"]
    assert clean(audit_native_call(native_set, model)) is model
    assert clean(audit_native_call(native_set, model, whiten=True, n_components=2)) is model
    assert clean(audit_native_call(native_fitted, model)) is False
    assert clean(audit_native_call(native_get, model))["whiten"] is True


@pytest.mark.parametrize("change", ["nan_input", "bad_rank", "bad_tol", "wrong_width", "bad_state", "negative_singular"])
def test_exact_refusals_are_whole_native_operations(change):
    model = PCA(2, whiten=True).fit(fixture())
    if change == "nan_input":
        x = fixture()
        x[1, 1] = np.nan
        clean(audit_native_call(native_fit, model, x), ValueError)
    elif change == "bad_rank":
        model.n_components = 0
        clean(audit_native_call(native_fit, model, fixture()), ValueError)
    elif change == "bad_tol":
        model.tol = 1
        clean(audit_native_call(native_fit, model, fixture()), NotImplementedError)
    elif change == "wrong_width":
        clean(audit_native_call(native_transform, model, fixture(cols=3)), ValueError)
    else:
        if change == "bad_state":
            model.components_ = np.asarray(model.components_).astype("f8")
        else:
            model.singular_values_ = np.array([-1, 1], "f4")
        clean(audit_native_call(native_transform, model, fixture()), ValueError)


def test_refuses_python_protocols_before_their_hooks(monkeypatch):
    calls = []

    class Poison(np.ndarray):
        @property
        def dtype(self):
            calls.append("dtype")
            raise AssertionError("subclass property reached")

    model = PCA(2)
    clean(audit_native_call(native_fit, model, fixture().view(Poison)), TypeError)

    class Scalar:
        def __float__(self):
            calls.append("float")
            raise AssertionError("scalar protocol reached")

    model.tol = Scalar()
    clean(audit_native_call(native_fit, model, fixture()), TypeError)
    model.tol = 0.0

    def poisoned_attribute(self, name):
        calls.append(name)
        raise AssertionError("Python model attribute access reached")

    with monkeypatch.context() as patch:
        patch.setattr(PCA, "__getattribute__", poisoned_attribute)
        assert clean(audit_native_call(native_fit, model, fixture())) is model
    assert calls == []


@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("destination", ["path", "bytesio"])
def test_archive_bytes_legacy_roundtrip_and_native_continuation(tmp_path, whiten, destination):
    x = fixture()
    model = PCA(2, whiten=whiten).fit(x)
    old = io.BytesIO()
    model.save(old)
    path = tmp_path / "pca.npz"
    target = path if destination == "path" else io.BytesIO()
    assert clean(audit_native_call(native_save, model, target)) is target
    actual = path.read_bytes() if destination == "path" else target.getvalue()
    assert actual == old.getvalue()
    restored = clean(audit_native_call(native_load, io.BytesIO(actual)))
    assert_state_equal(restored, model)
    expected = np.asarray(model.transform(x)).copy()
    np.testing.assert_array_equal(bits(clean(audit_native_call(native_transform, restored, x))), bits(expected))
    assert_state_equal(PCA.load(io.BytesIO(actual)), model)


def test_archive_corruption_does_not_change_a_live_model():
    model = PCA(2).fit(fixture())
    before = np.asarray(model.components_).copy()
    payload = io.BytesIO()
    model.save(payload)
    with np.load(io.BytesIO(payload.getvalue()), allow_pickle=False) as original:
        members = {name: original[name] for name in original.files}
    members["components"] = members["components"].astype("f8")
    broken = io.BytesIO()
    np.savez(broken, **members)
    clean(audit_native_call(native_load, io.BytesIO(broken.getvalue())), ValueError)
    np.testing.assert_array_equal(bits(model.components_), bits(before))


@pytest.mark.parametrize("rows,cols,rank,oversamples,power,normalizer,seed", [
    (24, 5, 2, 3, "auto", "auto", None),
    (24, 5, 2, 0, 1, "QR", 37),
    (5, 24, 2, 2, 2, "LU", -1),
    (24, 5, 2, 2, 0, "none", 2**70 + 3),
    (24, 5, 2, 3, -1, "auto", 37),
    (72, 65, 3, 14, 1, "QR", 7),
])
def test_randomized_complete_owner_preserves_seed_power_and_consumed_outputs(
        rows, cols, rank, oversamples, power, normalizer, seed, monkeypatch):
    x = fixture(rows, cols)
    kwargs = dict(n_components=rank, svd_solver="randomized", n_oversamples=oversamples,
                  iterated_power=power, power_iteration_normalizer=normalizer,
                  random_state=seed, whiten=True, numeric_mode="identical")
    expected = PCA(**kwargs).fit(x)
    projected = np.asarray(expected.transform(x)).copy()
    restored = np.asarray(expected.inverse_transform(projected)).copy()
    model = PCA(**kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python randomized PCA orchestration reached")

    from mojolearn import _expansion_decomp
    with monkeypatch.context() as patch:
        patch.setattr(_expansion_decomp, "_randomized_decompose", forbidden)
        patch.setattr(_expansion_decomp, "_rsvd_core", forbidden)
        patch.setattr(_expansion_decomp, "_Kit", forbidden)
        actual = clean(audit_native_call(native_fit_transform, model, x))
        inverse = clean(audit_native_call(native_inverse, model, actual))
    assert_state_equal(model, expected)
    np.testing.assert_array_equal(bits(actual), bits(projected))
    np.testing.assert_array_equal(bits(inverse), bits(restored))
    # Independent reconstruction equations consume the actual fitted state;
    # approximation is allowed when the randomized range is not complete.
    xd = x.astype("f8")
    centered = xd - xd.mean(axis=0)
    components = np.asarray(model.components_, dtype="f8")
    raw_projection = centered @ components.T
    scale = np.sqrt(rows - 1) / np.asarray(model.singular_values_, dtype="f8")
    np.testing.assert_allclose(actual, raw_projection * scale, rtol=5e-3, atol=5e-4)
    np.testing.assert_allclose(inverse, raw_projection @ components + xd.mean(axis=0), rtol=5e-3, atol=8e-4)
    np.testing.assert_allclose(components @ components.T, np.eye(rank), rtol=5e-3, atol=5e-4)


def test_randomized_rank_deficiency_preserves_zero_padding():
    base = fixture(20, 2)
    x = np.column_stack((base, base[:, 0] * 2, base[:, 1] * 4, base[:, 0] * -2)).astype("f4")
    kwargs = dict(n_components=3, svd_solver="randomized", n_oversamples=2,
                  iterated_power=1, random_state=17)
    expected = PCA(**kwargs).fit(x)
    model = PCA(**kwargs)
    actual = clean(audit_native_call(native_fit_transform, model, x))
    assert_state_equal(model, expected)
    np.testing.assert_array_equal(bits(actual), bits(expected.transform(x)))
    reconstructed = clean(audit_native_call(native_inverse, model, actual))
    np.testing.assert_allclose(reconstructed, x, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("change", ["normalizer", "seed", "zero"])
def test_randomized_native_refusals_and_failed_fit_preserve_previous_state(change):
    model = PCA(2).fit(fixture())
    original = vars(model)["components_"]
    model.svd_solver = "randomized"
    x = fixture()
    error = ValueError
    if change == "normalizer":
        model.power_iteration_normalizer = "invalid"
    elif change == "seed":
        model.random_state = True
        error = TypeError
    else:
        x.fill(0)
    clean(audit_native_call(native_fit, model, x), error)
    assert vars(model)["components_"] is original
