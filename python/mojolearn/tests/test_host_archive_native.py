# SPDX-License-Identifier: Apache-2.0
"""Native saved-model reconstruction preserves the existing loader's state.

The legacy base-class loader is an independent preparation oracle; only the
public Host*.load call is audited. No fit or inference is hidden in setup.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from mojolearn import Array
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call


def _f32(values):
    return np.asarray(values, dtype="<f4")


def _i64(values):
    return np.asarray(values, dtype="<i8")


def _f64(values):
    return np.asarray(values, dtype="<f8")


def _cases():
    return {
        "PCA": ("mojolearn-pca-1", dict(
            meta=_i64([2, 3, 8, 1]), svd_solver="covariance_eigh",
            components=_f32([[1, 0, 0], [0, 1, 0]]), mean=_f32([1, 2, 3]),
            singular_values=_f32([3, 2]), explained_variance=_f32([2, 1]),
            explained_variance_ratio=_f32([.6, .3]), noise_variance=_f64([.25]))),
        "TruncatedSVD": ("mojolearn-tsvd-1", dict(
            meta=_i64([2, 3]), algorithm="covariance_eigh",
            components=_f32([[1, 0, 0], [0, 1, 0]]), singular_values=_f32([3, 2]))),
        "ElasticNet": ("mojolearn-cd-1", dict(
            meta=_i64([3, 1, 17, 8]), hyper=_f64([.25, .4, .001]),
            coef=_f32([1, 2, 3]), intercept=_f64([1.5]))),
        "Lasso": ("mojolearn-cd-1", dict(
            meta=_i64([3, 0, 29, 11]), hyper=_f64([.75, 1., .003]),
            coef=_f32([-1, 0, 4]), intercept=_f64([0]))),
        "KernelRidge": ("mojolearn-kernel-ridge-1", dict(
            meta=_i64([2, 3, 2, 2, 3, 0, 0]), hyper=_f64([.1, 1., .25]),
            x_fit=_f32([[1, 2, 3], [4, 5, 6]]), dual=_f32([1, 2, 3, 4]))),
        "Nystroem": ("mojolearn-nystroem-1", dict(
            meta=_i64([2, 3, 2, 3, 4, 7]), hyper=_f64([.1, 1.]),
            components=_f32([[1, 2, 3], [4, 5, 6]]),
            component_indices=np.asarray([3, 1], dtype="<i4"),
            normalization=_f32([[1, 0], [0, 1]]), eigenvalues=_f32([1, 2]),
            eigenvectors=_f32([[0, 1], [1, 0]]))),
        "RBFSampler": ("mojolearn-rbf-sampler-1", dict(
            meta=_i64([3, 2, 4]), hyper=_f64([.1, .5, .25]),
            random_weights=_f32([[1, 2], [3, 4], [5, 6]]), random_offset=_f32([.1, .2]))),
        "MinMaxScaler": ("mojolearn-scaler-1", dict(
            meta=_i64([3, 17, 1]), feature_range=_f64([-2, 3]),
            data_min=_f32([1, 2, 3]), data_max=_f32([3, 4, 5]),
            data_range=_f32([2, 2, 2]), scale=_f32([2.5, 2.5, 2.5]), min=_f32([-4.5, -7, -9.5]))),
        "Cholesky": ("mojolearn-cholesky-1", dict(
            meta=_i64([2, 0, 16]), scalars=_f64([1.5, .0001]),
            jitter_param=_f64([-1]), L=_f32([[1, 0], [.25, 2]]))),
        "Embedding": ("mojolearn-embedding-1", dict(
            meta=_i64([3, 2, -1]), plan="auto",
            weight=_f32([[1, 2], [3, 4], [5, 6]]))),
        "KMeans": ("mojolearn-kmeans-1", dict(
            meta=_i64([2, 2, 3, 4, 0, 0, 20, 1, 4]),
            reals=_f64([.5, 1., 1., .0001, 2.]), metric="euclidean", init="k-means++",
            centers=_f32([[1, 2], [3, 4]]), labels=np.asarray([0, 1, 0], dtype="<i4"))),
        "GaussianMixture": ("mojolearn-gmm-1", dict(
            ints=_i64([2, 2, 7, 1, 20, 4]), reals=_f64([-.5, .001, .00001]),
            covariance_type="full", init_params="kmeans", weights=_f32([.5, .5]),
            means=_f32([1, 2, 3, 4]), covariances=_f32([1, 0, 0, 1] * 2),
            precisions_cholesky=_f32([1, 0, 0, 1] * 2), log_det_chol=_f32([0, 0]))),
        "KernelDensity": ("mojolearn-kde-1", dict(
            meta=_i64([2, 3, 1]), bandwidth=_f64([.75]), kernel="gaussian", metric="euclidean",
            x=_f32([[1, 2], [3, 4], [5, 6]]), weights=_f32([1, 2, 1]))),
        "DBSCAN": ("mojolearn-dbscan-1", dict(
            meta=_i64([2, 3, 2]), eps=_f64([.5]), metric="euclidean", algorithm="brute",
            components=_f32([[1, 2], [3, 4]]), core_sample_indices=np.asarray([0, 1], dtype="<i4"),
            core_labels=np.asarray([0, 1], dtype="<i4"), labels=np.asarray([0, 1, -1], dtype="<i4"))),
        "AgglomerativeClustering": ("mojolearn-agglomerative-1", dict(
            meta=_i64([2, 2, 3]), metric="euclidean", x=_f32([[1, 2], [3, 4], [5, 6]]),
            labels=np.asarray([0, 1, 1], dtype="<i4"))),
        "LogisticRegression": ("mojolearn-logistic-1", dict(
            meta=_i64([2, 1]), classes=np.asarray(["a", "b", "c"]),
            w=_f32([1, 2, 3, 4, 5, 6, .1, .2, .3]))),
    }


CASES = _cases()


def _archive(tmp_path, name, mode="identical", override=None):
    fmt, fields = CASES[name]
    fields = dict(fields, format=fmt, estimator=name, numeric_mode=mode)
    if override:
        fields.update(override)
    path = tmp_path / (name + ".npz")
    np.savez(path, **fields)
    return path


def _state(value):
    if isinstance(value, Array):
        return ("Array", value.dtype, value.shape, value.order, value.tobytes())
    if isinstance(value, np.ndarray):
        return ("Array", value.dtype.str, value.shape,
                "F" if value.flags.f_contiguous and not value.flags.c_contiguous else "C", value.tobytes())
    if type(value) is dict:
        return {key: _state(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_state(item) for item in value)
    return value


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("mode", ["identical", "fast"])
def test_public_host_load_matches_existing_loader_state(tmp_path, name, mode):
    path = _archive(tmp_path, name, mode)
    cls = getattr(host, "Host" + name)
    original = getattr(host, name).load.__func__(cls, path)
    restored = audit_native_call(cls.load, path).assert_native()
    assert type(restored) is cls
    assert _state(vars(restored)) == _state(vars(original))


@pytest.mark.parametrize("name", CASES)
def test_host_load_does_not_call_constructor_or_serialize_helpers(tmp_path, name, monkeypatch):
    path = _archive(tmp_path, name)
    cls = getattr(host, "Host" + name)

    def poisoned(*args, **kwargs):
        raise AssertionError("Python reconstruction was invoked")

    monkeypatch.setattr(cls, "__init__", poisoned)
    monkeypatch.setattr(host._serialize, "read_npz", poisoned)
    monkeypatch.setattr(host._serialize, "peek_npz", poisoned)
    monkeypatch.setattr(host._serialize, "exact", poisoned)
    audit_native_call(cls.load, path).assert_native()


@pytest.mark.parametrize("name", CASES)
def test_host_load_rejects_wrong_dtype_and_metadata_natively(tmp_path, name):
    key = "meta" if "meta" in CASES[name][1] else "ints"
    path = _archive(tmp_path, name, override={key: np.asarray([1], dtype="<f8")})
    audit = audit_native_call(getattr(host, "Host" + name).load, path)
    assert isinstance(audit.exception, ValueError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


def test_host_load_rejects_other_estimator(tmp_path):
    path = _archive(tmp_path, "PCA")
    audit = audit_native_call(host.HostTruncatedSVD.load, path)
    assert isinstance(audit.exception, ValueError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


def test_host_load_refuses_path_protocol_without_invoking_it():
    calls = []

    class PathHook:
        def __fspath__(self):
            calls.append("fspath")
            return "ignored.npz"

    audit = audit_native_call(host.HostPCA.load, PathHook())
    assert isinstance(audit.exception, TypeError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report
    assert calls == []


def test_host_load_refuses_dictionary_descriptor_before_instantiation(tmp_path):
    calls = []

    class CustomHost(host.HostPCA):
        @property
        def __dict__(self):
            calls.append("dictionary")
            return {}

    audit = audit_native_call(CustomHost.load, _archive(tmp_path, "PCA"))
    assert isinstance(audit.exception, TypeError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report
    assert calls == []


@pytest.mark.parametrize("labels", [np.asarray([2, 7], dtype="<i8"),
                                    np.asarray([.5, 1.5], dtype="<f8"),
                                    np.asarray(["left", "right"]),
                                    np.asarray(["a", "b", "c"])])
@pytest.mark.parametrize("intercept", [False, True])
def test_logistic_saved_label_types_and_coefficient_layout(tmp_path, labels, intercept):
    targets = 1 if len(labels) == 2 else len(labels)
    path = _archive(tmp_path, "LogisticRegression", override={
        "classes": labels, "meta": _i64([3, int(intercept)]),
        "w": np.arange((3 + int(intercept)) * targets, dtype="<f4")})
    original = host.LogisticRegression.load.__func__(host.HostLogisticRegression, path)
    restored = audit_native_call(host.HostLogisticRegression.load, path).assert_native()
    assert _state(vars(restored)) == _state(vars(original))
    before = restored._w.tobytes()
    restored.coef_._mv[0] = -999
    assert restored._w.tobytes() == before


def test_native_load_wrapper_is_one_binding_call():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "_NativeArchiveLoad")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef))
    assert len(function.body) == 2
    assert isinstance(function.body[-1], ast.Return)
    assert isinstance(function.body[-1].value, ast.Call)
    assert sum(isinstance(node, ast.Call) for node in ast.walk(function.body[-1])) == 1
