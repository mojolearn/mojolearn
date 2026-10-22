# SPDX-License-Identifier: Apache-2.0
"""Native saved-model reconstruction preserves the existing loader's state.

The legacy base-class loader is an independent preparation oracle; only the
public Host*.load call is audited. No fit or inference is hidden in setup.
"""
import ast
import struct
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
        "ARIMA": ("mojolearn-arima-1", dict(
            meta=_i64([1, 0, 0, 0, 0, 0, 0, 1, 2, 4, 30, 1]), trend="none", method="ML",
            y=_f32([[1, 2, 3, 4], [4, 5, 6, 7]]), params=_f32([[.5, 1, 2], [.25, 2, 1]]),
            x=_f32([[.5, 1, 2], [.25, 2, 1]]), x0=_f32([[0, 0, 1], [0, 0, 1]]),
            fx=_f32([2, 3]), n_iter=np.asarray([4, 5], dtype="<i4"), retcode=np.asarray([0, 0], dtype="<i4"),
            llf=_f64([-2, -3]), aic=_f64([4, 6]), bic=_f64([5, 7]))),
        "ExponentialSmoothing": ("mojolearn-holtwinters-1", dict(
            meta=_i64([5, 2, 2, 2]), eps=_f64([.00224]), seasonal="additive",
            initialization_method="estimated", components=np.arange(18, dtype="<f4"),
            sse=_f32([1, 2]), alpha=_f32([.2, .3]), beta=_f32([.1, .2]), gamma=_f32([.2, .4]),
            n_iter=np.asarray([4, 5], dtype="<i4"), criterion=np.asarray([0, 1], dtype="<i4"))),
        "IVFIndex": ("mojolearn-ivf-flat-1", dict(
            meta=_i64([3, 2, 2, 0, 1, 2, 10, 4]), centers=_f32([[1, 2], [3, 4]]),
            center_norms=_f32([5, 25]), list_offsets=np.asarray([0, 2, 3], dtype="<i4"),
            list_indices=np.asarray([0, 2, 1], dtype="<i4"), list_data=_f32([[1, 2], [5, 6], [3, 4]]))),
        "HDBSCAN": ("mojolearn-hdbscan-2", dict(
            ints=_i64([2, 1, 1, 2, 2, 3, 2, -1, 0, 1, 1]), reals=_f64([1., .25]),
            metric="euclidean", cluster_selection_method="eom", raw_data=_f32([[1, 2], [3, 4], [5, 6]]),
            core_distances=_f32([1, 2, 3]), labels=np.asarray([0, 0, -1], dtype="<i4"),
            parents=np.asarray([3, 3, 3], dtype="<i4"), lambdas=_f32([1, 1, .5]), deaths=_f32([1, .5]),
            selected_clusters=np.asarray([3], dtype="<i4"), index_into_children=np.asarray([0, 1, 2, 3], dtype="<i4"),
            exemplar_idx=np.asarray([0], dtype="<i4"), exemplar_label_offsets=np.asarray([0, 1], dtype="<i4"))),
        "NearestNeighbors": ("mojolearn-knn-1", dict(
            meta=_i64([2, 3, 2, 0, 0, 0]), p=_f64([2]), metric="euclidean", algorithm="rbc",
            index=_f32([[1, 2], [3, 4], [5, 6]]))),
        "KNeighborsClassifier": ("mojolearn-knn-1", dict(
            meta=_i64([2, 3, 2, 0, 1, 2]), p=_f64([2]), metric="euclidean", algorithm="brute",
            weights="distance", index=_f32([[1, 2], [3, 4], [5, 6]]),
            y_cols=np.asarray([[2, 7, 2], [-3, 5, 5]], dtype="<i4"),
            classes=_i64([2, 7, -3, 5]), class_counts=_i64([2, 2]))),
        "KNeighborsRegressor": ("mojolearn-knn-1", dict(
            meta=_i64([2, 3, 2, 0, 0, 1]), p=_f64([2]), metric="euclidean", algorithm="brute",
            weights="uniform", index=_f32([[1, 2], [3, 4], [5, 6]]), y_cols=_f32([[1, 2, 3]]))),
        "RadiusNeighbors": ("mojolearn-radius-1", dict(
            meta=_i64([2, 3]), radius=_f64([1.5]), p=_f64([2]), metric="euclidean", algorithm="auto",
            index=_f32([[1, 2], [3, 4], [5, 6]]))),
        "IsolationForest": ("mojolearn-iforest-1", dict(
            ints=_i64([10, 0, 256, -1, 0, 0, 1, 7, 1, 2, 3]),
            reals=_f64([1, 1, 0, -.5]), x=_f32([[1, 2], [3, 4], [5, 6]]))),
        "SpectralClustering": ("mojolearn-spectral-1", dict(
            meta=_i64([2, 2, 2, 3, 7, 0, 3, 2]), eigen_tol=_f64([1e-5]), affinity="nearest_neighbors",
            eigenvalues=_f32([1, 2]), eigenvectors=_f32([[1, 0], [0, 1], [1, 1]]), diag=_f32([1, 1, 2]),
            centroids=_f32([[1, 0], [0, 1]]), labels=np.asarray([0, 1, 0], dtype="<i4"),
            x=_f32([[1, 2], [3, 4], [5, 6]]))),
        "GaussianProcessRegressor": ("mojolearn-gp-1", dict(
            ints=_i64([2, 0, 1, 1]), reals=_f64([2**-20, .5, 1.5, -3.25]),
            X_train=_f32([[1, 2], [3, 4]]), L=_f32([1, 0, .25, 1]), alpha=_f32([.5, -.5]),
            kinds=np.asarray([0], dtype="<i4"), params=_f64([1]), ls_len=np.asarray([1], dtype="<i4"), ls=_f64([2]))),
        "GaussianProcessClassifier": ("mojolearn.gpc.v1", dict(
            meta=_i64([2, 2, 1, 100, 1, 2]), kinds=np.asarray([0], dtype="<i4"), kparams=_f32([1]),
            ls_len=np.asarray([1], dtype="<i4"), ls=_f32([2]), x=_f32([[1, 2], [3, 4]]),
            classes=np.asarray(["cold", "warm"]), y=_f32([[0, 1]]), L=_f32([[1, 0, .25, 1]]),
            pi=_f32([[.25, .75]]), wsr=_f32([[.5, .5]]), lml=_f64([-3.25]), n_iter=_i64([5]), nb=_i64([32]))),
        "UMAP": ("mojolearn-umap-1", dict(
            meta=_i64([3, 2, 0, 7, 5, 1, 1]), controls=_f32([.1, 1., 1., 1., 1., 1.]),
            init="spectral", metric="euclidean", training=np.arange(12, dtype="<f4").reshape(4, 3),
            embedding=np.arange(8, dtype="<f4").reshape(4, 2))),
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
    if type(value).__name__ in ("_SavedKernel", "_BinaryLaplace"):
        return (type(value).__module__, type(value).__name__, _state(vars(value)))
    if type(value) is dict:
        return {key: _state(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_state(item) for item in value)
    if type(value) is float:
        return ("float", struct.pack("=d", value))
    if type(value) in (bool, int, str):
        return (type(value).__name__, value)
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


def test_host_load_refuses_custom_finalizer_before_instantiation(tmp_path):
    calls = []

    class CustomHost(host.HostPCA):
        def __del__(self):
            calls.append("finalize")

    audit = audit_native_call(CustomHost.load, _archive(tmp_path, "PCA"))
    assert isinstance(audit.exception, TypeError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report
    assert calls == []


@pytest.mark.parametrize("name", ["Embedding", "KernelDensity", "IVFIndex", "NearestNeighbors"])
def test_loaded_host_model_finalizes_without_python(tmp_path, name):
    cls = getattr(host, "Host" + name)
    obj = audit_native_call(cls.load, _archive(tmp_path, name)).assert_native()
    audit_native_call(cls.__del__, obj).assert_native()
    assert cls.__del__ is not getattr(host, name).__del__


def test_ivf_host_finalizer_releases_the_native_resident_index(tmp_path):
    model = host.HostIVFIndex.load(_archive(tmp_path, "IVFIndex"))
    state = vars(model)
    native = host._backend.load_host_module("_mojolearn_ivf_search_host")
    arrays = [state[key] for key in ("centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_")]
    token = native.ivf_flat_index_prepare([a._addr for a in arrays],
        [state["n_rows_"], state["n_features_in_"], state["n_lists_"], state["metric_code_"], 0])

    class PoisonedCachedModule:
        def ivf_flat_index_release(self, token):
            raise AssertionError("cached Python release callback was invoked")

    state["_resident"] = ((), token, PoisonedCachedModule())
    audit_native_call(type(model).__del__, model).assert_native()
    assert state["_resident"] is None
    with pytest.raises(ValueError, match="released resident"):
        native.ivf_flat_index_release(token)


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


def test_arima_exogenous_archive_state_and_shape_refusal(tmp_path):
    changes = dict(format="mojolearn-arima-2",
                   meta=_i64([1, 0, 0, 0, 0, 0, 0, 1, 2, 4, 30, 1, 1]),
                   exog=np.ones((2, 4, 1), dtype="<f4"),
                   params=np.ones((2, 4), dtype="<f4"),
                   x=np.ones((2, 4), dtype="<f4"), x0=np.ones((2, 4), dtype="<f4"))
    path = _archive(tmp_path, "ARIMA", override=changes)
    original = host.ARIMA.load.__func__(host.HostARIMA, path)
    restored = audit_native_call(host.HostARIMA.load, path).assert_native()
    assert _state(vars(restored)) == _state(vars(original))
    changes["exog"] = np.ones((2, 4), dtype="<f4")
    path = _archive(tmp_path, "ARIMA", override=changes)
    audit = audit_native_call(host.HostARIMA.load, path)
    assert isinstance(audit.exception, ValueError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("name,changes,error", [
    ("ARIMA", {"trend": "n"}, ValueError),
    ("ARIMA", {"trend": "ct"}, NotImplementedError),
    ("ARIMA", {"method": "unknown"}, ValueError),
    ("ARIMA", {"exog": np.ones((2, 4, 1), dtype="<f4")}, ValueError),
    ("ExponentialSmoothing", {"seasonal": "bad"}, ValueError),
    ("ExponentialSmoothing", {"initialization_method": "bad"}, ValueError),
    ("ExponentialSmoothing", {"components": np.ones(17, dtype="<f4")}, ValueError),
])
def test_forecast_archive_rejects_inconsistent_configuration_natively(tmp_path, name, changes, error):
    path = _archive(tmp_path, name, override=changes)
    cls = getattr(host, "Host" + name)
    with pytest.raises(error):
        getattr(host, name).load.__func__(cls, path)
    audit = audit_native_call(cls.load, path)
    assert isinstance(audit.exception, error), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("name,changes", [
    ("NearestNeighbors", {"metric": "minkowski", "p": _f64([1.25])}),
    ("NearestNeighbors", {"metric": "minkowski", "p": _f64([.5]), "algorithm": "brute"}),
    ("RadiusNeighbors", {"metric": "manhattan"}),
    ("KNeighborsClassifier", {"y_cols": np.asfortranarray([[2, 7, 2], [-3, 5, 5]], dtype="<i4")}),
])
def test_neighbor_metric_and_strided_label_archive_parity(tmp_path, name, changes):
    path = _archive(tmp_path, name, override=changes)
    cls = getattr(host, "Host" + name)
    original = getattr(host, name).load.__func__(cls, path)
    restored = audit_native_call(cls.load, path).assert_native()
    assert _state(vars(restored)) == _state(vars(original))


@pytest.mark.parametrize("name,changes", [
    ("NearestNeighbors", {"metric": "cosine", "algorithm": "rbc"}),
    ("NearestNeighbors", {"metric": "minkowski", "p": _f64([.5]), "algorithm": "rbc"}),
    ("NearestNeighbors", {"metric": "euclidean", "p": _f64([3])}),
    ("KNeighborsRegressor", {"algorithm": "rbc"}),
    ("KNeighborsRegressor", {"metric": "inner_product", "weights": "distance"}),
    ("KNeighborsClassifier", {"classes": _i64([2, 7, 5, -3])}),
    ("KNeighborsClassifier", {"class_counts": _i64([1, 3])}),
    ("RadiusNeighbors", {"radius": _f64([float("inf")])}),
])
def test_neighbor_archive_refuses_inconsistent_configuration_natively(tmp_path, name, changes):
    path = _archive(tmp_path, name, override=changes)
    cls = getattr(host, "Host" + name)
    with pytest.raises(ValueError):
        getattr(host, name).load.__func__(cls, path)
    audit = audit_native_call(cls.load, path)
    assert isinstance(audit.exception, ValueError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("sample_mode", [0, 1, 2])
@pytest.mark.parametrize("feature_mode", [0, 1])
def test_isolation_forest_saved_integer_fractional_and_auto_controls(tmp_path, sample_mode, feature_mode):
    path = _archive(tmp_path, "IsolationForest", override={
        "ints": _i64([13, sample_mode, 2, 4, feature_mode, 1, 0, 4294967295, 0, 2, 2]),
        "reals": _f64([.5, .75, .2, -.25])})
    original = host.IsolationForest.load.__func__(host.HostIsolationForest, path)
    restored = audit_native_call(host.HostIsolationForest.load, path).assert_native()
    assert _state(vars(restored)) == _state(vars(original))


@pytest.mark.parametrize("index,value", [(0, 0), (1, 3), (3, 0), (4, 2), (6, 2), (7, 4294967296), (8, 2)])
def test_isolation_forest_invalid_saved_controls_are_refused(tmp_path, index, value):
    meta = CASES["IsolationForest"][1]["ints"].copy()
    meta[index] = value
    path = _archive(tmp_path, "IsolationForest", override={"ints": meta})
    with pytest.raises(ValueError):
        host.IsolationForest.load.__func__(host.HostIsolationForest, path)
    audit = audit_native_call(host.HostIsolationForest.load, path)
    assert isinstance(audit.exception, ValueError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("name,changes", [
    ("SpectralClustering", {"affinity": "precomputed"}),
    ("SpectralClustering", {"meta": _i64([2, 2, 2, 3, -999, 1, 3, 2])}),
    ("UMAP", {"metric": "minkowski", "extras": _f64([9, 1.5, 1, 2])}),
    ("UMAP", {"init": "array", "embedding": np.asfortranarray(np.arange(8, dtype="<f4").reshape(4, 2))}),
])
def test_saved_embedding_configuration_and_layout_parity(tmp_path, name, changes):
    path = _archive(tmp_path, name, override=changes)
    cls = getattr(host, "Host" + name)
    original = getattr(host, name).load.__func__(cls, path)
    restored = audit_native_call(cls.load, path).assert_native()
    assert _state(vars(restored)) == _state(vars(original))


@pytest.mark.parametrize("name,changes,error", [
    ("SpectralClustering", {"affinity": "rbf"}, NotImplementedError),
    ("SpectralClustering", {"eigen_tol": _f64([0])}, ValueError),
    ("SpectralClustering", {"eigenvalues": _f32([1])}, ValueError),
    ("UMAP", {"metric": "minkowski", "extras": _f64([9, 3.2, 0, 0])}, ValueError),
    ("UMAP", {"extras": _f64([-1, 2, 1, 0])}, ValueError),
    ("UMAP", {"controls": _f32([.1, 1, 1, 1, 0, 1])}, ValueError),
    ("UMAP", {"meta": _i64([3, 2, 0, 7, 5, 0, 1])}, ValueError),
    ("UMAP", {"embedding": np.ones((3, 2), dtype="<f4")}, ValueError),
])
def test_saved_embedding_configuration_refusals(tmp_path, name, changes, error):
    path = _archive(tmp_path, name, override=changes)
    cls = getattr(host, "Host" + name)
    with pytest.raises(error):
        getattr(host, name).load.__func__(cls, path)
    audit = audit_native_call(cls.load, path)
    assert isinstance(audit.exception, error), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


@pytest.mark.parametrize("name,changes", [
    ("GaussianProcessRegressor", {"ints": _i64([2, 0, 0, 0]), "ls_len": np.asarray([0], dtype="<i4"), "ls": _f64([1])}),
    ("GaussianProcessClassifier", {"meta": _i64([2, 2, 3, 100, 1, 3]), "classes": np.asarray([1, 2, 3], dtype="<i8"),
        "y": _f32([[0, 1], [1, 0], [0, 0]]), "L": np.asfortranarray(_f32([[1, 0, .25, 1]] * 3)),
        "pi": _f32([[.25, .75]] * 3), "wsr": _f32([[.5, .5]] * 3),
        "lml": _f64([-1e16, 1e16, -3.25]), "n_iter": _i64([5, 6, 7]), "nb": _i64([32, 32, 32])}),
])
def test_saved_gaussian_process_nodes_and_multiclass_fit_layout(tmp_path, name, changes):
    path = _archive(tmp_path, name, override=changes)
    cls = getattr(host, "Host" + name)
    original = getattr(host, name).load.__func__(cls, path)
    restored = audit_native_call(cls.load, path).assert_native()
    assert _state(vars(restored)) == _state(vars(original))


@pytest.mark.parametrize("name,changes", [
    ("GaussianProcessRegressor", {"ls_len": np.asarray([2], dtype="<i4")}),
    ("GaussianProcessRegressor", {"L": _f32([1, 2, 3])}),
    ("GaussianProcessClassifier", {"meta": _i64([2, 2, 1, 0, 1, 2])}),
    ("GaussianProcessClassifier", {"classes": np.asarray([1, 2, 3], dtype="<i8")}),
    ("GaussianProcessClassifier", {"y": _f32([0, 1])}),
])
def test_saved_gaussian_process_schema_refusals(tmp_path, name, changes):
    path = _archive(tmp_path, name, override=changes)
    cls = getattr(host, "Host" + name)
    with pytest.raises(ValueError):
        getattr(host, name).load.__func__(cls, path)
    audit = audit_native_call(cls.load, path)
    assert isinstance(audit.exception, ValueError), audit.report()
    report = audit.report()
    assert not any(report[key] for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads")), report


def test_native_load_wrapper_is_one_binding_call():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "_NativeArchiveLoad")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef))
    assert len(function.body) == 2
    assert isinstance(function.body[-1], ast.Return)
    assert isinstance(function.body[-1].value, ast.Call)
    assert sum(isinstance(node, ast.Call) for node in ast.walk(function.body[-1])) == 1
