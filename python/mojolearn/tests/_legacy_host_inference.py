# SPDX-License-Identifier: Apache-2.0
"""Frozen 0c1d790f3 preparation/orchestration for CPU inference test oracles.

Product execution never imports this module. Existing native numerical kernels
remain the independent reference while the surrounding operation is migrated.
"""
from mojolearn._buffer import as_f32_colmajor, as_f32_c, empty, addr, addr_ro, all_finite
from mojolearn.cluster import _METRIC_NAMES


def cd_predict(model, X):
    if not hasattr(model, "coef_"):
        raise ValueError("call fit before predict")
    work_x, _ = as_f32_colmajor(X, name="X")
    if work_x.shape[1] != model.n_features_in_:
        raise ValueError("feature count differs from fit")
    out = empty((work_x.shape[0],), "<f4")
    model._solver().cd_predict(addr_ro(work_x, name="X"), addr_ro(model.coef_, name="coef_"),
        addr(out, name="predictions"), [work_x.shape[0], work_x.shape[1], float(model.intercept_)])
    return out


def kmeans(model, X, transform=False):
    centers = getattr(model, "cluster_centers_", None)
    if centers is None:
        raise RuntimeError("this estimator is not fitted yet")
    metric = _METRIC_NAMES[model.metric] if isinstance(model.metric, str) else int(model.metric)
    x, _ = as_f32_c(X, ndim=2, name="X")
    if not all_finite(x):
        raise ValueError("X contains a NaN or infinity")
    n, d = x.shape
    if d != centers.shape[1]:
        raise ValueError("feature count differs from fitted centers")
    centers, _ = as_f32_c(centers, ndim=2, name="cluster_centers_")
    k = centers.shape[0]
    out = empty((n, k) if transform else (n,), "<f4" if transform else "<i4")
    binding = model._bind("_mojolearn")
    entry = binding.kmeans_transform if transform else binding.kmeans_predict
    entry(addr_ro(x, name="X"), addr_ro(centers, name="cluster_centers_"),
          addr(out, name="result"), [n, d, k, metric])
    return out


def public_cd_predict(model, X):
    # PublicOutputsMixin's completed-result conversion, frozen independently
    # of the native prediction owner. NumPy is a testing oracle only here.
    import numpy as np
    return np.asarray(cd_predict(model, X)).astype(np.float64)
