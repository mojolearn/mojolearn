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


def kernel_model(model, X, name, public=True):
    """Frozen 32011bb32 dense/sparse preparation and kernel entry contract."""
    from mojolearn._buffer import as_f32_dense_c
    import numpy as np
    x, _ = as_f32_dense_c(X, ndim=2, name="X")
    m, d = x.shape
    ext = model._extension()
    if name == "KernelRidge":
        n, t = model.X_fit_.shape[0], model.n_targets_
        kernel, degree, gamma, coef0, alpha = model._kernel_params
        dual = model.dual_coef_.reshape((n * t,))
        out = empty((m * t,), "<f4")
        ext.kernel_ridge_predict(
            [addr_ro(model.X_fit_, name="X_fit_"), addr_ro(dual, name="dual_coef_"),
             addr_ro(x, name="X"), addr(out, name="prediction")],
            [n, d, t, kernel, degree, gamma, coef0, alpha, model.info_, m])
        out = out if model._squeeze else out.reshape((m, t))
        return np.asarray(out).astype(np.float64) if public else out
    if name == "Nystroem":
        q = model.components_.shape[0]
        kernel, degree, gamma, coef0, seed = model._kernel_params
        out = empty((m * q,), "<f4")
        components = model.components_.reshape((q * d,))
        normalization = model.normalization_.reshape((q * q,))
        eigenvectors = model.eigenvectors_.reshape((q * q,))
        ext.nystroem_transform(
            [addr_ro(components, name="components_"), addr_ro(model.component_indices_, name="component_indices_"),
             addr_ro(normalization, name="normalization_"), addr_ro(model.eigenvalues_, name="eigenvalues_"),
             addr_ro(eigenvectors, name="eigenvectors_"), addr_ro(x, name="X"), addr(out, name="transform")],
            [q, d, kernel, degree, gamma, coef0, seed, model.sweeps_, m])
    else:
        q = model.random_weights_.shape[1]
        gamma, seed = model._params
        out = empty((m * q,), "<f4")
        weights = model.random_weights_.reshape((d * q,))
        ext.rbf_sampler_transform(
            [addr_ro(weights, name="random_weights_"), addr_ro(model.random_offset_, name="random_offset_"),
             addr_ro(x, name="X"), addr(out, name="transform")],
            [d, q, gamma, seed, model.sigma_, model.scale_, m])
    return out.reshape((m, q))


def projection(model, X, name, inverse=False):
    """Frozen 32011bb32 decomposition preparation and CPU kernel contract."""
    # Test-only oracle: use the unchanged numerical bindings while retaining
    # the original dense expansion and launch order independently of the owner.
    if hasattr(X, "toarray") and hasattr(X, "nnz") and hasattr(X, "format"):
        X = X.toarray()
    x, _ = as_f32_c(X, ndim=2, name="X")
    rows, features, components = x.shape[0], model.n_features_in_, model.n_components_
    if x.shape[1] != (components if inverse else features):
        raise ValueError("projection input width differs from fitted state")
    out = empty((rows, features if inverse else components), "<f4")
    binding = model._bind("_mojolearn_estimators")
    if name == "PCA" and model.whiten:
        model._validate_whiten_state()
        if inverse:
            binding.pca_whiten_inverse_transform(addr_ro(x, name="z"),
                addr_ro(model.components_, name="components_"), addr_ro(model.singular_values_, name="singular_values_"),
                addr_ro(model.mean_, name="mean_"), addr(out, name="out"),
                [rows, features, components, model.n_samples_])
        else:
            binding.pca_whiten_transform(addr_ro(x, name="X"), addr_ro(model.mean_, name="mean_"),
                addr_ro(model.components_, name="components_"), addr_ro(model.singular_values_, name="singular_values_"),
                addr(out, name="out"), [rows, features, components, model.n_samples_])
    elif inverse:
        binding.inverse_transform(addr_ro(x, name="z"), addr_ro(model.components_, name="components_"),
            addr_ro(model.mean_ if name == "PCA" else model.components_, name="mean_"),
            addr(out, name="out"), [rows, features, components, int(name == "PCA")])
    elif name == "PCA":
        binding.pca_transform(addr_ro(x, name="X"), addr_ro(model.mean_, name="mean_"),
            addr_ro(model.components_, name="components_"), addr(out, name="out"), [rows, features, components])
    else:
        binding.tsvd_transform(addr_ro(x, name="X"), addr_ro(model.components_, name="components_"),
            addr(out, name="out"), [rows, features, components])
    return out
