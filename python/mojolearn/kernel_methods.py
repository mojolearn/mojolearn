# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Kernel methods on the GPU: `KernelRidge`, `Nystroem`, `RBFSampler`
(workstream D, 2026-09-14).

The Python door of `kernel_methods/estimator.mojo` through
`bindings/_mojolearn_kernel_methods.mojo`. Every parameter means what the
scikit-learn parameter of that name means, or is named differently
(`kernel_methods/README.md` carries the mapping); every refusal is raised
by name, here for the shapes and the spellings, on the Mojo host for the
values (`km_validate_matrix`, `km_validate_kernel_params`, the alpha,
gamma, degree and n_components refusals, DEVIATION 1686).

THE MODEL IS ITS ARRAYS. Fit returns every field `kernel_methods/
estimator.mojo`'s structs carry, so a fitted object can be read, saved and
hashed, and predict/transform send the same arrays back down to rebuild
the struct. `Nystroem` keeps its eigenvalues, eigenvectors and Jacobi
sweep count, which scikit-learn discards, because the lane's identity
argument is which of those moved (the estimator's header, point 3).

The kernels are `svm/impl/svm_parameter.mojo`'s codes plus the lane's
laplacian: linear, poly, rbf, sigmoid, laplacian. 'precomputed' is refused
by name here and on the Mojo host. `gamma=None` is scikit-learn's
`1 / n_features`, one Python division, correctly rounded on every host.

NO SPEED CLAIM. The lane has no published number and this door adds none.
"""
from . import _backend, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

#: The saved-model formats (lane/inference-linear-svm, 2026-09-15):
#: `mojolearn.host_model(path)` predicts or transforms from each on a CPU.
_KERNEL_RIDGE_FORMAT = "mojolearn-kernel-ridge-1"
_NYSTROEM_FORMAT = "mojolearn-nystroem-1"
_RBF_SAMPLER_FORMAT = "mojolearn-rbf-sampler-1"


def _km_header(arrays, path, cls, meta_fields, hyper_fields):
    """The estimator, `meta` `<i8` and `hyper` `<f8` checks the three loads
    share; returns `(meta, hyper)`."""
    from .linear_model import _check_saved_by
    _check_saved_by(arrays, path, cls)
    meta = _serialize.exact(arrays, "meta", "<i8")
    if meta.size != meta_fields:
        raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, {meta_fields} are needed")
    hyper = _serialize.exact(arrays, "hyper", "<f8")
    if hyper.size != hyper_fields:
        raise ValueError(f"mojolearn: {path!r} hyper holds {hyper.size} fields, {hyper_fields} are needed")
    return meta, hyper


def _km_array(arrays, name, dtype, shape, path):
    value = _serialize.exact(arrays, name, dtype)
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"mojolearn: {path!r} {name} shape {tuple(value.shape)} is not {tuple(shape)}")
    return value


def _km_write(est, path, fmt, arrays):
    from .linear_model import _saved_mode
    arrays.update(format=fmt, estimator=type(est).__name__, numeric_mode=_saved_mode(est))
    return _serialize.write_npz(path, arrays)

#: `checks/numerics.mojo` codes, duplicated on purpose (the GP's reason):
#: the read-back must not share a table with the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

#: `kernel_methods/checks/kernel_matrix.mojo`'s KM_KERNEL_* codes:
#: `svm/impl/svm_parameter.mojo`'s enumeration (0 to 4) plus the lane's
#: laplacian at 5. A renumbering on either side is a WRONG KERNEL, not a
#: failure, which is why the surface test plants a fixture whose answer
#: moves if the codes move.
KERNEL_LINEAR = 0
KERNEL_POLYNOMIAL = 1
KERNEL_RBF = 2
KERNEL_SIGMOID = 3
KERNEL_PRECOMPUTED = 4
KERNEL_LAPLACIAN = 5

_KERNELS = {
    "linear": KERNEL_LINEAR,
    "poly": KERNEL_POLYNOMIAL,
    "polynomial": KERNEL_POLYNOMIAL,
    "rbf": KERNEL_RBF,
    "sigmoid": KERNEL_SIGMOID,
    "laplacian": KERNEL_LAPLACIAN,
}

#: Kernels that read gamma. Under 'linear' it is ignored on both sides,
#: exactly as scikit-learn's `pairwise_kernels` ignores it.
_NEEDS_GAMMA = frozenset({KERNEL_POLYNOMIAL, KERNEL_RBF, KERNEL_SIGMOID, KERNEL_LAPLACIAN})


def _kernel_code(kernel, where):
    if kernel == "precomputed":
        raise ValueError(
            f"mojolearn {where}: kernel='precomputed' is refused by name; the "
            "lane forms the kernel matrix on the device and takes no matrix "
            "from the caller (kernel_methods/NOT_IMPLEMENTED.tsv)"
        )
    if isinstance(kernel, str):
        if kernel not in _KERNELS:
            raise ValueError(
                f"mojolearn {where}: kernel must be one of "
                f"{sorted(_KERNELS)}, got {kernel!r}"
            )
        return _KERNELS[kernel]
    if isinstance(kernel, bool) or not isinstance(kernel, int):
        raise TypeError(
            f"mojolearn {where}: kernel must be a name or a KM_KERNEL code, "
            f"got {type(kernel).__name__}"
        )
    return int(kernel)


def _real(v, name, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(
            f"mojolearn {where}: {name} must be a real number, got "
            f"{type(v).__name__}"
        )
    return float(v)


def _gamma_for(gamma, n_features, where):
    """scikit-learn's `gamma=None` is `1 / n_features`."""
    if gamma is None:
        return 1.0 / float(n_features)
    return _real(gamma, "gamma", where)


class _KernelMethodBase(NumericModeMixin):
    _BINDING = "_mojolearn_kernel_methods"
    _WHERE = "kernel method"

    def _extension(self):
        """The binding for THIS estimator's tier, with the binary's own
        compile-time answer cross-checked against it (the GP's pattern)."""
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "kernel_methods_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn {self._WHERE}: numeric_mode={want!r} was requested "
                    f"but {mod.__name__} reports compile-time mode code {got}; "
                    "the binary and the directory it sits in disagree, rebuild "
                    "it with bash bindings/build_kernel_methods.sh"
                )
        return mod


class KernelRidge(_KernelMethodBase):
    """`sklearn.kernel_ridge.KernelRidge` (and cuML's) on the GPU.

    Parameters
    ----------
    alpha : float, default 1.0
        The ridge. This IS the Cholesky ridge (DEVIATION 1660); negative
        or NaN is refused by name on the Mojo host, and zero is accepted
        (inside scikit-learn's own interval). A kernel matrix that does
        not factor at the given alpha is refused by name, never solved by
        a least-squares fallback (DEVIATION 1662).
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'laplacian'}, default 'linear'
    gamma : float or None, default None
        None is scikit-learn's `1 / n_features`. Must be positive where
        the kernel reads it (refused by name on the Mojo host).
    degree : int, default 3
    coef0 : float, default 1.0

    Attributes
    ----------
    dual_coef_ : Array (n_samples, n_targets) float32
    X_fit_ : Array (n_samples, n_features) float32
        Predict needs it; a kernel method has no finite parameter vector.
    info_ : int
        LAPACK's info from the factorization; 0 on every model that
        comes back (the fit refuses anything else by name).
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

    _WHERE = "KernelRidge"

    def __init__(self, alpha=1.0, kernel="linear", gamma=None, degree=3, coef0=1.0):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _kp(self, n_features):
        k = _kernel_code(self.kernel, self._WHERE)
        if isinstance(self.degree, bool) or not isinstance(self.degree, int):
            raise TypeError(
                f"mojolearn {self._WHERE}: degree must be an int, got "
                f"{type(self.degree).__name__}"
            )
        return k, int(self.degree), _gamma_for(self.gamma, n_features, self._WHERE), _real(self.coef0, "coef0", self._WHERE)

    def fit(self, X, y):
        """Form `K`, ridge it, factor it, solve it (`kernel_ridge_fit_host`).
        `y` is `(n,)` or `(n, n_targets)`."""
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = x.shape
        yy, _ = as_f32_c(y, ndim=None, name="y")
        if yy.ndim == 1:
            t, squeeze = 1, True
        elif yy.ndim == 2:
            t, squeeze = yy.shape[1], False
        else:
            raise ValueError(f"mojolearn {self._WHERE}: y must be 1-D or 2-D, got {yy.ndim}-D")
        if yy.shape[0] != n:
            raise ValueError(
                f"mojolearn {self._WHERE}: y has {yy.shape[0]} rows, X has {n}"
            )
        kernel, degree, gamma, coef0 = self._kp(d)
        alpha = _real(self.alpha, "alpha", self._WHERE)
        flat_y = yy.reshape((n * t,))
        dual = empty((n * t,), "<f4")
        scalars = empty((1,), "<f8")
        info = self._extension().kernel_ridge_fit(
            # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_fit_binding.
            # x, y, dual_out, scalars_out
            [addr_ro(x, name="X"), addr_ro(flat_y, name="y"), addr(dual, name="dual_coef_"), addr(scalars, name="scalars")],
            # n, d, t, kernel, degree, gamma, coef0, alpha
            [n, d, t, kernel, degree, gamma, coef0, alpha],
        )
        self.X_fit_ = x
        self.n_features_in_ = d
        self.n_targets_ = t
        self._squeeze = squeeze
        self.dual_coef_ = dual if squeeze else dual.reshape((n, t))
        self.info_ = int(info)
        self._kernel_params = (kernel, degree, gamma, coef0, alpha)
        return self

    def predict(self, X):
        """`K(X, X_fit_) . dual_coef_` through the identical GEMM
        (DEVIATION 1680). Returns float32, `(q,)` or `(q, n_targets)` as
        `y` was."""
        if not hasattr(self, "dual_coef_"):
            raise ValueError(f"mojolearn {self._WHERE}: call fit before predict")
        xq, _ = as_f32_c(X, ndim=2, name="X")
        q, d = xq.shape
        if d != self.n_features_in_:
            raise ValueError(
                f"mojolearn {self._WHERE}: X has {d} features, the fit had {self.n_features_in_}"
            )
        n, t = self.X_fit_.shape[0], self.n_targets_
        kernel, degree, gamma, coef0, alpha = self._kernel_params
        flat_dual = self.dual_coef_.reshape((n * t,))
        out = empty((q * t,), "<f4")
        self._extension().kernel_ridge_predict(
            # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::kernel_ridge_predict_binding.
            # x_fit, dual, x_new, out
            [addr_ro(self.X_fit_, name="X_fit_"), addr_ro(flat_dual, name="dual_coef_"), addr_ro(xq, name="X"), addr(out, name="predict")],
            # n, d, t, kernel, degree, gamma, coef0, alpha, info, q
            [n, d, t, kernel, degree, gamma, coef0, alpha, self.info_, q],
        )
        if self._squeeze:
            return out
        return out.reshape((q, t))

    def save(self, path):
        """Write the fitted model to `path` as an npz: `X_fit_` and
        `dual_coef_` (flat, `n * n_targets`) as fitted, `hyper` `<f8`
        [gamma, coef0, alpha] as the fit resolved them, `meta` `<i8` [n,
        n_features, n_targets, kernel code, degree, info, 1-D target] and
        the numeric mode. `mojolearn.host_model(path)` predicts from it on a
        CPU with no GPU."""
        if not hasattr(self, "dual_coef_"):
            raise RuntimeError("this estimator is not fitted yet")
        kernel, degree, gamma, coef0, alpha = self._kernel_params
        n, t = self.X_fit_.shape[0], self.n_targets_
        return _km_write(self, path, _KERNEL_RIDGE_FORMAT, {
            "x_fit": self.X_fit_,
            "dual": self.dual_coef_.reshape((n * t,)),
            "hyper": Array.from_list([float(gamma), float(coef0), float(alpha)], "<f8"),
            "meta": Array.from_list([int(n), int(self.n_features_in_), int(t), int(kernel), int(degree),
                                     int(self.info_), 1 if self._squeeze else 0], "<i8"),
        })

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts; it does not
        refit."""
        from .linear_model import _restore_mode
        arrays = _serialize.read_npz(path, _KERNEL_RIDGE_FORMAT)
        meta, hyper = _km_header(arrays, path, cls, 7, 3)
        n, d, t, kernel, degree, info, squeeze = (int(v) for v in meta.tolist())
        gamma, coef0, alpha = (float(v) for v in hyper.tolist())
        obj = cls(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        _restore_mode(obj, arrays)
        obj.X_fit_ = _km_array(arrays, "x_fit", "<f4", (n, d), path)
        dual = _km_array(arrays, "dual", "<f4", (n * t,), path)
        obj.n_features_in_ = d
        obj.n_targets_ = t
        obj._squeeze = bool(squeeze)
        obj.dual_coef_ = dual if squeeze else dual.reshape((n, t))
        obj.info_ = info
        obj._kernel_params = (kernel, degree, gamma, coef0, alpha)
        return obj


class Nystroem(_KernelMethodBase):
    """`sklearn.kernel_approximation.Nystroem` on the GPU.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'laplacian'}, default 'rbf'
    gamma : float or None, default None
        None is scikit-learn's `1 / n_features`.
    degree : int, default 3
    coef0 : float, default 1.0
    n_components : int, default 100
        Refused by name on the Mojo host when it exceeds `n_samples` or is
        not positive.
    random_state : int, default 0
        The seed of `km_basis_indices`, the device permutation.

    Attributes
    ----------
    components_ : Array (n_components, n_features) float32
    component_indices_ : Array (n_components,) int32
    normalization_ : Array (n_components, n_components) float32
        NOT bitwise symmetric (DEVIATION 1674); transform uses its
        transpose, scikit-learn's arm.
    eigenvalues_ : Array (n_components,) float32
        The singular values of the basis kernel, `|lambda|` of its
        eigenvalues, descending, clipped at 1e-12 (DEVIATION 1670). A
        numerically negative eigenvalue keeps its sign in the right factor
        of `normalization_`, as scikit-learn's SVD does.
    eigenvectors_ : Array (n_components, n_components) float32
    sweeps_ : int
        Jacobi sweeps the device eigensolver ran. Part of the model.
    """

    _WHERE = "Nystroem"

    def __init__(self, kernel="rbf", gamma=None, degree=3, coef0=1.0, n_components=100, random_state=0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = x.shape
        kernel = _kernel_code(self.kernel, self._WHERE)
        if isinstance(self.degree, bool) or not isinstance(self.degree, int):
            raise TypeError(f"mojolearn {self._WHERE}: degree must be an int")
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, int):
            raise TypeError(f"mojolearn {self._WHERE}: n_components must be an int")
        q = int(self.n_components)
        if q < 1:
            raise ValueError(f"mojolearn {self._WHERE}: n_components must be positive, got {q}")
        gamma = _gamma_for(self.gamma, d, self._WHERE)
        coef0 = _real(self.coef0, "coef0", self._WHERE)
        seed = int(self.random_state)
        components = empty((q * d,), "<f4")
        indices = empty((q,), "<i4")
        normalization = empty((q * q,), "<f4")
        eigenvalues = empty((q,), "<f4")
        eigenvectors = empty((q * q,), "<f4")
        scalars = empty((1,), "<f8")
        sweeps = self._extension().nystroem_fit(
            # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_fit_binding.
            # x, components_out, indices_out, normalization_out, eigenvalues_out, eigenvectors_out, scalars_out
            [addr_ro(x, name="X"), addr(components, name="components_"), addr(indices, name="component_indices_"),
             addr(normalization, name="normalization_"), addr(eigenvalues, name="eigenvalues_"),
             addr(eigenvectors, name="eigenvectors_"), addr(scalars, name="scalars")],
            # n, d, kernel, degree, gamma, coef0, q, seed
            [n, d, kernel, int(self.degree), gamma, coef0, q, seed],
        )
        self.n_features_in_ = d
        self.components_ = components.reshape((q, d))
        self.component_indices_ = indices
        self.normalization_ = normalization.reshape((q, q))
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors.reshape((q, q))
        self.sweeps_ = int(sweeps)
        self._kernel_params = (kernel, int(self.degree), gamma, coef0, seed)
        return self

    def transform(self, X):
        """`K(X, components_) @ normalization_.T`, float32 `(m, n_components)`."""
        if not hasattr(self, "components_"):
            raise ValueError(f"mojolearn {self._WHERE}: call fit before transform")
        x, _ = as_f32_c(X, ndim=2, name="X")
        m, d = x.shape
        if d != self.n_features_in_:
            raise ValueError(
                f"mojolearn {self._WHERE}: X has {d} features, the fit had {self.n_features_in_}"
            )
        q = self.components_.shape[0]
        kernel, degree, gamma, coef0, seed = self._kernel_params
        out = empty((m * q,), "<f4")
        flat_c = self.components_.reshape((q * d,))
        flat_n = self.normalization_.reshape((q * q,))
        flat_e = self.eigenvectors_.reshape((q * q,))
        self._extension().nystroem_transform(
            # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::nystroem_transform_binding.
            # components, indices, normalization, eigenvalues, eigenvectors, x, out
            [addr_ro(flat_c, name="components_"), addr_ro(self.component_indices_, name="component_indices_"),
             addr_ro(flat_n, name="normalization_"), addr_ro(self.eigenvalues_, name="eigenvalues_"),
             addr_ro(flat_e, name="eigenvectors_"), addr_ro(x, name="X"), addr(out, name="transform")],
            # q, d, kernel, degree, gamma, coef0, seed, sweeps, m
            [q, d, kernel, degree, gamma, coef0, seed, self.sweeps_, m],
        )
        return out.reshape((m, q))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def save(self, path):
        """Write the fitted model to `path` as an npz: every model array as
        fitted, `hyper` `<f8` [gamma, coef0] as the fit resolved them, `meta`
        `<i8` [n_components, n_features, kernel code, degree, seed, sweeps]
        and the numeric mode. `mojolearn.host_model(path)` transforms from it
        on a CPU with no GPU."""
        if not hasattr(self, "components_"):
            raise RuntimeError("this estimator is not fitted yet")
        kernel, degree, gamma, coef0, seed = self._kernel_params
        q, d = self.components_.shape
        return _km_write(self, path, _NYSTROEM_FORMAT, {
            "components": self.components_,
            "component_indices": self.component_indices_,
            "normalization": self.normalization_,
            "eigenvalues": self.eigenvalues_,
            "eigenvectors": self.eigenvectors_,
            "hyper": Array.from_list([float(gamma), float(coef0)], "<f8"),
            "meta": Array.from_list([int(q), int(d), int(kernel), int(degree), int(seed), int(self.sweeps_)], "<i8"),
        })

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result transforms; it does not
        refit."""
        from .linear_model import _restore_mode
        arrays = _serialize.read_npz(path, _NYSTROEM_FORMAT)
        meta, hyper = _km_header(arrays, path, cls, 6, 2)
        q, d, kernel, degree, seed, sweeps = (int(v) for v in meta.tolist())
        gamma, coef0 = (float(v) for v in hyper.tolist())
        obj = cls(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, n_components=q, random_state=seed)
        _restore_mode(obj, arrays)
        obj.n_features_in_ = d
        obj.components_ = _km_array(arrays, "components", "<f4", (q, d), path)
        obj.component_indices_ = _km_array(arrays, "component_indices", "<i4", (q,), path)
        obj.normalization_ = _km_array(arrays, "normalization", "<f4", (q, q), path)
        obj.eigenvalues_ = _km_array(arrays, "eigenvalues", "<f4", (q,), path)
        obj.eigenvectors_ = _km_array(arrays, "eigenvectors", "<f4", (q, q), path)
        obj.sweeps_ = sweeps
        obj._kernel_params = (kernel, degree, gamma, coef0, seed)
        return obj


class RBFSampler(_KernelMethodBase):
    """`sklearn.kernel_approximation.RBFSampler` on the GPU: random
    Fourier features for the RBF kernel.

    Parameters
    ----------
    gamma : float, default 1.0
        Must be positive (refused by name on the Mojo host).
        scikit-learn's `gamma='scale'` is NOT implemented: it is a host
        variance over the data, a fold in front of every draw
        (`kernel_methods/NOT_IMPLEMENTED.tsv`).
    n_components : int, default 100
        Refused by name when not positive, before any buffer is made.
    random_state : int, default 0

    Attributes
    ----------
    random_weights_ : Array (n_features, n_components) float32
    random_offset_ : Array (n_components,) float32
    sigma_ : float
        `sqrt(2 gamma)`, computed once on the host (DEVIATION 1678).
    scale_ : float
        `sqrt(2 / n_components)`, same argument.
    """

    _WHERE = "RBFSampler"

    def __init__(self, gamma=1.0, n_components=100, random_state=0):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Reads `X.shape[1]` and nothing else of X, as scikit-learn's does."""
        x, _ = as_f32_c(X, ndim=2, name="X")
        _, d = x.shape
        if self.gamma == "scale":
            raise ValueError(
                f"mojolearn {self._WHERE}: gamma='scale' is not implemented; it is "
                "a host variance over the data in front of every draw "
                "(kernel_methods/NOT_IMPLEMENTED.tsv). Pass a positive float."
            )
        gamma = _real(self.gamma, "gamma", self._WHERE)
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, int):
            raise TypeError(f"mojolearn {self._WHERE}: n_components must be an int")
        q = int(self.n_components)
        if q < 1:
            # BY NAME, BEFORE ANY BUFFER. The first MI300X run of this surface
            # (2026-09-14) saw n_components=0 refused as "null float32 buffer
            # address": the zero-length weights buffer reached the binding's
            # address check before anything had said which argument was wrong.
            raise ValueError(f"mojolearn {self._WHERE}: n_components must be positive, got {q}")
        seed = int(self.random_state)
        weights = empty((d * q,), "<f4")
        offset = empty((q,), "<f4")
        scalars = empty((2,), "<f8")
        self._extension().rbf_sampler_fit(
            # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::rbf_sampler_fit_binding.
            # weights_out, offset_out, scalars_out
            [addr(weights, name="random_weights_"), addr(offset, name="random_offset_"), addr(scalars, name="scalars")],
            # d, q, gamma, seed
            [d, q, gamma, seed],
        )
        self.n_features_in_ = d
        self.random_weights_ = weights.reshape((d, q))
        self.random_offset_ = offset
        self.sigma_ = float(scalars[0])
        self.scale_ = float(scalars[1])
        self._params = (gamma, seed)
        return self

    def transform(self, X):
        """`scale_ * cos(X . random_weights_ + random_offset_)`, float32."""
        if not hasattr(self, "random_weights_"):
            raise ValueError(f"mojolearn {self._WHERE}: call fit before transform")
        x, _ = as_f32_c(X, ndim=2, name="X")
        m, d = x.shape
        if d != self.n_features_in_:
            raise ValueError(
                f"mojolearn {self._WHERE}: X has {d} features, the fit had {self.n_features_in_}"
            )
        q = self.random_weights_.shape[1]
        gamma, seed = self._params
        out = empty((m * q,), "<f4")
        flat_w = self.random_weights_.reshape((d * q,))
        self._extension().rbf_sampler_transform(
            # ORDER MATCHES bindings/_mojolearn_kernel_methods.mojo::rbf_sampler_transform_binding.
            # weights, offset, x, out
            [addr_ro(flat_w, name="random_weights_"), addr_ro(self.random_offset_, name="random_offset_"),
             addr_ro(x, name="X"), addr(out, name="transform")],
            # d, q, gamma, seed, sigma, scale, m
            [d, q, gamma, seed, self.sigma_, self.scale_, m],
        )
        return out.reshape((m, q))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def save(self, path):
        """Write the fitted sampler to `path` as an npz: the weights and
        offsets as drawn, `hyper` `<f8` [gamma, sigma_, scale_], `meta`
        `<i8` [n_features, n_components, seed] and the numeric mode.
        `mojolearn.host_model(path)` transforms from it on a CPU with no
        GPU."""
        if not hasattr(self, "random_weights_"):
            raise RuntimeError("this estimator is not fitted yet")
        gamma, seed = self._params
        d, q = self.random_weights_.shape
        return _km_write(self, path, _RBF_SAMPLER_FORMAT, {
            "random_weights": self.random_weights_,
            "random_offset": self.random_offset_,
            "hyper": Array.from_list([float(gamma), float(self.sigma_), float(self.scale_)], "<f8"),
            "meta": Array.from_list([int(d), int(q), int(seed)], "<i8"),
        })

    @classmethod
    def load(cls, path):
        """Load a sampler saved by `save`. The result transforms; it does
        not draw again."""
        from .linear_model import _restore_mode
        arrays = _serialize.read_npz(path, _RBF_SAMPLER_FORMAT)
        meta, hyper = _km_header(arrays, path, cls, 3, 3)
        d, q, seed = (int(v) for v in meta.tolist())
        gamma, sigma, scale = (float(v) for v in hyper.tolist())
        obj = cls(gamma=gamma, n_components=q, random_state=seed)
        _restore_mode(obj, arrays)
        obj.n_features_in_ = d
        obj.random_weights_ = _km_array(arrays, "random_weights", "<f4", (d, q), path)
        obj.random_offset_ = _km_array(arrays, "random_offset", "<f4", (q,), path)
        obj.sigma_ = sigma
        obj.scale_ = scale
        obj._params = (gamma, seed)
        return obj


__all__ = ["KernelRidge", "Nystroem", "RBFSampler"]
