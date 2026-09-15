# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`GaussianMixture` on the GPU (workstream D, 2026-09-14).

The Python door of `mixture/estimator.mojo` through
`bindings/_mojolearn_mixture.mojo`. THERE IS NO GPU REFERENCE: cuML, cuVS and
RAFT ship no Gaussian mixture at the pinned commits, so scikit-learn's
`_gaussian_mixture.py` and `_base.py` are the SEMANTICS and the oracle,
never the design source (the estimator's header). Every parameter here
means exactly what scikit-learn's parameter of that name means, or is
named differently; `mixture/README.md` carries the mapping.

WHAT IS IMPLEMENTED: `covariance_type='full'`, `init_params='kmeans'`
(through the identity-certified k-means) and `'random'` (position-mapped
Philox draws, DEVIATION 1733), `n_init=1`, `warm_start=False`. The other
covariance types and init methods are REFUSED BY NAME on the Mojo host
with the reason each costs; `n_init > 1`, `warm_start`, `means_init`,
`weights_init` and `precisions_init` are refused by name here, because
`GmmParams` has no field for them (DEVIATION 1734).

A COLLAPSED COMPONENT RAISES (DEVIATION 1723); the fit never returns a
model with a reset component. `n_iter_`, `converged_` and `lower_bound_`
are part of the model and of the identity card.

NO SPEED CLAIM. The lane has no published number and this door adds none.
"""
from . import _backend, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

#: The model file (the neighbors and density inference lane, 2026-09-15):
#: `weights`, `means`, `covariances`, `precisions_cholesky` and
#: `log_det_chol` as flat `<f4` in the order the scoring entries address
#: them, `ints` `<i8` [n_components, n_features, n_iter_, converged_,
#: max_iter, random_state], `reals` `<f8` [lower_bound_, tol, reg_covar],
#: `covariance_type` and `init_params` as text.
_GMM_FORMAT = "mojolearn-gmm-1"

#: Refused by absence on the Mojo side (`GmmParams` has no such field,
#: DEVIATION 1734), so refused by NAME here rather than accepted and
#: ignored, which the contract forbids.
_REFUSED_KNOBS = {
    "n_init": "n_init > 1 picks the best restart by a float comparison of "
              "lower bounds (_base.py:282), a second data-dependent branch "
              "with no identity argument yet (DEVIATION 1734)",
    "warm_start": "warm_start makes the fit depend on the object's history "
                  "rather than on its inputs (DEVIATION 1734)",
    "means_init": "means_init is not a field of GmmParams (mixture/NOT_IMPLEMENTED.tsv)",
    "weights_init": "weights_init is not a field of GmmParams (mixture/NOT_IMPLEMENTED.tsv)",
    "precisions_init": "precisions_init is not a field of GmmParams (mixture/NOT_IMPLEMENTED.tsv)",
}


def _real(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(
            f"mojolearn GaussianMixture: {name} must be a real number, got "
            f"{type(v).__name__}"
        )
    return float(v)


class GaussianMixture(NumericModeMixin):
    """`sklearn.mixture.GaussianMixture` semantics on the GPU.

    Parameters (scikit-learn's defaults, stated rather than matched)
    ----------
    n_components : int, default 1
    covariance_type : str, default 'full'
        Only 'full' is implemented; 'tied', 'diag' and 'spherical' are
        refused by name on the Mojo host with what each would cost.
    tol : float, default 1e-3
    reg_covar : float, default 1e-6
    max_iter : int, default 100
    init_params : {'kmeans', 'random'}, default 'kmeans'
        'k-means++' and 'random_from_data' are refused by name on the
        Mojo host.
    random_state : int, default 0
    n_init, warm_start, means_init, weights_init, precisions_init
        Refused by name when passed anything but scikit-learn's default.

    Attributes
    ----------
    weights_ : Array (n_components,) float32
    means_ : Array (n_components, n_features) float32
    covariances_ : Array (n_components, n_features, n_features) float32
        Includes `reg_covar` on the diagonal, as scikit-learn's does.
    precisions_cholesky_ : Array (n_components, n_features, n_features) float32
    log_det_chol_ : Array (n_components,) float32
        `_compute_log_det_cholesky`'s answer, from the device fold
        (DEVIATION 1726); never re-derived from the diagonal here.
    n_iter_ : int
    converged_ : bool
        False means the loop ran out of iterations; scikit-learn warns,
        this reports (DEVIATION 1746).
    lower_bound_ : float
    """

    _BINDING = "_mojolearn_mixture"

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        init_params="kmeans",
        random_state=0,
        n_init=1,
        warm_start=False,
        means_init=None,
        weights_init=None,
        precisions_init=None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.init_params = init_params
        self.random_state = random_state
        self.n_init = n_init
        self.warm_start = warm_start
        self.means_init = means_init
        self.weights_init = weights_init
        self.precisions_init = precisions_init

    def _extension(self):
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "mixture_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn GaussianMixture: numeric_mode={want!r} was requested "
                    f"but {mod.__name__} reports compile-time mode code {got}; "
                    "rebuild it with bash bindings/build_mixture.sh"
                )
        return mod

    def _refuse_knobs(self):
        if self.n_init != 1:
            raise ValueError("mojolearn GaussianMixture: " + _REFUSED_KNOBS["n_init"])
        if self.warm_start:
            raise ValueError("mojolearn GaussianMixture: " + _REFUSED_KNOBS["warm_start"])
        for name in ("means_init", "weights_init", "precisions_init"):
            if getattr(self, name) is not None:
                raise ValueError("mojolearn GaussianMixture: " + _REFUSED_KNOBS[name])

    def fit(self, X, y=None):
        """EM until `abs(lower_bound - prev) < tol` or `max_iter`, one shot
        on the device (`gaussian_mixture_fit`). Returns `self`."""
        self._refuse_knobs()
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = x.shape
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, int):
            raise TypeError("mojolearn GaussianMixture: n_components must be an int")
        if not isinstance(self.covariance_type, str) or not isinstance(self.init_params, str):
            raise TypeError(
                "mojolearn GaussianMixture: covariance_type and init_params are "
                "scikit-learn's strings; they are decoded, and refused, by name on "
                "the Mojo host"
            )
        k = int(self.n_components)
        if k < 1:
            raise ValueError(f"mojolearn GaussianMixture: n_components must be positive, got {k}")
        tol = _real(self.tol, "tol")
        reg_covar = _real(self.reg_covar, "reg_covar")
        max_iter = int(self.max_iter)
        weights = empty((k,), "<f4")
        means = empty((k * d,), "<f4")
        covariances = empty((k * d * d,), "<f4")
        precisions = empty((k * d * d,), "<f4")
        log_det = empty((k,), "<f4")
        scalars = empty((3,), "<f8")
        n_iter = self._extension().gmm_fit(
            # ORDER MATCHES bindings/_mojolearn_mixture.mojo::gmm_fit_binding.
            # x, weights_out, means_out, covariances_out, precisions_chol_out, log_det_chol_out, scalars_out
            [addr_ro(x, name="X"), addr(weights, name="weights_"), addr(means, name="means_"),
             addr(covariances, name="covariances_"), addr(precisions, name="precisions_cholesky_"),
             addr(log_det, name="log_det_chol_"), addr(scalars, name="scalars")],
            # n, d, n_components, covariance_type, tol, reg_covar, max_iter, init_params, random_state
            [n, d, k, self.covariance_type, tol, reg_covar, max_iter, self.init_params, int(self.random_state)],
        )
        self.n_features_in_ = d
        self.weights_ = weights
        self.means_ = means.reshape((k, d))
        self.covariances_ = covariances.reshape((k, d, d))
        self.precisions_cholesky_ = precisions.reshape((k, d, d))
        self.log_det_chol_ = log_det
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(int(scalars[1]))
        self.lower_bound_ = float(scalars[2])
        return self

    def _model_lists(self, X, out, name):
        if not hasattr(self, "weights_"):
            raise ValueError(f"mojolearn GaussianMixture: call fit before {name}")
        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = x.shape
        if d != self.n_features_in_:
            raise ValueError(
                f"mojolearn GaussianMixture: X has {d} features, the fit had {self.n_features_in_}"
            )
        k = self.weights_.shape[0]
        flat_m = self.means_.reshape((k * d,))
        flat_c = self.covariances_.reshape((k * d * d,))
        flat_p = self.precisions_cholesky_.reshape((k * d * d,))
        # ORDER MATCHES bindings/_mojolearn_mixture.mojo::_rebuild_model.
        # weights, means, covariances, precisions_chol, log_det_chol, x, out
        addrs = [addr_ro(self.weights_, name="weights_"), addr_ro(flat_m, name="means_"),
                 addr_ro(flat_c, name="covariances_"), addr_ro(flat_p, name="precisions_cholesky_"),
                 addr_ro(self.log_det_chol_, name="log_det_chol_"), addr_ro(x, name="X"), addr(out, name=name)]
        # k, d, n_iter, converged, lower_bound, n
        params = [k, d, self.n_iter_, 1 if self.converged_ else 0, self.lower_bound_, n]
        # The arrays addressed above stay alive through the caller's frame.
        return (x, flat_m, flat_c, flat_p), addrs, params, n, k

    def score_samples(self, X):
        """One log likelihood per row, float32 `(n,)`."""
        x, _ = as_f32_c(X, ndim=2, name="X")
        out = empty((x.shape[0],), "<f4")
        keep, addrs, params, n, k = self._model_lists(x, out, "score_samples")
        self._extension().gmm_score_samples(addrs, params)
        return out

    def predict_proba(self, X):
        """`exp(log_resp)`, float32 `(n, n_components)`."""
        x, _ = as_f32_c(X, ndim=2, name="X")
        k = self.weights_.shape[0] if hasattr(self, "weights_") else 0
        out = empty((x.shape[0] * k,), "<f4")
        keep, addrs, params, n, k = self._model_lists(x, out, "predict_proba")
        self._extension().gmm_predict_proba(addrs, params)
        return out.reshape((n, k))

    def predict(self, X):
        """Argmax of the weighted log probabilities, int32 `(n,)`; ties to
        the lowest component index."""
        x, _ = as_f32_c(X, ndim=2, name="X")
        out = empty((x.shape[0],), "<i4")
        keep, addrs, params, n, k = self._model_lists(x, out, "predict")
        self._extension().gmm_predict(addrs, params)
        return out

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def save(self, path):
        """Write the fitted model to `path` as an npz (`_GMM_FORMAT`, the
        neighbors and density inference lane, 2026-09-15): every array the
        scoring entries read, as fitted, plus `n_iter_`, `converged_` and
        `lower_bound_`, which the binding's model rebuild takes too.
        `mojolearn.host_model(path)` scores it on a CPU with no GPU."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("this estimator is not fitted yet")
        from .linear_model import _saved_mode
        k, d = self.means_.shape
        return _serialize.write_npz(path, {
            "format": _GMM_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "covariance_type": str(self.covariance_type),
            "init_params": str(self.init_params),
            "weights": self.weights_,
            "means": self.means_.reshape((k * d,)),
            "covariances": self.covariances_.reshape((k * d * d,)),
            "precisions_cholesky": self.precisions_cholesky_.reshape((k * d * d,)),
            "log_det_chol": self.log_det_chol_,
            "ints": Array.from_list([k, d, int(self.n_iter_), 1 if self.converged_ else 0,
                                     int(self.max_iter), int(self.random_state)], "<i8"),
            "reals": Array.from_list([float(self.lower_bound_), float(self.tol), float(self.reg_covar)], "<f8"),
        })

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`; every array at its saved dtype,
        never cast. The result answers score_samples, predict_proba,
        predict, score, bic and aic."""
        from .linear_model import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _GMM_FORMAT)
        _check_saved_by(arrays, path, cls)
        ints = _serialize.exact(arrays, "ints", "<i8")
        reals = _serialize.exact(arrays, "reals", "<f8")
        if ints.size != 6 or reals.size != 3:
            raise ValueError(f"mojolearn: {path!r} holds {ints.size} ints and {reals.size} reals, 6 and 3 are needed")
        k, d, n_iter, converged, max_iter, seed = (int(ints[i]) for i in range(6))
        if k < 1 or d < 1:
            raise ValueError(f"mojolearn: {path!r} holds an empty model")
        obj = cls(n_components=k, covariance_type=_serialize.scalar_str(arrays, "covariance_type"),
                  tol=float(reals[1]), reg_covar=float(reals[2]), max_iter=max_iter,
                  init_params=_serialize.scalar_str(arrays, "init_params"), random_state=seed)
        _restore_mode(obj, arrays)
        sizes = dict(weights=k, means=k * d, covariances=k * d * d, precisions_cholesky=k * d * d, log_det_chol=k)
        got = {}
        for name, size in sizes.items():
            a = _serialize.exact(arrays, name, "<f4")
            if a.ndim != 1 or a.size != size:
                raise ValueError(f"mojolearn: {path!r} {name} holds {a.size} values, {size} are needed")
            got[name] = a
        obj.n_features_in_ = d
        obj.weights_ = got["weights"]
        obj.means_ = got["means"].reshape((k, d))
        obj.covariances_ = got["covariances"].reshape((k, d, d))
        obj.precisions_cholesky_ = got["precisions_cholesky"].reshape((k, d, d))
        obj.log_det_chol_ = got["log_det_chol"]
        obj.n_iter_ = n_iter
        obj.converged_ = bool(converged)
        obj.lower_bound_ = float(reals[0])
        return obj

    def _score_bic_aic(self, X):
        x, _ = as_f32_c(X, ndim=2, name="X")
        out = empty((3,), "<f8")
        keep, addrs, params, n, k = self._model_lists(x, out, "score")
        self._extension().gmm_score_bic_aic(addrs, params)
        return float(out[0]), float(out[1]), float(out[2])

    def score(self, X, y=None):
        """The mean of `score_samples`, the host ascending fold."""
        return self._score_bic_aic(X)[0]

    def bic(self, X):
        return self._score_bic_aic(X)[1]

    def aic(self, X):
        return self._score_bic_aic(X)[2]

    def sample(self, n_samples=1):
        """`n_samples` random rows from the fitted mixture: `(X, y)`, `X`
        float32 `(n_samples, n_features)`, `y` int32 `(n_samples,)`.

        scikit-learn's `BaseMixture.sample` is the reference: the component
        counts are a multinomial draw over `weights_`, and the rows come out
        GROUPED BY COMPONENT ascending, `y` naming each row's component. The
        draws are position-mapped Philox keyed by `random_state` (DEVIATION
        2791, `mixture/checks/sample.mojo`) and the normals go through the
        fitted `precisions_cholesky_` (DEVIATION 2792), so the same model
        and `random_state` give the same bits on every vendor and on every
        call; they are not scikit-learn's bits. `X` is the model's float32
        and `y` is `predict`'s int32, where scikit-learn returns float64
        and int64. `n_samples < 1` is refused by name in Mojo.
        """
        if not hasattr(self, "weights_"):
            raise ValueError("mojolearn GaussianMixture: call fit before sample")
        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise TypeError("mojolearn GaussianMixture: n_samples must be an int")
        seed = self.random_state
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 64:
            raise ValueError(
                "mojolearn GaussianMixture: sample needs random_state to be an int in "
                f"[0, 2**64), got {seed!r}; it is the key of every draw (DEVIATION 2791)"
            )
        n = int(n_samples)
        k = self.weights_.shape[0]
        d = self.n_features_in_
        flat_m = self.means_.reshape((k * d,))
        flat_c = self.covariances_.reshape((k * d * d,))
        flat_p = self.precisions_cholesky_.reshape((k * d * d,))
        # Never a zero-length buffer: a refused n still addresses real memory.
        x = empty((max(n, 1) * d,), "<f4")
        y = empty((max(n, 1),), "<i4")
        self._extension().gmm_sample(
            # ORDER MATCHES bindings/_mojolearn_mixture.mojo::gmm_sample_binding.
            # weights, means, covariances, precisions_chol, log_det_chol, X out, y out
            [addr_ro(self.weights_, name="weights_"), addr_ro(flat_m, name="means_"),
             addr_ro(flat_c, name="covariances_"), addr_ro(flat_p, name="precisions_cholesky_"),
             addr_ro(self.log_det_chol_, name="log_det_chol_"), addr(x, name="X"), addr(y, name="y")],
            # k, d, n_iter, converged, lower_bound, n (0: no X is read)
            [k, d, self.n_iter_, 1 if self.converged_ else 0, self.lower_bound_, 0],
            # n_samples, random_state low 32 bits, random_state high 32 bits
            [n, seed & 0xFFFFFFFF, seed >> 32],
        )
        return x.reshape((n, d)), y


__all__ = ["GaussianMixture"]
