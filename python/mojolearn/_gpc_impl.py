# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gaussian process classification: the Laplace approximation.

PRIVATE MODULE. `GaussianProcessClassifier` is re-exported from
`mojolearn/__init__.py`; `HostGaussianProcessClassifier` is what
`mojolearn.host_model` returns for a saved classifier.

THE REFERENCE is scikit-learn 1.9.0 `sklearn/gaussian_process/_gpc.py`:
`_BinaryGaussianProcessClassifierLaplace` (`_gpc.py:36-513`, Rasmussen and
Williams Algorithms 3.1 and 3.2 with the logistic likelihood) and
`GaussianProcessClassifier` (`_gpc.py:516` on), which wraps the binary
estimator in `OneVsRestClassifier` past two classes. Every name here means
what scikit-learn means by it.

WHERE THE ARITHMETIC IS. One binary Laplace fit per binding call,
`gpc_fit` and `gpc_predict` on `_mojolearn_gp` (the GPU binding,
`gaussian_process/classifier.mojo`) or on `_mojolearn_gp_host` (the CPU host
binding, `gaussian_process/host/gpc_oracle.mojo`); the per-row steps of
both are `gaussian_process/host/gpc_steps.mojo`, which carries DEVIATIONS
2830 (the stop rule), 2831 (the pinned float32 orders) and 2832 (the
float64 probability and its erf). This file holds the class encoding, the
one-vs-rest composition (DEVIATION 2833) and the saved model.

DEVIATION 2833, THE ONE-VS-REST COMPOSITION, in plain Python float64 (every
value it reads is a float64 or an exact widening of a float32, and CPython
float arithmetic is IEEE double with no contraction, so it is the same on
every host):
  - column k of a K-class fit is the binary target `y == classes_[k]`, the
    `LabelBinarizer` column `OneVsRestClassifier` fits (`multiclass.py`);
  - `predict_proba` divides each row by its sum where the sum is not zero
    (`multiclass.py:559-560`), the sum taken over k ASCENDING from +0.0
    (NumPy's reduction order may differ in the last bits);
  - `predict` is the first k whose unnormalized class-k probability is
    strictly largest, which is NumPy's argmax tie rule that
    `OneVsRestClassifier.predict` reproduces (`multiclass.py:499-508`); a
    binary model predicts `classes_[1]` where the float32 latent mean is
    strictly positive (`_gpc.py:287-290`);
  - `log_marginal_likelihood_value_` is the mean of the K binary values
    (`_gpc.py:216-222`), summed ascending from +0.0 then divided by K.

WHAT IS HONORED AND WHAT IS REFUSED, one line per parameter:

    kernel                honored   RBF, Matern (nu in {0.5, 1.5, 2.5}),
                                    ConstantKernel and WhiteKernel via + and *;
                                    None is scikit-learn's default
                                    ConstantKernel(1.0) * RBF(1.0)
    optimizer             refused   anything but None (DEVIATION 1761). THE
                                    DEFAULT IS None, NOT 'fmin_l_bfgs_b', so
                                    the fitted kernel_ is the kernel passed
    n_restarts_optimizer  refused   anything but 0; it restarts the optimizer
    max_iter_predict      honored   at least 1; the Newton loop's cap
                                    (DEVIATION 2830 says how it stops early)
    warm_start            refused   True: it starts the Newton loop from the
                                    previous fit's latent values, so a second
                                    fit would depend on the first
    copy_X_train          refused   False: X always crosses to float32 memory
    random_state          refused   anything but None; only the refused
                                    optimizer restarts draw random numbers
    multi_class           honored   'one_vs_rest'; 'one_vs_one' is refused BY
                                    NAME (not implemented, and it has no
                                    predict_proba in the reference either)
    n_jobs                refused   anything but None; the binary fits run
                                    one after another on one device

THE INFERENCE BOUNDARY.
`fit` on a CPU-only install refuses unless the internal verifier's
`reference_training()` is active. `predict`, `predict_proba`,
`latent_mean_and_variance` and `load` are public on the CPU:
`GaussianProcessClassifier.load(path)` routes `_mojolearn_gp` to
`_mojolearn_gp_host` on a CPU-only install, and `mojolearn.host_model(path)`
returns `HostGaussianProcessClassifier`, bound to the host binding on any
box.
"""

from . import _backend, _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty, frombytes
from ._gp_impl import _MODE_CODE, ConstantKernel, Kernel, RBF
from ._labels import (
    classes_from_member,
    classes_member,
    decode_labels,
    encode_labels,
)
from ._mode import NumericModeMixin

#: The saved-model format tag.
_GPC_FORMAT = "mojolearn.gpc.v1"
_ESTIMATOR = "GaussianProcessClassifier"
_NAME = "mojolearn GaussianProcessClassifier"


def _gpc_all(ext):
    """True when the binding fits and predicts every class in one device
    session (`gpc_fit_all` / `gpc_predict_all`,
    `gaussian_process/gpc_ovr.mojo`, lane fam2-kernel-gp): IDENTICAL GPU
    builds. The host binding and a `-D MOJOLEARN_IDN_GPC_OVR_OFF` build
    answer no and the per-class binding calls below run; either way the
    binding, never Python, forms targets, codes and probabilities."""
    fn = getattr(ext, "gp_idn_caps", None)
    return fn is not None and (int(fn()) & 1) == 1


def _ovr_combine(ext, cols, n_star):
    """DEVIATION 2833 on the binding (`gpc_ovr_combine`, the device on a GPU
    install): the k unnormalized class-1 probability columns normalized per
    row and the first strictly largest column per row. Returns the
    (n_star, k) float64 probabilities and the n_star int32 codes."""
    k = len(cols)
    proba = empty((n_star, k), "<f8")
    codes = empty((n_star,), "<i4")
    ext.gpc_ovr_combine(
        # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_ovr_combine_binding.
        [addr(proba, name="proba_out"), addr(codes, name="codes_out")]
        + [addr_ro(c, name=f"class_{i}") for i, c in enumerate(cols)],  # glue: one class column address per class
        [int(n_star)],
    )
    return proba, codes


def _stacked(blocks, dtype, shape):
    """The fits' same-shape buffers as one (len(blocks),) + shape Array, joined
    as bytes (no per-element Python)."""
    raw = b"".join(a.tobytes() for a in blocks)  # glue: joins one buffer per class fit
    return frombytes(raw, dtype, (len(blocks),) + tuple(shape))


def _kernel_arrays(kernel):
    """The postfix spec as the four flat arrays `_gp_impl.py` sends
    (DEVIATION 1756), plus the length-scale count."""
    nodes = kernel._nodes()
    kinds = Array.from_list([int(k) for k, _, _ in nodes], "<i4")  # glue: packs the kernel's postfix nodes
    params = Array.from_list([float(p) for _, p, _ in nodes], "<f4")  # glue: packs the kernel's postfix nodes
    ls_len = Array.from_list([len(ls) for _, _, ls in nodes], "<i4")  # glue: packs the kernel's postfix nodes
    table = [v for _, _, ls in nodes for v in ls]  # glue: packs the kernel length-scale hyperparameters
    ls = Array.from_list(table if table else [1.0], "<f4")
    return kinds, params, ls_len, ls, len(table)


class _SavedKernel(Kernel):
    """A kernel read back from a model file: the postfix nodes exactly as
    they were saved (float32 values round-trip through Python floats
    exactly), so a loaded model sends the binding the bytes the fit sent."""

    def __init__(self, nodes):
        self._node_list = [(int(k), float(p), [float(v) for v in ls]) for k, p, ls in nodes]  # glue: copies the kernel's postfix nodes

    def _nodes(self):
        return [(k, p, list(ls)) for k, p, ls in self._node_list]  # glue: copies the kernel's postfix nodes

    def _name(self):
        return f"SavedKernel({len(self._node_list)} postfix nodes)"


class _BinaryLaplace:
    """One binary Laplace fit, the state `_BinaryGaussianProcessClassifierLaplace`
    keeps (`_gpc.py:263-265`): the 0/1 targets `y_train_`, `pi_`, `W_sr_`,
    `L_` (the lower factor of `B = I + W_sr K W_sr`), the likelihood, and the
    Newton iteration count and panel width that ran."""

    def __init__(self, y_train, L, pi, W_sr, lml, n_iter, nb):
        self.y_train_ = y_train
        self.L_ = L
        self.pi_ = pi
        self.W_sr_ = W_sr
        self.log_marginal_likelihood_value_ = lml
        self.n_iter_ = n_iter
        self.nb_ = nb


class GaussianProcessClassifier(NumericModeMixin):
    """Gaussian process classification by the Laplace approximation,
    scikit-learn's surface. See this module's docstring for the parameter
    table, the one-vs-rest composition and the CPU inference boundary.

    Attributes
    ----------
    classes_ : list
    n_classes_ : int
    X_train_ : Array (n_train, n_features) float32
    estimators_ : list of the binary fits (one for two classes, K past two),
        each with `y_train_`, `L_`, `pi_`, `W_sr_`,
        `log_marginal_likelihood_value_` and `n_iter_`
    L_, pi_, W_sr_, y_train_ : Array, two classes only (the one binary fit)
    log_marginal_likelihood_value_ : float
    n_iter_ : int for two classes, a list of K ints past two (DEVIATION 2830)
    kernel_ : Kernel, the kernel passed in (no optimizer)
    n_features_in_ : int
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "classifier"

    _BINDING = "_mojolearn_gp"

    def __init__(
        self,
        kernel=None,
        *,
        optimizer=None,
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
        multi_class="one_vs_rest",
        n_jobs=None,
    ):
        if kernel is None:
            # scikit-learn's default, `_gpc.py:186-189`, bounds "fixed"
            # there; there is no optimizer here, so the two coincide.
            kernel = ConstantKernel(1.0) * RBF(1.0)
        if not isinstance(kernel, Kernel):
            raise TypeError(
                f"{_NAME}: kernel must be a composition of mojolearn's RBF, "
                "Matern, ConstantKernel and WhiteKernel (DEVIATION 1761)"
            )
        if not (optimizer is None
                or (isinstance(optimizer, str) and optimizer.lower() == "none")):
            raise NotImplementedError(
                f"{_NAME}: optimizer={optimizer!r} is refused; only None runs "
                "(DEVIATION 1761). An optimizer's iteration count is data "
                "dependent and its line search, gradient fold and tolerance "
                "are not pinned. scikit-learn's default is 'fmin_l_bfgs_b', "
                "so this default DIFFERS from theirs and the fitted kernel_ "
                "is the kernel you passed"
            )
        if isinstance(n_restarts_optimizer, bool) or int(n_restarts_optimizer) != 0:
            raise NotImplementedError(
                f"{_NAME}: n_restarts_optimizer={n_restarts_optimizer!r} is "
                "refused; it restarts the optimizer, which is refused "
                "(DEVIATION 1761)"
            )
        if isinstance(max_iter_predict, bool) or not isinstance(max_iter_predict, int):
            raise TypeError(
                f"{_NAME}: max_iter_predict must be an int, got {max_iter_predict!r}"
            )
        if max_iter_predict < 1:
            raise ValueError(
                f"{_NAME}: max_iter_predict must be at least 1, got {max_iter_predict}"
            )
        if warm_start:
            raise NotImplementedError(
                f"{_NAME}: warm_start=True is refused. It starts the Newton "
                "loop from the previous fit's latent values (_gpc.py:454-459), "
                "so a second fit would depend on the first; every fit here "
                "starts from zero"
            )
        if not copy_X_train:
            raise NotImplementedError(
                f"{_NAME}: copy_X_train=False is refused; X always crosses to "
                "float32 C-order memory, so there is no reference to keep"
            )
        if random_state is not None:
            raise NotImplementedError(
                f"{_NAME}: random_state is refused; only the optimizer's "
                "restarts draw random numbers in the reference, and the "
                "optimizer is refused"
            )
        if multi_class == "one_vs_one":
            raise NotImplementedError(
                f"{_NAME}: multi_class='one_vs_one' is refused BY NAME; it is "
                "not implemented (and the reference gives it no "
                "predict_proba). 'one_vs_rest' runs (DEVIATION 2833)"
            )
        if multi_class != "one_vs_rest":
            raise ValueError(
                f"{_NAME}: multi_class must be 'one_vs_rest' or 'one_vs_one', "
                f"got {multi_class!r}"
            )
        if n_jobs is not None:
            raise NotImplementedError(
                f"{_NAME}: n_jobs={n_jobs!r} is refused; the one-vs-rest "
                "binary fits run one after another on one device"
            )
        self.kernel = kernel
        self.optimizer = None
        self.n_restarts_optimizer = 0
        self.max_iter_predict = int(max_iter_predict)
        self.warm_start = False
        self.copy_X_train = True
        self.random_state = None
        self.multi_class = "one_vs_rest"
        self.n_jobs = None

    # -- the binding ----------------------------------------------------------

    def _extension(self):
        """`_mojolearn_gp` (or the host binding) with its compile-time tier
        cross-checked against the requested one, `_gp_impl.py`'s rule."""
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "gp_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"{_NAME}: numeric_mode={want!r} was requested but "
                    f"{mod.__name__} reports compile-time mode code {got}; "
                    "rebuild it with bash bindings/build_gp.sh"
                )
        return mod

    # -- fit ------------------------------------------------------------------

    def fit(self, X, y):
        """`_gpc.py:164-226`: encode the classes, then one binary Laplace fit
        (two classes) or one per class against the rest. Returns `self`."""
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        n_rows, n_cols = x.shape
        classes, codes = encode_labels(y)
        if codes.size != n_rows:
            raise ValueError(f"{_NAME}: y has {codes.size} entries, X has {n_rows} rows")
        n_classes = len(classes)
        if n_classes == 1:
            raise ValueError(
                f"{_NAME} requires 2 or more distinct classes; got 1 class "
                f"(only class {classes[0]!r} is present)"
            )
        kinds, kparams, ls_len, ls, n_ls = _kernel_arrays(self.kernel)
        ext = self._extension()
        columns = [1] if n_classes == 2 else list(range(n_classes))
        codes32 = Array.from_list(codes, "<i4")
        if _gpc_all(ext):
            fits = self._fit_all(ext, x, codes32, kinds, kparams, ls_len, ls, n_ls, columns)
        else:
            fits = []
            for k in columns:  # glue: one binding call per one-vs-rest class
                fits.append(self._fit_binary(ext, x, codes32, kinds, kparams, ls_len, ls, n_ls, k=k))
        self._set_fitted(x, classes, fits)
        return self

    def _fit_all(self, ext, x, codes32, kinds, kparams, ls_len, ls, n_ls, columns):
        """Every binary fit in one binding call (`gpc_fit_all`): X uploaded
        and its kernel matrix computed once, the class targets built on the
        device. Returns the `_BinaryLaplace` list `_fit_binary` would."""
        n_rows, n_cols = x.shape
        codes, _ = as_i32_c(codes32, ndim=1, name="class codes")
        # Per fit: targets, factor, pi, W_sr and (lml, n_iter, nb). Every
        # array addressed below stays bound in `outs` for the call.
        outs = []
        addrs = [
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_fit_all_binding.
            addr_ro(x, name="x"),
            addr_ro(codes, name="codes"),
            addr_ro(kinds, name="kinds"),
            addr_ro(kparams, name="kparams"),
            addr_ro(ls_len, name="ls_len"),
            addr_ro(ls, name="ls"),
        ]
        for _k in columns:  # glue: allocates one output set per binary fit
            y01 = empty((n_rows,), "<f4")
            l_out = empty((n_rows * n_rows,), "<f4")
            pi = empty((n_rows,), "<f4")
            wsr = empty((n_rows,), "<f4")
            scalars = empty((3,), "<f8")
            outs.append((y01, l_out, pi, wsr, scalars))
            addrs += [
                addr(y01, name="y_out"),
                addr(l_out, name="l_out"),
                addr(pi, name="pi_out"),
                addr(wsr, name="wsr_out"),
                addr(scalars, name="scalars_out"),
            ]
        ext.gpc_fit_all(
            addrs,
            # n_train, n_features, n_nodes, n_ls, max_iter_predict, n_fits, then the classes
            [n_rows, n_cols, int(kinds.shape[0]), n_ls, self.max_iter_predict, len(columns)]
            + [int(k) for k in columns],  # glue: the class of each binary fit
        )
        return [
            _BinaryLaplace(y01, l_out.reshape((n_rows, n_rows)), pi, wsr,
                           float(scalars[0]), int(scalars[1]), int(scalars[2]))
            for y01, l_out, pi, wsr, scalars in outs  # glue: wraps each fit's outputs
        ]

    def _fit_binary(self, ext, x, y01, kinds, kparams, ls_len, ls, n_ls, k=None):
        """With `k`, `y01` is the int32 class codes and the targets are
        `code == k`: the binding builds them and writes them back for the
        model (every build: the Python fallback loop was removed by lane
        fam2-kernel-gp, a GPU route does no Python arithmetic)."""
        n_rows, n_cols = x.shape
        extra_addrs, extra_params, codes = [], [], None
        if k is not None:
            codes, _ = as_i32_c(y01, ndim=1, name="class codes")
            y01 = empty((n_rows,), "<f4")
            extra_addrs, extra_params = [addr(y01, name="y_out")], [int(k)]
        l_out = empty((n_rows * n_rows,), "<f4")
        pi = empty((n_rows,), "<f4")
        wsr = empty((n_rows,), "<f4")
        # lml, n_iter, nb, in that order.
        scalars = empty((3,), "<f8")
        # Every array addressed below is bound to a local of this frame.
        n_iter = ext.gpc_fit(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_fit_binding.
            [
                addr_ro(x, name="x"),
                addr_ro(codes if codes is not None else y01, name="y"),
                addr_ro(kinds, name="kinds"),
                addr_ro(kparams, name="kparams"),
                addr_ro(ls_len, name="ls_len"),
                addr_ro(ls, name="ls"),
                addr(l_out, name="l_out"),
                addr(pi, name="pi_out"),
                addr(wsr, name="wsr_out"),
                addr(scalars, name="scalars_out"),
            ] + extra_addrs,
            # n_train, n_features, n_nodes, n_ls, max_iter_predict (+ k)
            [n_rows, n_cols, int(kinds.shape[0]), n_ls, self.max_iter_predict] + extra_params,
        )
        return _BinaryLaplace(y01, l_out.reshape((n_rows, n_rows)), pi, wsr,
                              float(scalars[0]), int(n_iter), int(scalars[2]))

    def _set_fitted(self, x, classes, fits):
        self.X_train_ = x
        self.n_features_in_ = int(x.shape[1])
        self.classes_ = list(classes)
        self.n_classes_ = len(classes)
        self.estimators_ = fits
        self.kernel_ = self.kernel
        if self.n_classes_ == 2:
            only = fits[0]
            self.y_train_ = only.y_train_
            self.L_ = only.L_
            self.pi_ = only.pi_
            self.W_sr_ = only.W_sr_
            self.n_iter_ = only.n_iter_
            self.log_marginal_likelihood_value_ = only.log_marginal_likelihood_value_
        else:
            self.n_iter_ = [e.n_iter_ for e in fits]  # glue: iteration counts per class
            # the per-class values summed in storage order by the native
            # reduction (lane py-runtime-b: was a Python accumulation)
            lml = Array.from_list([float(e.log_marginal_likelihood_value_) for e in fits], "<f8")  # glue: one saved scalar per class
            self.log_marginal_likelihood_value_ = float(lml.sum()) / len(fits)  # glue: Array.sum is the native reduce_stat over the k class values

    # -- prediction -------------------------------------------------------------

    def _query(self, X):
        if not hasattr(self, "estimators_"):
            raise ValueError(f"{_NAME}: call fit() or load() first")
        q, _ = as_f32_c(X, ndim=2, name="X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"{_NAME}: X has {q.shape[1]} features, fit saw {self.n_features_in_}"
            )
        return q

    def _latent(self, ext, est, q, want_proba, out_kind=0):
        """(mean, var, proba) of one binary fit at the query rows; var and
        proba are written only with `want_proba`. `out_kind` 1 / 2 (lane
        apple-fast-py2mojo-cluster): a fourth item, the int64 class codes of
        `predict` / the float64 `[1 - p, p]` rows of `predict_proba`, from
        the binding (`gaussian_process/unnorm.mojo::gpc_binary_out`)."""
        n_star = int(q.shape[0])
        n_train, n_features = self.X_train_.shape
        kinds, kparams, ls_len, ls, n_ls = _kernel_arrays(self.kernel)
        mean = empty((n_star,), "<f4")
        var = empty((n_star,), "<f4")
        proba = empty((n_star,), "<f8")
        out = None
        if out_kind == 1:
            out = empty((n_star,), "<i8")
        elif out_kind == 2:
            out = empty((n_star, 2), "<f8")
        xt = self.X_train_
        y01 = est.y_train_
        pi = est.pi_
        wsr = est.W_sr_
        lf = est.L_
        ext.gpc_predict(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_predict_binding.
            [
                addr_ro(xt, name="xtrain"),
                addr_ro(y01, name="y"),
                addr_ro(pi, name="pi"),
                addr_ro(wsr, name="wsr"),
                addr_ro(lf, name="l"),
                addr_ro(q, name="xstar"),
                addr_ro(kinds, name="kinds"),
                addr_ro(kparams, name="kparams"),
                addr_ro(ls_len, name="ls_len"),
                addr_ro(ls, name="ls"),
                addr(mean, name="mean_out"),
                addr(var, name="var_out"),
                addr(proba, name="proba_out"),
            ] + ([addr(out, name="out")] if out is not None else []),
            # n_train, n_features, n_star, n_nodes, n_ls, want_proba (+ out_kind)
            [int(n_train), int(n_features), n_star, int(kinds.shape[0]), n_ls,
             1 if want_proba else 0] + ([out_kind] if out is not None else []),
        )
        if out is not None:
            return mean, var, proba, out
        return mean, var, proba

    def latent_mean_and_variance(self, X):
        """`_gpc.py:410-441` (binary models only, as in the reference,
        `_gpc.py:873-878`): the float32 latent mean and variance."""
        q = self._query(X)
        if self.n_classes_ > 2:
            raise ValueError(
                "Returning the mean and variance of the latent function f is "
                "only supported for binary classification, received "
                f"{self.n_classes_} classes."
            )
        mean, var, _ = self._latent(self._extension(), self.estimators_[0], q, True)
        return mean, var

    def predict_proba(self, X):
        """`_gpc.py:292-329` and `multiclass.py:523-562`, float64
        (DEVIATIONS 2832 and 2833)."""
        q = self._query(X)
        ext = self._extension()
        if _gpc_all(ext):
            return self._predict_all(ext, q, 2)
        if self.n_classes_ == 2:
            # `[1 - p, p]` from the binding (`gpc_binary_out`), every build
            return self._latent(ext, self.estimators_[0], q, True, 2)[3]
        cols = [self._latent(ext, e, q, True)[2] for e in self.estimators_]  # glue: one latent column per class
        proba, _ = _ovr_combine(ext, cols, int(q.shape[0]))
        return proba

    def _predict_all(self, ext, q, out_kind):
        """`gpc_predict_all`: every binary fit against one cross kernel in
        one device session. `out_kind` 1 returns the int64 class codes of
        `predict`, 2 the float64 `predict_proba` rows."""
        n_star = int(q.shape[0])
        n_train, n_features = self.X_train_.shape
        kinds, kparams, ls_len, ls, n_ls = _kernel_arrays(self.kernel)
        fits = self.estimators_
        k = len(fits)
        if out_kind == 1:
            out = empty((n_star,), "<i8")
        else:
            out = empty((n_star, max(k, 2)), "<f8")
        xt = self.X_train_
        addrs = [
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_predict_all_binding.
            addr_ro(xt, name="xtrain"),
            addr_ro(q, name="xstar"),
            addr_ro(kinds, name="kinds"),
            addr_ro(kparams, name="kparams"),
            addr_ro(ls_len, name="ls_len"),
            addr_ro(ls, name="ls"),
            addr(out, name="out"),
        ]
        for est in fits:  # glue: four model addresses per binary fit
            addrs += [
                addr_ro(est.y_train_, name="y"),
                addr_ro(est.pi_, name="pi"),
                addr_ro(est.W_sr_, name="wsr"),
                addr_ro(est.L_, name="l"),
            ]
        ext.gpc_predict_all(
            addrs,
            # n_train, n_features, n_star, n_nodes, n_ls, out_kind, k
            [int(n_train), int(n_features), n_star, int(kinds.shape[0]), n_ls, int(out_kind), k],
        )
        return out

    def predict(self, X):
        """Two classes: `classes_[1]` where the latent mean is positive
        (`_gpc.py:287-290`). Past two: the one-vs-rest argmax of the
        unnormalized class probabilities (DEVIATION 2833)."""
        q = self._query(X)
        ext = self._extension()
        if _gpc_all(ext):
            return decode_labels(self.classes_, self._predict_all(ext, q, 1))
        if self.n_classes_ == 2:
            # `mean > 0` codes from the binding (`gpc_binary_out`), every build
            return decode_labels(self.classes_, self._latent(ext, self.estimators_[0], q, False, 1)[3])
        else:
            cols = [self._latent(ext, e, q, True)[2] for e in self.estimators_]  # glue: one latent column per class
            _, codes32 = _ovr_combine(ext, cols, int(q.shape[0]))
            codes = codes32.astype("<i8")
        return decode_labels(self.classes_, codes)

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        """`_gpc.py:279-353`, the `theta is None` arm only."""
        if theta is not None:
            raise NotImplementedError(
                f"{_NAME}: log_marginal_likelihood(theta) at other "
                "hyperparameters is refused; it serves the optimizer, which is "
                "refused (DEVIATION 1761)"
            )
        if eval_gradient:
            raise ValueError("Gradient can only be evaluated for theta!=None")
        if not hasattr(self, "estimators_"):
            raise ValueError(f"{_NAME}: call fit() first")
        return self.log_marginal_likelihood_value_

    def score(self, X, y):
        """Mean accuracy, a host summary outside the identity claim."""
        from ._expansion_metrics import accuracy_fraction
        from ._metrics_impl import _shape_of
        pred = self.predict(X)
        n_truth = int(_shape_of(y)[0]) if _shape_of(y) else 0
        if n_truth != len(pred):
            raise ValueError(f"{_NAME}: y has {n_truth} entries for {len(pred)} rows")
        # the x_metrics binding's grouped match count over the row count
        return accuracy_fraction(y, pred)

    # -- save and load ----------------------------------------------------------

    def save(self, path):
        """Write the fitted model as an npz: the kernel's postfix arrays, the
        float32 training matrix, the classes, and per binary fit its 0/1
        targets, `L_`, `pi_`, `W_sr_`, likelihood, iteration count and panel
        width. What prediction reads, byte for byte."""
        if not hasattr(self, "estimators_"):
            raise RuntimeError(f"{_NAME}: call fit before save")
        mode = getattr(self, "numeric_mode", None) or _backend.default_mode()
        if not isinstance(mode, str) or mode.strip().lower() not in _MODE_CODE:
            raise ValueError(f"{_NAME}: cannot save invalid numeric_mode {mode!r}")
        kinds, kparams, ls_len, ls, n_ls = _kernel_arrays(self.kernel)
        n_train, n_features = self.X_train_.shape
        fits = self.estimators_
        arrays = {
            "format": _GPC_FORMAT,
            "estimator": _ESTIMATOR,
            "numeric_mode": mode.strip().lower(),
            "classes": classes_member(self.classes_),
            "x": self.X_train_,
            "kinds": kinds,
            "kparams": kparams,
            "ls_len": ls_len,
            "ls": ls,
            # n_train, n_features, n_estimators, max_iter_predict, n_ls, n_classes
            "meta": Array.from_list([int(n_train), int(n_features), len(fits),
                                     int(self.max_iter_predict), int(n_ls),
                                     int(self.n_classes_)], "<i8"),
            "y": _stacked([e.y_train_ for e in fits], "<f4", (n_train,)),  # glue: one buffer per class fit
            "L": _stacked([e.L_ for e in fits], "<f4", (n_train * n_train,)),  # glue: one buffer per class fit
            "pi": _stacked([e.pi_ for e in fits], "<f4", (n_train,)),  # glue: one buffer per class fit
            "wsr": _stacked([e.W_sr_ for e in fits], "<f4", (n_train,)),  # glue: one buffer per class fit
            "lml": Array.from_list([float(e.log_marginal_likelihood_value_) for e in fits], "<f8"),  # glue: saved per-class fit scalars
            "n_iter": Array.from_list([int(e.n_iter_) for e in fits], "<i8"),  # glue: saved per-class fit scalars
            "nb": Array.from_list([int(e.nb_) for e in fits], "<i8"),  # glue: saved per-class fit scalars
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`. The result predicts; every array
        is read at its saved dtype and never cast."""
        arrays = _serialize.read_npz(path, _GPC_FORMAT)
        saved_as = _serialize.scalar_str(arrays, "estimator")
        if saved_as != _ESTIMATOR:
            raise ValueError(f"mojolearn: {path!r} was saved by {saved_as}, not {_ESTIMATOR}")
        mode = _serialize.scalar_str(arrays, "numeric_mode")
        if mode not in _MODE_CODE:
            raise ValueError(f"mojolearn: invalid saved numeric_mode {mode!r}")
        meta = _serialize.exact(arrays, "meta", "<i8").tolist()
        if len(meta) != 6:
            raise ValueError(f"mojolearn: {path!r} meta holds {len(meta)} fields, 6 are needed")
        n_train, n_features, n_est, max_iter, n_ls, n_classes = (int(v) for v in meta)  # glue: six saved meta fields
        kinds = _serialize.exact(arrays, "kinds", "<i4").tolist()
        kparams = _serialize.exact(arrays, "kparams", "<f4").tolist()
        ls_len = _serialize.exact(arrays, "ls_len", "<i4").tolist()
        table = _serialize.exact(arrays, "ls", "<f4").tolist()
        if len(kparams) != len(kinds) or len(ls_len) != len(kinds) or sum(ls_len) != n_ls:  # glue: kernel node count check
            raise ValueError(f"mojolearn: {path!r} kernel arrays disagree with each other")
        nodes, off = [], 0
        for k, p, ln in zip(kinds, kparams, ls_len):  # glue: rebuilds the kernel's postfix nodes
            nodes.append((k, p, table[off:off + ln]))
            off += ln
        obj = cls(kernel=_SavedKernel(nodes), max_iter_predict=max_iter)
        obj.numeric_mode = mode
        x = _serialize.exact(arrays, "x", "<f4")
        if tuple(x.shape) != (n_train, n_features):
            raise ValueError(f"mojolearn: {path!r} x has shape {tuple(x.shape)}, meta says "
                             f"({n_train}, {n_features})")
        classes = classes_from_member(arrays["classes"])
        want_est = 1 if n_classes == 2 else n_classes
        if len(classes) != n_classes or n_classes < 2 or n_est != want_est:
            raise ValueError(f"mojolearn: {path!r} holds {len(classes)} classes and {n_est} "
                             f"binary fits for n_classes {n_classes}")

        def rows(name, dtype, width):
            a = _serialize.exact(arrays, name, dtype)
            if tuple(a.shape) != ((n_est, width) if width else (n_est,)):
                raise ValueError(f"mojolearn: {path!r} {name} has shape {tuple(a.shape)}")
            return a

        ys = rows("y", "<f4", n_train)
        ls_ = rows("L", "<f4", n_train * n_train)
        pis = rows("pi", "<f4", n_train)
        wsrs = rows("wsr", "<f4", n_train)
        lmls = rows("lml", "<f8", 0)
        iters = rows("n_iter", "<i8", 0)
        nbs = rows("nb", "<i8", 0)
        fits = []
        for e in range(n_est):  # glue: one loaded fit per class
            # row e of each saved block, copied by memoryview runs (no per-element Python)
            fits.append(_BinaryLaplace(
                ys[e],
                ls_[e].reshape((n_train, n_train)),
                pis[e],
                wsrs[e],
                float(lmls[e]), int(iters[e]), int(nbs[e]),
            ))
        obj._set_fitted(x, classes, fits)
        if n_est > 1:
            obj.log_marginal_likelihood_value_ = obj.log_marginal_likelihood_value_
        return obj


__all__ = ["GaussianProcessClassifier"]
