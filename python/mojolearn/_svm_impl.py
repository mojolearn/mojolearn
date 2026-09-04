# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Support vector classification and regression on the GPU, mirroring
cuML's SVC and SVR.

PRIVATE MODULE. `SVC` and `SVR` are named exactly as scikit-learn names
them; both are re-exported from `mojolearn/__init__.py`.

`SVR` LANDED 2026-09-01 AND THIS HEADER USED TO SAY IT COULD NOT. It read
"THERE IS NO `SVR` IN THIS MODULE, AND THAT IS NOT AN OVERSIGHT ... an
`SVR` class here would have nothing to call", which was true while
`svmType != C_SVC` raised. It stopped being true at `fea6becc`
(2026-08-31), when the six rung-2 pieces were gated 44 of 44 and the
refusal came out of `SmoSolver.solve`; what was missing after that was
only the Python path, and the path is `svr_fit_host` / `svr_predict_host`
in `svm/estimator.mojo`, `svr_fit` / `svr_predict` in
`bindings/_mojolearn_svm.mojo`, and the class below.

WHAT THE TWO CLASSES SHARE, AND WHAT THEY DO NOT. One `SmoSolver`, one
kernel-matrix path, one set of refusals in
`svm/impl/svm/svm_parameter.mojo::check_rung1_scope`. They differ in the
gradient initialization (`SvrInit` writes `f = +-epsilon - y` and a `+-1`
label vector), the domain size (`n_train = 2 * n_rows`) and how the
coefficients are combined (`CombineCoefs` folds the two alpha halves), and
in nothing else. On this surface that shows up as one extra parameter,
`epsilon`, and three absences: no `classes_`, no `decision_function` and no
`predict_proba`.

THE LANE'S STANDING: `svm/`'s CLASSIFIER carries a THREE-VENDOR IDENTICAL
card. THE REGRESSOR DOES NOT, and `SVR`'s own docstring says so rather than
inheriting the sentence below. Round
11 of the E3 judge (2026-08-23, commit `144aa5b`) records it in
`archive/evidence/E3_RESULTS.md`: the IDENTICAL card bit-identical on Apple M4 <-> NVIDIA
H100 and Apple M4 <-> AMD MI325X, 32 stages. Two thirds of that is
re-checkable from this repository and was re-checked: the NVIDIA leg's
`svm.identical.card` and the AMD leg's agree on every recorded line
(`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/lanes/` and
`.../2026-08-23_172650-mojolearn-e2-amd/lanes/`), while their FAST cards
differ, as the round says. The Apple leg the judge diffed against is NOT
committed here, so the Apple half of the sentence rests on the judge's
record rather than on a file anyone can re-diff today.

That property belongs to `MOJOLEARN_NUMERIC_MODE=identical`. The FAST
build, which is the default, makes no cross-vendor claim at all.
"""

import importlib.machinery
import importlib.util
import os
import sys

import numpy as np

from . import _backend
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c

# cuML's `KernelType` values (kernel_params.hpp). Only two are ported.
_KERNEL_LINEAR = 0
_KERNEL_POLYNOMIAL = 1
_KERNEL_RBF = 2
_KERNEL_TANH = 3
_KERNEL_PRECOMPUTED = 4

_KERNELS = {"linear": _KERNEL_LINEAR, "rbf": _KERNEL_RBF}

# The names cuML accepts that this port does not, and the reason each is
# refused. Refused BY NAME rather than silently downgraded to 'rbf'.
_REFUSED_KERNELS = {
    "poly": (
        "POLYNOMIAL is not ported in rung 1; it is one identical_pow away "
        "(svm/NOT_IMPLEMENTED.tsv spells the kernel out) and was left unported "
        "rather than written without a gate"
    ),
    "polynomial": (
        "POLYNOMIAL is not ported in rung 1 (svm/NOT_IMPLEMENTED.tsv); cuML spells "
        "this kernel 'poly'"
    ),
    "sigmoid": (
        "TANH is not ported in rung 1: there is no identical_tanh in "
        "checks/numerics.mojo, so the kernel has no bit-pinned spelling "
        "yet (svm/NOT_IMPLEMENTED.tsv)"
    ),
    "tanh": (
        "TANH is not ported in rung 1 (svm/NOT_IMPLEMENTED.tsv); cuML spells this "
        "kernel 'sigmoid'"
    ),
    "precomputed": (
        "PRECOMPUTED is not ported: kernelcache.cuh's "
        "extractColumnsForPrecomputed and svc_impl.cuh's precomputed "
        "predict arm are both unported (svm/NOT_IMPLEMENTED.tsv)"
    ),
}

_EXT_NAME = "_mojolearn_svm"
_PKG = __name__.rsplit(".", 1)[0]
_ext_cache = None


def _extension(mode=None):
    """The `_mojolearn_svm` extension, in the numeric mode the package
    asked for, cross-checked against the mode the binary was COMPILED in.

    WHY THIS IS NOT JUST `from . import _mojolearn_svm`. `_backend.py`'s
    `_MODULES` tuple lists the five older extensions and does not list
    this one, so under `MOJOLEARN_NUMERIC_MODE=identical` the selector
    installs nothing for it and a plain relative import would load the
    FAST binary sitting next to it -- an identical-mode process running
    FAST arithmetic and labelling it identical. That is the exact failure
    `_backend.py`'s own header calls "a mislabelled measurement".

    So this loads the right file itself and then asks the binary what it
    was compiled as. Adding `_mojolearn_svm` to `_MODULES` and
    `_build_script` in `_backend.py` is the tidier fix and belongs to
    whoever owns that file; when it lands, the module is already in
    `sys.modules` and this function uses it unchanged.

    Deliberately lazy, not called at import: `_backend.py` installs a stub
    that imports fine and raises BY NAME on first use when an identical
    binary is missing, so that one unbuilt extension does not take the
    whole package down. Same policy here.
    """
    global _ext_cache
    # PER-MODE CACHE, keyed by tier, since 2026-08-29. It was a single slot
    # holding whichever tier asked first, which is fine while the mode is a
    # process-wide environment variable and wrong the moment it is a
    # per-estimator parameter: the second tier to ask would have been handed
    # the first one's binary under its own name.
    if not isinstance(_ext_cache, dict):
        _ext_cache = {}
    mode = (mode or _backend.default_mode()).strip().lower()
    if mode in _ext_cache:
        return _ext_cache[mode]
    full = f"{_PKG}._sets.{mode}.{_EXT_NAME}" if mode != "fast" else f"{_PKG}.{_EXT_NAME}"
    mod = sys.modules.get(full)
    if mod is None:
        # The directory comes from `_backend.tier_dir`, the one place the
        # tier and vendor axes become a path (python/mojolearn/<vendor>/
        # <tier>/ on the Linux wheel, the package directory otherwise).
        path = os.path.join(_backend.tier_dir(mode), _EXT_NAME + ".so")
        if not os.path.exists(path):
            raise ImportError(
                f"mojolearn: {path} is not built; build it with\n    "
                + (f"MOJOLEARN_NUMERIC_MODE={mode} " if mode != "fast" else "")
                + "bash bindings/build_svm.sh"
            )
        loader = importlib.machinery.ExtensionFileLoader(full, path)
        spec = importlib.util.spec_from_loader(full, loader, origin=path)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        sys.modules[full] = mod
        # Publish under the PLAIN package attribute only for the tier this
        # process defaults to. Doing it unconditionally, as it did until
        # 2026-08-29, made `mojolearn._mojolearn_svm` mean "whichever tier
        # asked last" -- so one estimator built with numeric_mode="identical"
        # would silently repoint the name every other caller reads.
        if mode == _backend.default_mode():
            setattr(sys.modules[_PKG], _EXT_NAME, mod)
    # A NAME lookup, not a boolean: the middle tier reports 2 and the
    # old spelling called it "fast", so a deterministic binary matched
    # a fast request and the cross-check passed on the wrong arm.
    compiled = _backend._CODE_MODE.get(mod.svm_numeric_mode(), "unknown")
    requested = mode
    if compiled != requested:
        raise ImportError(
            f"mojolearn: {_EXT_NAME} was compiled {compiled} but "
            f"MOJOLEARN_NUMERIC_MODE asked for {requested} -- a binary is in "
            "the wrong directory; rebuild the sets with\n    "
            "bash bindings/build_svm.sh\n    "
            "MOJOLEARN_NUMERIC_MODE=deterministic bash bindings/build_svm.sh\n    "
            "MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_svm.sh"
        )
    _ext_cache[mode] = mod
    return mod


def _as_labels(y):
    """`y` as a 1-D float32 vector plus the two distinct labels in cuML's
    order (sorted ascending). Raises the binary-only refusal HERE, before
    any device work, because that is the most common way to reach this
    class by mistake."""
    a = np.asarray(y)
    if a.ndim != 1:
        raise ValueError(
            f"mojolearn SVC: y must be 1-D, got {a.ndim}-D shape {a.shape}"
        )
    classes = np.unique(a)
    if classes.shape[0] != 2:
        raise NotImplementedError(
            "mojolearn SVC: only binary classification is implemented, got "
            f"{classes.shape[0]} classes {classes.tolist()!r}. cuML's own C++ "
            "asserts the same thing (svc_impl.cuh: 'Only binary "
            "classification is implemented at the moment'); their multiclass "
            "is a Python-layer one-vs-one/one-vs-rest wrapper and is not "
            "ported (svm/NOT_IMPLEMENTED.tsv)"
        )
    f = np.ascontiguousarray(a, dtype=np.float32)
    if not np.isfinite(f).all():
        raise ValueError(
            "mojolearn SVC: y contains a non-finite value (DEVIATION 636: a "
            "NaN or inf cannot be fitted; a computed NaN carries a "
            "vendor-specific payload and cannot sit in a hashed stage)"
        )
    return f, classes


class SVC(NumericModeMixin):
    """Binary C-support vector classification, backed by the ported cuML
    SMO solver and cuVS kernel matrices (`svm/`, DEVIATIONS 630-637;
    `svm/README.md`), the scikit-learn surface.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter that is accepted and ignored is a wrong answer
    waiting for a caller (the house rule):

        C               honored   the penalty. Must be finite and positive;
                                  a non-finite C is refused by name
                                  (DEVIATION 636)
        kernel          honored   'linear' and 'rbf' only. 'poly',
                                  'sigmoid' and 'precomputed' are cuML's
                                  other three and each is REFUSED BY NAME
                                  with what is missing
        gamma           honored   a finite float >= 0, or the string 'auto'
                                  (= 1 / n_features, cuML's `_get_gamma`).
                                  'scale' is REFUSED -- see DEVIATION 870
                                  below, and note it is cuML's default
        degree          refused   read only by POLYNOMIAL, which is refused
        coef0           refused   read only by POLYNOMIAL and TANH, both
                                  refused
        tol             honored   the stopping tolerance; must be finite and
                                  positive (DEVIATION 636)
        cache_size      honored   ONLY as the prediction buffer, see
                                  DEVIATION 871 below
        class_weight    refused   upstream it becomes `sample_weight`, and
                                  `sample_weight` is not ported: the
                                  weighted `InitPenalty` arm (C_vec = C * w)
                                  has no port (svm/NOT_IMPLEMENTED.tsv)
        max_iter        honored   cuML's total inner-iteration cap; -1 (the
                                  default) is no limit
        nochange_steps  honored   cuML's convergence rule, transcribed with
                                  its n_small_diff counter
        verbose         refused   anything truthy. It selects LOG LINES
                                  upstream (CUML_LOG_DEBUG); this port
                                  prints none, so accepting it would be
                                  accepting-and-ignoring
        random_state    refused   anything but None. The binary C-SVC solver
                                  draws no random numbers; cuML threads
                                  `random_state` only into its multiclass
                                  wrapper, which is not ported
        decision_       refused   anything but 'ovo'. It picks between
          function_shape          cuML's one-vs-one and one-vs-rest
                                  multiclass wrappers; there is no
                                  multiclass here to shape
        probability     refused   Platt scaling is not in cuML's C++ surface
                                  at all (svm/NOT_IMPLEMENTED.tsv)
        output_type     refused   a cuML-internal array-type selector; this
                                  package returns NumPy
        sample_weight   refused   in fit(); see class_weight
        sparse X        refused   `svcFitSparse` / `svcPredictSparse` and
                                  every CSR arm are unported; dense
                                  row-major float32 only

    Non-finite cells of `X` are refused by name inside the Mojo entry
    (DEVIATION 636), naming the flat index, rather than being fitted.

    DEVIATION 870: `gamma='scale'` IS REFUSED AND THE DEFAULT HERE IS
    'auto', WHICH IS NOT cuML's DEFAULT. Theirs resolves 'scale' to
    `1 / (n_features * X.var())`, a float32 reduction over the whole
    matrix whose last bits are the reduction library's fold shape. Every
    bit of this fit is a function of gamma's bits -- the kernel matrix,
    the working sets, the alphas, `b` -- so a gamma whose last bit is the
    host's would put a host into the middle of a cross-vendor identity
    claim. `KernelDensity` refuses `bandwidth='scott'` for exactly this
    reason and with exactly this instruction: compute it yourself and pass
    the number, so the number that ran is the number you passed.

        gamma = 1.0 / (X.shape[1] * float(np.asarray(X, np.float32).var()))

    'auto' is kept because `1 / n_features` is an integer reciprocal in
    float64 and is the same bits on every host.

    DEVIATION 871: `cache_size` IS HONORED ONLY AT PREDICT. Upstream it is
    two things under one name: the training-time `raft::cache` LRU kernel
    cache, and the ceiling on the prediction kernel-tile buffer
    (`svm_base.pyx:554`, passed as `buffer_size` to `svcPredict`). The
    prediction half is honored here, exactly, and it is a real knob: it
    sets the prediction batch size, and `check_device_is_launch_invariant`
    holds the answer fixed over it from 0.001 MiB to 200 MiB. The training
    half is NOT ported -- the solver always runs cuML's own
    `n_cache_sets == 0` path -- so `cache_size` does not affect training
    time here the way it does upstream. It cannot affect training RESULTS
    upstream either (`svm/NOT_IMPLEMENTED.tsv` carries that determinism
    statement), so nothing numeric hangs on it.

    DEVIATION 873: the fitted model crosses back to the host and is
    uploaded again at every `predict` / `decision_function`, because the
    binding retains no device pointer. `fit` therefore allocates
    worst-case output buffers -- `n_samples` dual coefficients and
    `n_samples x n_features` support-vector floats -- since `n_support` is
    not known until the solve finishes. On a matrix that is mostly support
    vectors that is a second copy of X.

    Attributes
    ----------
    classes_ : ndarray (2,)
        The two distinct labels, sorted ascending, in the dtype of the `y`
        that was passed. `classes_[1]` is the one the solver maps to +1
        (`getOvrlabels(..., idx=1)`), so it fixes the sign of the decision
        function.
    support_ : ndarray (n_SV,) int32
        Indices of the support vectors in the training matrix.
    support_vectors_ : ndarray (n_SV, n_features) float32
    dual_coef_ : ndarray (1, n_SV) float32
        scikit-learn's 2-D layout of cuML's 1-D `dual_coefs`.
    intercept_ : ndarray (1,) float32
    n_support_ : int
        cuML's scalar count. scikit-learn's attribute of this name is a
        per-class array; this one is not, and the difference is here
        rather than in a surprise.
    n_iter_ : int
        Total inner SMO iterations.
    coef_ : ndarray (1, n_features) float32
        `dual_coef_ @ support_vectors_`, and only for `kernel='linear'`;
        raises AttributeError otherwise, as scikit-learn does.
    n_features_in_ : int
    """

    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="auto",
        coef0=0.0,
        tol=1e-3,
        cache_size=1024.0,
        max_iter=-1,
        nochange_steps=1000,
        verbose=False,
        output_type=None,
        random_state=None,
        class_weight=None,
        decision_function_shape="ovo",
        probability=False,
    ):
        if not isinstance(kernel, str):
            raise ValueError("mojolearn SVC: kernel is a name")
        k = kernel.lower()
        if k in _REFUSED_KERNELS:
            raise NotImplementedError(
                f"mojolearn SVC: kernel={kernel!r} is refused; "
                + _REFUSED_KERNELS[k]
            )
        if k not in _KERNELS:
            raise ValueError(
                f"mojolearn SVC: kernel={kernel!r} is not a kernel name; "
                f"this port carries {sorted(_KERNELS)} and refuses cuML's "
                f"other three by name ({sorted(_REFUSED_KERNELS)})"
            )
        if isinstance(gamma, str):
            g = gamma.lower()
            if g == "scale":
                raise NotImplementedError(
                    "mojolearn SVC: gamma='scale' is refused (DEVIATION 870). "
                    "It is 1 / (n_features * X.var()), a float32 reduction "
                    "whose last bits are the host reduction's fold shape, and "
                    "every bit of this fit is a function of gamma's bits. "
                    "Compute it yourself and pass the number:\n    "
                    "gamma = 1.0 / (X.shape[1] * float(np.asarray(X, "
                    "np.float32).var()))\n"
                    "'auto' (= 1 / n_features) is exact on every host and is "
                    "this class's default; note cuML's default is 'scale'."
                )
            if g != "auto":
                raise ValueError(
                    f"mojolearn SVC: gamma={gamma!r} is not a name; it is "
                    "'auto', or a float ('scale' is refused, DEVIATION 870)"
                )
            gamma = "auto"
        else:
            gamma = float(gamma)
            if not np.isfinite(gamma) or gamma < 0.0:
                raise ValueError(
                    "mojolearn SVC: gamma must be finite and >= 0 for the RBF "
                    f"kernel, got {gamma!r} (DEVIATION 636; scikit-learn's own "
                    "constraint is gamma >= 0)"
                )
        if degree != 3:
            raise NotImplementedError(
                f"mojolearn SVC: degree={degree!r} is refused; it is read only "
                "by the POLYNOMIAL kernel, which is not ported. Passing it "
                "with a ported kernel would be a parameter accepted and "
                "ignored"
            )
        if coef0 != 0.0:
            raise NotImplementedError(
                f"mojolearn SVC: coef0={coef0!r} is refused; it is read only "
                "by the POLYNOMIAL and TANH kernels, neither of which is "
                "ported"
            )
        C = float(C)
        if not np.isfinite(C):
            raise ValueError(
                f"mojolearn SVC: C must be finite, got {C!r} (DEVIATION 636)"
            )
        if C <= 0.0:
            raise ValueError(f"mojolearn SVC: C must be positive, got {C!r}")
        tol = float(tol)
        if not np.isfinite(tol):
            raise ValueError(
                f"mojolearn SVC: tol must be finite, got {tol!r} (DEVIATION 636)"
            )
        if tol <= 0.0:
            raise ValueError(f"mojolearn SVC: tol must be positive, got {tol!r}")
        cache_size = float(cache_size)
        if not (cache_size > 0.0) or not np.isfinite(cache_size):
            raise ValueError(
                "mojolearn SVC: cache_size is the prediction buffer here "
                f"(DEVIATION 871) and must be a positive finite MiB, got "
                f"{cache_size!r}"
            )
        max_iter = int(max_iter)
        if max_iter == 0 or max_iter < -1:
            raise ValueError(
                "mojolearn SVC: max_iter is a positive cap or -1 for no "
                f"limit (cuML's default), got {max_iter!r}"
            )
        nochange_steps = int(nochange_steps)
        if nochange_steps < 0:
            raise ValueError(
                "mojolearn SVC: nochange_steps cannot be negative, got "
                f"{nochange_steps!r}"
            )
        if verbose:
            raise NotImplementedError(
                "mojolearn SVC: verbose is refused; upstream it selects "
                "CUML_LOG_DEBUG lines and this port prints none, so honoring "
                "it is impossible and accepting it would be accepting-and-"
                "ignoring"
            )
        if output_type is not None:
            raise NotImplementedError(
                "mojolearn SVC: output_type is a cuML-internal array-type "
                "selector; this package returns NumPy"
            )
        if random_state is not None:
            raise NotImplementedError(
                "mojolearn SVC: random_state is refused; the binary C-SVC "
                "solver draws no random numbers, and cuML threads this "
                "parameter only into its multiclass wrapper, which is not "
                "ported"
            )
        if class_weight is not None:
            raise NotImplementedError(
                "mojolearn SVC: class_weight is refused; upstream it becomes "
                "sample_weight, and the weighted InitPenalty arm "
                "(C_vec = C * w) is not ported (svm/NOT_IMPLEMENTED.tsv)"
            )
        if decision_function_shape != "ovo":
            raise NotImplementedError(
                f"mojolearn SVC: decision_function_shape="
                f"{decision_function_shape!r} is refused; it picks between "
                "cuML's one-vs-one and one-vs-rest multiclass wrappers and "
                "there is no multiclass here to shape"
            )
        if probability:
            raise NotImplementedError(
                "mojolearn SVC: probability is refused; Platt scaling is not "
                "in cuML's C++ surface at all (svm/NOT_IMPLEMENTED.tsv)"
            )
        self.C = C
        self.kernel = k
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.nochange_steps = nochange_steps
        self.verbose = False
        self.output_type = None
        self.random_state = None
        self.class_weight = None
        self.decision_function_shape = "ovo"
        self.probability = False

    def _resolve_gamma(self, n_features):
        """cuML's `_get_gamma` minus the refused 'scale' arm. `1 / n_cols`
        in float64 is exact for every n_cols that is a power of two and
        correctly rounded otherwise, on every host."""
        if self.gamma == "auto":
            return 1.0 / float(n_features)
        return float(self.gamma)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn SVC: sample_weight is not ported; the weighted "
                "InitPenalty arm (C_vec = C * w) has no port "
                "(svm/NOT_IMPLEMENTED.tsv). class_weight is the same refusal"
            )
        x, self.input_copied_ = as_f32_c(X, "X")
        labels, classes = _as_labels(y)
        n_rows, n_cols = x.shape
        if labels.shape[0] != n_rows:
            raise ValueError(
                f"mojolearn SVC: y has {labels.shape[0]} entries, X has "
                f"{n_rows} rows"
            )
        gamma = self._resolve_gamma(n_cols)

        dual = np.empty(n_rows, dtype=np.float32)
        support = np.empty(n_rows, dtype=np.int32)
        sv = np.empty(n_rows * n_cols, dtype=np.float32)
        info = np.empty(5, dtype=np.float64)
        n_support = _extension(getattr(self, 'numeric_mode', None)).svc_fit(
            _addr_ro(x),
            _addr_ro(labels),
            _addr(dual),
            _addr(support),
            _addr(sv),
            _addr(info),
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_fit_binding.
            # n_rows, n_features, kernel, gamma, C, tol, max_iter,
            # nochange_steps
            [n_rows, n_cols, _KERNELS[self.kernel], gamma, self.C, self.tol,
             self.max_iter, self.nochange_steps],
        )
        n_support = int(n_support)
        b = np.float32(info[0])
        label0 = np.float32(info[3])
        label1 = np.float32(info[4])
        # THE LABEL PAIR IS CHECKED, NOT ASSUMED. The solver finds the two
        # distinct labels with its own host sort and maps the LARGER to +1;
        # if that disagreed with numpy's unique, `predict` would map the
        # device's answer onto the wrong class and the error would look like
        # a bad model rather than a bad boundary.
        if label0 != np.float32(classes[0]) or label1 != np.float32(classes[1]):
            raise RuntimeError(
                "mojolearn SVC: the solver's label pair "
                f"({float(label0)}, {float(label1)}) does not match numpy's "
                f"unique ({float(classes[0])}, {float(classes[1])}); the "
                "class mapping cannot be trusted"
            )
        self.classes_ = classes
        self.n_features_in_ = n_cols
        self.n_support_ = n_support
        self.n_iter_ = int(info[2])
        self.intercept_ = np.array([b], dtype=np.float32)
        self.support_ = np.ascontiguousarray(support[:n_support])
        self.support_vectors_ = np.ascontiguousarray(
            sv[: n_support * n_cols].reshape(n_support, n_cols)
        )
        self.dual_coef_ = np.ascontiguousarray(dual[:n_support]).reshape(1, n_support)
        self._gamma = gamma
        self._label0 = label0
        self._label1 = label1
        return self

    @property
    def coef_(self):
        if not hasattr(self, "dual_coef_"):
            raise AttributeError("mojolearn SVC: call fit() first")
        if self.kernel != "linear":
            raise AttributeError(
                "mojolearn SVC: coef_ is only available for kernel='linear'"
            )
        if self.n_support_ == 0:
            return np.zeros((1, self.n_features_in_), dtype=np.float32)
        return np.ascontiguousarray(self.dual_coef_ @ self.support_vectors_)

    def _run(self, X, predict_class):
        if not hasattr(self, "dual_coef_"):
            raise ValueError("mojolearn SVC: call fit() first")
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn SVC: X has {q.shape[1]} features, fit saw "
                f"{self.n_features_in_}"
            )
        n_rows = q.shape[0]
        out = np.empty(n_rows, dtype=np.float32)
        # Kept in locals so the arrays outlive the call; the Mojo side
        # borrows these addresses and owns nothing (`_arrays.py`).
        dual = self.dual_coef_
        sv = self.support_vectors_
        _extension(getattr(self, 'numeric_mode', None)).svc_predict(
            _addr_ro(q),
            _addr_ro(dual) if self.n_support_ > 0 else 0,
            _addr_ro(sv) if self.n_support_ > 0 else 0,
            _addr(out),
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_predict_binding.
            # n_rows, n_features, n_support, b, classes[0], classes[1],
            # kernel, gamma, predict_class, cache_size_mib
            [n_rows, self.n_features_in_, self.n_support_,
             float(self.intercept_[0]), float(self._label0),
             float(self._label1), _KERNELS[self.kernel], self._gamma,
             1 if predict_class else 0, self.cache_size],
        )
        return out

    def decision_function(self, X):
        """The raw `sum_j K(x, sv_j) dual_j + b`, one value per row.
        scikit-learn returns the same shape for a binary problem."""
        return self._run(X, False)

    def predict(self, X):
        """cuML's `applyPrediction` epilogue, ON THE DEVICE: the label is
        chosen there by `val + b < 0 ? classes_[0] : classes_[1]` and this
        maps the float32 label it wrote back into `classes_`'s dtype."""
        raw = self._run(X, True)
        return np.where(raw == self._label1, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        raise NotImplementedError(
            "mojolearn SVC: predict_proba is not available; Platt scaling is "
            "not in cuML's C++ surface at all (svm/NOT_IMPLEMENTED.tsv), which is "
            "why the constructor refuses probability=True"
        )

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


def _as_targets(y, n_rows):
    """`y` as a 1-D contiguous float32 vector of regression targets.

    NO `np.unique`, NO label pair, NO class count. `svr_impl.cuh:68` hands
    `y` straight to `SmoSolver::Solve`, so a target is any finite float and
    two rows sharing one are ordinary rather than interesting.

    NO FINITENESS CHECK HERE EITHER, and that is deliberate rather than an
    omission copied from `_as_labels`. `check_finite_list` in
    `svm/impl/svm/svr_impl.mojo::svr_fit` refuses a non-finite target BY
    NAME with the offending flat index (DEVIATION 636), before any device
    work reaches a kernel, and a
    duplicate check on this side would make that refusal unreachable from
    Python and therefore untestable. The house rule is to plumb the value
    through and let the named refusal fire.
    """
    a = np.asarray(y)
    if a.ndim != 1:
        raise ValueError(
            f"mojolearn SVR: y must be 1-D, got {a.ndim}-D shape {a.shape}"
        )
    if a.shape[0] != n_rows:
        raise ValueError(
            f"mojolearn SVR: y has {a.shape[0]} entries, X has {n_rows} rows"
        )
    return np.ascontiguousarray(a, dtype=np.float32)


class SVR(NumericModeMixin):
    """Epsilon-support vector regression, backed by the ported cuML SMO
    solver and cuVS kernel matrices (`svm/`, DEVIATIONS 630-637;
    `svm/README.md`), the scikit-learn surface.

    THE TUBE IS THE ALGORITHM. Rows whose prediction lands inside a band of
    half-width `epsilon` around the target cost nothing, so they never
    become support vectors; the fit is determined by the rows outside it.
    Raising `epsilon` widens the band and can only take rows out of the
    support set, never add them, which is the one property of this
    estimator a caller can check without a reference implementation, and
    `python/mojolearn/tests/test_svr_surface.py` checks it.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter that is accepted and ignored is a wrong answer
    waiting for a caller (the house rule). The file named on a refusal is
    the file that raises it.

        C               honored   the penalty. Refused if not finite or not
                                  positive, by `_svm_impl.py` and again by
                                  `svm/impl/svm/svm_parameter.mojo`
                                  (DEVIATION 636)
        epsilon         honored   the half-width of the insensitive tube,
                                  cuML's `param.epsilon`, the parameter the
                                  whole rung-2 port exists for. Passed
                                  through UNCLAMPED so that
                                  `svm/impl/svm/svm_parameter.mojo::
                                  check_rung1_scope`'s two refusals -- not
                                  finite, and negative -- stay reachable
                                  from this surface. They fire before any
                                  device context exists
        kernel          honored   'linear' and 'rbf' only. 'poly',
                                  'sigmoid' and 'precomputed' are cuML's
                                  other three and each is REFUSED BY NAME
                                  by `_svm_impl.py` with what is missing
        gamma           honored   a finite float >= 0, or the string 'auto'
                                  (= 1 / n_features, cuML's `_get_gamma`).
                                  'scale' is REFUSED by `_svm_impl.py` --
                                  see DEVIATION 870 below, and note it is
                                  both cuML's and scikit-learn's default
        degree          refused   `_svm_impl.py`. Read only by POLYNOMIAL,
                                  which is refused
        coef0           refused   `_svm_impl.py`. Read only by POLYNOMIAL
                                  and TANH, both refused
        tol             honored   the stopping tolerance. Refused if not
                                  finite or not positive, by `_svm_impl.py`
                                  and `svm/impl/svm/svm_parameter.mojo`
        cache_size      honored   ONLY as the prediction buffer, see
                                  DEVIATION 871 below. The TRAINING LRU it
                                  also names upstream is unported and
                                  `svm/impl/svm/svm_parameter.mojo` refuses
                                  a non-zero one; `svm/estimator.mojo` pins
                                  the training value at 0 so that refusal
                                  is never reached from here
        max_iter        honored   cuML's total inner-iteration cap; -1 (the
                                  default) is no limit. `_svm_impl.py`
                                  refuses 0 and anything below -1
        nochange_steps  honored   cuML's convergence rule, transcribed with
                                  its n_small_diff counter. NOT a
                                  scikit-learn parameter; it is cuML's, and
                                  it is here because the solver reads it
        verbose         refused   `_svm_impl.py`, for anything truthy. It
                                  selects LOG LINES upstream
                                  (CUML_LOG_DEBUG); this port prints none,
                                  so accepting it would be
                                  accepting-and-ignoring
        output_type     refused   `_svm_impl.py`. A cuML-internal
                                  array-type selector; this package returns
                                  NumPy
        shrinking       absent    NOT A PARAMETER OF THIS CLASS, so passing
                                  it is a TypeError naming it. It is
                                  scikit-learn's (libsvm's) shrinking
                                  heuristic and cuML has no such thing:
                                  there is no row for it in
                                  `svm/NOT_IMPLEMENTED.tsv` because there is
                                  nothing upstream of this port to leave
                                  unported. `SVC` omits it for the same
                                  reason
        sample_weight   refused   `_svm_impl.py`, in fit(). The weighted
                                  `InitPenalty` arm (C_vec = C * w) has no
                                  port (`svm/NOT_IMPLEMENTED.tsv`)
        sparse X        refused   `svrFitSparse` and every CSR arm are
                                  unported; dense row-major float32 only.
                                  `_arrays.py::as_f32_c` is what refuses
        non-finite X    refused   `svm/impl/svm/svr_impl.mojo` at fit and
                                  `svm/impl/svm/svc_impl.mojo` at predict
                                  name the flat index (DEVIATION 636)
                                  rather than fitting it
        non-finite y    refused   the same `check_finite_list`, same
                                  message, and it is the ONLY check `y`
                                  gets on this class

    THE PARAMETER ORDER IS scikit-learn's `SVR`, and every name it shares
    with scikit-learn means what scikit-learn means by it. TWO DEFAULTS
    DIFFER and both are stated where they bite: `gamma` is 'auto' here and
    'scale' there (DEVIATION 870, below), and `cache_size` is 1024.0 MiB
    here and 200 there because it is a different quantity on this surface
    (DEVIATION 871). `shrinking` is absent, above.

    THIS CLASS MAKES NO CROSS-VENDOR CLAIM, and that is a narrower statement
    than `SVC`'s. The classifier's 32-stage IDENTICAL card was diffed on
    Apple M4, NVIDIA H100 and AMD MI325X at `a0a0eee` (E3 round 13,
    2026-08-28). The REGRESSION path was gated after that leg and has not
    been in a three-vendor round: what stands behind it is 44 of 44 gates at
    `fea6becc` on one box, of which 26 are the SVR ones, including four
    property gates derived from the epsilon-insensitive formulation rather
    than from this solver (a dual objective that must not increase, a KKT
    gap, a tube bound, and a gradient recomputed from alpha alone that
    matches the solver's `f` to 1.5e-07). A three-vendor SVR card is OWED.

    DEVIATION 870: `gamma='scale'` IS REFUSED AND THE DEFAULT HERE IS
    'auto'. Theirs resolves 'scale' to `1 / (n_features * X.var())`, a
    float32 reduction over the whole matrix whose last bits are the
    reduction library's fold shape. Every bit of this fit is a function of
    gamma's bits, so a gamma whose last bit is the host's would put a host
    into the middle of an identity claim. Compute it yourself and pass the
    number, so the number that ran is the number you passed:

        gamma = 1.0 / (X.shape[1] * float(np.asarray(X, np.float32).var()))

    'auto' is kept because `1 / n_features` is an integer reciprocal in
    float64 and is the same bits on every host.

    DEVIATION 871: `cache_size` IS HONORED ONLY AT PREDICT, exactly as it is
    on `SVC`. Upstream it is two things under one name, the training-time
    `raft::cache` LRU and the ceiling on the prediction kernel-tile buffer
    (`svm_base.pyx:554`). The prediction half is honored here exactly and is
    a real knob: it sets the prediction batch size, and
    `check_svr_device_is_launch_invariant` holds the answer fixed over it
    from 0.001 MiB to 200 MiB ON THE REGRESSION PATH, which is a separate
    gate from the classifier's because SVR runs `UpdateF` twice per batch
    and so interleaves twice as many launches. The training half is NOT
    ported.

    DEVIATION 873: the fitted model crosses back to the host and is uploaded
    again at every `predict`, because the binding retains no device pointer.
    `fit` therefore allocates worst-case output buffers, `n_samples` dual
    coefficients and `n_samples x n_features` support-vector floats, since
    `n_support` is not known until the solve finishes. THOSE SIZES ARE NOT
    DOUBLED for the regressor even though its solver domain is: `Results`
    folds the two alpha halves before it selects, so at most `n_samples`
    coefficients can be written.

    Attributes
    ----------
    support_ : ndarray (n_SV,) int32
        Indices of the support vectors in the training matrix.
    support_vectors_ : ndarray (n_SV, n_features) float32
    dual_coef_ : ndarray (1, n_SV) float32
        scikit-learn's 2-D layout of cuML's 1-D folded `dual_coefs`. Each
        entry is `alpha_i - alpha*_i` for one row, which is why there are
        `n_SV` of them and not `2 * n_SV`.
    intercept_ : ndarray (1,) float32
    n_support_ : int
        cuML's scalar count.
    n_iter_ : int
        Total inner SMO iterations.
    coef_ : ndarray (1, n_features) float32
        `dual_coef_ @ support_vectors_`, and only for `kernel='linear'`;
        raises AttributeError otherwise, as scikit-learn does.
    n_features_in_ : int
    """

    def __init__(
        self,
        *,
        kernel="rbf",
        degree=3,
        gamma="auto",
        coef0=0.0,
        tol=1e-3,
        C=1.0,
        epsilon=0.1,
        cache_size=1024.0,
        verbose=False,
        max_iter=-1,
        nochange_steps=1000,
        output_type=None,
    ):
        if not isinstance(kernel, str):
            raise ValueError("mojolearn SVR: kernel is a name")
        k = kernel.lower()
        if k in _REFUSED_KERNELS:
            raise NotImplementedError(
                f"mojolearn SVR: kernel={kernel!r} is refused; "
                + _REFUSED_KERNELS[k]
            )
        if k not in _KERNELS:
            raise ValueError(
                f"mojolearn SVR: kernel={kernel!r} is not a kernel name; "
                f"this port carries {sorted(_KERNELS)} and refuses cuML's "
                f"other three by name ({sorted(_REFUSED_KERNELS)})"
            )
        if isinstance(gamma, str):
            g = gamma.lower()
            if g == "scale":
                raise NotImplementedError(
                    "mojolearn SVR: gamma='scale' is refused (DEVIATION 870). "
                    "It is 1 / (n_features * X.var()), a float32 reduction "
                    "whose last bits are the host reduction's fold shape, and "
                    "every bit of this fit is a function of gamma's bits. "
                    "Compute it yourself and pass the number:\n    "
                    "gamma = 1.0 / (X.shape[1] * float(np.asarray(X, "
                    "np.float32).var()))\n"
                    "'auto' (= 1 / n_features) is exact on every host and is "
                    "this class's default; note scikit-learn's default is "
                    "'scale'."
                )
            if g != "auto":
                raise ValueError(
                    f"mojolearn SVR: gamma={gamma!r} is not a name; it is "
                    "'auto', or a float ('scale' is refused, DEVIATION 870)"
                )
            gamma = "auto"
        else:
            gamma = float(gamma)
            if not np.isfinite(gamma) or gamma < 0.0:
                raise ValueError(
                    "mojolearn SVR: gamma must be finite and >= 0 for the RBF "
                    f"kernel, got {gamma!r} (DEVIATION 636; scikit-learn's own "
                    "constraint is gamma >= 0)"
                )
        if degree != 3:
            raise NotImplementedError(
                f"mojolearn SVR: degree={degree!r} is refused; it is read only "
                "by the POLYNOMIAL kernel, which is not ported. Passing it "
                "with a ported kernel would be a parameter accepted and "
                "ignored"
            )
        if coef0 != 0.0:
            raise NotImplementedError(
                f"mojolearn SVR: coef0={coef0!r} is refused; it is read only "
                "by the POLYNOMIAL and TANH kernels, neither of which is "
                "ported"
            )
        C = float(C)
        if not np.isfinite(C):
            raise ValueError(
                f"mojolearn SVR: C must be finite, got {C!r} (DEVIATION 636)"
            )
        if C <= 0.0:
            raise ValueError(f"mojolearn SVR: C must be positive, got {C!r}")
        tol = float(tol)
        if not np.isfinite(tol):
            raise ValueError(
                f"mojolearn SVR: tol must be finite, got {tol!r} (DEVIATION 636)"
            )
        if tol <= 0.0:
            raise ValueError(f"mojolearn SVR: tol must be positive, got {tol!r}")
        # EPSILON IS NOT VALIDATED HERE, ON PURPOSE. `float()` is a
        # conversion, not a policy: a string that is not a number still
        # raises here, and every value that IS a number goes to the Mojo
        # side unchanged so that `check_rung1_scope`'s "must be finite" and
        # "must be non-negative" refusals stay reachable from Python. That
        # is the same rule that keeps the non-finite-X refusal reachable.
        epsilon = float(epsilon)
        cache_size = float(cache_size)
        if not (cache_size > 0.0) or not np.isfinite(cache_size):
            raise ValueError(
                "mojolearn SVR: cache_size is the prediction buffer here "
                f"(DEVIATION 871) and must be a positive finite MiB, got "
                f"{cache_size!r}"
            )
        max_iter = int(max_iter)
        if max_iter == 0 or max_iter < -1:
            raise ValueError(
                "mojolearn SVR: max_iter is a positive cap or -1 for no "
                f"limit (cuML's default), got {max_iter!r}"
            )
        nochange_steps = int(nochange_steps)
        if nochange_steps < 0:
            raise ValueError(
                "mojolearn SVR: nochange_steps cannot be negative, got "
                f"{nochange_steps!r}"
            )
        if verbose:
            raise NotImplementedError(
                "mojolearn SVR: verbose is refused; upstream it selects "
                "CUML_LOG_DEBUG lines and this port prints none, so honoring "
                "it is impossible and accepting it would be accepting-and-"
                "ignoring"
            )
        if output_type is not None:
            raise NotImplementedError(
                "mojolearn SVR: output_type is a cuML-internal array-type "
                "selector; this package returns NumPy"
            )
        self.kernel = k
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.verbose = False
        self.max_iter = max_iter
        self.nochange_steps = nochange_steps
        self.output_type = None

    def _resolve_gamma(self, n_features):
        """cuML's `_get_gamma` minus the refused 'scale' arm, the same three
        lines `SVC._resolve_gamma` is. `1 / n_cols` in float64 is exact for
        every n_cols that is a power of two and correctly rounded otherwise,
        on every host."""
        if self.gamma == "auto":
            return 1.0 / float(n_features)
        return float(self.gamma)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn SVR: sample_weight is not ported; the weighted "
                "InitPenalty arm (C_vec = C * w) has no port "
                "(svm/NOT_IMPLEMENTED.tsv)"
            )
        x, self.input_copied_ = as_f32_c(X, "X")
        n_rows, n_cols = x.shape
        targets = _as_targets(y, n_rows)
        gamma = self._resolve_gamma(n_cols)

        # WORST-CASE OUTPUT BUFFERS, AND `n_rows` IS THE WORST CASE.
        # The solver's domain is `2 * n_rows` (alpha+ and alpha-), but
        # `Results::combine_coefs` folds the two halves before it selects,
        # so at most `n_rows` coefficients, `n_rows` indices and
        # `n_rows * n_cols` support-vector floats can ever be written. Every
        # size here is a function of `(n_rows, n_cols)` alone, which is what
        # lets this side allocate before the call.
        dual = np.empty(n_rows, dtype=np.float32)
        support = np.empty(n_rows, dtype=np.int32)
        sv = np.empty(n_rows * n_cols, dtype=np.float32)
        # THREE, not SVC's five: slots 3 and 4 there are the class labels.
        info = np.empty(3, dtype=np.float64)
        n_support = _extension(getattr(self, 'numeric_mode', None)).svr_fit(
            _addr_ro(x),
            _addr_ro(targets),
            _addr(dual),
            _addr(support),
            _addr(sv),
            _addr(info),
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svr_fit_binding.
            # n_rows, n_features, kernel, gamma, C, epsilon, tol, max_iter,
            # nochange_steps
            [n_rows, n_cols, _KERNELS[self.kernel], gamma, self.C,
             self.epsilon, self.tol, self.max_iter, self.nochange_steps],
        )
        n_support = int(n_support)
        b = np.float32(info[0])

        self.n_features_in_ = n_cols
        self.n_support_ = n_support
        self.n_iter_ = int(info[2])
        self.intercept_ = np.array([b], dtype=np.float32)
        self.support_ = np.ascontiguousarray(support[:n_support])
        self.support_vectors_ = np.ascontiguousarray(
            sv[: n_support * n_cols].reshape(n_support, n_cols)
        )
        self.dual_coef_ = np.ascontiguousarray(dual[:n_support]).reshape(1, n_support)
        self._gamma = gamma
        return self

    @property
    def coef_(self):
        if not hasattr(self, "dual_coef_"):
            raise AttributeError("mojolearn SVR: call fit() first")
        if self.kernel != "linear":
            raise AttributeError(
                "mojolearn SVR: coef_ is only available for kernel='linear'"
            )
        if self.n_support_ == 0:
            return np.zeros((1, self.n_features_in_), dtype=np.float32)
        return np.ascontiguousarray(self.dual_coef_ @ self.support_vectors_)

    def predict(self, X):
        """`sum_j K(x, sv_j) dual_j + b`, one value per row, float32.

        There is no `decision_function` beside this and no `predict_class`
        under it: on a regressor the decision value IS the prediction, and
        cuML reaches it through the same `svcPredict` with the class
        epilogue switched off.
        """
        if not hasattr(self, "dual_coef_"):
            raise ValueError("mojolearn SVR: call fit() first")
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn SVR: X has {q.shape[1]} features, fit saw "
                f"{self.n_features_in_}"
            )
        n_rows = q.shape[0]
        out = np.empty(n_rows, dtype=np.float32)
        # Kept in locals so the arrays outlive the call; the Mojo side
        # borrows these addresses and owns nothing (`_arrays.py`).
        dual = self.dual_coef_
        sv = self.support_vectors_
        _extension(getattr(self, 'numeric_mode', None)).svr_predict(
            _addr_ro(q),
            _addr_ro(dual) if self.n_support_ > 0 else 0,
            _addr_ro(sv) if self.n_support_ > 0 else 0,
            _addr(out),
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svr_predict_binding.
            # n_rows, n_features, n_support, b, kernel, gamma,
            # cache_size_mib
            [n_rows, self.n_features_in_, self.n_support_,
             float(self.intercept_[0]), _KERNELS[self.kernel], self._gamma,
             self.cache_size],
        )
        return out

    def score(self, X, y):
        """The coefficient of determination R^2, scikit-learn's definition,
        `1 - SS_res / SS_tot`.

        Accumulated in FLOAT64 from float32 predictions, which is what
        scikit-learn does and is not part of any identity claim: it is a
        summary of the answer, not the answer.
        """
        pred = np.asarray(self.predict(X), dtype=np.float64)
        t = np.asarray(y, dtype=np.float64)
        if t.ndim != 1:
            raise ValueError(
                f"mojolearn SVR: y must be 1-D, got {t.ndim}-D shape {t.shape}"
            )
        if t.shape[0] != pred.shape[0]:
            raise ValueError(
                f"mojolearn SVR: y has {t.shape[0]} entries, X has "
                f"{pred.shape[0]} rows"
            )
        ss_res = float(np.sum((t - pred) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        if ss_tot == 0.0:
            # scikit-learn returns 1.0 for a perfect constant fit and 0.0
            # otherwise; the ratio is undefined and this is its convention.
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
