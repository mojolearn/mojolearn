# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Support vector classification and regression on the GPU. Reference:
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
`svm/impl/svm_parameter.mojo::check_rung1_scope`. They differ in the
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

import array
import importlib.machinery
import importlib.util
import os
import sys

from . import _portable_math as math
import numbers

from . import _backend
from . import _serialize
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, empty, zeros
from ._labels import argmax_rows, classes_from_member, classes_member, decode_labels, sorted_classes
from ._mode import NumericModeMixin
from ._scale_gamma import scale_gamma
from .linear_model import (
    _accuracy_host,
    _check_saved_by,
    _labels_1d,
    _r2_sums,
    _restore_mode,
    _round_f32,
    _saved_mode,
    _shape_of,
)

#: `SVC.save`'s format tag (the kde svc host lane, 2026-09-14).
_SVC_FORMAT = "mojolearn-svc-1"
#: `SVR.save`'s format tag (lane/inference-svm, 2026-09-15).
_SVR_FORMAT = "mojolearn-svr-1"

# cuML's `KernelType` values (kernel_params.hpp). Only two are implemented.
_KERNEL_LINEAR = 0
_KERNEL_POLYNOMIAL = 1
_KERNEL_RBF = 2
_KERNEL_TANH = 3
_KERNEL_PRECOMPUTED = 4

_KERNELS = {"linear": _KERNEL_LINEAR, "rbf": _KERNEL_RBF}

# SVC also carries POLYNOMIAL (lane/cpu-training-small-gaps, 2026-09-15): the
# identical linear Gram, then kernel_methods' polynomial_epilogue_kernel
# (DEVIATION 1663). SVR keeps _KERNELS and refuses 'poly' by name.
_SVC_KERNELS = dict(_KERNELS, poly=_KERNEL_POLYNOMIAL, sigmoid=_KERNEL_TANH)

#: DEVIATION 1663's cap, kernel_methods/impl/distance/kernel_matrices.mojo::KM_MAX_DEGREE
#: and svm/impl/svm_parameter.mojo::SVM_MAX_POLY_DEGREE.
_MAX_POLY_DEGREE = 32

# The names cuML accepts that this implementation does not, and the reason each is
# refused. Refused BY NAME rather than silently downgraded to 'rbf'.
_REFUSED_KERNELS = {
    "poly": (
        "POLYNOMIAL is not implemented in rung 1; it is one identical_pow away "
        "(svm/NOT_IMPLEMENTED.tsv spells the kernel out) and was left unimplemented "
        "rather than written without a gate"
    ),
    "polynomial": (
        "POLYNOMIAL is not implemented in rung 1 (svm/NOT_IMPLEMENTED.tsv); cuML spells "
        "this kernel 'poly'"
    ),
    "sigmoid": (
        "TANH is not implemented for SVR: SVC carries it (kernel_methods' "
        "identical_tanh epilogue), but the SVR binding takes no coef0 yet "
        "(svm/NOT_IMPLEMENTED.tsv)"
    ),
    "tanh": (
        "TANH is not implemented in rung 1 (svm/NOT_IMPLEMENTED.tsv); cuML spells this "
        "kernel 'sigmoid'"
    ),
    "precomputed": (
        "PRECOMPUTED is not implemented: kernelcache.cuh's "
        "extractColumnsForPrecomputed and svc_impl.cuh's precomputed "
        "predict arm are both unimplemented (svm/NOT_IMPLEMENTED.tsv)"
    ),
}

# SVC's refusals: the shared table less 'poly', which SVC implements.
_SVC_REFUSED_KERNELS = {k: v for k, v in _REFUSED_KERNELS.items() if k not in ("poly", "sigmoid")}
_SVC_REFUSED_KERNELS["tanh"] = (
    "cuML and scikit-learn spell this kernel 'sigmoid', which SVC implements"
)
_SVC_REFUSED_KERNELS["polynomial"] = (
    "cuML and scikit-learn spell this kernel 'poly', which SVC implements"
)

# SVR carries the same kernels (2026-09-27): its bindings take degree and
# coef0 as optional trailing params, so every linear / rbf call is unchanged.
_SVR_KERNELS = _SVC_KERNELS
_SVR_REFUSED_KERNELS = dict(_SVC_REFUSED_KERNELS)
_SVR_REFUSED_KERNELS["polynomial"] = (
    "cuML and scikit-learn spell this kernel 'poly', which SVR implements"
)
_SVR_REFUSED_KERNELS["tanh"] = (
    "cuML and scikit-learn spell this kernel 'sigmoid', which SVR implements"
)

_EXT_NAME = "_mojolearn_svm"
_PKG = __name__.rsplit(".", 1)[0]


def _extension(mode=None):
    """The `_mojolearn_svm` extension for `mode` (the process default when
    None), through `_backend.binding`: the one choke point that refuses an
    identical-only lane by name under a lower tier, loads the set the tier
    and vendor axes name, and cross-checks the binary's compiled tier.

    This carried a private loader until DEVIATION 2490 (2026-09-10), written
    when `_backend._MODULES` did not list this extension. It does now, and
    the private path was how a stale FAST `_mojolearn_svm.so` on disk kept
    ANSWERING after the lane went identical only: the macOS release smoke
    caught it the day the rule landed. Nothing loads a binding by path any
    more.
    """
    return _backend.binding(_EXT_NAME, mode)


def _as_labels(y):
    """`y` as a 1-D float32 Array plus the two distinct labels in cuML's
    order (sorted ascending), as a Python LIST under the package-wide
    ORDER RULE (`_labels.sorted_classes`, DEVIATION 2340). Raises the
    binary-only refusal HERE, before any device work, because that is the
    most common way to reach this class by mistake."""
    labels, shape = _labels_1d(y)
    if labels is None:
        raise ValueError(
            f"mojolearn SVC: y must be 1-D, got {len(shape)}-D shape {shape}"
        )
    # A NaN label gets the DEVIATION 636 refusal BEFORE the order rule sees
    # it, so the message a caller reads does not depend on which check ran.
    if any(isinstance(v, float) and v != v for v in labels):
        raise ValueError(
            "mojolearn SVC: y contains a non-finite value (DEVIATION 636: a "
            "NaN or inf cannot be fitted; a computed NaN carries a "
            "vendor-specific payload and cannot sit in a hashed stage)"
        )
    classes, _codes = sorted_classes(labels)
    if len(classes) < 2:
        raise ValueError(
            f"mojolearn SVC: y has {len(classes)} class; at least two are needed"
        )
    if len(classes) > 2:
        # One-vs-one over the binary solver (`SVC._fit_ovo`): the caller
        # forms each pair's 0.0 / 1.0 labels from these codes.
        return None, classes, _codes
    if not all(isinstance(c, numbers.Real) for c in classes):
        # String (or other non-numeric) labels: the solver sees each row's
        # dense code, 0.0 or 1.0, so `classes_[1]` still maps to +1 and the
        # label pair it reports back is (0.0, 1.0). Numeric labels keep the
        # float32 path below, bit for bit. Found by the Sep 22 pip smoke:
        # `SVC().fit(X, ["a", "b", ...])` raised "buffer format '<U1'"
        # while LogisticRegression and LinearSVC took the same labels.
        f = Array.from_list([float(c) for c in _codes], "<f4")
        return f, classes, (0.0, 1.0)
    f, _ = as_f32_c(y, ndim=1, name="y")
    if not all_finite(f):
        raise ValueError(
            "mojolearn SVC: y contains a non-finite value (DEVIATION 636: a "
            "NaN or inf cannot be fitted; a computed NaN carries a "
            "vendor-specific payload and cannot sit in a hashed stage)"
        )
    return f, classes, (classes[0], classes[1])


def _c_rows(C, n_rows, sample_weight, class_weight=None, y=None, who="SVC"):
    """`InitPenalty`'s weighted arm: the per-row bounds `C * class_weight[y_i]
    * sample_weight_i` (scikit-learn's libsvm `C_i`; cuML's `C_vec = C * w`
    after its Python layer folds class_weight into sample_weight), formed in
    binary64 in that order and rounded ONCE to float32, so every host hands
    the solver the same bounds. None when nothing is weighted (the
    unweighted arm, C at every row, bit for bit the old fit). A zero weight
    pins that row's alpha at 0 (cuML keeps the row; libsvm drops it, and the
    solution is the same)."""
    if sample_weight is None and class_weight is None:
        return None
    if sample_weight is None:
        w = [1.0] * n_rows
    else:
        w = [float(v) for v in (sample_weight.tolist() if hasattr(sample_weight, "tolist") else sample_weight)]
        if len(w) != n_rows:
            raise ValueError(
                f"mojolearn {who}: sample_weight has {len(w)} entries, X has {n_rows} rows"
            )
        for v in w:
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(
                    f"mojolearn {who}: sample_weight must be finite and >= 0, got {v!r}"
                )
    if class_weight is not None:
        labels, _shape = _labels_1d(y)
        classes, codes = sorted_classes(labels)
        if isinstance(class_weight, str):
            if class_weight != "balanced":
                raise ValueError(
                    f"mojolearn {who}: class_weight is a dict, 'balanced' or None, got {class_weight!r}"
                )
            counts = [0] * len(classes)
            for c in codes:
                counts[c] += 1
            cw = [n_rows / (len(classes) * counts[k]) for k in range(len(classes))]
        else:
            cw = [1.0] * len(classes)
            for key, value in dict(class_weight).items():
                if key not in classes:
                    raise ValueError(
                        f"mojolearn {who}: class_weight names the label {key!r}, "
                        f"which is not in y's classes {classes!r}"
                    )
                cw[classes.index(key)] = float(value)
        w = [w[i] * cw[codes[i]] for i in range(n_rows)]
    C = float(C)
    return Array.from_list([C * v for v in w], "<f4")


def _dual_times_sv(dual_coef, support_vectors):
    """`dual_coef_ @ support_vectors_`: a `(1, n_support) x (n_support,
    n_features)` product, accumulated SEQUENTIALLY over the support vectors
    in Python float64 and rounded once to float32.

    DEVIATION 2372 -- A HOST REDUCTION, OUTSIDE THE IDENTITY CLAIM. NumPy's
    float32 matmul went through the platform BLAS (or NumPy's own blocked
    loop), whose fold shape was the host's; this is one written-down order
    instead, so the bits are the same on every host but MAY DIFFER from a
    NumPy-era `coef_` in the last place. Only reachable on
    `kernel='linear'`, and only through the `coef_` property. It is an
    O(n_support * n_features) Python loop, FLAGGED AS A DEFECT: the right
    home is a gemv binding.
    """
    d = dual_coef.tolist()[0]
    n_features = support_vectors.shape[1]
    acc = [0.0] * n_features
    for a, row in zip(d, support_vectors.tolist()):
        for j in range(n_features):
            acc[j] += a * row[j]
    return Array.from_list([acc], "<f4")


class SVC(NumericModeMixin):
    """C-support vector classification, binary and one-vs-one multiclass,
    backed by an SMO solver and GPU kernel matrices (`svm/`, DEVIATIONS 630-637;
    `svm/README.md`), the scikit-learn surface.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter that is accepted and ignored is a wrong answer
    waiting for a caller (the house rule):

        C               honored   the penalty. Must be finite and positive;
                                  a non-finite C is refused by name
                                  (DEVIATION 636)
        kernel          honored   'linear', 'rbf' and 'poly'. 'poly' is the
                                  identical linear Gram then
                                  `(gamma * K + coef0) ** degree` as
                                  kernel_methods' DEVIATION 1663 epilogue
                                  (one fused multiply-add, an ascending
                                  repeated product). 'sigmoid' is the
                                  linear Gram then tanh(gamma * K + coef0)
                                  (kernel_methods' TANH epilogue,
                                  identical_tanh). 'precomputed' is REFUSED
                                  BY NAME with what is missing
        gamma           honored   a finite float >= 0, or the string 'auto'
                                  (= 1 / n_features, cuML's `_get_gamma`).
                                  'scale' is resolved exactly, DEVIATION
                                  870 below (theirs is the default)
        degree          honored   with kernel='poly': an integer in
                                  [0, 32] (DEVIATION 1663). With any other
                                  kernel only the default 3 is accepted,
                                  since nothing would read it
        coef0           honored   with 'poly' and 'sigmoid': a finite float. With
                                  any other kernel only the default 0.0
                                  is accepted
        tol             honored   the stopping tolerance; must be finite and
                                  positive (DEVIATION 636)
        cache_size      honored   ONLY as the prediction buffer, see
                                  DEVIATION 871 below
        class_weight    honored   a dict {label: weight} or 'balanced'; it
                                  multiplies each row's bound, C * cw[y_i] *
                                  w_i (`_c_rows`)
        max_iter        honored   cuML's total inner-iteration cap; -1 (the
                                  default) is no limit
        nochange_steps  honored   cuML's convergence rule, with
                                  its n_small_diff counter
        verbose         refused   anything truthy. It selects LOG LINES
                                  in the reference (CUML_LOG_DEBUG); this implementation
                                  prints none, so accepting it would be
                                  accepting-and-ignoring
        random_state    refused   anything but None. The solver and its
                                  one-vs-one multiclass draw no random
                                  numbers; scikit-learn reads it only for
                                  probability=True, which is refused
        decision_       honored   'ovo' (the default here, cuML's) or 'ovr'
          function_shape          (scikit-learn's): with more than two
                                  classes, decision_function's pairwise
                                  (n, K(K-1)/2) or voted (n, K) form. No
                                  effect on two classes, as in scikit-learn
        break_ties      honored   with 'ovr' and more than two classes,
                                  predict is the argmax of the 'ovr'
                                  decision_function; refused with 'ovo'
                                  (scikit-learn's ValueError)
        multiclass      honored   one-vs-one, libsvm's scheme: one binary
                                  machine per class pair (i < j) on the
                                  rows of those two classes, gamma resolved
                                  on the whole X, the per-row bounds cut to
                                  the pair's rows; prediction by vote, ties
                                  to the lowest class (`_fit_ovo`)
        probability     refused   Platt scaling is not in cuML's C++ surface
                                  at all (svm/NOT_IMPLEMENTED.tsv)
        output_type     refused   a cuML-internal array-type selector; this
                                  package returns mojolearn Arrays
        sample_weight   honored   in fit(): the weighted `InitPenalty` arm,
                                  per-row bounds C * w formed in binary64 and
                                  rounded once to float32 (`_c_rows`)
        sparse X        refused   `svcFitSparse` / `svcPredictSparse` and
                                  every CSR arm are unimplemented; dense
                                  row-major float32 only

    Non-finite cells of `X` are refused by name inside the Mojo entry
    (DEVIATION 636), naming the flat index, rather than being fitted.

    DEVIATION 870: `gamma='scale'` IS RESOLVED EXACTLY, AND THE DEFAULT
    HERE IS STILL 'auto' (theirs is 'scale'; the default is kept so every
    recorded default fit keeps its bits). 'scale' is `1 / (n_features *
    X.var())` as theirs, but the variance is the EXACT population variance
    of the float32 cells, formed in integers, and the reciprocal is rounded
    once to binary64 (`_scale_gamma.scale_gamma`). Their float32 `X.var()`
    carries its reduction's fold shape in its last bits; this one has no
    fold to differ, so every host reads the same gamma bits, and it differs
    from theirs by at most that reduction's rounding.

    DEVIATION 871: `cache_size` IS HONORED ONLY AT PREDICT. In the reference it is
    two things under one name: the training-time `raft::cache` LRU kernel
    cache, and the ceiling on the prediction kernel-tile buffer
    (`svm_base.pyx:554`, passed as `buffer_size` to `svcPredict`). The
    prediction half is honored here, exactly, and it is a real knob: it
    sets the prediction batch size, and `check_device_is_launch_invariant`
    holds the answer fixed over it from 0.001 MiB to 200 MiB. The training
    half is NOT implemented -- the solver always runs cuML's own
    `n_cache_sets == 0` path -- so `cache_size` does not affect training
    time here the way it does in the reference. It cannot affect training RESULTS
    in the reference either (`svm/NOT_IMPLEMENTED.tsv` carries that determinism
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
    classes_ : list (K,)
        The distinct labels, sorted ascending, as Python scalars under
        the package-wide order rule (`_labels.sorted_classes`, DEVIATION
        2340; it was an ndarray in `y`'s dtype). `predict` returns an
        int64 / float64 Array for int / float labels and a Python list
        otherwise. `classes_[1]` is the one the solver maps to +1
        (`getOvrlabels(..., idx=1)`), so it fixes the sign of the decision
        function.
    support_ : Array (n_SV,) int32
        Indices of the support vectors in the training matrix.
    support_vectors_ : Array (n_SV, n_features) float32
    dual_coef_ : Array (1, n_SV) float32
        scikit-learn's 2-D layout of cuML's 1-D `dual_coefs`.
    intercept_ : Array (1,) float32
    n_support_ : int, or Array (K,) int32 for K > 2
        Two classes: cuML's scalar count (scikit-learn's is a per-class
        array; this one is not, and the difference is here rather than in
        a surprise). More: scikit-learn's per-class counts.
    n_iter_ : int, or Array (K(K-1)/2,) int32 for K > 2
        Total inner SMO iterations, per pair for more than two classes.

    MULTICLASS LAYOUT (K > 2) is scikit-learn's, in its orientation (a
    pair's decision positive toward its LOWER class): `support_` grouped
    by class, ascending within a class; `dual_coef_` (K - 1, n_SV);
    `intercept_` (K(K-1)/2,); `coef_` (K(K-1)/2, n_features) for 'linear'.
    coef_ : Array (1, n_features) float32
        `dual_coef_ @ support_vectors_`, and only for `kernel='linear'`;
        raises AttributeError otherwise, as scikit-learn does.
    n_features_in_ : int
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "classifier"

    #: This family's binding, for `NumericModeMixin._bind` (the kde svc host
    #: lane, 2026-09-14: `fit` and `_run` bind through `self._bind`, the same
    #: `_backend.binding(_EXT_NAME, numeric_mode)` that `_extension` resolves,
    #: so the host subclass in `_classical_host.py` answers the CPU binding by
    #: overriding one method and the Python around it is this class's own).
    _BINDING = _EXT_NAME

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
        break_ties=False,
    ):
        if not isinstance(kernel, str):
            raise ValueError("mojolearn SVC: kernel is a name")
        # The caller's own string object when it is already lower case, so
        # `get_params` hands `clone` back the object it was given.
        k = kernel if kernel == kernel.lower() else kernel.lower()
        if k in _SVC_REFUSED_KERNELS:
            raise NotImplementedError(
                f"mojolearn SVC: kernel={kernel!r} is refused; "
                + _SVC_REFUSED_KERNELS[k]
            )
        if k not in _SVC_KERNELS:
            raise ValueError(
                f"mojolearn SVC: kernel={kernel!r} is not a kernel name; "
                f"this implementation carries {sorted(_SVC_KERNELS)} and refuses "
                f"the others by name ({sorted(_SVC_REFUSED_KERNELS)})"
            )
        if isinstance(gamma, str):
            g = gamma.lower()
            if g not in ("auto", "scale"):
                raise ValueError(
                    f"mojolearn SVC: gamma={gamma!r} is not a name; it is "
                    "'auto', 'scale' or a float"
                )
            gamma = g
        else:
            gamma = float(gamma)
            if not math.isfinite(gamma) or gamma < 0.0:
                raise ValueError(
                    "mojolearn SVC: gamma must be finite and >= 0 for the RBF "
                    f"kernel, got {gamma!r} (DEVIATION 636; scikit-learn's own "
                    "constraint is gamma >= 0)"
                )
        if k == "poly":
            if isinstance(degree, bool) or not isinstance(degree, numbers.Integral):
                raise TypeError(
                    f"mojolearn SVC: degree={degree!r} must be an integer; the "
                    "polynomial power is an ascending repeated product (DEVIATION 1663)"
                )
            degree = int(degree)
            if not 0 <= degree <= _MAX_POLY_DEGREE:
                raise ValueError(
                    f"mojolearn SVC: degree must be in [0, {_MAX_POLY_DEGREE}], got "
                    f"{degree!r} (DEVIATION 1663)"
                )
            coef0 = float(coef0)
            if not math.isfinite(coef0):
                raise ValueError(
                    f"mojolearn SVC: coef0 must be finite, got {coef0!r} (DEVIATION 636)"
                )
        elif k == "sigmoid":
            # tanh(gamma * K + coef0), the kernel_methods lane's TANH epilogue.
            if degree != 3:
                raise NotImplementedError(
                    f"mojolearn SVC: degree={degree!r} is refused with kernel='sigmoid'; "
                    "it is read only by kernel='poly'"
                )
            coef0 = float(coef0)
            if not math.isfinite(coef0):
                raise ValueError(
                    f"mojolearn SVC: coef0 must be finite, got {coef0!r} (DEVIATION 636)"
                )
        else:
            if degree != 3:
                raise NotImplementedError(
                    f"mojolearn SVC: degree={degree!r} is refused with kernel={k!r}; "
                    "it is read only by kernel='poly'. Passing it with another kernel "
                    "would be a parameter accepted and ignored"
                )
            if coef0 != 0.0:
                raise NotImplementedError(
                    f"mojolearn SVC: coef0={coef0!r} is refused with kernel={k!r}; it "
                    "is read only by kernel='poly' and kernel='sigmoid'"
                )
        C = float(C)
        if not math.isfinite(C):
            raise ValueError(
                f"mojolearn SVC: C must be finite, got {C!r} (DEVIATION 636)"
            )
        if C <= 0.0:
            raise ValueError(f"mojolearn SVC: C must be positive, got {C!r}")
        tol = float(tol)
        if not math.isfinite(tol):
            raise ValueError(
                f"mojolearn SVC: tol must be finite, got {tol!r} (DEVIATION 636)"
            )
        if tol <= 0.0:
            raise ValueError(f"mojolearn SVC: tol must be positive, got {tol!r}")
        cache_size = float(cache_size)
        if not (cache_size > 0.0) or not math.isfinite(cache_size):
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
                "CUML_LOG_DEBUG lines and this implementation prints none, so honoring "
                "it is impossible and accepting it would be accepting-and-"
                "ignoring"
            )
        if output_type is not None:
            raise NotImplementedError(
                "mojolearn SVC: output_type is a cuML-internal array-type "
                "selector; this package returns mojolearn Arrays"
            )
        if random_state is not None:
            raise NotImplementedError(
                "mojolearn SVC: random_state is refused; the C-SVC solver and "
                "its one-vs-one multiclass draw no random numbers, and "
                "scikit-learn reads it only for probability=True, which is "
                "refused"
            )
        if class_weight is not None and not isinstance(class_weight, (dict, str)):
            raise ValueError(
                "mojolearn SVC: class_weight is a dict, 'balanced' or None, "
                f"got {type(class_weight).__name__}"
            )
        if decision_function_shape not in ("ovo", "ovr"):
            raise ValueError(
                "mojolearn SVC: decision_function_shape is 'ovo' or 'ovr', got "
                f"{decision_function_shape!r}"
            )
        break_ties = bool(break_ties)
        if break_ties and decision_function_shape == "ovo":
            raise ValueError(
                "mojolearn SVC: break_ties must be False when "
                "decision_function_shape is 'ovo' (scikit-learn's rule)"
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
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.probability = False
        self.break_ties = break_ties

    def _resolve_gamma(self, x):
        """cuML's `_get_gamma`. 'auto' is `1 / n_cols`, exact for every
        n_cols that is a power of two and correctly rounded otherwise;
        'scale' is `1 / (n_cols * X.var())` from the EXACT variance of the
        float32 cells, rounded once (`_scale_gamma.scale_gamma`, DEVIATION
        870). Both are the same bits on every host."""
        if self.gamma == "auto":
            return 1.0 / float(x.shape[1])
        if self.gamma == "scale":
            return scale_gamma(x.ravel().tolist(), x.shape[1])
        return float(self.gamma)

    def _solve(self, x, labels, gamma, c_rows):
        """ONE binary C-SVC solve through the binding: `labels` float32 with
        two distinct values, `c_rows` the per-row bounds or None. Returns
        (n_support, dual, support, sv, info) with the worst-case buffers
        uncut."""
        n_rows, n_cols = x.shape
        tail = [] if c_rows is None else [addr_ro(c_rows, name="C * sample_weight")]
        dual = empty((n_rows,), "<f4")
        support = empty((n_rows,), "<i4")
        sv = empty((n_rows * n_cols,), "<f4")
        info = empty((5,), "<f8")
        n_support = self._bind(_EXT_NAME).svc_fit(
            addr_ro(x, name="x"),
            addr_ro(labels, name="labels"),
            addr(dual, name="dual"),
            addr(support, name="support"),
            addr(sv, name="sv"),
            addr(info, name="info"),
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_fit_binding.
            # n_rows, n_features, kernel, gamma, C, tol, max_iter,
            # nochange_steps, degree, coef0[, the per-row bounds' address]
            [n_rows, n_cols, _SVC_KERNELS[self.kernel], gamma, self.C, self.tol,
             self.max_iter, self.nochange_steps, int(self.degree), float(self.coef0)] + tail,
        )
        return int(n_support), dual, support, sv, info

    def fit(self, X, y, sample_weight=None):
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        labels, classes, pair = _as_labels(y)
        n_rows, n_cols = x.shape
        n_y = len(pair) if labels is None else labels.shape[0]
        if n_y != n_rows:
            raise ValueError(
                f"mojolearn SVC: y has {n_y} entries, X has {n_rows} rows"
            )
        gamma = self._resolve_gamma(x)
        c_rows = _c_rows(self.C, n_rows, sample_weight, getattr(self, "class_weight", None), y, "SVC")
        if labels is None:
            return self._fit_ovo(x, classes, pair, gamma, c_rows)
        self.__dict__.pop("_pairs", None)
        n_support, dual, support, sv, info = self._solve(x, labels, gamma, c_rows)
        # `np.float32(...)`: one rounding of the float64 slot to binary32.
        b = _round_f32(info[0])
        label0 = _round_f32(info[3])
        label1 = _round_f32(info[4])
        # THE LABEL PAIR IS CHECKED, NOT ASSUMED. The solver finds the two
        # distinct labels with its own host sort and maps the LARGER to +1;
        # if that disagreed with this side's sorted unique, `predict` would
        # map the device's answer onto the wrong class and the error would
        # look like a bad model rather than a bad boundary.
        if label0 != _round_f32(pair[0]) or label1 != _round_f32(pair[1]):
            raise RuntimeError(
                "mojolearn SVC: the solver's label pair "
                f"({float(label0)}, {float(label1)}) does not match the "
                f"host's sorted unique ({float(pair[0])}, "
                f"{float(pair[1])}); the class mapping cannot be trusted"
            )
        self.classes_ = classes
        self.n_features_in_ = n_cols
        self.n_support_ = n_support
        self.n_iter_ = int(info[2])
        self.intercept_ = Array.from_list([b], "<f4")
        # A slice of an Array COPIES (the _array contract), so these three
        # are compact C-contiguous Arrays detached from the worst-case
        # buffers above, exactly as `np.ascontiguousarray` detached them.
        self.support_ = support[:n_support]
        self.support_vectors_ = sv[:n_support * n_cols].reshape(
            (n_support, n_cols))
        self.dual_coef_ = dual[:n_support].reshape((1, n_support))
        self._gamma = gamma
        self._label0 = label0
        self._label1 = label1
        return self

    def _fit_ovo(self, x, classes, codes, gamma, c_rows):
        """MULTICLASS: one-vs-one, libsvm's (and scikit-learn's SVC's)
        scheme, over the binary solver. For each class pair (i, j), i < j
        in `classes_` order, the rows of class i or j are copied out in
        their original order (exact bytes), class j is the solver's +1
        (label 1.0, class i label 0.0), the per-row bounds are the full
        fit's `_c_rows` cut to the same rows, and gamma is the one resolved
        on the WHOLE X (scikit-learn's `_get_gamma` runs once, before the
        pairs). Every number is the binary solver's; what is added here is
        exact bookkeeping (row selection, index remapping, negation).

        The public attributes are scikit-learn's multiclass layout, in its
        orientation (a pair's decision is positive toward class i, the
        negation of this solver's, which is exact): `support_` grouped by
        class, ascending index within a class; `dual_coef_` (K - 1, n_SV);
        `intercept_` (K (K - 1) / 2,); `n_support_` per class; `n_iter_`
        per pair."""
        n_rows, n_cols = x.shape
        k = len(classes)
        row_bytes = 4 * n_cols
        raw = x.tobytes()
        cb = None if c_rows is None else c_rows.tolist()
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                idx = [r for r in range(n_rows) if codes[r] == i or codes[r] == j]
                store = array.array("f")
                store.frombytes(b"".join(raw[r * row_bytes:(r + 1) * row_bytes] for r in idx))
                sub = Array._owned(store, (len(idx), n_cols), "<f4", "C")
                lab = Array.from_list([1.0 if codes[r] == j else 0.0 for r in idx], "<f4")
                sub_c = None if cb is None else Array.from_list([cb[r] for r in idx], "<f4")
                n_sv, dual, support, sv, info = self._solve(sub, lab, gamma, sub_c)
                if _round_f32(info[3]) != 0.0 or _round_f32(info[4]) != 1.0:
                    raise RuntimeError(
                        "mojolearn SVC: the solver's label pair for classes "
                        f"({classes[i]!r}, {classes[j]!r}) is not (0.0, 1.0); "
                        "the class mapping cannot be trusted")
                local = support[:n_sv].tolist()
                pairs.append(dict(
                    i=i, j=j,
                    dual=dual[:n_sv].reshape((1, n_sv)),
                    sv=sv[:n_sv * n_cols].reshape((n_sv, n_cols)),
                    support=[idx[s] for s in local],
                    b=_round_f32(info[0]),
                    n_iter=int(info[2]),
                ))
        self._set_ovo(classes, pairs, x, n_cols)
        self._gamma = gamma
        self._label0 = 0.0
        self._label1 = 1.0
        return self

    def _set_ovo(self, classes, pairs, x, n_cols):
        """scikit-learn's multiclass attributes from the per-pair machines
        (also `load`'s path, where `x` is None and the rows come from each
        pair's own support vectors)."""
        k = len(classes)
        per_class = [dict() for _ in range(k)]          # orig index -> row bytes
        for p in pairs:
            rows = p["sv"].tolist()
            d = p["dual"].tolist()[0]
            for pos, r in enumerate(p["support"]):
                # the class of a support vector is its pair side: this
                # solver's dual is +alpha on class j, -alpha on class i, and
                # a returned support vector has alpha > 0
                side = p["i"] if d[pos] < 0.0 else p["j"]
                per_class[side][r] = rows[pos]
        support, sv_rows, n_support, col_of = [], [], [], {}
        for c in range(k):
            keys = sorted(per_class[c])
            n_support.append(len(keys))
            for r in keys:
                col_of[r] = len(support)
                support.append(r)
                sv_rows.append(per_class[c][r])
        n_sv = len(support)
        dual = [[0.0] * n_sv for _ in range(k - 1)]
        for p in pairs:
            i, j = p["i"], p["j"]
            d = p["dual"].tolist()[0]
            for pos, r in enumerate(p["support"]):
                side = i if d[pos] < 0.0 else j
                # a class-i vector's coefficient for pair (i, j) sits in
                # row j - 1, a class-j vector's in row i; negated into
                # scikit-learn's orientation
                dual[j - 1 if side == i else i][col_of[r]] = -d[pos]
        self.classes_ = classes
        self.n_features_in_ = n_cols
        self._pairs = pairs
        self.support_ = Array.from_list(support, "<i4")
        self.support_vectors_ = (Array.from_list(sv_rows, "<f4") if n_sv
                                 else zeros((0, n_cols), "<f4"))
        self.dual_coef_ = Array.from_list(dual, "<f4") if n_sv else zeros((k - 1, 0), "<f4")
        self.intercept_ = Array.from_list([-p["b"] for p in pairs], "<f4")
        self.n_support_ = Array.from_list(n_support, "<i4")
        self.n_iter_ = Array.from_list([p["n_iter"] for p in pairs], "<i4")

    @property
    def coef_(self):
        if not hasattr(self, "dual_coef_"):
            raise AttributeError("mojolearn SVC: call fit() first")
        if self.kernel != "linear":
            raise AttributeError(
                "mojolearn SVC: coef_ is only available for kernel='linear'"
            )
        pairs = self.__dict__.get("_pairs")
        if pairs is not None:
            # one row per pair in scikit-learn's orientation: the negation
            # of each pair machine's own dual_coef_ @ support_vectors_
            rows = []
            for p in pairs:
                if p["dual"].shape[1] == 0:
                    rows.append([0.0] * self.n_features_in_)
                else:
                    rows.append([-v for v in _dual_times_sv(p["dual"], p["sv"]).tolist()[0]])
            return Array.from_list(rows, "<f4")
        if self.n_support_ == 0:
            return zeros((1, self.n_features_in_), "<f4")
        return _dual_times_sv(self.dual_coef_, self.support_vectors_)

    def _query(self, X):
        if not hasattr(self, "dual_coef_"):
            raise ValueError("mojolearn SVC: call fit() first")
        q, _ = as_f32_c(X, ndim=2, name="X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn SVC: X has {q.shape[1]} features, fit saw "
                f"{self.n_features_in_}"
            )
        return q

    def _machine(self, q, dual, sv, n_support, b, label0, label1, predict_class):
        n_rows = q.shape[0]
        out = empty((n_rows,), "<f4")
        # Kept in locals so the arrays outlive the call; the Mojo side
        # borrows these addresses and owns nothing (`_buffer.py`).
        self._bind(_EXT_NAME).svc_predict(
            addr_ro(q, name="q"),
            addr_ro(dual, name="dual") if n_support > 0 else 0,
            addr_ro(sv, name="sv") if n_support > 0 else 0,
            addr(out, name="out"),
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_predict_binding.
            # n_rows, n_features, n_support, b, classes[0], classes[1],
            # kernel, gamma, predict_class, cache_size_mib, degree, coef0
            [n_rows, self.n_features_in_, n_support,
             float(b), float(label0), float(label1),
             _SVC_KERNELS[self.kernel], self._gamma,
             1 if predict_class else 0, self.cache_size,
             int(self.degree), float(self.coef0)],
        )
        return out

    def _run(self, X, predict_class):
        q = self._query(X)
        return self._machine(q, self.dual_coef_, self.support_vectors_, self.n_support_,
                             self.intercept_[0], self._label0, self._label1, predict_class)

    def _pair_decisions(self, X):
        """Each pair machine's raw decision on X (this solver's
        orientation: >= 0 toward class j), as lists of float32 values."""
        q = self._query(X)
        return [self._machine(q, p["dual"], p["sv"], p["dual"].shape[1], p["b"],
                              0.0, 1.0, False).tolist() for p in self._pairs]

    def _ovr_scores(self, decisions, n_rows):
        """scikit-learn's `_ovr_decision_function(dec < 0, -dec, K)` on the
        one-vs-one decisions `dec` (= -d here): votes plus the summed
        confidences squashed into (-1/3, 1/3), in binary64, accumulated in
        pair order. Pure host arithmetic in one written order."""
        k = len(self.classes_)
        out = []
        for r in range(n_rows):
            votes = [0.0] * k
            conf = [0.0] * k
            for p, d in zip(self._pairs, decisions):
                v = d[r]
                i, j = p["i"], p["j"]
                conf[i] -= v
                conf[j] += v
                if v > 0.0:
                    votes[j] += 1.0
                else:
                    votes[i] += 1.0
            out.append([votes[c] + conf[c] / (3.0 * (abs(conf[c]) + 1.0)) for c in range(k)])
        return out

    def decision_function(self, X):
        """Binary: the raw `sum_j K(x, sv_j) dual_j + b`, one value per row.
        Multiclass: `decision_function_shape='ovo'` gives (n, K (K - 1) / 2)
        float32, each pair's decision in scikit-learn's orientation
        (positive toward the lower class of the pair); 'ovr' gives (n, K)
        float64, scikit-learn's vote-plus-confidence transform of those."""
        if self.__dict__.get("_pairs") is None:
            return self._run(X, False)
        decisions = self._pair_decisions(X)
        n_rows = len(decisions[0])
        if self.decision_function_shape == "ovo":
            return Array.from_list(
                [[-d[r] for d in decisions] for r in range(n_rows)], "<f4")
        return Array.from_list(self._ovr_scores(decisions, n_rows), "<f8")

    def predict(self, X):
        """Binary: cuML's `applyPrediction` epilogue, ON THE DEVICE: the
        label is chosen there by `val + b < 0 ? classes_[0] :
        classes_[1]` and this maps the float32 label it wrote back into
        `classes_`'s dtype. Multiclass: libsvm's vote (a pair's decision
        < 0 here votes class i, otherwise class j, the binary epilogue's
        rule), the most votes winning and ties to the lowest class; with
        `break_ties=True` (shape 'ovr') the argmax of the 'ovr'
        decision_function instead, first maximum winning."""
        pairs = self.__dict__.get("_pairs")
        if pairs is None:
            raw = self._run(X, True)
            label1 = self._label1
            return decode_labels(self.classes_,
                                 [1 if v == label1 else 0 for v in raw.tolist()])
        decisions = self._pair_decisions(X)
        n_rows = len(decisions[0])
        if self.break_ties:
            return decode_labels(self.classes_,
                                 argmax_rows(Array.from_list(self._ovr_scores(decisions, n_rows), "<f8")))
        k = len(self.classes_)
        codes = []
        for r in range(n_rows):
            votes = [0] * k
            for p, d in zip(pairs, decisions):
                votes[p["j"] if d[r] >= 0.0 else p["i"]] += 1
            best = 0
            for c in range(1, k):
                if votes[c] > votes[best]:
                    best = c
            codes.append(best)
        return decode_labels(self.classes_, codes)

    def predict_proba(self, X):
        raise NotImplementedError(
            "mojolearn SVC: predict_proba is not available; Platt scaling is "
            "not in cuML's C++ surface at all (svm/NOT_IMPLEMENTED.tsv), which is "
            "why the constructor refuses probability=True"
        )

    def score(self, X, y):
        """Accuracy, a Python count over O(rows) labels (DEVIATION 2365)."""
        return _accuracy_host(self.predict(X), y)

    def save(self, path):
        """Write the fitted model to `path` as an npz: `dual_coef_`,
        `support_vectors_`, `intercept_` as fitted (float32), `support_`
        `<i4`, `classes_` as `_labels.classes_member`, `labels` `<f8` (the
        float32 label pair the solver returned, `_label0` and `_label1`),
        `hyper` `<f8` [C, tol, cache_size, the gamma the FIT resolved],
        `kernel` and `gamma` (the constructor's setting, 'auto' or a
        number) as strings, `meta` `<i8` [n_features_in_, n_support_,
        n_iter_, max_iter, nochange_steps, degree] (the kde svc host lane,
        2026-09-14). `mojolearn.host_model(path)` predicts from it on a CPU
        with no GPU through `_mojolearn_svm_host.svc_predict`."""
        if not hasattr(self, "dual_coef_"):
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": _SVC_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "kernel": str(self.kernel),
            "gamma": str(self.gamma),
            "dual_coef": self.dual_coef_,
            "support_vectors": self.support_vectors_,
            "support": self.support_,
            "intercept": self.intercept_,
            "classes": classes_member(self.classes_),
            "labels": Array.from_list([float(self._label0), float(self._label1)], "<f8"),
            "hyper": Array.from_list(
                [float(self.C), float(self.tol), float(self.cache_size), float(self._gamma)],
                "<f8",
            ),
        }
        if self.__dict__.get("_pairs") is None:
            arrays["meta"] = Array.from_list(
                [int(self.n_features_in_), int(self.n_support_), int(self.n_iter_),
                 int(self.max_iter), int(self.nochange_steps), int(self.degree)],
                "<i8",
            )
        if self.kernel in ("poly", "sigmoid"):
            # Only for poly and sigmoid, so every linear and rbf file keeps its bytes.
            arrays["coef0"] = Array.from_list([float(self.coef0)], "<f8")
        pairs = self.__dict__.get("_pairs")
        if pairs is not None:
            # MULTICLASS ONLY, so every binary file keeps its bytes: the
            # one-vs-one machines themselves, concatenated in pair order
            # ([i, j, n_support, n_iter] per pair), from which `load`
            # rebuilds the public attributes exactly as `fit` built them.
            nf = int(self.n_features_in_)
            arrays["meta"] = Array.from_list(
                [nf, len(self.support_.tolist()), sum(p["n_iter"] for p in pairs),
                 int(self.max_iter), int(self.nochange_steps), int(self.degree)], "<i8")
            arrays["labels"] = Array.from_list([0.0, 1.0], "<f8")
            arrays["shape"] = str(self.decision_function_shape)
            arrays["break_ties"] = Array.from_list([1 if self.break_ties else 0], "<i8")
            arrays["pair_meta"] = Array.from_list(
                [v for p in pairs for v in (p["i"], p["j"], p["dual"].shape[1], p["n_iter"])], "<i8")
            arrays["pair_b"] = Array.from_list([p["b"] for p in pairs], "<f4")
            arrays["pair_support"] = Array.from_list([r for p in pairs for r in p["support"]], "<i4")
            arrays["pair_dual"] = Array.from_list([v for p in pairs for v in p["dual"].tolist()[0]], "<f4")
            arrays["pair_sv"] = Array.from_list(
                [v for p in pairs for row in p["sv"].tolist() for v in row], "<f4")
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts; every array
        is read at its saved dtype and never cast."""
        arrays = _serialize.read_npz(path, _SVC_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 6:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 6 are needed")
        hyper = _serialize.exact(arrays, "hyper", "<f8")
        if hyper.size != 4:
            raise ValueError(f"mojolearn: {path!r} hyper holds {hyper.size} fields, 4 are needed")
        gamma_setting = _serialize.scalar_str(arrays, "gamma")
        kernel_setting = _serialize.scalar_str(arrays, "kernel")
        coef0 = 0.0
        if kernel_setting in ("poly", "sigmoid"):
            coef0_arr = _serialize.exact(arrays, "coef0", "<f8")
            if coef0_arr.size != 1:
                raise ValueError(f"mojolearn: {path!r} coef0 must hold one float64")
            coef0 = float(coef0_arr[0])
        obj = cls(
            C=float(hyper[0]),
            kernel=_serialize.scalar_str(arrays, "kernel"),
            gamma=gamma_setting if gamma_setting in ("auto", "scale") else float(gamma_setting),
            tol=float(hyper[1]),
            cache_size=float(hyper[2]),
            max_iter=int(meta[3]),
            nochange_steps=int(meta[4]),
            degree=int(meta[5]),
            coef0=coef0,
        )
        _restore_mode(obj, arrays)
        nf, n_support = int(meta[0]), int(meta[1])
        if "pair_meta" in arrays:
            return cls._load_ovo(obj, arrays, path, nf, hyper)
        dual = _serialize.exact(arrays, "dual_coef", "<f4")
        if dual.ndim != 2 or tuple(dual.shape) != (1, n_support):
            raise ValueError(f"mojolearn: {path!r} dual_coef shape {tuple(dual.shape)} is not (1, {n_support})")
        sv = _serialize.exact(arrays, "support_vectors", "<f4")
        if sv.ndim != 2 or tuple(sv.shape) != (n_support, nf):
            raise ValueError(f"mojolearn: {path!r} support_vectors shape {tuple(sv.shape)} is not ({n_support}, {nf})")
        support = _serialize.exact(arrays, "support", "<i4")
        if support.ndim != 1 or support.size != n_support:
            raise ValueError(f"mojolearn: {path!r} support does not match n_support_")
        intercept = _serialize.exact(arrays, "intercept", "<f4")
        if intercept.ndim != 1 or intercept.size != 1:
            raise ValueError(f"mojolearn: {path!r} intercept must hold one float32")
        labels = _serialize.exact(arrays, "labels", "<f8")
        if labels.size != 2:
            raise ValueError(f"mojolearn: {path!r} labels must hold the two solver labels")
        obj.classes_ = classes_from_member(arrays["classes"])
        if len(obj.classes_) != 2:
            raise ValueError(f"mojolearn: {path!r} must carry exactly two classes")
        obj.n_features_in_ = nf
        obj.n_support_ = n_support
        obj.n_iter_ = int(meta[2])
        obj.intercept_ = intercept
        obj.support_ = support
        obj.support_vectors_ = sv
        obj.dual_coef_ = dual
        obj._gamma = float(hyper[3])
        obj._label0 = float(labels[0])
        obj._label1 = float(labels[1])
        return obj

    @classmethod
    def _load_ovo(cls, obj, arrays, path, nf, hyper):
        """The multiclass half of `load`: the per-pair machines, then the
        public attributes through `_set_ovo`, as `fit` sets them."""
        classes = classes_from_member(arrays["classes"])
        k = len(classes)
        pm = _serialize.exact(arrays, "pair_meta", "<i8").tolist()
        n_pairs = k * (k - 1) // 2
        if k < 3 or len(pm) != 4 * n_pairs:
            raise ValueError(f"mojolearn: {path!r} pair_meta does not hold {n_pairs} pairs")
        pb = _serialize.exact(arrays, "pair_b", "<f4").tolist()
        ps = _serialize.exact(arrays, "pair_support", "<i4").tolist()
        pd = _serialize.exact(arrays, "pair_dual", "<f4").tolist()
        pv = _serialize.exact(arrays, "pair_sv", "<f4").tolist()
        total = sum(pm[4 * q + 2] for q in range(n_pairs))
        if len(pb) != n_pairs or len(ps) != total or len(pd) != total or len(pv) != total * nf:
            raise ValueError(f"mojolearn: {path!r} pair arrays do not match pair_meta")
        pairs, at, q = [], 0, 0
        for i in range(k):
            for j in range(i + 1, k):
                pi, pj, n_sv, n_iter = pm[4 * q:4 * q + 4]
                if (pi, pj) != (i, j):
                    raise ValueError(f"mojolearn: {path!r} pair {q} is ({pi}, {pj}), not ({i}, {j})")
                pairs.append(dict(
                    i=i, j=j,
                    dual=Array.from_list([pd[at:at + n_sv]], "<f4").reshape((1, n_sv)),
                    sv=Array.from_list(pv[at * nf:(at + n_sv) * nf], "<f4").reshape((n_sv, nf)),
                    support=ps[at:at + n_sv], b=pb[q], n_iter=int(n_iter)))
                at += n_sv
                q += 1
        shape = _serialize.scalar_str(arrays, "shape")
        ties = bool(_serialize.exact(arrays, "break_ties", "<i8").tolist()[0])
        obj.decision_function_shape = "ovo" if shape == "ovo" else "ovr"
        obj.break_ties = ties and obj.decision_function_shape == "ovr"
        obj._set_ovo(classes, pairs, None, nf)
        obj._gamma = float(hyper[3])
        obj._label0 = 0.0
        obj._label1 = 1.0
        return obj


def _as_targets(y, n_rows):
    """`y` as a 1-D contiguous float32 vector of regression targets.

    NO `np.unique`, NO label pair, NO class count. `svr_impl.cuh:68` hands
    `y` straight to `SmoSolver::Solve`, so a target is any finite float and
    two rows sharing one are ordinary rather than interesting.

    NO FINITENESS CHECK HERE EITHER, and that is deliberate rather than an
    omission copied from `_as_labels`. `check_finite_list` in
    `svm/impl/svr_impl.mojo::svr_fit` refuses a non-finite target BY
    NAME with the offending flat index (DEVIATION 636), before any device
    work reaches a kernel, and a
    duplicate check on this side would make that refusal unreachable from
    Python and therefore untestable. The house rule is to plumb the value
    through and let the named refusal fire.
    """
    shape = _shape_of(y)
    if len(shape) != 1:
        raise ValueError(
            f"mojolearn SVR: y must be 1-D, got {len(shape)}-D shape {shape}"
        )
    a, _ = as_f32_c(y, ndim=1, name="y")
    if a.shape[0] != n_rows:
        raise ValueError(
            f"mojolearn SVR: y has {a.shape[0]} entries, X has {n_rows} rows"
        )
    return a


class SVR(NumericModeMixin):
    """Epsilon-support vector regression, backed by an SMO
    solver and GPU kernel matrices (`svm/`, DEVIATIONS 630-637;
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
                                  `svm/impl/svm_parameter.mojo`
                                  (DEVIATION 636)
        epsilon         honored   the half-width of the insensitive tube,
                                  cuML's `param.epsilon`, the parameter the
                                  whole rung-2 implementation exists for. Passed
                                  through UNCLAMPED so that
                                  `svm/impl/svm_parameter.mojo::
                                  check_rung1_scope`'s two refusals -- not
                                  finite, and negative -- stay reachable
                                  from this surface. They fire before any
                                  device context exists
        kernel          honored   'linear', 'rbf', 'poly' and 'sigmoid', the
                                  SVC's Gram and epilogues (DEVIATION 1663,
                                  identical_tanh). 'precomputed' is REFUSED
                                  BY NAME with what is missing
        gamma           honored   a finite float >= 0, or the string 'auto'
                                  (= 1 / n_features, cuML's `_get_gamma`).
                                  'scale' is resolved exactly, DEVIATION
                                  870 below (theirs is the default)
        degree          honored   with kernel='poly': an integer in [0, 32]
        coef0           honored   with 'poly' and 'sigmoid': a finite float
        tol             honored   the stopping tolerance. Refused if not
                                  finite or not positive, by `_svm_impl.py`
                                  and `svm/impl/svm_parameter.mojo`
        cache_size      honored   ONLY as the prediction buffer, see
                                  DEVIATION 871 below. The TRAINING LRU it
                                  also names in the reference is unimplemented and
                                  `svm/impl/svm_parameter.mojo` refuses
                                  a non-zero one; `svm/estimator.mojo` pins
                                  the training value at 0 so that refusal
                                  is never reached from here
        max_iter        honored   cuML's total inner-iteration cap; -1 (the
                                  default) is no limit. `_svm_impl.py`
                                  refuses 0 and anything below -1
        nochange_steps  honored   cuML's convergence rule, with
                                  its n_small_diff counter. NOT a
                                  scikit-learn parameter; it is cuML's, and
                                  it is here because the solver reads it
        verbose         refused   `_svm_impl.py`, for anything truthy. It
                                  selects LOG LINES in the reference
                                  (CUML_LOG_DEBUG); this implementation prints none,
                                  so accepting it would be
                                  accepting-and-ignoring
        output_type     refused   `_svm_impl.py`. A cuML-internal
                                  array-type selector; this package returns
                                  mojolearn Arrays
        shrinking       absent    NOT A PARAMETER OF THIS CLASS, so passing
                                  it is a TypeError naming it. It is
                                  scikit-learn's (libsvm's) shrinking
                                  heuristic and cuML has no such thing:
                                  there is no row for it in
                                  `svm/NOT_IMPLEMENTED.tsv` because there is
                                  nothing in the reference to leave
                                  unimplemented. `SVC` omits it for the same
                                  reason
        sample_weight   honored   in fit(): the weighted `InitPenalty` arm,
                                  row i's bound C * w_i at alpha_i and
                                  alpha*_i (`_c_rows`)
        sparse X        refused   `svrFitSparse` and every CSR arm are
                                  unimplemented; dense row-major float32 only.
                                  `_buffer.py::as_f32_c` is what refuses
        non-finite X    refused   `svm/impl/svr_impl.mojo` at fit and
                                  `svm/impl/svc_impl.mojo` at predict
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

    DEVIATION 870: `gamma='scale'` IS RESOLVED EXACTLY, AND THE DEFAULT
    HERE IS STILL 'auto' (theirs is 'scale'; the default is kept so every
    recorded default fit keeps its bits). 'scale' is `1 / (n_features *
    X.var())` as theirs, but the variance is the EXACT population variance
    of the float32 cells, formed in integers, and the reciprocal is rounded
    once to binary64 (`_scale_gamma.scale_gamma`). Their float32 `X.var()`
    carries its reduction's fold shape in its last bits; this one has no
    fold to differ, so every host reads the same gamma bits, and it differs
    from theirs by at most that reduction's rounding.

    DEVIATION 871: `cache_size` IS HONORED ONLY AT PREDICT, exactly as it is
    on `SVC`. Upstream it is two things under one name, the training-time
    `raft::cache` LRU and the ceiling on the prediction kernel-tile buffer
    (`svm_base.pyx:554`). The prediction half is honored here exactly and is
    a real knob: it sets the prediction batch size, and
    `check_svr_device_is_launch_invariant` holds the answer fixed over it
    from 0.001 MiB to 200 MiB ON THE REGRESSION PATH, which is a separate
    gate from the classifier's because SVR runs `UpdateF` twice per batch
    and so interleaves twice as many launches. The training half is NOT
    implemented.

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
    support_ : Array (n_SV,) int32
        Indices of the support vectors in the training matrix.
    support_vectors_ : Array (n_SV, n_features) float32
    dual_coef_ : Array (1, n_SV) float32
        scikit-learn's 2-D layout of cuML's 1-D folded `dual_coefs`. Each
        entry is `alpha_i - alpha*_i` for one row, which is why there are
        `n_SV` of them and not `2 * n_SV`.
    intercept_ : Array (1,) float32
    n_support_ : int
        cuML's scalar count.
    n_iter_ : int
        Total inner SMO iterations.
    coef_ : Array (1, n_features) float32
        `dual_coef_ @ support_vectors_`, and only for `kernel='linear'`;
        raises AttributeError otherwise, as scikit-learn does.
    n_features_in_ : int
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

    #: This family's binding, for `NumericModeMixin._bind`
    #: (lane/inference-svm, 2026-09-15): `predict` binds through
    #: `self._bind`, which resolves what `_extension` resolves, so
    #: `_classical_host.HostSVR` answers the CPU binding as `HostSVC` does.
    _BINDING = _EXT_NAME

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
        # The caller's own string object when it is already lower case, so
        # `get_params` hands `clone` back the object it was given.
        k = kernel if kernel == kernel.lower() else kernel.lower()
        if k in _SVR_REFUSED_KERNELS:
            raise NotImplementedError(
                f"mojolearn SVR: kernel={kernel!r} is refused; "
                + _SVR_REFUSED_KERNELS[k]
            )
        if k not in _SVR_KERNELS:
            raise ValueError(
                f"mojolearn SVR: kernel={kernel!r} is not a kernel name; "
                f"this implementation carries {sorted(_SVR_KERNELS)} and refuses "
                f"the others by name ({sorted(_SVR_REFUSED_KERNELS)})"
            )
        if isinstance(gamma, str):
            g = gamma.lower()
            if g not in ("auto", "scale"):
                raise ValueError(
                    f"mojolearn SVR: gamma={gamma!r} is not a name; it is "
                    "'auto', 'scale' or a float"
                )
            gamma = g
        else:
            gamma = float(gamma)
            if not math.isfinite(gamma) or gamma < 0.0:
                raise ValueError(
                    "mojolearn SVR: gamma must be finite and >= 0 for the RBF "
                    f"kernel, got {gamma!r} (DEVIATION 636; scikit-learn's own "
                    "constraint is gamma >= 0)"
                )
        if k == "poly":
            if isinstance(degree, bool) or not isinstance(degree, numbers.Integral):
                raise TypeError(
                    f"mojolearn SVR: degree={degree!r} must be an integer; the "
                    "polynomial power is an ascending repeated product (DEVIATION 1663)"
                )
            degree = int(degree)
            if not 0 <= degree <= _MAX_POLY_DEGREE:
                raise ValueError(
                    f"mojolearn SVR: degree must be in [0, {_MAX_POLY_DEGREE}], got "
                    f"{degree!r} (DEVIATION 1663)"
                )
        elif degree != 3:
            raise NotImplementedError(
                f"mojolearn SVR: degree={degree!r} is refused with kernel={k!r}; it "
                "is read only by kernel='poly'. Passing it with another kernel "
                "would be a parameter accepted and ignored"
            )
        if k in ("poly", "sigmoid"):
            coef0 = float(coef0)
            if not math.isfinite(coef0):
                raise ValueError(
                    f"mojolearn SVR: coef0 must be finite, got {coef0!r} (DEVIATION 636)"
                )
        elif coef0 != 0.0:
            raise NotImplementedError(
                f"mojolearn SVR: coef0={coef0!r} is refused with kernel={k!r}; it "
                "is read only by kernel='poly' and kernel='sigmoid'"
            )
        C = float(C)
        if not math.isfinite(C):
            raise ValueError(
                f"mojolearn SVR: C must be finite, got {C!r} (DEVIATION 636)"
            )
        if C <= 0.0:
            raise ValueError(f"mojolearn SVR: C must be positive, got {C!r}")
        tol = float(tol)
        if not math.isfinite(tol):
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
        if not (cache_size > 0.0) or not math.isfinite(cache_size):
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
                "CUML_LOG_DEBUG lines and this implementation prints none, so honoring "
                "it is impossible and accepting it would be accepting-and-"
                "ignoring"
            )
        if output_type is not None:
            raise NotImplementedError(
                "mojolearn SVR: output_type is a cuML-internal array-type "
                "selector; this package returns mojolearn Arrays"
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

    _resolve_gamma = SVC._resolve_gamma

    def _kernel_tail(self):
        """degree and coef0 as the bindings' optional trailing params: only
        for 'poly' and 'sigmoid', so every linear and rbf call keeps its
        exact params list."""
        if self.kernel in ("poly", "sigmoid"):
            return [int(self.degree), float(self.coef0)]
        return []

    def _fit_tail(self, c_rows):
        """svr_fit's optional slots: 9 the per-row bounds' address (0 = the
        unweighted arm), then 10 degree and 11 coef0."""
        kt = self._kernel_tail()
        if c_rows is None and not kt:
            return []
        head = [0 if c_rows is None else addr_ro(c_rows, name="C * sample_weight")]
        return head + kt

    def fit(self, X, y, sample_weight=None):
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        n_rows, n_cols = x.shape
        targets = _as_targets(y, n_rows)
        gamma = self._resolve_gamma(x)
        c_rows = _c_rows(self.C, n_rows, sample_weight, who="SVR")

        # WORST-CASE OUTPUT BUFFERS, AND `n_rows` IS THE WORST CASE.
        # The solver's domain is `2 * n_rows` (alpha+ and alpha-), but
        # `Results::combine_coefs` folds the two halves before it selects,
        # so at most `n_rows` coefficients, `n_rows` indices and
        # `n_rows * n_cols` support-vector floats can ever be written. Every
        # size here is a function of `(n_rows, n_cols)` alone, which is what
        # lets this side allocate before the call.
        dual = empty((n_rows,), "<f4")
        support = empty((n_rows,), "<i4")
        sv = empty((n_rows * n_cols,), "<f4")
        # THREE, not SVC's five: slots 3 and 4 there are the class labels.
        info = empty((3,), "<f8")
        n_support = _extension(getattr(self, 'numeric_mode', None)).svr_fit(
            addr_ro(x, name="x"),
            addr_ro(targets, name="targets"),
            addr(dual, name="dual"),
            addr(support, name="support"),
            addr(sv, name="sv"),
            addr(info, name="info"),
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svr_fit_binding.
            # n_rows, n_features, kernel, gamma, C, epsilon, tol, max_iter,
            # nochange_steps[, the per-row bounds' address]
            [n_rows, n_cols, _SVR_KERNELS[self.kernel], gamma, self.C,
             self.epsilon, self.tol, self.max_iter, self.nochange_steps] + self._fit_tail(c_rows),
        )
        n_support = int(n_support)
        b = _round_f32(info[0])

        self.n_features_in_ = n_cols
        self.n_support_ = n_support
        self.n_iter_ = int(info[2])
        self.intercept_ = Array.from_list([b], "<f4")
        # Slices COPY (the _array contract): compact, C-contiguous, detached.
        self.support_ = support[:n_support]
        self.support_vectors_ = sv[:n_support * n_cols].reshape(
            (n_support, n_cols))
        self.dual_coef_ = dual[:n_support].reshape((1, n_support))
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
            return zeros((1, self.n_features_in_), "<f4")
        return _dual_times_sv(self.dual_coef_, self.support_vectors_)

    def predict(self, X):
        """`sum_j K(x, sv_j) dual_j + b`, one value per row, float32.

        There is no `decision_function` beside this and no `predict_class`
        under it: on a regressor the decision value IS the prediction, and
        cuML reaches it through the same `svcPredict` with the class
        epilogue switched off.
        """
        if not hasattr(self, "dual_coef_"):
            raise ValueError("mojolearn SVR: call fit() first")
        q, _ = as_f32_c(X, ndim=2, name="X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn SVR: X has {q.shape[1]} features, fit saw "
                f"{self.n_features_in_}"
            )
        n_rows = q.shape[0]
        out = empty((n_rows,), "<f4")
        # Kept in locals so the arrays outlive the call; the Mojo side
        # borrows these addresses and owns nothing (`_buffer.py`).
        dual = self.dual_coef_
        sv = self.support_vectors_
        self._bind(_EXT_NAME).svr_predict(
            addr_ro(q, name="q"),
            addr_ro(dual, name="dual") if self.n_support_ > 0 else 0,
            addr_ro(sv, name="sv") if self.n_support_ > 0 else 0,
            addr(out, name="out"),
            # ORDER MATCHES bindings/_mojolearn_svm.mojo::svr_predict_binding.
            # n_rows, n_features, n_support, b, kernel, gamma,
            # cache_size_mib
            [n_rows, self.n_features_in_, self.n_support_,
             float(self.intercept_[0]), _SVR_KERNELS[self.kernel], self._gamma,
             self.cache_size] + self._kernel_tail(),
        )
        return out

    def score(self, X, y):
        """The coefficient of determination R^2, scikit-learn's definition,
        `1 - SS_res / SS_tot`.

        Accumulated in FLOAT64 from float32 predictions, which is what
        scikit-learn does and is not part of any identity claim: it is a
        summary of the answer, not the answer. SEQUENTIALLY, in Python
        (`linear_model._r2_sums`, DEVIATION 2365): the NumPy-era pairwise
        sum could differ in the last bits, so a recorded R^2 is
        RE-BASELINE OWED (test_svr_surface.py only thresholds it).
        """
        pred = self.predict(X)
        shape = _shape_of(y)
        if len(shape) != 1:
            raise ValueError(
                f"mojolearn SVR: y must be 1-D, got {len(shape)}-D shape {shape}"
            )
        if shape[0] != pred.shape[0]:
            raise ValueError(
                f"mojolearn SVR: y has {shape[0]} entries, X has "
                f"{pred.shape[0]} rows"
            )
        ss_res, ss_tot = _r2_sums(pred, y)
        if ss_tot == 0.0:
            # scikit-learn returns 1.0 for a perfect constant fit and 0.0
            # otherwise; the ratio is undefined and this is its convention.
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    def save(self, path):
        """Write the fitted model to `path` as an npz: `dual_coef_`,
        `support_vectors_`, `intercept_` as fitted (float32), `support_`
        `<i4`, `hyper` `<f8` [C, epsilon, tol, cache_size, the gamma the FIT
        resolved], `kernel` and `gamma` (the constructor's setting, 'auto'
        or a number) as strings, `meta` `<i8` [n_features_in_, n_support_,
        n_iter_, max_iter, nochange_steps] (lane/inference-svm, 2026-09-15).
        `mojolearn.host_model(path)` predicts from it on a CPU with no GPU
        through `_mojolearn_svm_host.svr_predict`."""
        if not hasattr(self, "dual_coef_"):
            raise RuntimeError("this estimator is not fitted yet")
        return _serialize.write_npz(path, {
            "format": _SVR_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "kernel": str(self.kernel),
            "gamma": str(self.gamma),
            "dual_coef": self.dual_coef_,
            "support_vectors": self.support_vectors_,
            "support": self.support_,
            "intercept": self.intercept_,
            "hyper": Array.from_list(
                [float(self.C), float(self.epsilon), float(self.tol),
                 float(self.cache_size), float(self._gamma)],
                "<f8",
            ),
            "meta": Array.from_list(
                [int(self.n_features_in_), int(self.n_support_), int(self.n_iter_),
                 int(self.max_iter), int(self.nochange_steps)],
                "<i8",
            ),
            # Only for poly and sigmoid, so every linear and rbf file keeps its bytes.
            **({"kernel_params": Array.from_list([float(self.degree), float(self.coef0)], "<f8")}
               if self.kernel in ("poly", "sigmoid") else {}),
        })

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts; every array
        is read at its saved dtype and never cast."""
        arrays = _serialize.read_npz(path, _SVR_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 5:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 5 are needed")
        hyper = _serialize.exact(arrays, "hyper", "<f8")
        if hyper.size != 5:
            raise ValueError(f"mojolearn: {path!r} hyper holds {hyper.size} fields, 5 are needed")
        gamma_setting = _serialize.scalar_str(arrays, "gamma")
        kernel_setting = _serialize.scalar_str(arrays, "kernel")
        degree, coef0 = 3, 0.0
        if kernel_setting in ("poly", "sigmoid"):
            kpar = _serialize.exact(arrays, "kernel_params", "<f8")
            if kpar.size != 2:
                raise ValueError(f"mojolearn: {path!r} kernel_params must hold degree and coef0")
            degree, coef0 = int(kpar[0]), float(kpar[1])
        obj = cls(
            C=float(hyper[0]),
            epsilon=float(hyper[1]),
            kernel=kernel_setting,
            degree=degree,
            coef0=coef0,
            gamma=gamma_setting if gamma_setting in ("auto", "scale") else float(gamma_setting),
            tol=float(hyper[2]),
            cache_size=float(hyper[3]),
            max_iter=int(meta[3]),
            nochange_steps=int(meta[4]),
        )
        _restore_mode(obj, arrays)
        nf, n_support = int(meta[0]), int(meta[1])
        dual = _serialize.exact(arrays, "dual_coef", "<f4")
        if dual.ndim != 2 or tuple(dual.shape) != (1, n_support):
            raise ValueError(f"mojolearn: {path!r} dual_coef shape {tuple(dual.shape)} is not (1, {n_support})")
        sv = _serialize.exact(arrays, "support_vectors", "<f4")
        if sv.ndim != 2 or tuple(sv.shape) != (n_support, nf):
            raise ValueError(f"mojolearn: {path!r} support_vectors shape {tuple(sv.shape)} is not ({n_support}, {nf})")
        support = _serialize.exact(arrays, "support", "<i4")
        if support.ndim != 1 or support.size != n_support:
            raise ValueError(f"mojolearn: {path!r} support does not match n_support_")
        intercept = _serialize.exact(arrays, "intercept", "<f4")
        if intercept.ndim != 1 or intercept.size != 1:
            raise ValueError(f"mojolearn: {path!r} intercept must hold one float32")
        obj.n_features_in_ = nf
        obj.n_support_ = n_support
        obj.n_iter_ = int(meta[2])
        obj.intercept_ = intercept
        obj.support_ = support
        obj.support_vectors_ = sv
        obj.dual_coef_ = dual
        obj._gamma = float(hyper[4])
        return obj
