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
from . import _native_archive_bridge
from . import _native_family_bridge
from ._array import Array
from ._buffer import _native, addr, addr_ro, empty, zeros
from ._mode import NumericModeMixin, available_if, is_fitted as _is_fitted

#: `SVC.save`'s format tag (the kde svc host lane, 2026-09-14).
_SVC_FORMAT = "mojolearn-svc-1"
#: `SVR.save`'s format tag (lane/inference-svm, 2026-09-15).
_SVR_FORMAT = "mojolearn-svr-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schemas registered at the end of this module;
# host_archive_native.mojo builds the loaded model with the refusals and
# messages of the Python loaders it replaced).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()

# The compiled SVM family owner (bindings/svc_native.mojo), connected once at
# import: SVC's and SVR's fit, decision_function, predict, predict_proba,
# predict_log_proba and score are one native call each, handed the
# `_mojolearn_svm` binding `_bind` (or `_extension`) resolves.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()

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
_SVC_KERNELS = dict(_KERNELS, poly=_KERNEL_POLYNOMIAL, sigmoid=_KERNEL_TANH,
                    precomputed=_KERNEL_PRECOMPUTED)

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
_SVC_REFUSED_KERNELS = {k: v for k, v in _REFUSED_KERNELS.items() if k not in ("poly", "sigmoid", "precomputed")}
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


def _i32(values):
    """`values` (an int32 Array or a sequence of ints) as an int32 Array."""
    if isinstance(values, Array):
        return values
    # lane py-runtime-b: array('i') converts the ints in C (no Python
    # comprehension over the indices); integral floats take int() as before
    try:
        store = array.array("i", values)
    except TypeError:
        store = array.array("i", map(int, values))
    return Array._owned(store, (len(store),), "<i4", "C")


def _gather(binding, x, rows=None, cols=None):
    """`x[rows][:, cols]`, an exact copy of float32 cells, gathered in the
    binding (on the device on a GPU install; cgfin-c-svm). An absent index
    list keeps that axis whole."""
    n_rows, n_cols = x.shape
    ra = None if rows is None else _i32(rows)
    ca = None if cols is None else _i32(cols)
    nor = n_rows if ra is None else ra.shape[0]
    noc = n_cols if ca is None else ca.shape[0]
    out = empty((nor, noc), "<f4")
    if nor and noc:
        binding.svc_pair_epilogue(
            addr_ro(x, name="x"),
            0 if ra is None else addr_ro(ra, name="row indices"),
            0 if ca is None else addr_ro(ca, name="column indices"),
            addr(out, name="gathered"),
            [_GLUE_GATHER, n_rows, n_cols, nor, noc])
    return out


def _precomputed_columns(binding, q, support):
    """kernel='precomputed' at predict: `X[:, support_]`, the query's
    cross-kernel columns at the support vectors, copied exactly (float32
    bytes, no arithmetic). What the solver's decision reads as its kernel
    tile."""
    # `support` is the fitted int32 Array (a fit's own rows, or a load's,
    # checked against n_features_in_ by the native loader); q has
    # n_features_in_ columns (`_query`), so every index is in range and no
    # per-predict Python pass over it runs (lane cpu2-l10-linear)
    return _gather(binding, q, None, _i32(support))


#: `svc_pair_epilogue` modes (svm/impl/svc_rows.mojo): the epilogues
#: (lane/py-dn-svm, 2026-09-28) and the row glue (cgfin-c-svm, 2026-10-02)
(_EPI_OVO, _EPI_OVR, _EPI_VOTES, _EPI_PROBA, _EPI_LOG_PROBA, _EPI_BINARY,
 _GLUE_GATHER, _GLUE_SELECT, _GLUE_C_ROWS, _GLUE_FOLD, _GLUE_FOLD_FINISH) = range(11)


def _native_perm(binding, n, seed):
    """`tests/_svm_reference.py::_splitmix_perm(n, seed)` in Mojo (`svc_splitmix_perm`): the same
    integers, an int32 `array.array`."""
    out = array.array("i", bytes(4 * n))
    seed &= 0xFFFFFFFFFFFFFFFF
    if n:
        binding.svc_splitmix_perm(out.buffer_info()[0], n, seed & 0xFFFFFFFF, seed >> 32)
    return out


def _native_sigmoid_train(binding, dec, labels):
    """`tests/_svm_reference.py::_sigmoid_train(dec, labels)` in Mojo (`svc_platt_train`): the same
    binary64 operations in the same order, the same `(A, B)` bits."""
    d = array.array("d", dec)
    lab = array.array("d", labels)
    if len(d) != len(lab):
        raise ValueError("decision values and labels differ in length")
    ab = array.array("d", [0.0, 0.0])
    binding.svc_platt_train(d.buffer_info()[0] if len(d) else 0,
                            lab.buffer_info()[0] if len(lab) else 0, ab.buffer_info()[0], len(d))
    return ab[0], ab[1]


# Maintainer notes (moved verbatim from the SVC docstring):
# C-support vector classification, binary and one-vs-one multiclass,
# backed by an SMO solver and GPU kernel matrices (`svm/`, DEVIATIONS 630-637;
# `svm/README.md`), the scikit-learn surface.
#
# WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
# because a parameter that is accepted and ignored is a wrong answer
# waiting for a caller (the house rule):
#
#     C               honored   the penalty. Must be finite and positive;
#                               a non-finite C is refused by name
#                               (DEVIATION 636)
#     kernel          honored   'linear', 'rbf' and 'poly'. 'poly' is the
#                               identical linear Gram then
#                               `(gamma * K + coef0) ** degree` as
#                               kernel_methods' DEVIATION 1663 epilogue
#                               (one fused multiply-add, an ascending
#                               repeated product). 'sigmoid' is the
#                               linear Gram then tanh(gamma * K + coef0)
#                               (kernel_methods' TANH epilogue,
#                               identical_tanh). 'precomputed': X is the
#                               n x n kernel matrix at fit (the solver's
#                               tiles are exact copies of its cells,
#                               cuML's extractColumnsForPrecomputed) and
#                               the q x n cross-kernel against the
#                               training rows at predict, whose support
#                               columns the decision reads.
#                               support_vectors_ then holds the kernel
#                               matrix's support ROWS (scikit-learn's is
#                               empty)
#     gamma           honored   a finite float >= 0, or the string 'auto'
#                               (= 1 / n_features, cuML's `_get_gamma`).
#                               'scale' is resolved exactly, DEVIATION
#                               870 below (theirs is the default)
#     degree          honored   with kernel='poly': an integer in
#                               [0, 32] (DEVIATION 1663). With any other
#                               kernel only the default 3 is accepted,
#                               since nothing would read it
#     coef0           honored   with 'poly' and 'sigmoid': a finite float. With
#                               any other kernel only the default 0.0
#                               is accepted
#     tol             honored   the stopping tolerance; must be finite and
#                               positive (DEVIATION 636)
#     cache_size      honored   ONLY as the prediction buffer, see
#                               DEVIATION 871 below
#     class_weight    honored   a dict {label: weight} or 'balanced'; it
#                               multiplies each row's bound, C * cw[y_i] *
#                               w_i (`_c_rows`)
#     max_iter        honored   cuML's total inner-iteration cap; -1 (the
#                               default) is no limit
#     nochange_steps  honored   cuML's convergence rule, with
#                               its n_small_diff counter
#     verbose         refused   anything truthy. It selects LOG LINES
#                               in the reference (CUML_LOG_DEBUG); this implementation
#                               prints none, so accepting it would be
#                               accepting-and-ignoring
#     random_state    honored   with probability=True: the key of the
#                               folds' SplitMix64 shuffle (None = 0);
#                               refused otherwise, since nothing else
#                               draws a random number
#     decision_       honored   'ovo' (the default here, cuML's) or 'ovr'
#       function_shape          (scikit-learn's): with more than two
#                               classes, decision_function's pairwise
#                               (n, K(K-1)/2) or voted (n, K) form. No
#                               effect on two classes, as in scikit-learn
#     break_ties      honored   with 'ovr' and more than two classes,
#                               predict is the argmax of the 'ovr'
#                               decision_function; refused with 'ovo'
#                               (scikit-learn's ValueError)
#     multiclass      honored   one-vs-one, libsvm's scheme: one binary
#                               machine per class pair (i < j) on the
#                               rows of those two classes, gamma resolved
#                               on the whole X, the per-row bounds cut to
#                               the pair's rows; prediction by vote, ties
#                               to the lowest class (`_fit_ovo`)
#     probability     honored   libsvm's Platt scaling: a seeded 5-fold
#                               cross-validation of each class pair, the
#                               folds' decision values from the binary
#                               solver, then sigmoid_train and (more than
#                               two classes) pairwise coupling in binary64
#                               with the portable exp/log
#                               (`_fit_probability`); predict_proba and
#                               predict_log_proba. The same bits on every
#                               column; not libsvm's bits (its fold
#                               shuffle is C rand())
#     output_type     refused   a cuML-internal array-type selector; this
#                               package returns mojolearn Arrays
#     sample_weight   honored   in fit(): the weighted `InitPenalty` arm,
#                               per-row bounds C * w formed in binary64 and
#                               rounded once to float32 (`_c_rows`)
#     sparse X        honored   densified exactly (`as_f32_dense_c`: the
#                               implicit entries are zeros), so it takes
#                               its dense twin's path and bits; cuML's
#                               CSR solver arms are not implemented
#
# Non-finite cells of `X` are refused by name inside the Mojo entry
# (DEVIATION 636), naming the flat index, rather than being fitted.
#
# DEVIATION 870: `gamma='scale'` IS RESOLVED EXACTLY, AND THE DEFAULT
# HERE IS STILL 'auto' (theirs is 'scale'; the default is kept so every
# recorded default fit keeps its bits). 'scale' is `1 / (n_features *
# X.var())` as theirs, but the variance is the EXACT population variance
# of the float32 cells, formed in integers, and the reciprocal is rounded
# once to binary64 (`_scale_gamma.scale_gamma`). Their float32 `X.var()`
# carries its reduction's fold shape in its last bits; this one has no
# fold to differ, so every host reads the same gamma bits, and it differs
# from theirs by at most that reduction's rounding.
#
# DEVIATION 871: `cache_size` IS HONORED ONLY AT PREDICT. In the reference it is
# two things under one name: the training-time `raft::cache` LRU kernel
# cache, and the ceiling on the prediction kernel-tile buffer
# (`svm_base.pyx:554`, passed as `buffer_size` to `svcPredict`). The
# prediction half is honored here, exactly, and it is a real knob: it
# sets the prediction batch size, and `check_device_is_launch_invariant`
# holds the answer fixed over it from 0.001 MiB to 200 MiB. The training
# half is NOT implemented -- the solver always runs cuML's own
# `n_cache_sets == 0` path -- so `cache_size` does not affect training
# time here the way it does in the reference. It cannot affect training RESULTS
# in the reference either (`svm/NOT_IMPLEMENTED.tsv` carries that determinism
# statement), so nothing numeric hangs on it.
#
# DEVIATION 873: the fitted model crosses back to the host and is
# uploaded again at every `predict` / `decision_function`, because the
# binding retains no device pointer. `fit` therefore allocates
# worst-case output buffers -- `n_samples` dual coefficients and
# `n_samples x n_features` support-vector floats -- since `n_support` is
# not known until the solve finishes. On a matrix that is mostly support
# vectors that is a second copy of X.
#
# Attributes
# ----------
# classes_ : list (K,)
#     The distinct labels, sorted ascending, as Python scalars under
#     the package-wide order rule (`_labels.sorted_classes`, DEVIATION
#     2340; it was an ndarray in `y`'s dtype). `predict` returns an
#     int64 / float64 Array for int / float labels and a Python list
#     otherwise. `classes_[1]` is the one the solver maps to +1
#     (`getOvrlabels(..., idx=1)`), so it fixes the sign of the decision
#     function.
# support_ : Array (n_SV,) int32
#     Indices of the support vectors in the training matrix.
# support_vectors_ : Array (n_SV, n_features) float32
# dual_coef_ : Array (1, n_SV) float32
#     scikit-learn's 2-D layout of cuML's 1-D `dual_coefs`.
# intercept_ : Array (1,) float32
# n_support_ : int, or Array (K,) int32 for K > 2
#     Two classes: cuML's scalar count (scikit-learn's is a per-class
#     array; this one is not, and the difference is here rather than in
#     a surprise). More: scikit-learn's per-class counts.
# n_iter_ : int, or Array (K(K-1)/2,) int32 for K > 2
#     Total inner SMO iterations, per pair for more than two classes.
#
# MULTICLASS LAYOUT (K > 2) is scikit-learn's, in its orientation (a
# pair's decision positive toward its LOWER class): `support_` grouped
# by class, ascending within a class; `dual_coef_` (K - 1, n_SV);
# `intercept_` (K(K-1)/2,); `coef_` (K(K-1)/2, n_features) for 'linear'.
# coef_ : Array (1, n_features) float32
#     `dual_coef_ @ support_vectors_`, and only for `kernel='linear'`;
#     raises AttributeError otherwise, as scikit-learn does.
# n_features_in_ : int
#
def _check_proba(self):
    """scikit-learn's `SVC._check_proba`: the AttributeError names why, and
    `available_if` chains it under "has no attribute 'predict_proba'"."""
    if not self.probability:
        raise AttributeError("predict_proba is not available when probability=False")
    return True


class SVC(NumericModeMixin):
    """C-support vector classification.

    Follows scikit-learn's `SVC`, with cuML's `SVC` as the reference
    implementation: an SMO solver with the kernel matrices computed on the
    GPU. Two classes train one machine; more train one machine per class
    pair (one-vs-one, libsvm's scheme) and predict by vote, ties going to
    the lowest class. `X` is converted to float32 (sparse input is
    densified), and non-finite values in `X` raise an error naming the
    first bad cell. Supports the `'identical'` (default) and `'fast'`
    numeric modes; `'fast'` is available on Apple GPUs only.

    Defaults that differ from scikit-learn: `gamma='auto'` (scikit-learn
    uses 'scale'); `cache_size` is the prediction buffer in MiB and does not
    affect training. There is no `shrinking` parameter. Every parameter is
    keyword-only. A parameter that would be ignored is refused rather than
    accepted: `degree` with any kernel but 'poly', `coef0` with any kernel
    but 'poly' or 'sigmoid', and a truthy `verbose` or any `output_type` all raise
    `NotImplementedError`.

    Parameters
    ----------
    C : float, default 1.0
        Regularization penalty. Must be finite and positive.
    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default 'rbf'
        The kernel. 'poly' is `(gamma * <x, y> + coef0) ** degree`;
        'sigmoid' is `tanh(gamma * <x, y> + coef0)`. With 'precomputed',
        `X` is the n x n kernel matrix at fit and the q x n kernel against
        the training rows at predict; `support_vectors_` then holds the
        support rows of the kernel matrix (scikit-learn's is empty). Other
        names, including 'polynomial' and 'tanh', raise.
    degree : int, default 3
        Degree of the 'poly' kernel, an integer in [0, 32]. Must stay 3
        with any other kernel.
    gamma : float, 'auto' or 'scale', default 'auto'
        Kernel coefficient, a finite float >= 0. 'auto' is
        `1 / n_features`; 'scale' is `1 / (n_features * X.var())`, computed
        from the exact variance of the float32 data, so it can differ from
        scikit-learn's float32 `X.var()` in the last bits.
    coef0 : float, default 0.0
        Independent term of the 'poly' and 'sigmoid' kernels, finite. Must
        stay 0.0 with any other kernel.
    tol : float, default 1e-3
        Stopping tolerance. Must be finite and positive.
    cache_size : float, default 1024.0
        Size in MiB of the kernel tile buffer used at prediction, which
        sets the prediction batch size; results do not depend on it. It
        has no training-time kernel cache role here. Must be positive and
        finite.
    max_iter : int, default -1
        Cap on the total inner SMO iterations (cuML's meaning); -1 means no
        limit. 0 and values below -1 are refused.
    nochange_steps : int, default 1000
        Follows cuML's ``nochange_steps`` convergence rule. Must be >= 0.
    verbose : bool, default False
        Only False is accepted; this estimator prints no log lines.
    output_type : None, default None
        Only None is accepted; results are mojolearn Arrays.
    random_state : int or None, default None
        With `probability=True`, the seed of the cross-validation fold
        shuffle (None means 0); an int in [0, 2**64). Unused when
        `probability` is False.
    class_weight : dict, 'balanced' or None, default None
        Per-class weights; row i's penalty becomes
        `C * class_weight[y_i] * sample_weight_i`.
    decision_function_shape : {'ovr', 'ovo'}, default 'ovr'
        With more than two classes, `decision_function` returns the
        voted (n, K) form ('ovr') or the pairwise (n, K(K-1)/2) form
        ('ovo'). No effect with two classes.
    probability : bool, default False
        Fit Platt scaling (libsvm's method: a seeded 5-fold
        cross-validation per class pair, then a sigmoid fit and, for more
        than two classes, pairwise coupling) so that `predict_proba` and
        `predict_log_proba` are available. The fold shuffle differs from
        libsvm's, so the probabilities are not bit-equal to libsvm's.
    break_ties : bool, default False
        With `decision_function_shape='ovr'` and more than two classes,
        predict the argmax of the 'ovr' decision function instead of the
        vote. Must be False with 'ovo' (scikit-learn's rule).
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator; None uses the process default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    classes_ : list of shape (K,)
        The distinct labels, sorted ascending, as Python scalars.
        `classes_[1]` is the positive class of the binary decision
        function.
    support_ : Array of shape (n_SV,), int32
        Indices of the support vectors in the training data.
    support_vectors_ : Array of shape (n_SV, n_features), float32
        The support vectors.
    dual_coef_ : Array of shape (1, n_SV) or (K - 1, n_SV), float32
        Dual coefficients, in scikit-learn's layout.
    intercept_ : Array of shape (1,) or (K(K-1)/2,), float32
        Intercepts of the decision functions.
    coef_ : Array of shape (1, n_features) or (K(K-1)/2, n_features), float32
        Primal weights `dual_coef_ @ support_vectors_`, only for
        `kernel='linear'`; raises `AttributeError` otherwise, as
        scikit-learn does.
    n_support_ : int, or Array of shape (K,), int32
        Number of support vectors. With two classes this is a single count
        (cuML's form; scikit-learn returns a per-class array); with more,
        the per-class counts.
    n_iter_ : int, or Array of shape (K(K-1)/2,), int32
        Total inner SMO iterations, per class pair with more than two
        classes.
    probA_, probB_ : Array of shape (K(K-1)/2,), float64
        Platt sigmoid parameters per class pair, when `probability=True`.
    n_features_in_ : int
        Number of features seen during `fit`.

    Notes
    -----
    With more than two classes the fitted arrays follow scikit-learn's
    layout and orientation: a pair's decision is positive toward its lower
    class, `support_` is grouped by class (ascending within a class),
    `dual_coef_` is (K - 1, n_SV) and `intercept_` is (K(K-1)/2,).

    `predict` returns an int64 or float64 Array for integer or float
    labels, and a Python list for other label types.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int64)
    >>> model = mojolearn.SVC(C=1.0, kernel="rbf").fit(X, y)
    >>> labels = model.predict(X[:5])
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
        decision_function_shape="ovr",
        probability=False,
        break_ties=False,
    ):
        # Stored verbatim (scikit-learn's convention); `_validate_params`
        # checks them when `fit` (or `load`) runs and keeps the checked
        # values the solver reads as `_hp_<name>`.
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.nochange_steps = nochange_steps
        self.verbose = verbose
        self.output_type = output_type
        self.random_state = random_state
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.probability = probability
        self.break_ties = break_ties

    def __setstate__(self, state):
        return _NATIVE(self, "family_svc_public_svc_setstate", (state,), _NATIVE_CONTEXT)

    def _validate_params(self):
        """The constructor checks, run at the start of `fit` and in `load`."""
        C = self.C
        kernel = self.kernel
        degree = self.degree
        gamma = self.gamma
        coef0 = self.coef0
        tol = self.tol
        cache_size = self.cache_size
        max_iter = self.max_iter
        nochange_steps = self.nochange_steps
        verbose = self.verbose
        output_type = self.output_type
        random_state = self.random_state
        class_weight = self.class_weight
        decision_function_shape = self.decision_function_shape
        probability = self.probability
        break_ties = self.break_ties

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
            g = gamma if gamma == gamma.lower() else gamma.lower()
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
        probability = bool(probability)
        if not probability:
            # scikit-learn accepts random_state always and reads it only for
            # probability's cross-validation folds; the C-SVC solver draws no
            # random numbers, so without probability it is accepted and unused
            random_state = None
        if random_state is not None and (
                isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral)
                or not 0 <= int(random_state) < 2 ** 64):
            raise ValueError(
                "mojolearn SVC: random_state must be None or an int in [0, 2**64) "
                f"(the key of probability's fold shuffle), got {random_state!r}"
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
        self._hp_C = C
        self._hp_kernel = k
        self._hp_degree = degree
        self._hp_gamma = gamma
        self._hp_coef0 = coef0
        self._hp_tol = tol
        self._hp_cache_size = cache_size
        self._hp_max_iter = max_iter
        self._hp_nochange_steps = nochange_steps
        self._hp_random_state = None if random_state is None else int(random_state)
        self._hp_probability = probability
        self._hp_break_ties = break_ties

    def fit(self, X, y, sample_weight=None):
        """Fit the support vector classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32. With
            `kernel='precomputed'`, the square n x n kernel matrix.
        y : array-like of shape (n_samples,)
            Class labels; two or more distinct values.
        sample_weight : array-like of shape (n_samples,), default None
            Per-sample weights; row i's penalty becomes `C * w_i` (times
            its class weight). None weights every sample equally.

        Returns
        -------
        self : SVC
            The fitted estimator.
        """
        return _NATIVE(self, "family_svc_svc_fit", (None, X, y, sample_weight), _NATIVE_CONTEXT)

    def _dual_times_sv(self, dual_coef, support_vectors, out=None, row=0, negate=False):
        """`dual_coef_ @ support_vectors_` in the binding (`svc_dual_gemv`,
        on the device on a GPU install: an ascending binary64 chain per
        feature), into row `row` of `out` (a new (1, d) Array when None),
        negated when `negate` (exact)."""
        native = self._bind(_EXT_NAME)
        n_sv, n_features = support_vectors.shape
        if out is None:
            out = empty((1, n_features), "<f4")
        if n_features:
            native.svc_dual_gemv(
                addr_ro(dual_coef, name="dual_coef_") if n_sv else 0,
                addr_ro(support_vectors, name="support_vectors_") if n_sv else 0,
                [n_sv, n_features, 1 if negate else 0], addr(out, name="coef_") + 4 * row * n_features)
        return out

    @property
    def coef_(self):
        """Primal weights, only for `kernel='linear'`.

        Shape (1, n_features) for two classes and (K(K-1)/2, n_features)
        for K > 2 classes, float32. Raises `AttributeError` for other
        kernels or before `fit`.
        """
        return _NATIVE(self, "family_svc_svc_coef", (None,), _NATIVE_CONTEXT)

    def decision_function(self, X):
        """Evaluate the decision function for the rows of `X`.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Query data. With `kernel='precomputed'`, the kernel between the
            queries and the training rows, shape (n_rows, n_samples).

        Returns
        -------
        Array
            Two classes: shape (n_rows,), the raw
            `sum_j K(x, sv_j) dual_j + b`, positive toward `classes_[1]`.
            More classes with `decision_function_shape='ovo'`: shape
            (n_rows, K(K-1)/2), float32, one column per class pair,
            positive toward the pair's lower class. With 'ovr': shape
            (n_rows, K), float64, scikit-learn's vote-plus-confidence
            transform of the pairwise values.
        """
        return _NATIVE(self, "family_svc_svc_decision", (None, X), _NATIVE_CONTEXT)

    # Maintainer notes: binary predict is cuML's `applyPrediction` epilogue,
    # ON THE DEVICE: the label is chosen there by `val + b < 0 ? classes_[0] :
    # classes_[1]` and this maps the float32 label it wrote back into
    # `classes_`'s dtype. Multiclass: libsvm's vote (a pair's decision < 0
    # here votes class i, otherwise class j, the binary epilogue's rule).
    def predict(self, X):
        """Predict class labels for the rows of `X`.

        Two classes: `classes_[1]` where the decision function is >= 0,
        otherwise `classes_[0]`. More classes: one-vs-one vote, the most
        votes winning and ties going to the lowest class; with
        `break_ties=True` (and `decision_function_shape='ovr'`), the argmax
        of the 'ovr' decision function, the first maximum winning.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Query data (the cross-kernel with `kernel='precomputed'`).

        Returns
        -------
        Array or list of shape (n_rows,)
            Labels from `classes_`: an int64 or float64 Array for integer
            or float labels, a Python list otherwise.
        """
        return _NATIVE(self, "family_svc_svc_predict", (None, X), _NATIVE_CONTEXT)

    # Maintainer note: libsvm's `svm_predict_probability`; the coupling is
    # `_multiclass_probability`.
    @available_if(_check_proba)
    def predict_proba(self, X):
        """Class probabilities from the fitted Platt scaling.

        Follows libsvm: each class pair's decision value goes through its
        fitted sigmoid (clamped to [1e-7, 1 - 1e-7]); with more than two
        classes the pairwise probabilities are then coupled.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Query data.

        Returns
        -------
        Array of shape (n_rows, K), float64
            Probabilities, columns in `classes_` order.

        Raises
        ------
        AttributeError
            If the model was fitted with `probability=False`.
        """
        return _NATIVE(self, "family_svc_svc_proba", (None, X), _NATIVE_CONTEXT)

    # Maintainer note: the portable binary64 log (its Mojo twin, `pm_log`, in
    # the same epilogue call); a probability <= 0 raises as
    # `_portable_math.log` does.
    @available_if(_check_proba)
    def predict_log_proba(self, X):
        """Natural log of `predict_proba(X)`.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Query data.

        Returns
        -------
        Array of shape (n_rows, K), float64
            Log probabilities, columns in `classes_` order.

        Raises
        ------
        AttributeError
            If the model was fitted with `probability=False`.
        """
        return _NATIVE(self, "family_svc_svc_log_proba", (None, X), _NATIVE_CONTEXT)

    # Maintainer note: a Python count over O(rows) labels (DEVIATION 2365).
    def score(self, X, y):
        """Mean accuracy of `predict(X)` against `y`.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Query data.
        y : array-like of shape (n_rows,)
            True labels.

        Returns
        -------
        float
            Fraction of rows predicted correctly.
        """
        return _NATIVE(self, "family_svc_svc_score", (None, X, y), _NATIVE_CONTEXT)

    # Maintainer note, file layout: `dual_coef_`, `support_vectors_`,
    # `intercept_` as fitted (float32), `support_` `<i4`, `classes_` as
    # `_labels.classes_member`, `labels` `<f8` (the float32 label pair the
    # solver returned, `_label0` and `_label1`), `hyper` `<f8` [C, tol,
    # cache_size, the gamma the FIT resolved], `kernel` and `gamma` (the
    # constructor's setting, 'auto' or a number) as strings, `meta` `<i8`
    # [n_features_in_, n_support_, n_iter_, max_iter, nochange_steps, degree].
    # `mojolearn.host_model(path)` predicts through
    # `_mojolearn_svm_host.svc_predict`.
    def save(self, path):
        """Save the fitted model to `path` as an `.npz` file.

        The file holds the fitted arrays, the hyperparameters, the gamma
        resolved at fit and the numeric mode. Load it with `SVC.load`, or
        predict from it on a CPU without a GPU with
        `mojolearn.host_model(path)`.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        Every array is read at its saved dtype, never cast.

        Parameters
        ----------
        path : str or path-like
            File written by `SVC.save`.

        Returns
        -------
        SVC
            A fitted model, ready to predict, with the saved numeric mode.

        Raises
        ------
        ValueError
            If the file is not a valid saved `SVC`.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# Maintainer notes (moved verbatim from the SVR docstring). NOTE: the kernel row
# below says 'precomputed' is refused; the code accepts it (_SVR_KERNELS is
# _SVC_KERNELS, and fit/predict handle it), so that row is stale.
# Epsilon-support vector regression, backed by an SMO
# solver and GPU kernel matrices (`svm/`, DEVIATIONS 630-637;
# `svm/README.md`), the scikit-learn surface.
#
# THE TUBE IS THE ALGORITHM. Rows whose prediction lands inside a band of
# half-width `epsilon` around the target cost nothing, so they never
# become support vectors; the fit is determined by the rows outside it.
# Raising `epsilon` widens the band and can only take rows out of the
# support set, never add them, which is the one property of this
# estimator a caller can check without a reference implementation, and
# `python/mojolearn/tests/test_svr_surface.py` checks it.
#
# WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
# because a parameter that is accepted and ignored is a wrong answer
# waiting for a caller (the house rule). The file named on a refusal is
# the file that raises it.
#
#     C               honored   the penalty. Refused if not finite or not
#                               positive, by `_svm_impl.py` and again by
#                               `svm/impl/svm_parameter.mojo`
#                               (DEVIATION 636)
#     epsilon         honored   the half-width of the insensitive tube,
#                               cuML's `param.epsilon`, the parameter the
#                               whole rung-2 implementation exists for. Passed
#                               through UNCLAMPED so that
#                               `svm/impl/svm_parameter.mojo::
#                               check_rung1_scope`'s two refusals -- not
#                               finite, and negative -- stay reachable
#                               from this surface. They fire before any
#                               device context exists
#     kernel          honored   'linear', 'rbf', 'poly' and 'sigmoid', the
#                               SVC's Gram and epilogues (DEVIATION 1663,
#                               identical_tanh). 'precomputed' is REFUSED
#                               BY NAME with what is missing
#     gamma           honored   a finite float >= 0, or the string 'auto'
#                               (= 1 / n_features, cuML's `_get_gamma`).
#                               'scale' is resolved exactly, DEVIATION
#                               870 below (theirs is the default)
#     degree          honored   with kernel='poly': an integer in [0, 32]
#     coef0           honored   with 'poly' and 'sigmoid': a finite float
#     tol             honored   the stopping tolerance. Refused if not
#                               finite or not positive, by `_svm_impl.py`
#                               and `svm/impl/svm_parameter.mojo`
#     cache_size      honored   ONLY as the prediction buffer, see
#                               DEVIATION 871 below. The TRAINING LRU it
#                               also names in the reference is unimplemented and
#                               `svm/impl/svm_parameter.mojo` refuses
#                               a non-zero one; `svm/estimator.mojo` pins
#                               the training value at 0 so that refusal
#                               is never reached from here
#     max_iter        honored   cuML's total inner-iteration cap; -1 (the
#                               default) is no limit. `_svm_impl.py`
#                               refuses 0 and anything below -1
#     nochange_steps  honored   cuML's convergence rule, with
#                               its n_small_diff counter. NOT a
#                               scikit-learn parameter; it is cuML's, and
#                               it is here because the solver reads it
#     verbose         refused   `_svm_impl.py`, for anything truthy. It
#                               selects LOG LINES in the reference
#                               (CUML_LOG_DEBUG); this implementation prints none,
#                               so accepting it would be
#                               accepting-and-ignoring
#     output_type     refused   `_svm_impl.py`. A cuML-internal
#                               array-type selector; this package returns
#                               mojolearn Arrays
#     shrinking       absent    NOT A PARAMETER OF THIS CLASS, so passing
#                               it is a TypeError naming it. It is
#                               scikit-learn's (libsvm's) shrinking
#                               heuristic and cuML has no such thing:
#                               there is no row for it in
#                               `svm/NOT_IMPLEMENTED.tsv` because there is
#                               nothing in the reference to leave
#                               unimplemented. `SVC` omits it for the same
#                               reason
#     sample_weight   honored   in fit(): the weighted `InitPenalty` arm,
#                               row i's bound C * w_i at alpha_i and
#                               alpha*_i (`_c_rows`)
#     sparse X        honored   densified exactly (`as_f32_dense_c`), its
#                               dense twin's path and bits; cuML's CSR
#                               arms are not implemented
#     non-finite X    refused   `svm/impl/svr_impl.mojo` at fit and
#                               `svm/impl/svc_impl.mojo` at predict
#                               name the flat index (DEVIATION 636)
#                               rather than fitting it
#     non-finite y    refused   the same `check_finite_list`, same
#                               message, and it is the ONLY check `y`
#                               gets on this class
#
# THE PARAMETER ORDER IS scikit-learn's `SVR`, and every name it shares
# with scikit-learn means what scikit-learn means by it. TWO DEFAULTS
# DIFFER and both are stated where they bite: `gamma` is 'auto' here and
# 'scale' there (DEVIATION 870, below), and `cache_size` is 1024.0 MiB
# here and 200 there because it is a different quantity on this surface
# (DEVIATION 871). `shrinking` is absent, above.
#
# THIS CLASS MAKES NO CROSS-VENDOR CLAIM, and that is a narrower statement
# than `SVC`'s. The classifier's 32-stage IDENTICAL card was diffed on
# Apple M4, NVIDIA H100 and AMD MI325X at `a0a0eee` (E3 round 13,
# 2026-08-28). The REGRESSION path was gated after that leg and has not
# been in a three-vendor round: what stands behind it is 44 of 44 gates at
# `fea6becc` on one box, of which 26 are the SVR ones, including four
# property gates derived from the epsilon-insensitive formulation rather
# than from this solver (a dual objective that must not increase, a KKT
# gap, a tube bound, and a gradient recomputed from alpha alone that
# matches the solver's `f` to 1.5e-07). A three-vendor SVR card is OWED.
#
# DEVIATION 870: `gamma='scale'` IS RESOLVED EXACTLY, AND THE DEFAULT
# HERE IS STILL 'auto' (theirs is 'scale'; the default is kept so every
# recorded default fit keeps its bits). 'scale' is `1 / (n_features *
# X.var())` as theirs, but the variance is the EXACT population variance
# of the float32 cells, formed in integers, and the reciprocal is rounded
# once to binary64 (`_scale_gamma.scale_gamma`). Their float32 `X.var()`
# carries its reduction's fold shape in its last bits; this one has no
# fold to differ, so every host reads the same gamma bits, and it differs
# from theirs by at most that reduction's rounding.
#
# DEVIATION 871: `cache_size` IS HONORED ONLY AT PREDICT, exactly as it is
# on `SVC`. Upstream it is two things under one name, the training-time
# `raft::cache` LRU and the ceiling on the prediction kernel-tile buffer
# (`svm_base.pyx:554`). The prediction half is honored here exactly and is
# a real knob: it sets the prediction batch size, and
# `check_svr_device_is_launch_invariant` holds the answer fixed over it
# from 0.001 MiB to 200 MiB ON THE REGRESSION PATH, which is a separate
# gate from the classifier's because SVR runs `UpdateF` twice per batch
# and so interleaves twice as many launches. The training half is NOT
# implemented.
#
# DEVIATION 873: the fitted model crosses back to the host and is uploaded
# again at every `predict`, because the binding retains no device pointer.
# `fit` therefore allocates worst-case output buffers, `n_samples` dual
# coefficients and `n_samples x n_features` support-vector floats, since
# `n_support` is not known until the solve finishes. THOSE SIZES ARE NOT
# DOUBLED for the regressor even though its solver domain is: `Results`
# folds the two alpha halves before it selects, so at most `n_samples`
# coefficients can be written.
#
# Attributes
# ----------
# support_ : Array (n_SV,) int32
#     Indices of the support vectors in the training matrix.
# support_vectors_ : Array (n_SV, n_features) float32
# dual_coef_ : Array (1, n_SV) float32
#     scikit-learn's 2-D layout of cuML's 1-D folded `dual_coefs`. Each
#     entry is `alpha_i - alpha*_i` for one row, which is why there are
#     `n_SV` of them and not `2 * n_SV`.
# intercept_ : Array (1,) float32
# n_support_ : int
#     cuML's scalar count.
# n_iter_ : int
#     Total inner SMO iterations.
# coef_ : Array (1, n_features) float32
#     `dual_coef_ @ support_vectors_`, and only for `kernel='linear'`;
#     raises AttributeError otherwise, as scikit-learn does.
# n_features_in_ : int
#
class SVR(NumericModeMixin):
    """Epsilon-support vector regression.

    Follows scikit-learn's `SVR`, with cuML's `SVR` as the reference
    implementation: an SMO solver with the kernel matrices computed on the
    GPU. Rows whose prediction falls within `epsilon` of the target cost
    nothing and never become support vectors, so raising `epsilon` can only
    remove rows from the support set. `X` is converted to float32 (sparse
    input is densified); non-finite values in `X` or `y` raise an error
    naming the first bad cell. Supports the `'identical'` (default) and
    `'fast'` numeric modes; `'fast'` is available on Apple GPUs only.

    The parameters keep scikit-learn's order and meaning, with two
    defaults that differ: `gamma='auto'` (scikit-learn uses 'scale') and
    `cache_size=1024.0`, which here is the prediction buffer in MiB and
    does not affect training (scikit-learn's 200 is a training cache).
    There is no `shrinking` parameter; passing it raises `TypeError`.
    `nochange_steps` is cuML's. Every parameter is keyword-only. A
    parameter that would be ignored is refused: `degree` with any kernel
    but 'poly', `coef0` with any kernel but 'poly' or 'sigmoid', and a
    truthy `verbose` or any `output_type` raise `NotImplementedError`.

    Parameters
    ----------
    kernel : {'linear', 'rbf', 'poly', 'sigmoid', 'precomputed'}, default 'rbf'
        The kernel. 'poly' is `(gamma * <x, y> + coef0) ** degree`;
        'sigmoid' is `tanh(gamma * <x, y> + coef0)`. With 'precomputed',
        `X` is the n x n kernel matrix at fit and the q x n kernel against
        the training rows at predict. Other names, including 'polynomial'
        and 'tanh', raise.
    degree : int, default 3
        Degree of the 'poly' kernel, an integer in [0, 32]. Must stay 3
        with any other kernel.
    gamma : float, 'auto' or 'scale', default 'auto'
        Kernel coefficient, a finite float >= 0. 'auto' is
        `1 / n_features`; 'scale' is `1 / (n_features * X.var())`, computed
        from the exact variance of the float32 data, so it can differ from
        scikit-learn's float32 `X.var()` in the last bits.
    coef0 : float, default 0.0
        Independent term of the 'poly' and 'sigmoid' kernels, finite. Must
        stay 0.0 with any other kernel.
    tol : float, default 1e-3
        Stopping tolerance. Must be finite and positive.
    C : float, default 1.0
        Regularization penalty. Must be finite and positive.
    epsilon : float, default 0.1
        Half-width of the epsilon-insensitive tube. Must be finite and
        >= 0; an invalid value is refused when `fit` runs.
    cache_size : float, default 1024.0
        Size in MiB of the kernel tile buffer used at prediction, which
        sets the prediction batch size; results do not depend on it. Must
        be positive and finite.
    verbose : bool, default False
        Only False is accepted; this estimator prints no log lines.
    max_iter : int, default -1
        Cap on the total inner SMO iterations (cuML's meaning); -1 means no
        limit. 0 and values below -1 are refused.
    nochange_steps : int, default 1000
        Follows cuML's ``nochange_steps`` convergence rule. Must be >= 0.
    output_type : None, default None
        Only None is accepted; results are mojolearn Arrays.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this estimator; None uses the process default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    support_ : Array of shape (n_SV,), int32
        Indices of the support vectors in the training data.
    support_vectors_ : Array of shape (n_SV, n_features), float32
        The support vectors (rows of the kernel matrix with
        `kernel='precomputed'`).
    dual_coef_ : Array of shape (1, n_SV), float32
        `alpha_i - alpha*_i` for each support vector, in scikit-learn's
        layout.
    intercept_ : Array of shape (1,), float32
        Intercept of the decision function.
    coef_ : Array of shape (1, n_features), float32
        Primal weights `dual_coef_ @ support_vectors_`, only for
        `kernel='linear'`; raises `AttributeError` otherwise, as
        scikit-learn does.
    n_support_ : int
        Number of support vectors (cuML's scalar form; scikit-learn's is an
        array).
    n_iter_ : int
        Total inner SMO iterations.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X.sum(axis=1)
    >>> model = mojolearn.SVR(C=1.0, epsilon=0.1).fit(X, y)
    >>> pred = model.predict(X[:5])
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
        # Stored verbatim (scikit-learn's convention); `_validate_params`
        # checks them when `fit` (or `load`) runs and keeps the checked
        # values the solver reads as `_hp_<name>`.
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.nochange_steps = nochange_steps
        self.output_type = output_type

    def __setstate__(self, state):
        return _NATIVE(self, "family_svc_public_svr_setstate", (state,), _NATIVE_CONTEXT)

    def _validate_params(self):
        """The constructor checks, run at the start of `fit` and in `load`."""
        kernel = self.kernel
        degree = self.degree
        gamma = self.gamma
        coef0 = self.coef0
        tol = self.tol
        C = self.C
        epsilon = self.epsilon
        cache_size = self.cache_size
        verbose = self.verbose
        max_iter = self.max_iter
        nochange_steps = self.nochange_steps
        output_type = self.output_type

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
            g = gamma if gamma == gamma.lower() else gamma.lower()
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
        self._hp_kernel = k
        self._hp_degree = degree
        self._hp_gamma = gamma
        self._hp_coef0 = coef0
        self._hp_tol = tol
        self._hp_C = C
        self._hp_epsilon = epsilon
        self._hp_cache_size = cache_size
        self._hp_max_iter = max_iter
        self._hp_nochange_steps = nochange_steps

    _dual_times_sv = SVC._dual_times_sv

    def fit(self, X, y, sample_weight=None):
        """Fit the support vector regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32. With
            `kernel='precomputed'`, the square n x n kernel matrix.
        y : array-like of shape (n_samples,)
            Finite target values.
        sample_weight : array-like of shape (n_samples,), default None
            Per-sample weights; row i's penalty becomes `C * w_i`. None
            weights every sample equally.

        Returns
        -------
        self : SVR
            The fitted estimator.
        """
        return _NATIVE(self, "family_svc_svr_fit", (None, X, y, sample_weight), _NATIVE_CONTEXT)

    @property
    def coef_(self):
        """Primal weights, shape (1, n_features), float32.

        Only for `kernel='linear'`; raises `AttributeError` for other
        kernels or before `fit`.
        """
        return _NATIVE(self, "family_svc_svr_coef", (None,), _NATIVE_CONTEXT)

    # Maintainer note: there is no `decision_function` beside this and no
    # `predict_class` under it: on a regressor the decision value IS the
    # prediction, and cuML reaches it through the same `svcPredict` with the
    # class epilogue switched off.
    def predict(self, X):
        """Predict target values for the rows of `X`.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Query data. With `kernel='precomputed'`, the kernel between the
            queries and the training rows, shape (n_rows, n_samples).

        Returns
        -------
        Array of shape (n_rows,), float32
            `sum_j K(x, sv_j) dual_j + b` for each row.
        """
        return _NATIVE(self, "family_svc_svr_predict", (None, X), _NATIVE_CONTEXT)

    # Maintainer notes: accumulated in FLOAT64 from float32 predictions, which
    # is what scikit-learn does and is not part of any identity claim.
    # SEQUENTIALLY, in Python (`linear_model._r2_sums`, DEVIATION 2365): the
    # NumPy-era pairwise sum could differ in the last bits.
    def score(self, X, y):
        """Coefficient of determination R^2 of `predict(X)` against `y`.

        scikit-learn's definition, `1 - SS_res / SS_tot`, accumulated in
        float64. A constant `y` scores 1.0 when predicted exactly and 0.0
        otherwise.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_features)
            Query data.
        y : array-like of shape (n_rows,)
            True target values.

        Returns
        -------
        float
            The R^2 score.
        """
        return _NATIVE(self, "family_svc_svr_score", (None, X, y), _NATIVE_CONTEXT)

    # Maintainer note, file layout: `dual_coef_`, `support_vectors_`,
    # `intercept_` as fitted (float32), `support_` `<i4`, `hyper` `<f8` [C,
    # epsilon, tol, cache_size, the gamma the FIT resolved], `kernel` and
    # `gamma` (the constructor's setting, 'auto' or a number) as strings,
    # `meta` `<i8` [n_features_in_, n_support_, n_iter_, max_iter,
    # nochange_steps], `kernel_params` [degree, coef0] for poly and sigmoid.
    # `mojolearn.host_model(path)` predicts through
    # `_mojolearn_svm_host.svr_predict`.
    def save(self, path):
        """Save the fitted model to `path` as an `.npz` file.

        The file holds the fitted arrays, the hyperparameters, the gamma
        resolved at fit and the numeric mode. Load it with `SVR.load`, or
        predict from it on a CPU without a GPU with
        `mojolearn.host_model(path)`.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        Every array is read at its saved dtype, never cast.

        Parameters
        ----------
        path : str or path-like
            File written by `SVR.save`.

        Returns
        -------
        SVR
            A fitted model, ready to predict, with the saved numeric mode.

        Raises
        ------
        ValueError
            If the file is not a valid saved `SVR`.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


_native_archive_bridge.register(
    SVC, _SVC_FORMAT, (
        _native_archive_bridge.fitted("dual_coef_"),
        *_native_archive_bridge.head(_SVC_FORMAT),
        ("str", "kernel", "_hp_kernel"),
        ("str", "gamma", "_hp_gamma"),
        ("array", "dual_coef", "dual_coef_"),
        ("array", "support_vectors", "support_vectors_"),
        ("array", "support", "support_"),
        ("array", "intercept", "intercept_"),
        ("classes", "classes", "classes_"),
        ("f8", "labels", (("float", "_label0"), ("float", "_label1"))),
        ("f8", "hyper", (("float", "_hp_C"), ("float", "_hp_tol"), ("float", "_hp_cache_size"),
                         ("float", "_gamma"))),
        ("when_none", "_pairs", (
            ("i8", "meta", (("int", "n_features_in_"), ("int", "n_support_"), ("int", "n_iter_"),
                            ("int", "_hp_max_iter"), ("int", "_hp_nochange_steps"), ("int", "_hp_degree"))),
        )),
        # Only for poly and sigmoid, so every linear and rbf file keeps its bytes.
        ("when_in", "_hp_kernel", ("poly", "sigmoid"), (
            ("f8", "coef0", (("float", "_hp_coef0"),)),
        )),
        # probability=True only: the per-pair sigmoid (A, B), binary64.
        ("when_set", "_prob_ab", (
            ("rows", "prob_ab", "<f8", "_prob_ab", (("float", ("_", 0)), ("float", ("_", 1)))),
        )),
        # Multiclass only: the one-vs-one machines, concatenated in pair
        # order ([i, j, n_support, n_iter] per pair), from which `load`
        # rebuilds the public attributes exactly as `fit` built them.
        ("when_set", "_pairs", (
            ("i8", "meta", (("int", "n_features_in_"), ("dim", "support_", 0), ("sum", "_pairs", "n_iter"),
                            ("int", "_hp_max_iter"), ("int", "_hp_nochange_steps"), ("int", "_hp_degree"))),
            ("f8", "labels", (("const", 0.0), ("const", 1.0))),
            ("str", "shape", "decision_function_shape"),
            ("i8", "break_ties", (("bool", "_hp_break_ties"),)),
            ("rows", "pair_meta", "<i8", "_pairs", (("int", ("_", "i")), ("int", ("_", "j")),
                                                     ("dim", ("_", "dual"), 1), ("int", ("_", "n_iter")))),
            ("rows", "pair_b", "<f4", "_pairs", (("float", ("_", "b")),)),
            ("concat", "pair_support", "<i4", "_pairs", "support"),
            ("concat", "pair_dual", "<f4", "_pairs", "dual"),
            ("concat", "pair_sv", "<f4", "_pairs", "sv"),
        )),
    ),
    host_archive_svm_kernels=dict(_SVC_KERNELS),
    host_archive_svm_refused=dict(_SVC_REFUSED_KERNELS),
    host_archive_svr_refused=dict(_SVR_REFUSED_KERNELS),
)
_native_archive_bridge.register(
    SVR, _SVR_FORMAT, (
        _native_archive_bridge.fitted("dual_coef_"),
        *_native_archive_bridge.head(_SVR_FORMAT),
        ("str", "kernel", "_hp_kernel"),
        ("str", "gamma", "_hp_gamma"),
        ("array", "dual_coef", "dual_coef_"),
        ("array", "support_vectors", "support_vectors_"),
        ("array", "support", "support_"),
        ("array", "intercept", "intercept_"),
        ("f8", "hyper", (("float", "_hp_C"), ("float", "_hp_epsilon"), ("float", "_hp_tol"),
                         ("float", "_hp_cache_size"), ("float", "_gamma"))),
        ("i8", "meta", (("int", "n_features_in_"), ("int", "n_support_"), ("int", "n_iter_"),
                        ("int", "_hp_max_iter"), ("int", "_hp_nochange_steps"))),
        # Only for poly and sigmoid, so every linear and rbf file keeps its bytes.
        ("when_in", "_hp_kernel", ("poly", "sigmoid"), (
            ("f8", "kernel_params", (("float", "_hp_degree"), ("float", "_hp_coef0"))),
        )),
    ),
)

# The compiled SVM family owner (bindings/svc_native.mojo) reads these fixed
# tables: the base binding's host-helper resolver (`_buffer._native`, the
# entries `gather_i64`, `threshold_labels_i64`, ... the Python called) and
# the number ABCs of the label and class-weight checks.
_native_family_bridge.tables(family_svc_native=_native, family_svc_real=numbers.Real,
                             family_svc_integral=numbers.Integral, family_svc_kernels=dict(_SVC_KERNELS),
                             family_svc_refused=dict(_SVC_REFUSED_KERNELS),
                             family_svr_refused=dict(_SVR_REFUSED_KERNELS))

# Import connection only; each native public operation admits the live registry.
_native_family_bridge.connect_bindings("_mojolearn_svm")
