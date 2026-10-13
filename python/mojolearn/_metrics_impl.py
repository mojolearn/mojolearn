# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Scoring functions: classification, regression, ranking and clustering metrics.

``mojolearn.metrics`` is shaped like ``sklearn.metrics``: plain functions, no
estimator class. Function and argument names follow scikit-learn, and the
work runs on the GPU. Where cuML has the same function, its defaults and
semantics are followed, and each function states where scikit-learn and cuML
differ. Three differences matter most:

* ``entropy`` and ``mutual_info_score`` are in nats.
* ``r2_score`` returns scikit-learn's finite values for constant targets
  (``force_finite=True``) by default.
* ``kl_divergence`` does not normalize ``P`` and ``Q``, as in cuML.

A parameter a function cannot honor raises an error naming it; nothing is
silently accepted and ignored. Inputs are checked here: non-finite values are
refused where the docstring says so.

Functions that take ``numeric_mode`` accept ``'identical'`` (the same bits
on Apple, NVIDIA and AMD GPUs) or ``'fast'`` (Apple only); None
uses the library default. Other functions run in the library default mode.

Examples
--------
>>> import numpy as np
>>> from mojolearn import metrics
>>> rng = np.random.default_rng(0)
>>> y_true = rng.integers(0, 2, 1000).astype(np.int32)
>>> y_pred = rng.integers(0, 2, 1000).astype(np.int32)
>>> acc = metrics.accuracy_score(y_true, y_pred)
"""

# Maintainer notes (moved from the module docstring):
# HISTORICAL QUALIFICATION (excludes new regression-error and A2
# classification kernels). The original metrics kernels were bit-identical
# across Apple M4, NVIDIA H100 and AMD MI325X at leg 11
# (`archive/evidence/E3_RESULTS.md` round 11, commit 144aa5b,
# `tools/e3_round_judge.sh` section 7, 34 card stages, both boxes) under
# MOJOLEARN_NUMERIC_MODE=identical. The card has since grown to 61 stages and
# the three-vendor leg on the grown card is owed (`metrics/README.md` Status).
# The FAST arm makes no cross-vendor claim; FAST cards differ between vendors.
# Evidence caveat: the two box cards are committed and byte-identical
# (`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/lanes/metrics.identical.card`
# and its `..._172650-mojolearn-e2-amd` sibling, commit 144aa5b, 34 stages).
# The Apple reference directory that round was judged against is not in the
# repository (`archive/evidence/E3_RESULTS.md:253` names it `<mac 123452>`), so
# NVIDIA <-> AMD is reproducible from committed artifacts and the Apple
# comparisons rest on the recorded judge output.
# Input validation lives here deliberately: the Mojo entries do not scan their
# device inputs (`metrics/README.md` HAND-OFF ask 3), so these functions do the
# `check_array` work (finite X for silhouette and trustworthiness, finite
# y/y_pred for r2_score, finite non-negative P/Q for kl_divergence). The
# kernel-level answers for out-of-contract values (a NaN reaching `sil_op` as
# +0.0 through DEVIATION 656, r2_score's canonical NaN 0x7fc00000 through
# DEVIATION 657) are therefore not reachable from Python; they are gated in
# `metrics/checks/regression_metrics_check.mojo` and `silhouette_check.mojo`.
# cuML defaults come from the pinned v26.08.00 checkout.

from . import _portable_math as math
import numbers
from ._buffer import _materialize, _native, materialize_f32_lists
from ._labels import is_bool, flatten_labels, flat_view
from ._arrays import _addr, _addr_ro

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, as_i64_c, empty
from ._labels import encode_labels, sorted_classes
from .linear_model import (
    _buffer_format, _dtype_name, _flatten, _is_integer_labels, _shape_of,
)

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "log_loss",
    "roc_auc_score",
    "precision_recall_curve",
    "adjusted_rand_score",
    "completeness_score",
    "entropy",
    "fowlkes_mallows_score",
    "homogeneity_completeness_v_measure",
    "homogeneity_score",
    "kl_divergence",
    "mutual_info_score",
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "rand_score",
    "silhouette_samples",
    "silhouette_score",
    "trustworthiness",
    "v_measure_score",
    # the metrics lane (lane/metrics, 2026-09-27; python/mojolearn/_expansion_metrics.py)
    "balanced_accuracy_score",
    "class_likelihood_ratios",
    "classification_report",
    "cohen_kappa_score",
    "fbeta_score",
    "hamming_loss",
    "jaccard_score",
    "matthews_corrcoef",
    "multilabel_confusion_matrix",
    "precision_recall_fscore_support",
    "zero_one_loss",
    "d2_absolute_error_score",
    "d2_pinball_score",
    "d2_tweedie_score",
    "explained_variance_score",
    "max_error",
    "mean_absolute_percentage_error",
    "mean_gamma_deviance",
    "mean_pinball_loss",
    "mean_poisson_deviance",
    "mean_squared_log_error",
    "mean_tweedie_deviance",
    "median_absolute_error",
    "root_mean_squared_log_error",
    "auc",
    "average_precision_score",
    "brier_score_loss",
    "coverage_error",
    "d2_brier_score",
    "d2_log_loss_score",
    "dcg_score",
    "det_curve",
    "hinge_loss",
    "label_ranking_average_precision_score",
    "label_ranking_loss",
    "ndcg_score",
    "roc_curve",
    "top_k_accuracy_score",
    "adjusted_mutual_info_score",
    "calinski_harabasz_score",
    "contingency_matrix",
    "davies_bouldin_score",
    "normalized_mutual_info_score",
    "pair_confusion_matrix",
]


# ---------------------------------------------------------------------------
# Loading the binding, mode-aware.
# ---------------------------------------------------------------------------
# All metric calls share the estimator loader. Resolve the current default at
# call time; an explicit mode selects its own compiled artifact without changing
# the process default. The backend checks vendor provenance and never falls back.
def _get_binding(numeric_mode=None):
    if numeric_mode is not None and (
        not isinstance(numeric_mode, str)
        or numeric_mode.strip().lower() not in ("fast", "identical")
    ):
        raise ValueError("numeric_mode must be 'fast' or 'identical'")
    expected = (numeric_mode or _backend.default_mode()).strip().lower()
    mod = _backend.binding("_mojolearn_metrics", expected)
    read_mode = getattr(mod, "metrics_numeric_mode", None)
    if read_mode is None:
        read_mode = getattr(mod, "umap_numeric_mode", None)
    if read_mode is not None:
        actual = {0: "fast", 1: "identical"}.get(read_mode())
        if actual != expected:
            raise RuntimeError(
                f"mojolearn metrics: requested {expected}, binary reports {actual}; rebuild it"
            )
    return mod


# ---------------------------------------------------------------------------
# Array handling. `_buffer.py` owns the 2-D float32 contract and is used for
# X; these are its 1-D siblings, under the same rule: the returned Array MUST
# be kept alive in a local across the Mojo call, because the Mojo side takes
# a raw address, borrows it, and holds nothing after the call returns.
# NumPy-free since DEVIATION 2377: the shape and dtype questions are asked
# of the caller's object through `_shape_of` / the buffer protocol BEFORE
# it is converted, so every message below keeps its wording.
# ---------------------------------------------------------------------------


def _vector_shape(x, name):
    """`(n, column)`: the length of a 1-D input, or of an `(n, 1)` column,
    which is accepted and flattened as before; every other shape refused."""
    shape = _shape_of(x)
    column = len(shape) == 2 and shape[1] == 1
    if len(shape) != 1 and not column:
        raise ValueError(
            f"mojolearn metrics: {name} must be 1-D, got shape {shape}"
        )
    if shape[0] == 0:
        raise ValueError(f"mojolearn metrics: {name} is empty")
    return shape, column


def _is_float64(x):
    """`numpy.asarray(x).dtype == numpy.float64`: a buffer of format 'd', or a
    list/tuple with a Python float among its leaves (NumPy made float64 of
    that too)."""
    fmt = _buffer_format(x)
    if fmt is not None:
        return fmt == "d"
    return any(type(v) is float for v in _flatten(x))  # glue: type probe of a list argument


def _as_i32_1d(x, name):
    shape, column = _vector_shape(x, name)
    if not _is_integer_labels(x):
        raise ValueError(
            f"mojolearn metrics: {name} must be an integer label array, got "
            f"{_dtype_name(x)}; the ported kernels take int32 labels"
        )
    a, _ = as_i64_c(x, ndim=len(shape), name=name)
    if column:
        a = a.reshape((shape[0],))
    if a.min() < -(1 << 31) or a.max() > (1 << 31) - 1:
        raise ValueError(
            f"mojolearn metrics: {name} does not fit in int32; the implemented "
            "kernels are the int32 instantiation (the int64 overload of "
            "adjusted_rand_index is the same code at a wider type and is "
            "not instantiated, metrics/NOT_IMPLEMENTED.tsv)"
        )
    # int64 -> int32 through `Array.astype`: the core helper `cast_elements`
    # (DEVIATION 3100), else `array.array`'s item loop, which has no Python
    # loop BODY but makes one Python int per element (69 ms per 1,000,000
    # rows on an EPYC 7713 before the helper, 1.9 ms with it). Exact after
    # the range check.
    return a.astype("<i4")


def _as_f32_1d(x, name, *, require_finite=True):
    shape, column = _vector_shape(x, name)
    if _is_float64(x):
        # Named rather than silent: cuML has a float64 overload of r2_score,
        # kl_divergence and silhouette_score and this implementation does not, because
        # Apple's GPU has no float64 (mojolearn-hardware-limits). The cast is
        # the caller's to make, so the precision they run at is the precision
        # they asked for.
        raise TypeError(
            f"mojolearn metrics: {name} is float64; the float64 overloads of "
            "r2_score, kl_divergence and silhouette_score are NOT implemented "
            "(no float64 on this GPU). Cast to float32 yourself so the "
            "precision you run at is the one you chose."
        )
    out, _ = as_f32_c(x, ndim=len(shape), name=name)
    if column:
        out = out.reshape((shape[0],))
    if require_finite and not all_finite(out):
        raise ValueError(
            f"mojolearn metrics: {name} contains NaN or infinity; refused "
            "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
            "reach a kernel whose answer for it is a washed sentinel"
        )
    return out


def _sample_weight_f32(sample_weight, n, caller):
    """Weights for the pinned-sum weighted arms (`metrics/impl/weighted_scores.mojo`):
    one finite, non-negative Float32 per sample with a positive total, the
    rule `GradientBoosting.fit` applies to its own sample_weight. Cast to
    Float32 like the estimators' weights. scikit-learn's `np.average` also
    accepts negative weights; those are refused here by name."""
    shape = _shape_of(sample_weight)
    if len(shape) != 1:
        raise ValueError(
            f"mojolearn {caller}: sample_weight must be 1-D, got shape {shape}"
        )
    if shape[0] != n:
        raise ValueError(
            f"mojolearn {caller}: sample_weight has {shape[0]} entries for {n} samples"
        )
    w, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
    # `Array.min` / `Array.max` answer as `min()` / `max()` over the storage
    # view did, without one Python float per weight (DEVIATION 3101).
    if not all_finite(w) or w.min() < 0:
        raise ValueError(
            f"mojolearn {caller}: sample_weight must have finite nonnegative entries"
        )
    if not w.max() > 0:
        raise ValueError(
            f"mojolearn {caller}: sample_weight must have positive total weight"
        )
    return w


def _pair_1d(a, b, name_a, name_b, loader):
    x = loader(a, name_a)
    y = loader(b, name_b)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"mojolearn metrics: {name_a} has {x.shape[0]} entries and "
            f"{name_b} has {y.shape[0]}; they must be the same length"
        )
    return x, y


def _prepare_cluster_labels(labels_true, labels_pred):
    """cuML 26.08's `prepare_cluster_metric_inputs`
    (`python/cuml/cuml/metrics/cluster/utils.py`), same behavior.

    Both arrays are remapped onto the CONTIGUOUS range `[0, n_classes - 1]`
    over the UNION of their distinct labels, and the range handed to the
    kernel is `(0, n_classes - 1)`. That is what keeps the contingency
    matrix `n_classes` square instead of `(max - min + 1)` square.

    NOTE FOR ANYONE READING `metrics/README.md`'s HAND-OFF: that paragraph
    says the Python side passes `min` and `max` of the two arrays "exactly
    as cuML's .pyx files do". That is true of `entropy` and false of these
    four at the 26.08 pin, where the remap above was introduced. The remap
    is what runs here.
    """
    yt = _as_i32_1d(labels_true, "labels_true")
    yp = _as_i32_1d(labels_pred, "labels_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"mojolearn metrics: labels_true has {yt.shape[0]} entries and "
            f"labels_pred has {yp.shape[0]}; they must be the same length"
        )
    # `np.unique` of the union, then `np.searchsorted`: the package-wide
    # order rule (`_labels.sorted_classes`, DEVIATION 2340) over the
    # concatenation and a dict lookup, O(rows) label loops (DEVIATION 2377).
    fast = _native_union_codes(yt, yp)
    if fast is None:
        return yt, yp, 0, 0, -1   # both sides empty: no classes
    return fast[0], fast[1], int(yt.shape[0]), 0, fast[2] - 1


#: DEVIATION 3107 (lane/python-hotpath, 2026-09-17). Below this many labels
#: the Python label routines were the only path; lane apple-fast-py2mojo-core
#: (2026-10-03) sends every non-empty label buffer to the native encoder.
_NATIVE_MIN_LABELS = 1


def _native_union_codes(yt, yp):
    """`(codes_true, codes_pred, n_classes)` over the sorted union of two
    int32 label Arrays, or None when both are empty (lane pyglue-sweep: one
    empty side is encoded too; the Python dict route is gone).

    lane apple-fast-py2mojo-core (2026-10-03): the two arrays are laid end
    to end and encoded ONCE by the base binding's `unique_inverse` (the
    device sort, flag/scan compaction and gather on a GPU install, the core
    host twin on a CPU-only one): its sorted distinct values ARE the sorted
    union and its codes, split at len(yt), are each side's codes on it. No
    class cap (the 4096-class encoder and the Python merge of the two class
    lists are gone, and with them the dict fallback for more classes)."""
    import ctypes
    from ._labels import unique_inverse

    nt, npred = int(yt.size), int(yp.size)
    if nt + npred == 0:
        return None
    yt, yp = yt._as_c(), yp._as_c()
    both = empty((nt + npred,), "<i4")
    ctypes.memmove(_addr(both), _addr_ro(yt), 4 * nt)
    ctypes.memmove(_addr(both) + 4 * nt, _addr_ro(yp), 4 * npred)
    classes, codes = unique_inverse(both)
    return codes[0:nt], codes[nt:nt + npred], int(classes.size)


# ===========================================================================
# Group A: the label metrics
# ===========================================================================


# Maintainer notes: backed by `ML::Metrics::accuracy_score_py` (cuML
# `accuracy_score.cu`): one fused subtract-and-count with an INTEGER atomic,
# exact and order-free on every vendor; the returned float32 is
# `count * (1 / n)`. The int32 instantiation is the one cuML's Python passes.
# sample_weight: pinned-sum path (metrics/impl/weighted_scores.mojo), the
# weights of agreeing samples and all weights folded by the PINNED_SUM_W tree,
# then one division; RAFT has no weighted arm and cuML weights in cupy, so this
# arm follows scikit-learn. normalize=False goes to the x_metrics binding.
# `n == 0` is refused inside the kernel (`count / n` is 0/0 in RAFT and a NaN
# may not reach a recorded value).
def accuracy_score(
    y_true, y_pred, *, normalize=True, sample_weight=None, numeric_mode=None
):
    """Accuracy: the fraction (or number) of positions where two label arrays agree.

    Follows scikit-learn's ``accuracy_score``; the unweighted fraction uses
    cuML's kernel. The count of matches is exact on every device.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Integer labels of the same length that fit in int32. Larger labels
        raise ``ValueError``. With ``normalize=False``, string labels are
        accepted too.
    normalize : bool, default True
        True returns the fraction of matches. False returns the number of
        matches (an int), or with ``sample_weight`` their summed weight.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights, as in scikit-learn's
        ``np.average(y_true == y_pred, weights=sample_weight)``, computed in
        float32. Weights must be finite, non-negative and have a positive
        total; negative weights, which scikit-learn accepts, raise
        ``ValueError``.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    score : float or int

    Raises
    ------
    ValueError
        For empty input or arrays of different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> metrics.accuracy_score(np.array([0, 1, 1, 0], dtype=np.int32),
    ...                        np.array([0, 1, 0, 0], dtype=np.int32))
    0.75
    """
    if not is_bool(normalize):
        raise ValueError("normalize must be a bool")
    if not normalize:
        # The COUNT (lane/metrics, 2026-09-27): exact integer matches, or the
        # Float32 PairSum of the matching rows' weights, from the x_metrics
        # binding (strings or integers, like scikit-learn). The fraction
        # keeps its original kernel and bits.
        from ._expansion_metrics import accuracy_count
        return accuracy_count(y_true, y_pred, sample_weight, numeric_mode)
    yt, yp = _pair_1d(y_true, y_pred, "y_true", "y_pred", _as_i32_1d)
    if sample_weight is not None:
        w = _sample_weight_f32(sample_weight, int(yt.shape[0]), "accuracy_score")
        # ORDER MATCHES bindings/_mojolearn_metrics.mojo::accuracy_score_weighted_binding.
        # n
        return float(
            _get_binding(numeric_mode).accuracy_score_weighted(
                _addr_ro(yt), _addr_ro(yp), _addr_ro(w), [int(yt.shape[0])]
            )
        )
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::accuracy_score_binding.
    # n
    return float(
        _get_binding(numeric_mode).accuracy_score(
            _addr_ro(yt), _addr_ro(yp), [int(yt.shape[0])]
        )
    )


# Maintainer notes: backed by `ML::Metrics::rand_index` (cuML
# `rand_index.cu`, DEVIATION 652: per-block partials plus a host Int64 sum,
# because Apple has no 64-bit float atomic). Raw labels, no label range on
# this entry. cuML's Python exports only adjusted_rand_score; the unadjusted
# index has a C++ entry and is exposed under scikit-learn's name.
def rand_score(labels_true, labels_pred):
    """Rand index: the fraction of sample pairs on which two clusterings agree.

    Follows scikit-learn's ``rand_score``. Labels are used as given; they
    do not need to be contiguous.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Integer cluster labels of the same length that fit in int32.

    Returns
    -------
    score : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> score = metrics.rand_score(np.array([0, 0, 1, 1], dtype=np.int32),
    ...                            np.array([1, 1, 0, 0], dtype=np.int32))
    """
    yt, yp = _pair_1d(
        labels_true, labels_pred, "labels_true", "labels_pred", _as_i32_1d
    )
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::rand_score_binding.
    # n
    return float(
        _get_binding().rand_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [int(yt.shape[0])]
        )
    )


# Maintainer notes: backed by `ML::Metrics::adjusted_rand_index`, the int32
# instantiation (the int64 overload is the same code at a wider label type and
# is not instantiated). Builds its own contingency matrix over its own label
# range from raw labels, as cuML does. A difference of large numbers over a
# difference of large numbers: the three exact integers are card stages
# (`metrics.ari.pair_sums`), and the host epilogue is float64
# divide/multiply/subtract only, identity-safe in both modes.
def adjusted_rand_score(labels_true, labels_pred):
    """Rand index adjusted for chance.

    Follows scikit-learn's and cuML's ``adjusted_rand_score``: 1.0 for
    identical clusterings (up to relabeling), about 0.0 for random ones.
    Labels are used as given.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Integer cluster labels of the same length that fit in int32.

    Returns
    -------
    score : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> rng = np.random.default_rng(0)
    >>> a = rng.integers(0, 4, 1000).astype(np.int32)
    >>> score = metrics.adjusted_rand_score(a, a)
    """
    yt, yp = _pair_1d(
        labels_true, labels_pred, "labels_true", "labels_pred", _as_i32_1d
    )
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::adjusted_rand_score_binding.
    # n
    return float(
        _get_binding().adjusted_rand_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [int(yt.shape[0])]
        )
    )


# Maintainer notes: backed by `ML::Metrics::entropy` (cuML `entropy.cu`, RAFT
# `entropy.cuh`, DEVIATIONS 650 and 651). `base` uses cuML's `entropy.pyx:72-74`
# spelling exactly, `math.log(math.exp(S), base)`, rather than the algebraically
# equal `S / math.log(base)`. The min..max histogram is cuML's own behavior
# (`entropy.pyx:61-62` passes min and max with no remapping, unlike the four
# cluster metrics). Under identical the float epilogue is Float32 through
# `identical_log` and differs from FAST's Float64 in the seventh digit by
# design (DEVIATION 651): `checks/numerics.mojo` has no portable Float64 log
# yet; `metrics/README.md` hand-off ask 1.
def entropy(clustering, *, base=None):
    """Entropy of a labeling, in nats unless ``base`` is given.

    Follows cuML's ``cuml.metrics.cluster.entropy``; scikit-learn has no
    public equivalent.

    The label histogram spans ``max(labels) - min(labels) + 1`` bins, not the
    number of distinct labels, as in cuML. Sparse, large-valued labels
    therefore allocate that many bins; encode them as ``0..k-1`` first if
    that matters.

    In the ``'identical'`` numeric mode the final logarithms run in float32,
    so the result can differ from the other modes in about the seventh
    significant digit.

    Parameters
    ----------
    clustering : array-like of shape (n_samples,)
        Integer labels that fit in int32.
    base : float or None, default None
        Logarithm base. None means nats.

    Returns
    -------
    entropy : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> h = metrics.entropy(np.array([0, 0, 1, 1], dtype=np.int32), base=2)
    """
    lab = _as_i32_1d(clustering, "clustering")
    lower = int(lab.min())
    upper = int(lab.max())
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::entropy_binding.
    # n, lower_class_range, upper_class_range
    value = float(
        _get_binding().entropy(
            addr_ro(lab, name="lab"), [int(lab.shape[0]), lower, upper]
        )
    )
    if base is not None:
        value = math.log(math.exp(value), base)
    return value


# Maintainer notes: backed by `ML::Metrics::mutual_info_score` (DEVIATIONS
# 650 and 651): the contingency matrix is built with INTEGER atomics on the
# device, exact and order-free everywhere, and the `c (log(nc) - log(ab))`
# epilogue runs serially and ascending on the host. The labels are remapped
# as cuML 26.08's `prepare_cluster_metric_inputs` does.
def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    """Mutual information between two clusterings, in nats.

    Follows scikit-learn's and cuML's ``mutual_info_score``, in nats as cuML
    computes it. Both label arrays are first mapped onto ``0..k-1`` over the
    union of their distinct labels, as cuML does. The contingency counts
    are exact on every device.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Integer cluster labels of the same length that fit in int32.
    contingency : None, default None
        A precomputed contingency matrix is not supported; passing one
        raises ``NotImplementedError``.

    Returns
    -------
    mi : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> mi = metrics.mutual_info_score(np.array([0, 0, 1, 1], dtype=np.int32),
    ...                                np.array([0, 0, 1, 2], dtype=np.int32))
    """
    if contingency is not None:
        raise NotImplementedError(
            "mojolearn mutual_info_score: contingency= is refused; the "
            "implemented entry builds the contingency matrix on the device "
            "(metrics/impl/stats/detail/contingency_matrix.mojo) and has "
            "no arm that consumes a precomputed one"
        )
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::mutual_info_score_binding.
    # n, lower_class_range, upper_class_range
    return float(
        _get_binding().mutual_info_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [n, lower, upper]
        )
    )


# Maintainer notes: reference `sklearn/metrics/cluster/_supervised.py`. The
# contingency matrix is built with INTEGER atomics on the device (the
# mutual_info_score kernel); tk = sum_ij c_ij^2 - n, pk = sum_j colsum^2 - n,
# qk = sum_i rowsum^2 - n are exact Int64 sums on the host and the score is
# Float64.
def fowlkes_mallows_score(labels_true, labels_pred, *, sparse="deprecated"):
    """Fowlkes-Mallows index between two clusterings.

    Follows scikit-learn's ``fowlkes_mallows_score`` (cuML has no such
    metric): the geometric mean of pairwise precision and recall,
    ``sqrt(tk / pk) * sqrt(tk / qk)``. Returns 0.0 when ``tk == 0`` (every
    cluster a singleton, a single sample, or no samples), as scikit-learn
    does. Labels are mapped onto ``0..k-1`` over their union first.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Integer cluster labels of the same length that fit in int32.
    sparse : default 'deprecated'
        Deprecated and ignored in scikit-learn; any explicit value raises
        ``NotImplementedError``.

    Returns
    -------
    score : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> score = metrics.fowlkes_mallows_score(np.array([0, 0, 1, 1], dtype=np.int32),
    ...                                       np.array([0, 0, 1, 2], dtype=np.int32))
    """
    if sparse != "deprecated":
        raise NotImplementedError(
            "mojolearn fowlkes_mallows_score: sparse= is refused; scikit-learn "
            "deprecated and ignores it, and this surface has no sparse arm"
        )
    try:
        empty = len(labels_true) == 0 and len(labels_pred) == 0
    except TypeError:
        empty = False
    if empty:
        # scikit-learn's check_clusterings allows no samples
        # (ensure_min_samples=0); the empty contingency matrix gives
        # tk == 0 and the score 0.0. The label loaders refuse empty input,
        # so the case is answered here, before them.
        return 0.0
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::fowlkes_mallows_score_binding.
    # n, lower_class_range, upper_class_range
    return float(
        _get_binding().fowlkes_mallows_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [n, lower, upper]
        )
    )


# Maintainer note: backed by `ML::Metrics::homogeneity_score`.
def homogeneity_score(labels_true, labels_pred):
    """Homogeneity: whether each cluster holds members of a single class.

    ``MI(true, pred) / H(true)``, in ``[0, 1]``. Follows scikit-learn's and
    cuML's ``homogeneity_score``. Labels are mapped onto ``0..k-1`` over their
    union first, as cuML does.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Integer labels of the same length that fit in int32.

    Returns
    -------
    score : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> h = metrics.homogeneity_score(np.array([0, 0, 1, 1], dtype=np.int32),
    ...                               np.array([0, 0, 1, 2], dtype=np.int32))
    """
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::homogeneity_score_binding.
    # n, lower_class_range, upper_class_range
    return float(
        _get_binding().homogeneity_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [n, lower, upper]
        )
    )


# Maintainer notes: backed by `ML::Metrics::completeness_score`, which RAFT
# computes as homogeneity_score with the two arrays swapped. The swapped call
# folds the TRANSPOSED contingency matrix, so the host's serial ascending walk
# visits the cells in a different sequence: the same quantity in exact
# arithmetic, not necessarily the same float32 bits. Implemented as theirs;
# both folds are stages on the metrics card.
def completeness_score(labels_true, labels_pred):
    """Completeness: whether all members of a class are in the same cluster.

    ``MI(pred, true) / H(pred)``, in ``[0, 1]``. Follows scikit-learn's and
    cuML's ``completeness_score``. Labels are mapped onto ``0..k-1`` over
    their union first, as cuML does.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Integer labels of the same length that fit in int32.

    Returns
    -------
    score : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> c = metrics.completeness_score(np.array([0, 0, 1, 1], dtype=np.int32),
    ...                                np.array([0, 0, 1, 2], dtype=np.int32))
    """
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::completeness_score_binding.
    # n, lower_class_range, upper_class_range
    return float(
        _get_binding().completeness_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [n, lower, upper]
        )
    )


# Maintainer note: backed by `ML::Metrics::v_measure` (cuML `v_measure.cu`).
def v_measure_score(labels_true, labels_pred, *, beta=1.0):
    """V-measure: the weighted harmonic mean of homogeneity and completeness.

    ``(1 + beta) * h * c / (beta * h + c)``. Follows scikit-learn's and
    cuML's ``v_measure_score``. Labels are mapped onto ``0..k-1`` over their
    union first.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Integer labels of the same length that fit in int32.
    beta : float, default 1.0
        Above 1 weights completeness more, below 1 weights homogeneity more.

    Returns
    -------
    score : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> v = metrics.v_measure_score(np.array([0, 0, 1, 1], dtype=np.int32),
    ...                             np.array([0, 0, 1, 2], dtype=np.int32))
    """
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::v_measure_score_binding.
    # n, lower_class_range, upper_class_range, beta
    return float(
        _get_binding().v_measure_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [n, lower, upper, float(beta)]
        )
    )


def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0):
    """Homogeneity, completeness and V-measure in one call.

    Returns ``(homogeneity_score, completeness_score, v_measure_score)``, as
    scikit-learn's function of the same name. Each value is computed by its
    own call, so this costs about three separate metric calls; call the one
    you need if speed matters.

    Parameters
    ----------
    labels_true, labels_pred : array-like of shape (n_samples,)
        Integer labels of the same length that fit in int32.
    beta : float, default 1.0
        Passed to ``v_measure_score``.

    Returns
    -------
    homogeneity, completeness, v_measure : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> h, c, v = metrics.homogeneity_completeness_v_measure(
    ...     np.array([0, 0, 1, 1], dtype=np.int32), np.array([0, 0, 1, 2], dtype=np.int32))
    """
    return (
        homogeneity_score(labels_true, labels_pred),
        completeness_score(labels_true, labels_pred),
        v_measure_score(labels_true, labels_pred, beta=beta),
    )


# ===========================================================================
# Group B: r2 and KL divergence
# ===========================================================================


# Maintainer notes: the 1-D default call is backed by `ML::Metrics::r2_score_py`,
# the float overload (DEVIATIONS 653 and 657). The three sums (y_bar, sse,
# ssto) are folded by a PINNED_SUM_W = 256 slab tree whose width is a constant
# of the source, not of the launch, gated launch-invariant across two block
# sizes and two grid shapes. float64 is refused: cuML has a double overload and
# this implementation does not (no float64 on Apple GPUs). sample_weight:
# pinned-sum path (metrics/impl/weighted_scores.mojo; RAFT's r2_score has no
# weighted arm), `sum w (y - y_pred)^2` over `sum w (y - y_avg)^2` with the same
# force_finite guards. force_finite=True: ssto == 0 returns 1.0 when sse == 0
# and 0.0 otherwise. multioutput, 2-D targets and force_finite=False go to the
# x_metrics binding, whose NaN / -inf are chosen by value in the binary64
# epilogue (DEVIATION 6103). An overflow-scale y (|y - y_hat| above ~1.8e19)
# returns the one canonical NaN payload 0x7fc00000 on every vendor.
def r2_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    force_finite=True,
    numeric_mode=None,
):
    """Coefficient of determination (R^2), computed in float32.

    Follows scikit-learn's ``r2_score``. When the targets are constant
    (total sum of squares 0), the default ``force_finite=True`` returns 1.0
    for a perfect prediction and 0.0 otherwise, as scikit-learn and cuML do.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Finite float32 targets and predictions of the same shape. float64
        arrays raise ``TypeError`` (for 1-D input, so do Python lists of
        floats): cast to float32 first. NaN or infinity raises
        ``ValueError``.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights, as in scikit-learn. They must be finite,
        non-negative and have a positive total.
    multioutput : {'uniform_average', 'raw_values', 'variance_weighted'} or array-like, default 'uniform_average'
        How to combine the scores of several outputs, as in scikit-learn.
    force_finite : bool, default True
        False returns scikit-learn's raw values for constant targets
        (``nan`` or ``-inf``).
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    score : float or Array
        A float, or one float64 value per output for ``'raw_values'``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> rng = np.random.default_rng(0)
    >>> y = rng.random(1000, dtype=np.float32)
    >>> pred = y + rng.normal(0, 0.1, 1000).astype(np.float32)
    >>> r2 = metrics.r2_score(y, pred)
    """
    if (not isinstance(multioutput, str) or multioutput != "uniform_average" or not force_finite
            or len(_shape_of(y_true)) == 2 and _shape_of(y_true)[1] != 1):
        # lane/metrics: multioutput ('raw_values', 'variance_weighted', an
        # array of weights), 2-D targets and force_finite=False through the
        # x_metrics binding. force_finite=False's NaN / -inf are chosen BY
        # VALUE in the binary64 epilogue, never computed on a device, so no
        # vendor payload reaches the result (DEVIATION 6103). The 1-D default
        # call keeps its kernel and bits.
        from ._expansion_metrics import r2_score_options
        return r2_score_options(y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode)
    y, yh = _pair_1d(y_true, y_pred, "y_true", "y_pred", _as_f32_1d)
    if sample_weight is not None:
        w = _sample_weight_f32(sample_weight, int(y.shape[0]), "r2_score")
        # ORDER MATCHES bindings/_mojolearn_metrics.mojo::r2_score_weighted_binding.
        # n
        return float(
            _get_binding(numeric_mode).r2_score_weighted(
                _addr_ro(y), _addr_ro(yh), _addr_ro(w), [int(y.shape[0])]
            )
        )
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::r2_score_binding.
    # n
    return float(
        _get_binding(numeric_mode).r2_score(_addr_ro(y), _addr_ro(yh), [int(y.shape[0])])
    )


def _as_regression_f32_1d(x, name):
    # A plain Python list or tuple has no float32 spelling; it is cast once,
    # the same rounding `as_f32_c` applies (`materialize_f32_lists`). A
    # buffer keeps the float32-only contract (Sep 22 pip smoke: plain lists
    # were refused as "must have dtype float32").
    if _buffer_format(x) != "f" and not isinstance(x, (list, tuple)):
        raise TypeError(f"mojolearn regression metrics: {name} must have dtype float32; cast explicitly before scoring")
    a = materialize_f32_lists(x, "input")[0]
    if a.dtype != "<f4":
        raise TypeError(
            f"mojolearn regression metrics: {name} must have dtype float32, "
            f"got {a.dtype}; cast explicitly before scoring"
        )
    return _as_f32_1d(a, name)


def _regression_error(name, y_true, y_pred, sample_weight, multioutput, numeric_mode):
    if (sample_weight is not None or not isinstance(multioutput, str)
            or multioutput != "uniform_average" or len(_shape_of(y_true)) == 2 and _shape_of(y_true)[1] != 1):
        # lane/metrics: sample_weight, multioutput and 2-D targets through the
        # x_metrics binding; the 1-D unweighted call keeps its kernel and bits.
        from ._expansion_metrics import regression_error_options
        return regression_error_options(name, y_true, y_pred, sample_weight, multioutput, numeric_mode)
    yt, yp = _pair_1d(y_true, y_pred, "y_true", "y_pred", _as_regression_f32_1d)
    return float(getattr(_get_binding(numeric_mode), name)(
        _addr_ro(yt), _addr_ro(yp), [int(yt.size)]
    ))


# Maintainer notes: IDENTICAL pins the reduction schedule and flushes
# subnormal arithmetic. The 1-D unweighted call keeps its own kernel; weights,
# multioutput and 2-D targets go to the x_metrics binding
# (regression_error_options). The new error metrics had local Metal
# qualification; the cross-vendor leg was pending when this was written.
def mean_squared_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", numeric_mode=None
):
    """Mean squared error, computed in float32.

    Follows scikit-learn's ``mean_squared_error``. The residuals, their
    squares and the sum are float32, with no wider accumulator, so very
    large residuals can overflow to ``inf``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Finite, nonempty float32 arrays of the same shape. A float64 or other
        non-float32 buffer raises ``TypeError``: cast it first. Python lists
        are converted to float32. NaN or infinity raises ``ValueError``.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights, as in scikit-learn.
    multioutput : {'uniform_average', 'raw_values'} or array-like, default 'uniform_average'
        How to combine several outputs, as in scikit-learn.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    loss : float or Array
        A float, or one float64 value per output for ``'raw_values'``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> y = np.array([3.0, -0.5, 2.0, 7.0], dtype=np.float32)
    >>> pred = np.array([2.5, 0.0, 2.0, 8.0], dtype=np.float32)
    >>> metrics.mean_squared_error(y, pred)
    0.375
    """
    return _regression_error(
        "mean_squared_error", y_true, y_pred, sample_weight, multioutput, numeric_mode
    )


def mean_absolute_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", numeric_mode=None
):
    """Mean absolute error, computed in float32.

    Follows scikit-learn's ``mean_absolute_error``, with the same input and
    overflow behavior as ``mean_squared_error``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Finite, nonempty float32 arrays of the same shape. A float64 or other
        non-float32 buffer raises ``TypeError``: cast it first. Python lists
        are converted to float32. NaN or infinity raises ``ValueError``.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights, as in scikit-learn.
    multioutput : {'uniform_average', 'raw_values'} or array-like, default 'uniform_average'
        How to combine several outputs, as in scikit-learn.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    loss : float or Array
        A float, or one float64 value per output for ``'raw_values'``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> y = np.array([3.0, -0.5, 2.0, 7.0], dtype=np.float32)
    >>> pred = np.array([2.5, 0.0, 2.0, 8.0], dtype=np.float32)
    >>> metrics.mean_absolute_error(y, pred)
    0.5
    """
    return _regression_error(
        "mean_absolute_error", y_true, y_pred, sample_weight, multioutput, numeric_mode
    )


def root_mean_squared_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", numeric_mode=None
):
    """Root mean squared error, computed in float32.

    Follows scikit-learn's ``root_mean_squared_error``: the square root of
    ``mean_squared_error``, with the same input behavior. The squares can
    overflow to ``inf`` even when the true RMSE would fit in float32.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Finite, nonempty float32 arrays of the same shape. A float64 or other
        non-float32 buffer raises ``TypeError``: cast it first. Python lists
        are converted to float32. NaN or infinity raises ``ValueError``.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights, as in scikit-learn.
    multioutput : {'uniform_average', 'raw_values'} or array-like, default 'uniform_average'
        How to combine several outputs, as in scikit-learn.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    loss : float or Array
        A float, or one float64 value per output for ``'raw_values'``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> y = np.array([3.0, -0.5, 2.0, 7.0], dtype=np.float32)
    >>> pred = np.array([2.5, 0.0, 2.0, 8.0], dtype=np.float32)
    >>> rmse = metrics.root_mean_squared_error(y, pred)
    """
    return _regression_error(
        "root_mean_squared_error", y_true, y_pred, sample_weight, multioutput, numeric_mode
    )


# Maintainer notes: backed by `ML::Metrics::kl_divergence`, the float overload
# (DEVIATIONS 653 and 658), on the same pinned fold as r2_score. float64 is
# refused (no float64 on Apple GPUs). Negative entries are refused here, before
# the kernel, per `metrics/README.md` HAND-OFF ask 3. RAFT's kernel does not
# normalize either; p_i == 0 contributes zero (their branch).
def kl_divergence(P, Q):
    """Kullback-Leibler divergence ``sum p_i * (log p_i - log q_i)``, in float32.

    Follows cuML's ``cuml.metrics.kl_divergence``. ``P`` and ``Q`` are not
    normalized, as in cuML; divide each by its sum first if you want the
    divergence between distributions. A term with ``p_i == 0`` contributes
    zero, and ``q_i == 0`` with ``p_i > 0`` gives ``inf``.

    Parameters
    ----------
    P, Q : array-like of shape (n,)
        Finite, non-negative float32 arrays of the same length. float64
        input raises ``TypeError`` (cast to float32 first); a negative,
        NaN or infinite entry raises ``ValueError``.

    Returns
    -------
    divergence : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> P = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    >>> Q = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    >>> kl = metrics.kl_divergence(P, Q)
    """
    p, q = _pair_1d(P, Q, "P", "Q", _as_f32_1d)
    for name, arr in (("P", p), ("Q", q)):  # glue: validates the two named arguments
        if arr.min() < 0:
            raise ValueError(
                f"mojolearn kl_divergence: {name} has a negative entry; "
                "refused here (metrics/README.md HAND-OFF ask 3) rather "
                "than letting log of a negative reach the fold"
            )
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::kl_divergence_binding.
    # n
    return float(
        _get_binding().kl_divergence(
            addr_ro(p, name="p"), addr_ro(q, name="q"), [int(p.shape[0])]
        )
    )


# ===========================================================================
# Group C: silhouette
# ===========================================================================

_SILHOUETTE_METRICS = ("euclidean", "l2")


def _silhouette(X, labels, metric, chunksize, caller):
    if not isinstance(metric, str) or metric.lower() not in _SILHOUETTE_METRICS:
        raise NotImplementedError(
            f"mojolearn {caller}: metric={metric!r} is refused; only "
            "'euclidean' and 'l2' are implemented (they are the same "
            "DistanceType::L2SqrtUnexpanded, which is the arm cuML's "
            "silhouette_score.pyx dispatches by default). 'cityblock', "
            "'cosine', 'l1', 'manhattan' and 'sqeuclidean' are other "
            "distance kernels and are not in metrics/ (NOT_IMPLEMENTED.tsv)"
        )
    x, _copied = as_f32_c(X, ndim=2, name="X")
    if not all_finite(x):
        raise ValueError(
            f"mojolearn {caller}: X contains NaN or infinity; refused here "
            "(metrics/README.md HAND-OFF ask 3), because a NaN distance "
            "reaches sil_op as +0.0 (DEVIATION 656) rather than as an error"
        )
    lab = _as_i32_1d(labels, "labels")
    if lab.shape[0] != x.shape[0]:
        raise ValueError(
            f"mojolearn {caller}: labels has {lab.shape[0]} entries and X "
            f"has {x.shape[0]} rows"
        )
    # Same behavior as cuML's silhouette_score.pyx:99-101: monotonic labels via
    # cp.unique(..., return_inverse=True), and n_labels is how many distinct
    # labels there are. `_labels.encode_labels` gives both under the order
    # rule (DEVIATION 2340), natively over the int32 buffer (lane py-shared
    # retired DEVIATION 2377's O(rows) Python pass here; `sorted_classes`
    # stays the definition, `tests/test_labels_native.py`).
    unique_labels, mapped = encode_labels(lab)
    n_labels = len(unique_labels)
    chunk = 40000 if chunksize is None else int(chunksize)
    if chunk < 1:
        raise ValueError(
            f"mojolearn {caller}: chunksize must be at least 1, got {chunk}"
        )
    scores = empty((x.shape[0],), "<f4")
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::silhouette_binding.
    # n_rows, n_cols, n_labels, chunksize
    mean = float(
        _get_binding().silhouette(
            addr_ro(x, name="x"),
            addr_ro(mapped, name="mapped"),
            addr(scores, name="scores"),
            [int(x.shape[0]), int(x.shape[1]), n_labels, chunk],
        )
    )
    return mean, scores


# Maintainer notes: backed by `ML::Metrics::Batched::silhouette_score`, the
# float batched entry cuML's Python dispatches (DEVIATIONS 654 and 656). The
# per-cell distance runs one thread per cell through `identical_mul_add` /
# `ftz` / `identical_sqrt`, and the a/b accumulation is one fixed tree per
# (row, cluster); gated bitwise against a host model over 1031 samples and
# launch-invariant across eight launches. Labels are mapped as cuML's
# silhouette_score.pyx does; the kernel asserts 2 <= n_labels <= n_rows - 1.
# 'euclidean' and 'l2' are the same DistanceType::L2SqrtUnexpanded.
# chunksize is scheduling only: no distance tile is materialized, and chunk
# 1 / 7 / 40000 are gated to one byte pattern. sample_size / random_state are
# refused so no random draw happens where no check sees it.
def silhouette_score(
    X, labels, *, metric="euclidean", sample_size=None, random_state=None,
    chunksize=None,
):
    """Mean silhouette coefficient over all samples.

    Follows scikit-learn's and cuML's ``silhouette_score`` with Euclidean
    distance. Values lie in ``[-1, 1]``; higher means better-separated
    clusters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite data, converted to float32 row-major. NaN or infinity raises
        ``ValueError``.
    labels : array-like of shape (n_samples,)
        Integer cluster labels that fit in int32; they are mapped onto
        ``0..k-1``. There must be at least 2 and at most ``n_samples - 1``
        distinct labels.
    metric : {'euclidean', 'l2'}, default 'euclidean'
        Both mean Euclidean distance. Any other metric raises
        ``NotImplementedError``.
    sample_size, random_state : None, default None
        scikit-learn's subsampling is not supported; any other value raises
        ``NotImplementedError``. Subsample X yourself if you need it.
    chunksize : int or None, default None
        Accepted for compatibility with cuML (default 40000, must be at
        least 1). It does not change the result or the work done.

    Returns
    -------
    score : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> labels = rng.integers(0, 4, 1000).astype(np.int32)
    >>> s = metrics.silhouette_score(X, labels)
    """
    if sample_size is not None or random_state is not None:
        raise NotImplementedError(
            "mojolearn silhouette_score: sample_size / random_state are "
            "refused; subsample X yourself so the rows that were scored are "
            "the rows you chose (cuML's surface has no subsampling either)"
        )
    mean, _scores = _silhouette(X, labels, metric, chunksize, "silhouette_score")
    return mean


# Maintainer notes: the same kernel call as silhouette_score, which computes
# both. Values that come out +0.0: a singleton cluster, an exact tie a == b,
# a == b == 0 (all points identical), and a row whose distances were
# non-finite (DEVIATION 656). No fixture has produced a -0.0.
def silhouette_samples(X, labels, *, metric="euclidean", chunksize=None):
    """Silhouette coefficient of every sample.

    The per-sample values that ``silhouette_score`` averages; parameters,
    input rules and errors are those of ``silhouette_score``. A sample in a
    singleton cluster gets 0.0, as in scikit-learn.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite data, converted to float32 row-major.
    labels : array-like of shape (n_samples,)
        Integer cluster labels that fit in int32.
    metric : {'euclidean', 'l2'}, default 'euclidean'
        Both mean Euclidean distance.
    chunksize : int or None, default None
        Accepted for compatibility with cuML; does not change the result.

    Returns
    -------
    scores : Array of shape (n_samples,), float32

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> labels = rng.integers(0, 4, 1000).astype(np.int32)
    >>> s = metrics.silhouette_samples(X, labels)
    """
    _mean, scores = _silhouette(
        X, labels, metric, chunksize, "silhouette_samples"
    )
    return scores


# ===========================================================================
# Group D: trustworthiness
# ===========================================================================


# Maintainer notes: backed by `ML::Metrics::trustworthiness_score<float,
# L2SqrtUnexpanded>` (DEVIATION 655). The rank of each embedded neighbor is
# COUNTED with the stable sort's tie-break instead of sorting every row of the
# n x n distance matrix, and the closed form runs in float64 on the host;
# measured equal to scikit-learn 1.9's value on the lane's fixture.
# n_neighbors refusals: < 1; 2 * n_neighbors >= n_samples (cuML's
# trustworthiness.pyx:114 refuses the same, keeping the closed form's
# denominator positive); n_neighbors + 1 > 256 (the k+1 distances and ranks
# live in a threadgroup slab). metric: cuML's trustworthiness.pyx:86 refuses
# other names too. batch_size sizes nothing (ranks are counted, not sorted).
# Known soft spot: the embedded k-NN goes through this repository's one k-NN
# entry, which uses the EXPANDED L2 distance, while RAFT's run_knn uses the
# unexpanded one, so near-tied embedded neighbors can order differently in
# the last bit (measured 0 rows differing against a float64 host k-NN on the
# lane's fixture); a second neighbors/ arm for the unexpanded metric is the
# neighbors lane's to add.
def trustworthiness(
    X, X_embedded, *, n_neighbors=5, metric="euclidean", batch_size=512
):
    """Trustworthiness: how well an embedding preserves local neighborhoods.

    Follows scikit-learn's and cuML's ``trustworthiness`` with Euclidean
    distance. Values lie in ``[0, 1]``; 1 means every embedded neighbor is
    also a near neighbor in the original space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite original data, converted to float32 row-major.
    X_embedded : array-like of shape (n_samples, n_components)
        Finite embedding with the same number of rows, converted to float32.
    n_neighbors : int, default 5
        Neighbors considered, as in cuML. Must be at least 1, less than
        ``n_samples / 2``, and at most 255; otherwise an error names it.
    metric : {'euclidean'}, default 'euclidean'
        Any other metric raises ``NotImplementedError``, as in cuML.
    batch_size : int, default 512
        Accepted for compatibility with cuML (must be at least 1). It does
        not change the result or the work done.

    Returns
    -------
    score : float

    Raises
    ------
    ValueError
        If X and X_embedded have different numbers of rows, contain NaN or
        infinity, or ``batch_size < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> t = metrics.trustworthiness(X, X[:, :2], n_neighbors=5)
    """
    if not isinstance(metric, str) or metric.lower() != "euclidean":
        raise NotImplementedError(
            f"mojolearn trustworthiness: metric={metric!r} is refused; only "
            "'euclidean' is implemented, and cuML's trustworthiness.pyx:86 "
            "refuses every other name too"
        )
    x, _cx = as_f32_c(X, ndim=2, name="X")
    emb, _ce = as_f32_c(X_embedded, ndim=2, name="X_embedded")
    if x.shape[0] != emb.shape[0]:
        raise ValueError(
            f"mojolearn trustworthiness: X has {x.shape[0]} rows and "
            f"X_embedded has {emb.shape[0]}"
        )
    for name, arr in (("X", x), ("X_embedded", emb)):  # glue: validates the two named arguments
        if not all_finite(arr):
            raise ValueError(
                f"mojolearn trustworthiness: {name} contains NaN or "
                "infinity; refused here (metrics/README.md HAND-OFF ask 3)"
            )
    if int(batch_size) < 1:
        raise ValueError(
            "mojolearn trustworthiness: batch_size must be at least 1, got "
            f"{batch_size!r}"
        )
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::trustworthiness_binding.
    # n, m, d, n_neighbors, batch_size
    return float(
        _get_binding().trustworthiness(
            addr_ro(x, name="x"),
            addr_ro(emb, name="emb"),
            [
                int(x.shape[0]),
                int(x.shape[1]),
                int(emb.shape[1]),
                int(n_neighbors),
                int(batch_size),
            ],
        )
    )


# ---------------------------------------------------------------------------
# Unweighted single-label classification. Labels are encoded on the host;
# confusion counts, ratios and averaging are computed by the GPU binding.
# ---------------------------------------------------------------------------
def _classification_labels(values, name, *, allow_empty=False):
    shape = _shape_of(values)
    if len(shape) != 1:
        raise ValueError(f"{name} must be one-dimensional single-label data")
    labels = flatten_labels(values)
    if not allow_empty and not labels:
        raise ValueError(f"{name} must contain at least one label")
    if all(isinstance(v, str) for v in labels):  # glue: type validation of list labels
        return [str(v) for v in labels], "string"  # cpu-route: str labels, the explicit label input step
    if all(isinstance(v, numbers.Integral) or is_bool(v) for v in labels):  # glue: type validation of list labels
        return [int(v) for v in labels], "integer"  # cpu-route: Python int labels, the explicit label input step
    raise TypeError(f"{name} must contain only strings or only integers; "
                    "floating labels, missing labels and mixed types are unsupported")


class _EncodedLabels:
    """Integer labels held as `classes` (sorted distinct Python ints) and
    `codes` (one int32 index into `classes` per row): what
    `_classification_labels` returns as a list of n Python ints, without the
    n objects (DEVIATION 3107). `len()` and iteration answer as that list
    does, so a consumer that walks labels still works; the metric functions
    below never walk it, they go through `_label_set` and `_label_map`."""

    __slots__ = ("classes", "codes")

    def __init__(self, classes, codes):
        self.classes = classes
        self.codes = codes

    def __len__(self):
        return int(self.codes.size)

    def __iter__(self):
        return map(self.classes.__getitem__, self.codes._values())


def _classification_encoded(values, name, *, allow_empty=False):
    """`_classification_labels`, with an integer or bool BUFFER of at least
    `_NATIVE_MIN_LABELS` labels answered as `_EncodedLabels` through the
    native encoder. Everything else (lists, str labels, float buffers and
    their refusal, empty input and its refusal, more distinct labels than
    the encoder holds) is `_classification_labels` itself."""
    fast = _native_classification_labels(values)
    if fast is not None:
        return fast, "integer"
    return _classification_labels(values, name, allow_empty=allow_empty)


def _native_classification_labels(values):
    from ._buffer import _has_buffer, hotpath_enabled
    from ._labels import _encode_labels_native

    if isinstance(values, (str, bytes)) or not hotpath_enabled():
        return None
    if type(values) in (list, tuple):
        # lane apple-fast-py2mojo-core: a flat list of exact Python ints is
        # packed into an int64 buffer in C and encoded like one (the list
        # route's `[int(v) for v in labels]` and its dict per row are gone
        # for it); str, bool, nested or mixed lists keep the list route
        if not values or set(map(type, values)) != {int}:
            return None
        import array
        try:
            store = array.array("q", values)
        except OverflowError:
            return None
        values = Array._owned(store, (len(store),), "<i8", "C")
    if not isinstance(values, Array) and not _has_buffer(values):
        return None
    if len(_shape_of(values)) != 1:
        return None
    try:
        arr = _materialize(values, "input")[0]
    except (TypeError, ValueError, OverflowError):
        return None  # `_classification_labels` names the refusal
    if arr.ndim != 1 or arr.size < _NATIVE_MIN_LABELS:
        return None
    if arr.dtype in ("<u2", "<i1"):
        arr = arr.astype("<i4")  # exact; the encoder has no 8- or 16-bit arm
    if arr.dtype not in ("<i4", "<i8", "<u4", "<u1"):
        return None  # floats are refused by `_classification_labels`
    try:
        encoded = _encode_labels_native(arr)
    except (ImportError, AttributeError):
        return None
    if encoded is None:
        return None
    classes, codes = encoded
    # a bool buffer's classes come back as bools; the list routine's are ints
    return _EncodedLabels([int(c) for c in classes], codes)  # glue: reads the native class table (classes-sized: distinct classes)


def _label_set(labels):
    """`set(labels)`."""
    if isinstance(labels, _EncodedLabels):
        return set(labels.classes)
    return set(labels)


def _label_map(labels, fn):
    """`Array.from_list([fn(v) for v in labels], '<i4')`, `fn` a pure
    function of the label: evaluated once per CLASS and gathered per row for
    `_EncodedLabels` (DEVIATION 3107), once per row for a list."""
    if isinstance(labels, _EncodedLabels):
        from ._buffer import _native
        gather = _native("gather_i32")
        table = [fn(c) for c in labels.classes]  # glue: maps the class table through the label function
        table = Array.from_list(table, "<i4")
        out = empty((len(labels),), "<i4")
        gather(_addr_ro(table), len(labels.classes), _addr_ro(labels.codes),
               len(labels), _addr(out))
        return out
    return Array.from_list([fn(v) for v in labels], "<i4")  # cpu-route: Python object labels mapped per row, the explicit label input step


def _classification_pair(y_true, y_pred, sample_weight):
    if sample_weight is not None:
        raise NotImplementedError("classification metrics do not yet support sample_weight")
    true, kind = _classification_encoded(y_true, "y_true")
    pred, pred_kind = _classification_encoded(y_pred, "y_pred")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred lengths differ")
    if kind != pred_kind:
        raise TypeError("y_true and y_pred must use the same label type")
    if len(true) > 2147483647:
        raise ValueError("classification counts require at most INT32_MAX rows")
    return true, pred, kind, sorted(_label_set(true) | _label_set(pred))  # glue: sorts the distinct class labels



# Maintainer notes: the probability check (finite, in [0, 1], rows summing to
# one), the binary [1 - p, p] layout, the clipped -log p_true per row, its
# Float32 PairSum and the mean (divided by n in binary64) run in ONE x_metrics
# device program (x_metrics/tail.mojo proba_rows, ranking.mojo row_metric,
# rank_epi.mojo MEAN). sample_weight goes to log_loss_options.
def log_loss(y_true, y_pred, *, normalize=True, sample_weight=None, labels=None,
             numeric_mode=None):
    """Log loss (cross-entropy) of predicted probabilities.

    Follows scikit-learn's ``log_loss`` for single-label targets. A 1-D (or
    one-column) ``y_pred`` holds the probability of the positive class and
    is expanded to ``[1 - p, p]``. Otherwise each column is a class: in the
    order of ``labels`` when given, else the sorted labels seen in
    ``y_true``. Probabilities are checked, not renormalized: each must lie
    in ``[0, 1]`` and each row must sum to one within ``sqrt(float32 eps)``.
    The selected probabilities are clipped to ``[eps, 1 - eps]`` (float32
    epsilon) before the logarithm.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        String or integer class labels, nonempty.
    y_pred : array-like of shape (n_samples,), (n_samples, 1) or (n_samples, n_classes)
        float32 probabilities. Other dtypes raise ``TypeError``; Python lists
        are converted to float32.
    normalize : bool, default True
        True returns the mean loss per sample, False the sum.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights, as in scikit-learn.
    labels : array-like, default None
        The class labels, unique, in column order. Needed when ``y_true``
        holds a single class. At least two labels are required.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    loss : float

    Raises
    ------
    ValueError
        For fewer than two labels, a label in ``y_true`` missing from
        ``labels``, a column count that does not match the labels, or
        probabilities that are not finite, outside ``[0, 1]`` or do not sum
        to one.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> y = np.array([0, 1, 1, 0], dtype=np.int32)
    >>> p = np.array([0.1, 0.9, 0.8, 0.35], dtype=np.float32)
    >>> loss = metrics.log_loss(y, p)
    """
    if sample_weight is not None:
        # lane/metrics: the weighted mean of the clipped -log p_true rows.
        if not is_bool(normalize):
            raise ValueError("normalize must be a bool")
        from ._expansion_metrics import log_loss_options
        return log_loss_options(y_true, y_pred, normalize, sample_weight, labels, numeric_mode)
    if not is_bool(normalize):
        raise ValueError("normalize must be a bool")
    true, kind = _classification_encoded(y_true, "y_true")
    selected = _selected_labels(labels, kind, sorted(_label_set(true)))  # glue: sorts the distinct class labels
    if len(selected) < 2:
        raise ValueError("log_loss requires at least two labels; pass labels for one observed class")
    mapping = {label: i for i, label in enumerate(selected)}  # glue: class index table
    if any(label not in mapping for label in _label_set(true)):  # glue: checks the label set against the selected labels
        raise ValueError("y_true contains a label missing from labels")
    probabilities = materialize_f32_lists(y_pred, "input")[0]
    if probabilities.dtype != "<f4":
        raise TypeError("log_loss probabilities must have dtype float32")
    if probabilities.ndim not in (1, 2):
        raise ValueError("log_loss probabilities must have shape (n,), (n,1) or (n,k)")
    if probabilities.shape[0] != len(true):
        raise ValueError("y_true and probability row counts differ")
    binary = probabilities.ndim == 1 or probabilities.shape[1] == 1
    if binary:
        if len(selected) != 2:
            raise ValueError("one-column probabilities require exactly two labels")
    elif probabilities.shape[1] < 2 or probabilities.shape[1] != len(selected):
        raise ValueError("probability columns must match the label count (at least two)")
    if len(true) > 2147483647 or probabilities.size > 2147483647 or (binary and 2 * len(true) > 2147483647):
        raise ValueError("log_loss exceeds the native Int32 indexing bound")
    probabilities = as_f32_c(probabilities, ndim=probabilities.ndim, name="probabilities")[0]
    if not len(true):
        raise ValueError("log_loss requires at least one sample")
    # lane cpu2-l7-metrics: the probabilities' check (finite, in [0, 1], rows
    # summing to one), the binary [1 - p, p] layout, the clipped -log p_true
    # per row, its PairSum and the mean run in ONE x_metrics device program
    # (x_metrics/tail.mojo proba_rows, ranking.mojo row_metric, rank_epi.mojo
    # MEAN); no host probability_rows_f32 pass
    from ._expansion_metrics import _Prog, _put_proba, _proba_code, _row_mean
    encoded = _label_map(true, mapping.__getitem__)
    n, k = len(true), len(selected)
    prog = _Prog()
    S, flags = _put_proba(prog, probabilities, n, k, binary)
    Y = prog.put_i32(encoded)

    def check(prog):
        code = _proba_code(prog, flags)
        if code:
            reasons = {1: "must be finite", 2: "must lie in [0,1]",
                       3: "rows must sum to one within sqrt(float32 eps)"}
            raise ValueError("log_loss probabilities " + reasons.get(code, "failed native validation"))
    return float(_row_mean(S, k, Y, n, "logloss", None, numeric_mode, prog=prog, normalize=bool(normalize),
                           after=(check,)))


def _binary_ranking_inputs(y_true, y_score, sample_weight):
    if sample_weight is not None:
        raise NotImplementedError("binary ranking metrics do not yet support sample_weight")
    true, kind = _classification_encoded(y_true, "y_true")
    classes = sorted(_label_set(true))  # glue: sorts the distinct class labels
    if len(classes) > 2:
        raise ValueError("binary ranking metrics support at most two observed classes")
    scores = materialize_f32_lists(y_score, "input")[0]
    if scores.dtype != "<f4":
        raise TypeError("binary ranking scores must have dtype float32")
    if scores.ndim != 1 or len(scores) != len(true):
        raise ValueError("binary ranking scores must be one-dimensional with the same length as y_true")
    if not all_finite(scores):
        raise ValueError("binary ranking scores must be finite")
    if len(true) > 2147483647:
        raise ValueError("binary ranking metrics exceed the native Int32 indexing bound")
    return true, kind, classes, as_f32_c(scores, ndim=scores.ndim, name="scores")[0]


# Maintainer notes: the binary unweighted full-AUC call keeps its own kernel
# and returns the device Float32 scalar. sample_weight, partial AUC (max_fpr),
# multiclass one-vs-rest / one-vs-one, labels and averages go to the x_metrics
# device curve (roc_auc_options).
def roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="raise", labels=None, numeric_mode=None):
    """Area under the ROC curve.

    Follows scikit-learn's ``roc_auc_score``. In the binary case, ``y_true``
    has two classes, the larger sorted label is the positive class, and
    ``y_score`` holds one finite score per sample (probabilities or any
    decision scores); tied scores are grouped. Multiclass input (a 2-D
    ``y_score`` of class probabilities) needs ``multi_class='ovr'`` or
    ``'ovo'``, as in scikit-learn.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        String or integer labels. Both classes must be present in the binary
        case.
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Finite float32 scores. Other dtypes raise ``TypeError``.
    average : {'macro', 'weighted', 'micro', 'samples'} or None, default 'macro'
        Averaging for multiclass input, as in scikit-learn.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights.
    max_fpr : float or None, default None
        In (0, 1]: return the standardized partial AUC up to this false
        positive rate (binary only), as in scikit-learn.
    multi_class : {'raise', 'ovr', 'ovo'}, default 'raise'
        Multiclass strategy.
    labels : array-like, default None
        Class labels for multiclass input, as in scikit-learn.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    auc : float

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> y = np.array([0, 0, 1, 1], dtype=np.int32)
    >>> s = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    >>> auc = metrics.roc_auc_score(y, s)
    """
    if (average != "macro" or multi_class != "raise" or labels is not None or sample_weight is not None
            or not (max_fpr is None or (not is_bool(max_fpr) and isinstance(max_fpr, numbers.Real)
                                        and max_fpr == 1))
            or len(_shape_of(y_score)) == 2):
        # lane/metrics: sample_weight, partial AUC (max_fpr), multiclass
        # one-vs-rest / one-vs-one, labels and averages, from the x_metrics
        # device curve; the binary unweighted full-AUC call keeps its kernel.
        from ._expansion_metrics import roc_auc_options
        return roc_auc_options(y_true, y_score, average, sample_weight, max_fpr, multi_class, labels,
                               numeric_mode)
    true, _, classes, scores = _binary_ranking_inputs(y_true, y_score, sample_weight)
    if len(classes) != 2:
        raise ValueError("roc_auc_score requires both positive and negative classes")
    encoded = _label_map(true, lambda v: int(v == classes[1]))
    result = empty(1, '<f4')
    _get_binding(numeric_mode).roc_auc_score(
        _addr_ro(encoded), _addr_ro(scores), _addr(result), [len(true)])
    return float(result[0])


# Maintainer notes: the default call keeps its own kernel and Float32 outputs
# (no CPU sorting or reduction). sample_weight and drop_intermediate=True go to
# precision_recall_curve_options: Float64 curves from the x_metrics device sort
# and counts.
def precision_recall_curve(y_true, y_score, *, pos_label=None, sample_weight=None,
                           drop_intermediate=False, numeric_mode=None):
    """Precision-recall pairs for each score threshold (binary).

    Follows scikit-learn's ``precision_recall_curve``. Thresholds are the
    distinct scores in ascending order; ``precision`` and ``recall`` have
    one more element than ``thresholds`` and end at ``(1, 0)``.

    Without ``pos_label``, ``y_true`` must be integers whose observed values
    are a subset of ``{0, 1}`` or ``{-1, 1}``, and 1 is the positive class;
    other labels need an explicit ``pos_label`` of the same type. If the
    positive class is absent, recall is set to one for every threshold and
    a ``UserWarning`` is issued.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        String or integer labels with at most two classes.
    y_score : array-like of shape (n_samples,)
        Finite float32 scores (not necessarily probabilities). Other dtypes
        raise ``TypeError``.
    pos_label : int, str or None, default None
        The positive class.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights.
    drop_intermediate : bool, default False
        Drop thresholds that do not change the curve's shape, as in
        scikit-learn.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    precision : Array of shape (n_thresholds + 1,)
    recall : Array of shape (n_thresholds + 1,)
    thresholds : Array of shape (n_thresholds,)
        float32 by default; float64 when ``sample_weight`` is given or
        ``drop_intermediate=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> y = np.array([0, 0, 1, 1], dtype=np.int32)
    >>> s = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
    >>> precision, recall, thresholds = metrics.precision_recall_curve(y, s)
    """
    if not is_bool(drop_intermediate):
        raise ValueError("drop_intermediate must be a bool")
    if drop_intermediate or sample_weight is not None:
        # lane/metrics: Float64 curves from the x_metrics device sort and
        # counts; the default call keeps its kernel and Float32 outputs.
        from ._expansion_metrics import precision_recall_curve_options
        return precision_recall_curve_options(y_true, y_score, pos_label, sample_weight,
                                              drop_intermediate, numeric_mode)
    true, kind, classes, scores = _binary_ranking_inputs(y_true, y_score, sample_weight)
    if pos_label is None:
        if kind != "integer" or classes not in ([0], [1], [-1], [0, 1], [-1, 1]):
            raise ValueError("pos_label must be specified for labels outside {0,1} or {-1,1}")
        positive = 1
    else:
        values, positive_kind = _classification_labels([pos_label], "pos_label")
        if kind != positive_kind:
            raise ValueError("pos_label must use the same type as y_true")
        positive = values[0]
    encoded = _label_map(true, lambda v: int(v == positive))
    n = len(true)
    precision = empty(n + 1, '<f4')
    recall = empty(n + 1, '<f4')
    thresholds = empty(n, '<f4')
    size = int(_get_binding(numeric_mode).precision_recall_curve(
        _addr_ro(encoded), _addr_ro(scores), _addr(precision), _addr(recall),
        _addr(thresholds), [n]))
    if not 1 <= size <= n:
        raise RuntimeError("precision_recall_curve returned an invalid threshold count")
    if positive not in classes:
        import warnings
        warnings.warn("No positive class found in y_true; recall is set to one "
                      "for all thresholds.", UserWarning, stacklevel=2)
    return precision[:size + 1].copy(), recall[:size + 1].copy(), thresholds[:size].copy()


def _selected_labels(labels, kind, observed):
    if labels is None:
        return observed.copy()
    selected, selected_kind = _classification_labels(labels, "labels")
    if selected_kind != kind:
        raise TypeError("labels must use the same type as the targets")
    if len(set(selected)) != len(selected):
        raise ValueError("labels must be unique")
    return selected


def _encode_classification(true, pred, labels):
    mapping = {label: i for i, label in enumerate(labels)}  # glue: class index table
    return (_label_map(true, lambda v: mapping.get(v, -1)),
            _label_map(pred, lambda v: mapping.get(v, -1)))


# Maintainer notes: unweighted counts come from the metrics binding (at most
# INT32_MAX input rows). sample_weight goes to confusion_matrix_weighted:
# Float64 cells, each the Float32 PairSum of its rows' weights, normalized in
# binary64.
def confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None,
                     normalize=None, numeric_mode=None):
    """Confusion matrix of single-label predictions.

    Follows scikit-learn's ``confusion_matrix``: row ``i`` is the true label
    and column ``j`` the predicted label. By default the labels are the
    sorted union of those seen in ``y_true`` and ``y_pred``. With ``labels``,
    their order is kept and rows whose true or predicted label is not in
    ``labels`` are left out.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        String or integer labels of the same type and length, nonempty.
        Float labels raise ``TypeError``.
    labels : array-like, default None
        Unique labels to index the matrix. At least one must occur in
        ``y_true``. Without ``sample_weight``, at most 4096 labels.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights; the cells are then float64 sums of weights.
    normalize : {'true', 'pred', 'all'} or None, default None
        Normalize over rows, columns or the whole matrix. Rows or columns
        with no mass give zeros.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    C : Array of shape (n_labels, n_labels)
        int64 counts; float32 when normalized; float64 with
        ``sample_weight``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> C = metrics.confusion_matrix(np.array([0, 1, 2, 2], dtype=np.int32),
    ...                              np.array([0, 2, 2, 1], dtype=np.int32))
    """
    if normalize is not None and (not isinstance(normalize, str) or
                                  normalize not in ("true", "pred", "all")):
        raise ValueError("normalize must be None, 'true', 'pred' or 'all'")
    if sample_weight is not None:
        # lane/metrics: Float64 cells, each the Float32 PairSum of its rows'
        # weights (x_metrics binding); the unweighted call keeps its bits.
        from ._expansion_metrics import confusion_matrix_weighted
        return confusion_matrix_weighted(y_true, y_pred, labels, sample_weight, normalize, numeric_mode)
    true, pred, kind, observed = _classification_pair(y_true, y_pred, sample_weight)
    selected = _selected_labels(labels, kind, observed)
    if not set(selected).intersection(_label_set(true)):
        raise ValueError("At least one label specified must be in y_true")
    if len(selected) > 4096:
        raise ValueError("confusion_matrix supports at most 4096 output labels")
    yt, yp = _encode_classification(true, pred, selected)
    norm = {None: 0, "true": 1, "pred": 2, "all": 3}[normalize]
    result = empty((len(selected), len(selected)), "<i8" if norm == 0 else "<f4")
    _get_binding(numeric_mode).confusion_matrix(
        _addr_ro(yt), _addr_ro(yp), _addr(result), [len(true), len(selected), norm])
    return result


def _precision_recall_fscore(y_true, y_pred, *, labels, pos_label, average,
                            sample_weight, zero_division, numeric_mode, metric):
    if sample_weight is not None:
        # lane/metrics: the weighted sums come from the x_metrics binding;
        # the unweighted call keeps its binding and its bits.
        from ._expansion_metrics import _prf_weighted
        return _prf_weighted(y_true, y_pred, labels, pos_label, average, sample_weight,
                             zero_division, numeric_mode, metric)
    averages = {None: 0, "binary": 1, "micro": 2, "macro": 3, "weighted": 4}
    if average is not None and (not isinstance(average, str) or average not in averages):
        raise ValueError("average must be 'binary', 'micro', 'macro', 'weighted' or None")
    warn = isinstance(zero_division, str) and zero_division == "warn"
    if not warn and (is_bool(zero_division) or
                     not isinstance(zero_division, numbers.Real) or
                     zero_division not in (0, 1)):
        raise ValueError("zero_division must be 'warn', 0 or 1")
    true, pred, kind, observed = _classification_pair(y_true, y_pred, sample_weight)
    positive = 0
    if average == "binary":
        if len(observed) > 2:
            raise ValueError("average='binary' requires at most two observed classes")
        positive_labels, positive_kind = _classification_labels([pos_label], "pos_label")
        if positive_kind != kind:
            raise ValueError("pos_label must use the same type as the targets")
        positive_label = positive_labels[0]
        if positive_label not in observed and len(observed) == 2:
            raise ValueError("pos_label is not a valid observed label")
        selected = observed.copy()
        if positive_label not in selected:
            selected.append(positive_label)
        positive = selected.index(positive_label)
    else:
        selected = _selected_labels(labels, kind, observed)
    selected_count = len(selected)
    # Keep classes outside the requested output set for false-positive and
    # false-negative accounting. Truncating confusion first would be wrong.
    selected_set = set(selected)
    all_labels = selected + [v for v in observed if v not in selected_set]  # glue: appends the observed class labels
    if len(all_labels) > 715827882:
        raise ValueError("classification metrics exceed the native class-index bound")
    yt, yp = _encode_classification(true, pred, all_labels)
    width = selected_count if average is None else 1
    output = empty(3 * width + 3, '<f4')
    _get_binding(numeric_mode).precision_recall_fscore(
        _addr_ro(yt), _addr_ro(yp), _addr(output),
        [len(true), len(all_labels), averages[average], positive,
         0 if warn else int(zero_division), selected_count])
    if warn and output[3 * width + metric] != 0:
        import warnings
        from ._expansion_metrics import _undefined_category
        name = ("Precision", "Recall", "F-score")[metric]
        warnings.warn(f"{name} is ill-defined and being set to 0.0 due to zero "
                      "denominator; use zero_division to control this behavior.",
                      _undefined_category(), stacklevel=3)
    values = output[metric * width:(metric + 1) * width]
    return values.copy() if average is None else float(values[0])


def precision_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
                    sample_weight=None, zero_division="warn", numeric_mode=None):
    """Precision: ``tp / (tp + fp)``.

    Follows scikit-learn's ``precision_score`` for single-label targets.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        String or integer labels of the same type and length, nonempty.
        Float and multilabel input are refused.
    labels : array-like, default None
        Labels to report, in order. Labels left out still count as errors
        for the reported ones. Ignored for ``average='binary'``.
    pos_label : int or str, default 1
        The positive class for ``average='binary'``.
    average : {'binary', 'micro', 'macro', 'weighted'} or None, default 'binary'
        How to combine per-label scores; None returns one score per label.
        ``'binary'`` needs at most two observed classes.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights.
    zero_division : {'warn', 0, 1}, default 'warn'
        The value for an undefined ratio; ``'warn'`` gives 0 and a warning.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    score : float or Array of shape (n_labels,)
        A float, or a float32 Array per label when ``average=None``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> p = metrics.precision_score(np.array([0, 1, 1, 0], dtype=np.int32),
    ...                             np.array([0, 1, 0, 1], dtype=np.int32))
    """
    return _precision_recall_fscore(y_true, y_pred, labels=labels, pos_label=pos_label,
        average=average, sample_weight=sample_weight, zero_division=zero_division,
        numeric_mode=numeric_mode, metric=0)


def recall_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
                 sample_weight=None, zero_division="warn", numeric_mode=None):
    """Recall: ``tp / (tp + fn)``.

    Follows scikit-learn's ``recall_score`` for single-label targets.
    Labels, averaging and ``zero_division`` work as in ``precision_score``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        String or integer labels of the same type and length, nonempty.
        Float and multilabel input are refused.
    labels : array-like, default None
        Labels to report, in order. Labels left out still count as errors
        for the reported ones. Ignored for ``average='binary'``.
    pos_label : int or str, default 1
        The positive class for ``average='binary'``.
    average : {'binary', 'micro', 'macro', 'weighted'} or None, default 'binary'
        How to combine per-label scores; None returns one score per label.
        ``'binary'`` needs at most two observed classes.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights.
    zero_division : {'warn', 0, 1}, default 'warn'
        The value for an undefined ratio; ``'warn'`` gives 0 and a warning.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    score : float or Array of shape (n_labels,)
        A float, or a float32 Array per label when ``average=None``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> r = metrics.recall_score(np.array([0, 1, 1, 0], dtype=np.int32),
    ...                          np.array([0, 1, 0, 1], dtype=np.int32))
    """
    return _precision_recall_fscore(y_true, y_pred, labels=labels, pos_label=pos_label,
        average=average, sample_weight=sample_weight, zero_division=zero_division,
        numeric_mode=numeric_mode, metric=1)


def f1_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
             sample_weight=None, zero_division="warn", numeric_mode=None):
    """F1 score: the harmonic mean of precision and recall.

    Follows scikit-learn's ``f1_score`` for single-label targets. Labels,
    averaging and ``zero_division`` work as in ``precision_score``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        String or integer labels of the same type and length, nonempty.
        Float and multilabel input are refused.
    labels : array-like, default None
        Labels to report, in order. Labels left out still count as errors
        for the reported ones. Ignored for ``average='binary'``.
    pos_label : int or str, default 1
        The positive class for ``average='binary'``.
    average : {'binary', 'micro', 'macro', 'weighted'} or None, default 'binary'
        How to combine per-label scores; None returns one score per label.
        ``'binary'`` needs at most two observed classes.
    sample_weight : array-like of shape (n_samples,), default None
        Per-sample weights.
    zero_division : {'warn', 0, 1}, default 'warn'
        The value for an undefined ratio; ``'warn'`` gives 0 and a warning.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for this call (``'fast'`` is Apple only). None uses the
        library default.

    Returns
    -------
    score : float or Array of shape (n_labels,)
        A float, or a float32 Array per label when ``average=None``.

    Examples
    --------
    >>> import numpy as np
    >>> from mojolearn import metrics
    >>> f1 = metrics.f1_score(np.array([0, 1, 1, 0], dtype=np.int32),
    ...                       np.array([0, 1, 0, 1], dtype=np.int32), average='macro')
    """
    return _precision_recall_fscore(y_true, y_pred, labels=labels, pos_label=pos_label,
        average=average, sample_weight=sample_weight, zero_division=zero_division,
        numeric_mode=numeric_mode, metric=2)


# ===========================================================================
# NAMED ABSENCES. Everything a caller might reasonably reach for in
# `sklearn.metrics` or `cuml.metrics` that this module does NOT have, with
# the reason and where the thing that exists stops. Omitting them silently
# would leave a caller to discover it as an AttributeError.
# ===========================================================================

_UNSUPPORTED = {
    "pairwise_distances": (
        "cuML's pairwise_distance.cu is a front-end over cuVS distances; "
        "neighbors/ owns distances in this repository "
        "(neighbors/checks/pinned_distance_tile.mojo), and it is not a "
        "metric of a model (metrics/NOT_IMPLEMENTED.tsv)"
    ),
}


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise AttributeError(
            f"mojolearn.metrics.{name} does not exist: {_UNSUPPORTED[name]}. "
            "See SUPPORT_MATRIX.md for the supported metrics."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# THE METRICS LANE'S FUNCTIONS (lane/metrics, 2026-09-27), computed by the
# x_metrics binding; see python/mojolearn/_expansion_metrics.py.
from ._expansion_metrics import (  # noqa: E402
    ranking_report, regression_report, balanced_accuracy_score, class_likelihood_ratios, classification_report, cohen_kappa_score,
    fbeta_score, hamming_loss, jaccard_score, matthews_corrcoef, multilabel_confusion_matrix,
    precision_recall_fscore_support, zero_one_loss,
    d2_absolute_error_score, d2_pinball_score, d2_tweedie_score, explained_variance_score, max_error, mean_absolute_percentage_error, mean_gamma_deviance, mean_pinball_loss, mean_poisson_deviance, mean_squared_log_error, mean_tweedie_deviance, median_absolute_error, root_mean_squared_log_error,
    auc, average_precision_score, brier_score_loss, coverage_error, d2_brier_score, d2_log_loss_score, dcg_score, det_curve, hinge_loss, label_ranking_average_precision_score, label_ranking_loss, ndcg_score, roc_curve, top_k_accuracy_score,
    adjusted_mutual_info_score, calinski_harabasz_score, contingency_matrix, davies_bouldin_score, normalized_mutual_info_score, pair_confusion_matrix,
)
