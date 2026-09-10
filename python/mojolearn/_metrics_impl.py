# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU scoring functions, including cuML implements and native regression errors.

**metrics IS NOT AN ESTIMATOR.** It is a set of scoring functions, so this
module is shaped like `sklearn.metrics` -- plain functions, no class, no
`fit` -- and each one carries its own WHAT IS HONORED / REFUSED note, the
same house rule the estimator classes follow. A parameter the kernel does
not carry is refused BY NAME with a reason, never accepted and ignored.

WHERE THE NAMES AND THE DEFAULTS COME FROM. Function and argument names
follow scikit-learn. New regression-error
functions document their bounded Float32 contract below. For the original
implements, **the defaults and semantics are cuML's**, from the pinned `v26.08.00`
checkout, and where the two libraries differ the difference is written on
the function. Three of those differences matter:

  * `entropy` and `mutual_info_score` are in NATS, as RAFT computes them.
  * `r2_score` has scikit-learn's and cuML's `force_finite=True` behavior
    BAKED IN (DEVIATION 657) and `force_finite=False` is refused by name.
  * `kl_divergence` does NOT normalize `P` and `Q`, exactly as cuML's
    `cuml.metrics.kl_divergence` does not.

HISTORICAL QUALIFICATION (excludes new regression-error and A2 classification kernels).
The original metrics kernels are bit-identical
across Apple M4, NVIDIA H100 and AMD MI325X, measured at leg 11
(`archive/evidence/E3_RESULTS.md` round 11, commit 144aa5b, `tools/e3_round_judge.sh`
section 7, 34 card stages, both boxes) under `MOJOLEARN_NUMERIC_MODE=
identical`. That result is for the 34-stage card of that commit; the card
has since grown to 61 stages and the three-vendor leg on the GROWN card is
OWED (`metrics/README.md` Status). **The FAST arm, which is not the default
and must be selected explicitly, makes no cross-vendor claim at all** -- the FAST cards differ between
vendors and that is recorded, not a defect.

ONE CAVEAT ABOUT THAT EVIDENCE, since it is checkable and worth checking.
The two BOX cards are committed and are byte-identical to each other:
`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/lanes/
metrics.identical.card` and its `..._172650-mojolearn-e2-amd` sibling, both
at commit `144aa5b`, 34 stages. **The APPLE reference directory that round
was judged against is NOT in the repository** -- `archive/evidence/E3_RESULTS.md:253` names
it `<mac 123452>` and no directory under `bench/results/e1/` carries that
stamp or any other `lanes/metrics.identical.card`. So NVIDIA <-> AMD is
reproducible from committed artifacts today and the two APPLE comparisons
rest on the judge output recorded in `archive/evidence/E3_RESULTS.md`, not on a file
anybody can re-diff.

INPUT VALIDATION LIVES HERE, DELIBERATELY. The Mojo entries do not scan
their device inputs (`metrics/README.md` HAND-OFF ask 3), so these
functions do the `check_array` work: finite `X` for silhouette and
trustworthiness, finite `y`/`y_pred` for `r2_score`, finite and
non-negative `P`/`Q` for `kl_divergence`. One consequence worth naming:
the kernel-level answers for out-of-contract values (a NaN reaching
`sil_op` as `+0.0` through DEVIATION 656, `r2_score` returning the
canonical NaN `0x7fc00000` through DEVIATION 657) are therefore NOT
reachable from Python. They are gated where they live, in
`metrics/checks/regression_metrics_check.mojo` and
`silhouette_check.mojo`.
"""

import math
import numbers
from ._buffer import _materialize, _native
from ._labels import is_bool, flatten_labels
from ._arrays import _addr, _addr_ro

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, as_i64_c, empty
from ._labels import sorted_classes
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
        or numeric_mode.strip().lower() not in ("fast", "deterministic", "identical")
    ):
        raise ValueError("numeric_mode must be 'fast', 'deterministic' or 'identical'")
    expected = (numeric_mode or _backend.default_mode()).strip().lower()
    mod = _backend.binding("_mojolearn_metrics", expected)
    read_mode = getattr(mod, "metrics_numeric_mode", None)
    if read_mode is None:
        read_mode = getattr(mod, "umap_numeric_mode", None)
    if read_mode is not None:
        actual = {0: "fast", 1: "identical", 2: "deterministic"}.get(read_mode())
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
    return any(type(v) is float for v in _flatten(x))


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
    # int64 -> int32 is `array.array`'s C item loop (`Array.astype`), exact
    # after the range check above; no Python loop.
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
    (`python/cuml/cuml/metrics/cluster/utils.py`), mirrored exactly.

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
    tl = yt.tolist()
    pl = yp.tolist()
    classes, _ = sorted_classes(tl + pl)
    index = {c: i for i, c in enumerate(classes)}
    yt = Array.from_list([index[v] for v in tl], "<i4")
    yp = Array.from_list([index[v] for v in pl], "<i4")
    return yt, yp, len(tl), 0, len(classes) - 1


# ===========================================================================
# Group A: the label metrics
# ===========================================================================


def accuracy_score(
    y_true, y_pred, *, normalize=True, sample_weight=None, numeric_mode=None
):
    """The fraction of positions where two label arrays agree.

    numeric_mode selects a compiled tier; None uses the current library default.

    Backed by `ML::Metrics::accuracy_score_py` (cuML `accuracy_score.cu`):
    one fused subtract-and-count with an INTEGER atomic, so the count is
    exact and order-free on every vendor, and the returned float32 is
    `count * (1 / n)`.

        y_true, y_pred  honored   int32 labels, same length. A label array
                                  that does not fit in int32 is refused by
                                  name (the int32 instantiation is the one
                                  cuML's Python passes).
        normalize       honored   True only. False (return the raw count)
                                  is REFUSED by name: the kernel returns the
                                  fraction, and recovering an integer count
                                  from a float32 fraction is a different
                                  computation, not this one.
        sample_weight   refused   not implemented. cuML 26.08's own Python
                                  `accuracy_score` supports it in pure cupy
                                  and does not call this kernel at all;
                                  weighting here would be a host formula
                                  wearing a GPU metric's name.

    `n == 0` is refused by name inside the kernel (`count / n` is `0 / 0`
    in RAFT and a NaN may not reach a recorded value).
    """
    if sample_weight is not None:
        raise NotImplementedError(
            "mojolearn accuracy_score: sample_weight is not implemented "
            "(metrics/impl/stats/detail/scores.mojo has no weighted arm)"
        )
    if not normalize:
        raise NotImplementedError(
            "mojolearn accuracy_score: normalize=False is refused; the "
            "implemented kernel returns the FRACTION of agreeing positions and "
            "the count is not recoverable from it exactly. Use "
            "int((y_true == y_pred).sum()) if you want the count."
        )
    yt, yp = _pair_1d(y_true, y_pred, "y_true", "y_pred", _as_i32_1d)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::accuracy_score_binding.
    # n
    return float(
        _get_binding(numeric_mode).accuracy_score(
            _addr_ro(yt), _addr_ro(yp), [int(yt.shape[0])]
        )
    )


def rand_score(labels_true, labels_pred):
    """The (unadjusted) Rand index between two clusterings.

    Backed by `ML::Metrics::rand_index` (cuML `rand_index.cu`, DEVIATION
    652: per-block partials plus a host Int64 sum, because Apple has no
    64-bit float atomic). Takes RAW labels; there is no label range on this
    entry because theirs has none.

        labels_true,    honored   int32 labels, same length
        labels_pred

    cuML's Python exports only `adjusted_rand_score`; the unadjusted index
    has a C++ entry and this surface exposes it under scikit-learn's name.
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


def adjusted_rand_score(labels_true, labels_pred):
    """The Rand index adjusted for chance.

    Backed by `ML::Metrics::adjusted_rand_index`, the int32 instantiation
    (the int64 overload is the same code at a wider label type and is not
    instantiated). Builds its own contingency matrix over its own label
    range and takes RAW labels, exactly as theirs does.

        labels_true,    honored   int32 labels, same length
        labels_pred

    It is a DIFFERENCE OF LARGE NUMBERS OVER A DIFFERENCE OF LARGE NUMBERS;
    the three exact integers those differences are formed from are card
    stages (`metrics.ari.pair_sums`), and the host epilogue is float64
    divide/multiply/subtract only, which is identity-safe in both modes.
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


def entropy(clustering, *, base=None):
    """The entropy of a labelling, IN NATS unless `base` is given.

    Backed by `ML::Metrics::entropy` (cuML `entropy.cu`, RAFT
    `entropy.cuh`, DEVIATIONS 650 and 651). cuML's name for this is
    `cuml.metrics.cluster.entropy`; scikit-learn has no public equivalent.

        clustering  honored   int32 labels
        base        honored   cuML's `entropy.pyx:72-74` spelling exactly,
                              `math.log(math.exp(S), base)`. Kept as theirs
                              rather than the algebraically equal
                              `S / math.log(base)`:
                              None (the default) means nats.

    THE HISTOGRAM SPANS `max(labels) - min(labels) + 1` BINS, not the number
    of distinct labels. That is cuML's own behavior (`entropy.pyx:61-62`
    passes `min` and `max` with NO remapping, unlike the four cluster
    metrics below), and it means entropy of sparse large-valued labels
    allocates that many integers. Factorize your labels first if that
    matters to you.

    Under `MOJOLEARN_NUMERIC_MODE=identical` the float epilogue is Float32
    through `identical_log` and differs from the FAST arm's Float64 in the
    seventh digit BY DESIGN (DEVIATION 651). That is the price of the same
    bits everywhere; `checks/numerics.mojo` has no portable Float64 log
    yet, and `metrics/README.md`'s hand-off ask 1 is exactly that.
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


def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    """The mutual information between two clusterings, IN NATS.

    Backed by `ML::Metrics::mutual_info_score` (DEVIATIONS 650 and 651):
    the contingency matrix is built with INTEGER atomics on the device, so
    it is exact and order-free everywhere, and the `c (log(nc) - log(ab))`
    epilogue runs serially and ascending on the host.

        labels_true,   honored   int32 labels, same length. Both are
        labels_pred              remapped onto [0, n_classes - 1] over the
                                 union of their distinct labels, exactly as
                                 cuML 26.08's `prepare_cluster_metric_inputs`
                                 does, and the kernel is handed that range.
        contingency    refused   scikit-learn lets you pass a precomputed
                                 contingency matrix instead of labels; the
                                 implemented entry builds its own on the device
                                 and there is no arm that consumes one.

    NATS, not bits: RAFT computes it in nats and so does this.
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


def homogeneity_score(labels_true, labels_pred):
    """Homogeneity: `MI(true, pred) / H(true)`, in `[0, 1]`.

    Backed by `ML::Metrics::homogeneity_score`.

        labels_true,   honored   int32 labels, same length, remapped onto
        labels_pred              [0, n_classes - 1] as cuML 26.08 does
    """
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::homogeneity_score_binding.
    # n, lower_class_range, upper_class_range
    return float(
        _get_binding().homogeneity_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [n, lower, upper]
        )
    )


def completeness_score(labels_true, labels_pred):
    """Completeness: `MI(pred, true) / H(pred)`, in `[0, 1]`.

    Backed by `ML::Metrics::completeness_score`, which RAFT computes as
    `homogeneity_score` WITH THE TWO ARRAYS SWAPPED. That is not a
    relabelling of the same computation: the swapped call folds the
    TRANSPOSED contingency matrix, so the host's serial ascending walk
    visits the cells in a different sequence, and the two are the same
    quantity in exact arithmetic and not necessarily the same float32 bits.
    Implemented as theirs, and both folds are stages on the metrics card.

        labels_true,   honored   int32 labels, same length, remapped onto
        labels_pred              [0, n_classes - 1] as cuML 26.08 does
    """
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::completeness_score_binding.
    # n, lower_class_range, upper_class_range
    return float(
        _get_binding().completeness_score(
            addr_ro(yt, name="yt"), addr_ro(yp, name="yp"), [n, lower, upper]
        )
    )


def v_measure_score(labels_true, labels_pred, *, beta=1.0):
    """The V-measure: `(1 + beta) h c / (beta h + c)`.

    Backed by `ML::Metrics::v_measure` (cuML `v_measure.cu`).

        labels_true,   honored   int32 labels, same length, remapped onto
        labels_pred              [0, n_classes - 1] as cuML 26.08 does
        beta           honored   float64, cuML's and scikit-learn's default
                                 1.0. Above 1 weights completeness more,
                                 below 1 weights homogeneity more.
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
    """`(homogeneity, completeness, v_measure)`, scikit-learn's convenience.

    THREE SEPARATE DEVICE CALLS, not one fused pass, and therefore three
    device contexts. scikit-learn computes all three from one contingency
    matrix; the implemented C++ has three entries and this calls them. If you
    care about the cost, call the one you need.
    """
    return (
        homogeneity_score(labels_true, labels_pred),
        completeness_score(labels_true, labels_pred),
        v_measure_score(labels_true, labels_pred, beta=beta),
    )


# ===========================================================================
# Group B: r2 and KL divergence
# ===========================================================================


def r2_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput="uniform_average",
    force_finite=True,
    numeric_mode=None,
):
    """The coefficient of determination, float32.

    numeric_mode selects a compiled tier; None uses the current library default.

    Backed by `ML::Metrics::r2_score_py`, the float overload (DEVIATIONS
    653 and 657). The three sums (`y_bar`, `sse`, `ssto`) are folded by a
    `PINNED_SUM_W = 256` slab tree whose width is a constant of the source
    and not of the launch, gated launch-invariant across two block sizes
    and two grid shapes.

        y_true, y_pred  honored   1-D float32, same length, finite. float64
                                  is REFUSED by name: cuML has a double
                                  overload and this implementation does not (no
                                  float64 on this GPU), so the cast is
                                  yours to make.
        multioutput     refused   anything but 'uniform_average' with 1-D
                                  input. 2-D targets are not implemented; the
                                  kernel takes one flat pair of arrays.
        sample_weight   refused   not implemented (RAFT's r2_score has no
                                  weighted arm).
        force_finite    honored   True only, and it is BAKED IN. `ssto == 0`
                                  returns 1.0 when `sse == 0` and 0.0
                                  otherwise, which is scikit-learn's and
                                  cuML's own Python surface's behavior.
                                  False (RAFT's raw `1 - sse/ssto`, which
                                  gives -inf or NaN there) is REFUSED by
                                  name, because a recorded scalar carrying
                                  a vendor's NaN payload would give a
                                  different card per vendor on one legal
                                  input (DEVIATION 657).

    An overflow-scale `y` (`|y - y_hat|` above about 1.8e19, so the square
    is +inf) still returns a NaN, but the ONE canonical payload
    `0x7fc00000` on every vendor.
    """
    if sample_weight is not None:
        raise NotImplementedError(
            "mojolearn r2_score: sample_weight is not implemented "
            "(metrics/impl/stats/detail/scores.mojo has no weighted arm)"
        )
    if multioutput != "uniform_average":
        raise NotImplementedError(
            f"mojolearn r2_score: multioutput={multioutput!r} is refused; "
            "the implemented kernel takes one flat pair of 1-D arrays and there "
            "is no multioutput arm"
        )
    if not force_finite:
        raise NotImplementedError(
            "mojolearn r2_score: force_finite=False is refused; the implemented "
            "epilogue bakes in the force_finite=True behavior (DEVIATION "
            "657) so that a constant y cannot put a vendor-specific NaN "
            "payload into a recorded value"
        )
    y, yh = _pair_1d(y_true, y_pred, "y_true", "y_pred", _as_f32_1d)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::r2_score_binding.
    # n
    return float(
        _get_binding(numeric_mode).r2_score(_addr_ro(y), _addr_ro(yh), [int(y.shape[0])])
    )


def _as_regression_f32_1d(x, name):
    if _buffer_format(x) != "f":
        raise TypeError(f"mojolearn regression metrics: {name} must have dtype float32; cast explicitly before scoring")
    a = _materialize(x, "input")[0]
    if a.dtype != "<f4":
        raise TypeError(
            f"mojolearn regression metrics: {name} must have dtype float32, "
            f"got {a.dtype}; cast explicitly before scoring"
        )
    return _as_f32_1d(a, name)


def _regression_error(name, y_true, y_pred, sample_weight, multioutput, numeric_mode):
    if sample_weight is not None:
        raise NotImplementedError(f"mojolearn {name}: sample_weight is not supported yet")
    if not isinstance(multioutput, str) or multioutput != "uniform_average":
        raise NotImplementedError(
            f"mojolearn {name}: multioutput={multioutput!r} is not supported; "
            "only single-output uniform_average is implemented"
        )
    yt, yp = _pair_1d(y_true, y_pred, "y_true", "y_pred", _as_regression_f32_1d)
    return float(getattr(_get_binding(numeric_mode), name)(
        _addr_ro(yt), _addr_ro(yp), [int(yt.size)]
    ))


def mean_squared_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", numeric_mode=None
):
    """Mean squared residual, computed and reduced on the GPU in Float32.

    Inputs must be finite, nonempty Float32 arrays of equal length, shaped
    (n,) or (n, 1). Other dtypes require an explicit cast. Weights and multiple
    outputs are not supported yet. The result is a Python float containing
    the Float32 scalar. Residuals, squares and the sum can overflow to +inf;
    this implementation does not use a wider or scaled accumulator.

    numeric_mode selects the compiled tier per call; None uses the current
    library default. IDENTICAL pins the reduction schedule and flushes
    subnormal arithmetic. New error metrics have local Metal qualification;
    cross-vendor qualification remains pending.
    """
    return _regression_error(
        "mean_squared_error", y_true, y_pred, sample_weight, multioutput, numeric_mode
    )


def mean_absolute_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", numeric_mode=None
):
    """Mean absolute residual on the GPU in Float32.

    Uses the input, mode and overflow contract of mean_squared_error, with
    absolute residuals instead of squares. Weights/multiple outputs are
    unsupported; numeric_mode=None resolves the current library default.
    """
    return _regression_error(
        "mean_absolute_error", y_true, y_pred, sample_weight, multioutput, numeric_mode
    )


def root_mean_squared_error(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", numeric_mode=None
):
    """GPU Float32 square root of mean_squared_error's GPU result.

    Uses mean_squared_error's input, mode and overflow contract. The square
    root also runs on the GPU. Squaring can overflow even when the exact
    RMSE would fit in Float32; no scaled sum-of-squares algorithm is claimed.
    """
    return _regression_error(
        "root_mean_squared_error", y_true, y_pred, sample_weight, multioutput, numeric_mode
    )


def kl_divergence(P, Q):
    """The Kullback-Leibler divergence `sum p_i (log p_i - log q_i)`, float32.

    Backed by `ML::Metrics::kl_divergence`, the float overload (DEVIATIONS
    653 and 658), on the same pinned fold as `r2_score`.

        P, Q   honored   1-D float32, same length, finite and non-negative.
                         float64 is REFUSED by name (no float64 on this
                         GPU). Negative entries are refused HERE, before
                         the kernel, per `metrics/README.md` HAND-OFF ask 3.

    **IT DOES NOT NORMALIZE.** cuML's `cuml.metrics.kl_divergence` does not
    normalize `P` and `Q` either, and neither does RAFT's kernel. If you
    want a divergence between distributions, divide by the sums yourself.
    `p_i == 0` contributes zero (their branch); `q_i == 0` with `p_i > 0`
    gives `+inf`, as theirs.
    """
    p, q = _pair_1d(P, Q, "P", "Q", _as_f32_1d)
    for name, arr in (("P", p), ("Q", q)):
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
    # cuML's silhouette_score.pyx:99-101, mirrored: monotonic labels via
    # cp.unique(..., return_inverse=True), and n_labels is how many distinct
    # labels there are. `_labels.sorted_classes` (DEVIATION 2340) gives
    # both in one O(rows) pass (DEVIATION 2377).
    unique_labels, codes = sorted_classes(lab.tolist())
    mapped = Array.from_list(codes, "<i4")
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


def silhouette_score(
    X, labels, *, metric="euclidean", sample_size=None, random_state=None,
    chunksize=None,
):
    """The mean silhouette coefficient over all samples.

    Backed by `ML::Metrics::Batched::silhouette_score`, the float batched
    entry cuML's Python dispatches (DEVIATIONS 654 and 656). The per-cell
    distance runs one thread per cell through `identical_mul_add` / `ftz` /
    `identical_sqrt`, and the `a`/`b` accumulation is one fixed tree per
    (row, cluster); gated bitwise against a host model over 1031 samples and
    launch-invariant across eight launches.

        X          honored   2-D float32 row-major, finite. float64 is
                             refused by name (no float64 on this GPU).
        labels     honored   any integer labels; they are mapped onto
                             [0, n_labels - 1] by a host sort here, exactly
                             as cuML's silhouette_score.pyx does. The kernel
                             asserts 2 <= n_labels <= n_rows - 1 and refuses
                             otherwise by name.
        metric     honored   'euclidean' or 'l2' (the same DistanceType).
                             Every other metric is REFUSED by name.
        chunksize  honored   cuML's chunk, default 40000, validated >= 1.
                             SCHEDULING ONLY: this implementation materializes no
                             distance tile, so there is nothing for it to
                             size, and chunk 1 / 7 / 40000 are gated to one
                             byte pattern. It is accepted because cuML's
                             surface has it, and it is documented as inert
                             rather than pretending to be a knob.
        sample_size,  refused  scikit-learn's subsampling. Sample X
        random_state           yourself if you want it; doing it inside
                               would put a random draw where no check sees
                               it.
    """
    if sample_size is not None or random_state is not None:
        raise NotImplementedError(
            "mojolearn silhouette_score: sample_size / random_state are "
            "refused; subsample X yourself so the rows that were scored are "
            "the rows you chose (cuML's surface has no subsampling either)"
        )
    mean, _scores = _silhouette(X, labels, metric, chunksize, "silhouette_score")
    return mean


def silhouette_samples(X, labels, *, metric="euclidean", chunksize=None):
    """The silhouette coefficient of every sample, float32 `(n_samples,)`.

    The same kernel call as `silhouette_score`, which computes both: the
    per-sample coefficients are what the mean is taken over. Parameters and
    refusals are `silhouette_score`'s.

    Values that come out `+0.0` and are worth knowing about: a singleton
    cluster, an exact tie `a == b`, `a == b == 0` (all points identical),
    and a row whose distances were non-finite (DEVIATION 656). No fixture
    in this lane has produced a `-0.0`.
    """
    _mean, scores = _silhouette(
        X, labels, metric, chunksize, "silhouette_samples"
    )
    return scores


# ===========================================================================
# Group D: trustworthiness
# ===========================================================================


def trustworthiness(
    X, X_embedded, *, n_neighbors=5, metric="euclidean", batch_size=512
):
    """How well an embedding preserves the original space's neighborhoods.

    Backed by `ML::Metrics::trustworthiness_score<float,
    L2SqrtUnexpanded>` (DEVIATION 655). The rank of each embedded neighbor
    is COUNTED with the stable sort's tie-break instead of sorting every row
    of the n x n distance matrix, and the closed form runs in float64 on the
    host. Measured equal to scikit-learn 1.9's value on this lane's fixture.

        X            honored   2-D float32 row-major `(n, m)`, finite
        X_embedded   honored   2-D float32 row-major `(n, d)`, finite
        n_neighbors  honored   cuML's default 5. REFUSED by name when
                               `n_neighbors < 1`, when `2 * n_neighbors >=
                               n_samples` (cuML's trustworthiness.pyx:114
                               refuses the same, and it is what keeps the
                               closed form's denominator positive so no NaN
                               reaches the answer), or when
                               `n_neighbors + 1 > 256` (the k+1 distances
                               and ranks live in a threadgroup slab).
        metric       honored   'euclidean' only, as cuML's own surface;
                               every other name is REFUSED by name.
        batch_size   honored   validated >= 1, and it SIZES NOTHING:
                               DEVIATION 655 counts ranks instead of
                               sorting, so there is no batch. Accepted
                               because cuML's surface has it, documented as
                               inert rather than pretending to be a knob.

    ONE KNOWN SOFT SPOT, from the lane's own hand-off. The embedded k-NN
    goes through this repository's one k-NN entry, which uses the EXPANDED
    L2 distance, while RAFT's `run_knn` here uses the unexpanded one; near-
    tied embedded neighbors can therefore order differently in the last
    bit. Measured 0 rows differing against a float64 host k-NN on the
    lane's fixture, and a second `neighbors/` arm for the unexpanded metric
    is that lane's to add.
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
    for name, arr in (("X", x), ("X_embedded", emb)):
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
    if all(isinstance(v, str) for v in labels):
        return [str(v) for v in labels], "string"
    if all(isinstance(v, numbers.Integral) or is_bool(v) for v in labels):
        return [int(v) for v in labels], "integer"
    raise TypeError(f"{name} must contain only strings or only integers; "
                    "floating labels, missing labels and mixed types are unsupported")


def _classification_pair(y_true, y_pred, sample_weight):
    if sample_weight is not None:
        raise NotImplementedError("classification metrics do not yet support sample_weight")
    true, kind = _classification_labels(y_true, "y_true")
    pred, pred_kind = _classification_labels(y_pred, "y_pred")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred lengths differ")
    if kind != pred_kind:
        raise TypeError("y_true and y_pred must use the same label type")
    if len(true) > 2147483647:
        raise ValueError("classification counts require at most INT32_MAX rows")
    return true, pred, kind, sorted(set(true) | set(pred))



def log_loss(y_true, y_pred, *, normalize=True, sample_weight=None, labels=None,
             numeric_mode=None):
    """GPU log loss for strict Float32 probabilities and single-label targets.

    A vector or one-column matrix contains binary positive-class probabilities;
    columns expand to [1-p, p]. Otherwise columns follow explicit unique labels
    in caller order, or sorted observed target labels when labels is omitted.
    At least two labels are required; explicit labels permit one observed class.
    Probabilities must lie in [0,1], with row sums within sqrt(Float32 epsilon)
    of one. Validation does not renormalize. The GPU clips selected probabilities
    to [Float32 epsilon, 1-epsilon] and returns a Float32 mean (normalize=True)
    or sum. Sample weights and empty inputs are not supported.
    """
    if sample_weight is not None:
        raise NotImplementedError("log_loss does not yet support sample_weight")
    if not is_bool(normalize):
        raise ValueError("normalize must be a bool")
    true, kind = _classification_labels(y_true, "y_true")
    selected = _selected_labels(labels, kind, sorted(set(true)))
    if len(selected) < 2:
        raise ValueError("log_loss requires at least two labels; pass labels for one observed class")
    mapping = {label: i for i, label in enumerate(selected)}
    if any(label not in mapping for label in true):
        raise ValueError("y_true contains a label missing from labels")
    probabilities = _materialize(y_pred, "input")[0]
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
    packed = empty((len(true), 2), "<f4") if binary else None
    validate = _native("probability_rows_f32")
    if validate is None:
        raise RuntimeError("rebuild the base binding for NumPy-free probability validation")
    code = int(validate(_addr_ro(probabilities), _addr(packed) if binary else 0,
                        len(true), 1 if binary else len(selected), int(binary)))
    if code:
        reasons = {1: "must be finite", 2: "must lie in [0,1]",
                   3: "rows must sum to one within sqrt(float32 eps)"}
        raise ValueError("log_loss probabilities " + reasons.get(code, "failed native validation"))
    if binary:
        probabilities = packed
    encoded = Array.from_list([mapping[label] for label in true], '<i4')
    probabilities = as_f32_c(probabilities, ndim=probabilities.ndim, name="probabilities")[0]
    result = empty(1, '<f4')
    _get_binding(numeric_mode).log_loss(
        _addr_ro(encoded), _addr_ro(probabilities), _addr(result),
        [len(true), len(selected), int(normalize)])
    return float(result[0])



def _binary_ranking_inputs(y_true, y_score, sample_weight):
    if sample_weight is not None:
        raise NotImplementedError("binary ranking metrics do not yet support sample_weight")
    true, kind = _classification_labels(y_true, "y_true")
    classes = sorted(set(true))
    if len(classes) > 2:
        raise ValueError("binary ranking metrics support at most two observed classes")
    scores = _materialize(y_score, "input")[0]
    if scores.dtype != "<f4":
        raise TypeError("binary ranking scores must have dtype float32")
    if scores.ndim != 1 or len(scores) != len(true):
        raise ValueError("binary ranking scores must be one-dimensional with the same length as y_true")
    if not all_finite(scores):
        raise ValueError("binary ranking scores must be finite")
    if len(true) > 2147483647:
        raise ValueError("binary ranking metrics exceed the native Int32 indexing bound")
    return true, kind, classes, as_f32_c(scores, ndim=scores.ndim, name="scores")[0]


def roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="raise", labels=None, numeric_mode=None):
    """GPU binary ROC AUC from finite one-dimensional Float32 scores.

    Both observed classes are required. Targets are strings or integers and
    the larger sorted class is positive. Scores may be probabilities or
    arbitrary finite decision scores. Ties are grouped before integration.
    The result is a Python float representing the GPU Float32 scalar.
    Only full binary AUC is implemented: weights, explicit labels, nondefault
    averaging/multiclass options and partial AUC are refused.
    """
    if average != "macro" or multi_class != "raise" or labels is not None:
        raise NotImplementedError("roc_auc_score currently supports binary defaults "
                                  "average='macro', multi_class='raise', labels=None only")
    if max_fpr is not None and (is_bool(max_fpr) or
            not isinstance(max_fpr, numbers.Real) or max_fpr != 1):
        raise NotImplementedError("roc_auc_score supports only full AUC (max_fpr=None or 1)")
    true, _, classes, scores = _binary_ranking_inputs(y_true, y_score, sample_weight)
    if len(classes) != 2:
        raise ValueError("roc_auc_score requires both positive and negative classes")
    encoded = Array.from_list([int(v == classes[1]) for v in true], '<i4')
    result = empty(1, '<f4')
    _get_binding(numeric_mode).roc_auc_score(
        _addr_ro(encoded), _addr_ro(scores), _addr(result), [len(true)])
    return float(result[0])


def precision_recall_curve(y_true, y_score, *, pos_label=None, sample_weight=None,
                           drop_intermediate=False, numeric_mode=None):
    """GPU binary PR curve with ascending distinct Float32 score thresholds.

    Precision/recall have one more element than thresholds and end at (1,0).
    Targets are strings or integers. Without pos_label, only observed subsets
    [0], [1], [-1], [0,1], or [-1,1] are accepted and 1 is positive.
    Other binary labels require an explicit same-type pos_label. A missing
    positive class is allowed: threshold recalls are one, with terminal zero,
    and a warning is emitted. Weights and drop_intermediate=True are refused.
    All three output arrays are Float32. Scores must be finite, not necessarily
    probabilities; no CPU sorting or metric reduction is used.
    """
    if not is_bool(drop_intermediate):
        raise ValueError("drop_intermediate must be a bool")
    if drop_intermediate:
        raise NotImplementedError("precision_recall_curve does not yet support drop_intermediate=True")
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
    encoded = Array.from_list([int(v == positive) for v in true], '<i4')
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
    mapping = {label: i for i, label in enumerate(labels)}
    return (Array.from_list([mapping.get(v, -1) for v in true], '<i4'),
            Array.from_list([mapping.get(v, -1) for v in pred], '<i4'))


def confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None,
                     normalize=None, numeric_mode=None):
    """GPU confusion counts for unweighted strings/integer single labels.

    Rows are true labels; columns are predictions. The default order is the
    sorted observed union. Explicit unique labels preserve caller order and
    exclude rows whose true or predicted label is outside that set.
    Counts are Int64 (at most INT32_MAX input rows); normalized results are
    Float32. Normalization accepts None, 'true', 'pred', or 'all'; zero-mass
    rows/columns return zeros. At most 4096 output labels are supported.
    Empty inputs and float labels are refused.
    """
    if normalize is not None and (not isinstance(normalize, str) or
                                  normalize not in ("true", "pred", "all")):
        raise ValueError("normalize must be None, 'true', 'pred' or 'all'")
    true, pred, kind, observed = _classification_pair(y_true, y_pred, sample_weight)
    selected = _selected_labels(labels, kind, observed)
    if not set(selected).intersection(true):
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
    all_labels = selected + [v for v in observed if v not in selected_set]
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
        try:
            from sklearn.exceptions import UndefinedMetricWarning
        except ImportError:
            UndefinedMetricWarning = RuntimeWarning
        name = ("Precision", "Recall", "F-score")[metric]
        warnings.warn(f"{name} is ill-defined and being set to 0.0 due to zero "
                      "denominator; use zero_division to control this behavior.",
                      UndefinedMetricWarning, stacklevel=3)
    values = output[metric * width:(metric + 1) * width]
    return values.copy() if average is None else float(values[0])


def precision_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
                    sample_weight=None, zero_division="warn", numeric_mode=None):
    """GPU precision; unweighted strings/integers, Float32 ratios/averages.

    Supported average values are binary, micro, macro, weighted and None.
    Explicit labels preserve order and retain errors against excluded labels;
    binary averaging uses pos_label and ignores labels. zero_division accepts
    'warn' (zero with a warning), 0 or 1. Empty/multilabel inputs are refused.
    """
    return _precision_recall_fscore(y_true, y_pred, labels=labels, pos_label=pos_label,
        average=average, sample_weight=sample_weight, zero_division=zero_division,
        numeric_mode=numeric_mode, metric=0)


def recall_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
                 sample_weight=None, zero_division="warn", numeric_mode=None):
    """GPU recall with the label/average/zero-division contract of precision_score."""
    return _precision_recall_fscore(y_true, y_pred, labels=labels, pos_label=pos_label,
        average=average, sample_weight=sample_weight, zero_division=zero_division,
        numeric_mode=numeric_mode, metric=1)


def f1_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
             sample_weight=None, zero_division="warn", numeric_mode=None):
    """GPU F1 with the label/average/zero-division contract of precision_score."""
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
    "median_absolute_error": (
        "a GPU selection/ordering path with a validated median contract "
        "is not implemented for regression targets yet"
    ),
    "pairwise_distances": (
        "cuML's pairwise_distance.cu is a front-end over cuVS distances; "
        "neighbors/ owns distances in this repository "
        "(neighbors/checks/pinned_distance_tile.mojo), and it is not a "
        "metric of a model (metrics/NOT_IMPLEMENTED.tsv)"
    ),
    "normalized_mutual_info_score": "normalization conventions and public validation are not implemented",
    "adjusted_mutual_info_score": "expected mutual information and its public contract are not implemented",
    "fowlkes_mallows_score": "public score and normalization checks are not implemented",
    "hinge_loss": "GPU margin reduction and its public label contract are not implemented",
}


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise AttributeError(
            f"mojolearn.metrics.{name} does not exist: {_UNSUPPORTED[name]}. "
            "See docs/lanes/GPU_PIPELINE_PLAN.md for implementation scope."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
