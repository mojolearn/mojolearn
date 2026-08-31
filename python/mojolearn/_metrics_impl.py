# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Scoring functions backed by the ported cuML `cpp/src/metrics/` kernels.

**metrics IS NOT AN ESTIMATOR.** It is a set of scoring functions, so this
module is shaped like `sklearn.metrics` -- plain functions, no class, no
`fit` -- and each one carries its own WHAT IS HONORED / REFUSED note, the
same house rule the estimator classes follow. A parameter the kernel does
not carry is refused BY NAME with a reason, never accepted and ignored.

WHERE THE NAMES AND THE DEFAULTS COME FROM. The function names and argument
names are scikit-learn's, because that is what a caller types. **The
DEFAULTS and the SEMANTICS are cuML's**, from the pinned `v26.08.00`
checkout, and where the two libraries differ the difference is written on
the function. Three of those differences matter:

  * `entropy` and `mutual_info_score` are in NATS, as RAFT computes them.
  * `r2_score` has scikit-learn's and cuML's `force_finite=True` behavior
    BAKED IN (DEVIATION 657) and `force_finite=False` is refused by name.
  * `kl_divergence` does NOT normalize `P` and `Q`, exactly as cuML's
    `cuml.metrics.kl_divergence` does not.

WHAT THIS LANE IS CERTIFIED TO BE. The metrics kernels are bit-identical
across Apple M4, NVIDIA H100 and AMD MI325X, measured at leg 11
(`E3_RESULTS.md` round 11, commit 144aa5b, `tools/e3_round_judge.sh`
section 7, 34 card stages, both boxes) under `MOJOLEARN_NUMERIC_MODE=
identical`. That result is for the 34-stage card of that commit; the card
has since grown to 61 stages and the three-vendor leg on the GROWN card is
OWED (`metrics/README.md` Status). **The FAST arm, which is the default,
makes no cross-vendor claim at all** -- the FAST cards differ between
vendors and that is recorded, not a defect.

ONE CAVEAT ABOUT THAT EVIDENCE, since it is checkable and worth checking.
The two BOX cards are committed and are byte-identical to each other:
`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/lanes/
metrics.identical.card` and its `..._172650-mojolearn-e2-amd` sibling, both
at commit `144aa5b`, 34 stages. **The APPLE reference directory that round
was judged against is NOT in the repository** -- `E3_RESULTS.md:253` names
it `<mac 123452>` and no directory under `bench/results/e1/` carries that
stamp or any other `lanes/metrics.identical.card`. So NVIDIA <-> AMD is
reproducible from committed artifacts today and the two APPLE comparisons
rest on the judge output recorded in `E3_RESULTS.md`, not on a file
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

import importlib.machinery
import importlib.util
import math
import os
import sys

import numpy as np

from . import _backend
from ._arrays import _addr, _addr_ro, as_f32_c

__all__ = [
    "accuracy_score",
    "adjusted_rand_score",
    "completeness_score",
    "entropy",
    "homogeneity_completeness_v_measure",
    "homogeneity_score",
    "kl_divergence",
    "mutual_info_score",
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
# `_backend.select()` installs the identical binaries under the canonical
# module names for the FIVE extensions it knows about, and `_mojolearn_metrics`
# is not one of them -- `_backend._MODULES` is a convergence point this lane
# does not own. So a plain `from . import _mojolearn_metrics` under
# MOJOLEARN_NUMERIC_MODE=identical would load the FAST binary under the
# identical name, which is a MISLABELLED MEASUREMENT and exactly the failure
# `_backend.py` exists to prevent. This loader mirrors that module's rule
# without editing it: identical mode loads `identical/_mojolearn_metrics.so`
# explicitly and RAISES BY NAME if it is not built. It never falls back.
#
# THE OPERATOR SHOULD DELETE THIS AND ADD `_mojolearn_metrics` TO
# `_backend._MODULES` AND `_backend._build_script` INSTEAD; this is a
# stand-in, not a second mechanism, and two mechanisms for one decision is
# how they drift.
_BINDING = None


def _get_binding():
    global _BINDING
    if _BINDING is not None:
        return _BINDING
    mode = _backend.requested_mode()
    pkg_name = __name__.rsplit(".", 1)[0]
    full = f"{pkg_name}._mojolearn_metrics"
    if mode == "fast":
        from . import _mojolearn_metrics as mod
    else:
        # The directory comes from `_backend.tier_dir`, which folds in the
        # vendor axis (python/mojolearn/<vendor>/<tier>/ on the Linux wheel).
        path = os.path.join(_backend.tier_dir(mode), "_mojolearn_metrics.so")
        if not os.path.exists(path):
            raise ImportError(
                f"mojolearn: MOJOLEARN_NUMERIC_MODE={mode} but "
                f"{path} is not built; build it with\n    "
                f"MOJOLEARN_NUMERIC_MODE={mode} bash "
                "bindings/build_metrics.sh"
            )
        loader = importlib.machinery.ExtensionFileLoader(full, path)
        spec = importlib.util.spec_from_loader(full, loader, origin=path)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        sys.modules[full] = mod
    _BINDING = mod
    return _BINDING


# ---------------------------------------------------------------------------
# Array handling. `_arrays.py` owns the 2-D float32 contract and is used for
# X; these are its 1-D siblings, under the same rule: the returned array MUST
# be kept alive in a local across the Mojo call, because the Mojo side takes
# a raw address, borrows it, and holds nothing after the call returns.
# ---------------------------------------------------------------------------


def _as_i32_1d(x, name):
    a = np.asarray(x)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(
            f"mojolearn metrics: {name} must be 1-D, got shape {a.shape}"
        )
    if a.size == 0:
        raise ValueError(f"mojolearn metrics: {name} is empty")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(
            f"mojolearn metrics: {name} must be an integer label array, got "
            f"dtype {a.dtype}; the ported kernels take int32 labels"
        )
    out = np.ascontiguousarray(a, dtype=np.int32)
    if not np.array_equal(out.astype(a.dtype, copy=False), a):
        raise ValueError(
            f"mojolearn metrics: {name} does not fit in int32; the ported "
            "kernels are the int32 instantiation (the int64 overload of "
            "adjusted_rand_index is the same code at a wider type and is "
            "not instantiated, metrics/NOT_IMPLEMENTED.tsv)"
        )
    return out


def _as_f32_1d(x, name, *, require_finite=True):
    a = np.asarray(x)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(
            f"mojolearn metrics: {name} must be 1-D, got shape {a.shape}"
        )
    if a.size == 0:
        raise ValueError(f"mojolearn metrics: {name} is empty")
    if a.dtype == np.float64:
        # Named rather than silent: cuML has a float64 overload of r2_score,
        # kl_divergence and silhouette_score and this port does not, because
        # Apple's GPU has no float64 (mojolearn-hardware-limits). The cast is
        # the caller's to make, so the precision they run at is the precision
        # they asked for.
        raise TypeError(
            f"mojolearn metrics: {name} is float64; the float64 overloads of "
            "r2_score, kl_divergence and silhouette_score are NOT ported "
            "(no float64 on this GPU). Cast to float32 yourself so the "
            "precision you run at is the one you chose."
        )
    out = np.ascontiguousarray(a, dtype=np.float32)
    if require_finite and not np.isfinite(out).all():
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
    classes = np.unique(np.concatenate([np.unique(yt), np.unique(yp)]))
    yt = np.ascontiguousarray(np.searchsorted(classes, yt), dtype=np.int32)
    yp = np.ascontiguousarray(np.searchsorted(classes, yp), dtype=np.int32)
    return yt, yp, int(yt.shape[0]), 0, int(len(classes) - 1)


# ===========================================================================
# Group A: the label metrics
# ===========================================================================


def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    """The fraction of positions where two label arrays agree.

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
        sample_weight   refused   not ported. cuML 26.08's own Python
                                  `accuracy_score` supports it in pure cupy
                                  and does not call this kernel at all;
                                  weighting here would be a host formula
                                  wearing a GPU metric's name.

    `n == 0` is refused by name inside the kernel (`count / n` is `0 / 0`
    in RAFT and a NaN may not reach a recorded value).
    """
    if sample_weight is not None:
        raise NotImplementedError(
            "mojolearn accuracy_score: sample_weight is not ported "
            "(metrics/impl/stats/detail/scores.mojo has no weighted arm)"
        )
    if not normalize:
        raise NotImplementedError(
            "mojolearn accuracy_score: normalize=False is refused; the "
            "ported kernel returns the FRACTION of agreeing positions and "
            "the count is not recoverable from it exactly. Use "
            "int((y_true == y_pred).sum()) if you want the count."
        )
    yt, yp = _pair_1d(y_true, y_pred, "y_true", "y_pred", _as_i32_1d)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::accuracy_score_binding.
    # n
    return float(
        _get_binding().accuracy_score(
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
            _addr_ro(yt), _addr_ro(yp), [int(yt.shape[0])]
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
            _addr_ro(yt), _addr_ro(yp), [int(yt.shape[0])]
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
                              `S / math.log(base)`: COPY, DO NOT IMPROVE.
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
            _addr_ro(lab), [int(lab.shape[0]), lower, upper]
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
                                 ported entry builds its own on the device
                                 and there is no arm that consumes one.

    NATS, not bits: RAFT computes it in nats and so does this.
    """
    if contingency is not None:
        raise NotImplementedError(
            "mojolearn mutual_info_score: contingency= is refused; the "
            "ported entry builds the contingency matrix on the device "
            "(metrics/impl/stats/detail/contingency_matrix.mojo) and has "
            "no arm that consumes a precomputed one"
        )
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::mutual_info_score_binding.
    # n, lower_class_range, upper_class_range
    return float(
        _get_binding().mutual_info_score(
            _addr_ro(yt), _addr_ro(yp), [n, lower, upper]
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
            _addr_ro(yt), _addr_ro(yp), [n, lower, upper]
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
    Ported as theirs, and both folds are stages on the metrics card.

        labels_true,   honored   int32 labels, same length, remapped onto
        labels_pred              [0, n_classes - 1] as cuML 26.08 does
    """
    yt, yp, n, lower, upper = _prepare_cluster_labels(labels_true, labels_pred)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::completeness_score_binding.
    # n, lower_class_range, upper_class_range
    return float(
        _get_binding().completeness_score(
            _addr_ro(yt), _addr_ro(yp), [n, lower, upper]
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
            _addr_ro(yt), _addr_ro(yp), [n, lower, upper, float(beta)]
        )
    )


def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0):
    """`(homogeneity, completeness, v_measure)`, scikit-learn's convenience.

    THREE SEPARATE DEVICE CALLS, not one fused pass, and therefore three
    device contexts. scikit-learn computes all three from one contingency
    matrix; the ported C++ has three entries and this calls them. If you
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
):
    """The coefficient of determination, float32.

    Backed by `ML::Metrics::r2_score_py`, the float overload (DEVIATIONS
    653 and 657). The three sums (`y_bar`, `sse`, `ssto`) are folded by a
    `PINNED_SUM_W = 256` slab tree whose width is a constant of the source
    and not of the launch, gated launch-invariant across two block sizes
    and two grid shapes.

        y_true, y_pred  honored   1-D float32, same length, finite. float64
                                  is REFUSED by name: cuML has a double
                                  overload and this port does not (no
                                  float64 on this GPU), so the cast is
                                  yours to make.
        multioutput     refused   anything but 'uniform_average' with 1-D
                                  input. 2-D targets are not ported; the
                                  kernel takes one flat pair of arrays.
        sample_weight   refused   not ported (RAFT's r2_score has no
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
            "mojolearn r2_score: sample_weight is not ported "
            "(metrics/impl/stats/detail/scores.mojo has no weighted arm)"
        )
    if multioutput != "uniform_average":
        raise NotImplementedError(
            f"mojolearn r2_score: multioutput={multioutput!r} is refused; "
            "the ported kernel takes one flat pair of 1-D arrays and there "
            "is no multioutput arm"
        )
    if not force_finite:
        raise NotImplementedError(
            "mojolearn r2_score: force_finite=False is refused; the ported "
            "epilogue bakes in the force_finite=True behavior (DEVIATION "
            "657) so that a constant y cannot put a vendor-specific NaN "
            "payload into a recorded value"
        )
    y, yh = _pair_1d(y_true, y_pred, "y_true", "y_pred", _as_f32_1d)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::r2_score_binding.
    # n
    return float(
        _get_binding().r2_score(_addr_ro(y), _addr_ro(yh), [int(y.shape[0])])
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
        if (arr < 0).any():
            raise ValueError(
                f"mojolearn kl_divergence: {name} has a negative entry; "
                "refused here (metrics/README.md HAND-OFF ask 3) rather "
                "than letting log of a negative reach the fold"
            )
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::kl_divergence_binding.
    # n
    return float(
        _get_binding().kl_divergence(
            _addr_ro(p), _addr_ro(q), [int(p.shape[0])]
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
            "'euclidean' and 'l2' are ported (they are the same "
            "DistanceType::L2SqrtUnexpanded, which is the arm cuML's "
            "silhouette_score.pyx dispatches by default). 'cityblock', "
            "'cosine', 'l1', 'manhattan' and 'sqeuclidean' are other "
            "distance kernels and are not in metrics/ (NOT_IMPLEMENTED.tsv)"
        )
    x, _copied = as_f32_c(X, "X")
    if not np.isfinite(x).all():
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
    # labels there are.
    unique_labels, inverse = np.unique(lab, return_inverse=True)
    mapped = np.ascontiguousarray(inverse, dtype=np.int32)
    n_labels = int(unique_labels.shape[0])
    chunk = 40000 if chunksize is None else int(chunksize)
    if chunk < 1:
        raise ValueError(
            f"mojolearn {caller}: chunksize must be at least 1, got {chunk}"
        )
    scores = np.empty(x.shape[0], dtype=np.float32)
    # ORDER MATCHES bindings/_mojolearn_metrics.mojo::silhouette_binding.
    # n_rows, n_cols, n_labels, chunksize
    mean = float(
        _get_binding().silhouette(
            _addr_ro(x),
            _addr_ro(mapped),
            _addr(scores),
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
                             [0, n_labels - 1] with np.unique here, exactly
                             as cuML's silhouette_score.pyx does. The kernel
                             asserts 2 <= n_labels <= n_rows - 1 and refuses
                             otherwise by name.
        metric     honored   'euclidean' or 'l2' (the same DistanceType).
                             Every other metric is REFUSED by name.
        chunksize  honored   cuML's chunk, default 40000, validated >= 1.
                             SCHEDULING ONLY: this port materializes no
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
            "'euclidean' is ported, and cuML's trustworthiness.pyx:86 "
            "refuses every other name too"
        )
    x, _cx = as_f32_c(X, "X")
    emb, _ce = as_f32_c(X_embedded, "X_embedded")
    if x.shape[0] != emb.shape[0]:
        raise ValueError(
            f"mojolearn trustworthiness: X has {x.shape[0]} rows and "
            f"X_embedded has {emb.shape[0]}"
        )
    for name, arr in (("X", x), ("X_embedded", emb)):
        if not np.isfinite(arr).all():
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
            _addr_ro(x),
            _addr_ro(emb),
            [
                int(x.shape[0]),
                int(x.shape[1]),
                int(emb.shape[1]),
                int(n_neighbors),
                int(batch_size),
            ],
        )
    )


# ===========================================================================
# NAMED ABSENCES. Everything a caller might reasonably reach for in
# `sklearn.metrics` or `cuml.metrics` that this module does NOT have, with
# the reason and where the thing that exists stops. Omitting them silently
# would leave a caller to discover it as an AttributeError.
# ===========================================================================

_NOT_PORTED = {
    "mean_absolute_error": (
        "RAFT's regression_metrics (scores.cuh) is NOT PORTED "
        "(metrics/NOT_IMPLEMENTED.tsv); it was not in this lane's brief"
    ),
    "mean_squared_error": (
        "RAFT's regression_metrics (scores.cuh) is NOT PORTED "
        "(metrics/NOT_IMPLEMENTED.tsv); it was not in this lane's brief"
    ),
    "root_mean_squared_error": (
        "RAFT's regression_metrics (scores.cuh) is NOT PORTED "
        "(metrics/NOT_IMPLEMENTED.tsv)"
    ),
    "median_absolute_error": (
        "RAFT's regression_metrics (scores.cuh) is NOT PORTED, and the "
        "median arm would additionally need a radix sort this repository "
        "does not have on device (metrics/NOT_IMPLEMENTED.tsv)"
    ),
    "pairwise_distances": (
        "cuML's pairwise_distance.cu is a front-end over cuVS distances; "
        "neighbors/ owns distances in this repository "
        "(neighbors/checks/pinned_distance_tile.mojo), and it is not a "
        "metric of a model (metrics/NOT_IMPLEMENTED.tsv)"
    ),
    "confusion_matrix": (
        "the CONTINGENCY matrix is ported and is what the label metrics "
        "consume (metrics/impl/stats/detail/contingency_matrix.mojo), but "
        "it has no entry of its own; cuML's confusion_matrix.py is pure "
        "cupy on the host and is not a kernel this lane ported"
    ),
    "normalized_mutual_info_score": (
        "no C++ entry in cuML's cpp/src/metrics/ and no RAFT header; it "
        "would be a host formula over the ported mutual_info_score and "
        "entropy, which belongs to whoever wants to gate it"
    ),
    "adjusted_mutual_info_score": (
        "same as normalized_mutual_info_score, plus an expected-MI term "
        "that nothing here computes"
    ),
    "fowlkes_mallows_score": "no C++ entry in cuML's cpp/src/metrics/",
    "roc_auc_score": (
        "cuML has one in Python (_ranking.py) computed with cupy on the "
        "host; there is no kernel in cpp/src/metrics/ for this lane to port"
    ),
    "log_loss": (
        "cuML has one in Python (_classification.py) computed with cupy; no "
        "kernel in cpp/src/metrics/"
    ),
    "precision_recall_curve": "cuML's is Python/cupy; no kernel to port",
    "hinge_loss": "cuML's is Python/cupy; no kernel to port",
}


def __getattr__(name):
    if name in _NOT_PORTED:
        raise AttributeError(
            f"mojolearn.metrics.{name} does not exist: {_NOT_PORTED[name]}. "
            "Binding it anyway would put an ungated host formula behind a "
            "GPU metric's name."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
