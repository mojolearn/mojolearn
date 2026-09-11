# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML's `KernelDensity`: the six log-kernels, their norms, the logsumexp.

FOLLOWS cuML `python/cuml/cuml/neighbors/kernel_density.py` at cuML
`00094f7` (the 25.08 Python layer: `*_log_kernel` at `:43-99`,
`logVn`/`logSn`/`norm_log_probabilities` at `:112-141`,
`logsumexp_kernel` at `:144-156`, `KernelDensity.fit` at `:220-262`,
`KernelDensity.score_samples` at `:264-363`). Five
numbered departures: DEVIATIONS 600-602 (below), 603 (`logsumexp_kernel`)
and 604 (the refusals above `kde_float32_min`).

WHY THE 25.08 PYTHON FILE AND NOT THE 26.08 C++
------------------------------------------------
The brief names cuML v26.08's `cpp/src/kde/kde.cu`. That file is 83 lines
and does ONE thing: it casts the enums and calls
`cuvs::distance::kde(...)` (`kde.cu:45-55`); the Cython
`kernel_density.pyx` (26.08) validates and forwards. The algorithm itself
lives in cuVS 26.08, and the cuVS checkout this tree is pinned to
(`ENGINEERING_RULES.md` 0a: `upstream/cuvs` at `94c2819`, 25.08) predates it
and has no `kde`. The 25.08 cuML Python file above IS the algorithm the
26.08 fused kernel was written to reproduce -- the same six log-kernels
with the same `FLOAT_MIN` sentinel, the same normalization, the same
per-row logsumexp -- and is the version read symbol by symbol here.
`kde/impl/kde.mojo` carries the 26.08 entry's shape (enum values,
signature, `sum_weights` passed in) over this algorithm. When cuVS 26.08
is cloned the fused kernel is the next implementation; `kde/NOT_IMPLEMENTED.tsv` names it.

THE SIX LOG-KERNELS, TRANSCRIBED WITH THEIR CUPY SEMANTICS
----------------------------------------------------------
Their kernels are `cp.fuse` elementwise functions over a float32 distance
matrix with Python-float `h`. Three things about them are not obvious and
are kept ON PURPOSE (sklearn's `_binary_tree.pxi:377-414` differs on all
three, and sklearn is the oracle for SEMANTICS, not bits):

1. **The sentinel is `np.finfo(float32).min`, not `-inf`.** `tophat`,
   `epanechnikov`, `linear`, `cosine` write `-3.4028235e38` outside the
   support (`:55`, `:66`, `:85`, `:96`), sklearn writes `NEG_INF`. A query
   outside every training point's support therefore scores about
   `-3.4e38`, not `-inf`. Same sign, both "zero density".
2. **A bool-times-float product, so a kernel that is 1 inside the support
   stores `-0.0`, not `+0.0`.** `tophat` is `(x >= h) * FLOAT_MIN`, and
   for `x < h` that is `0.0 * (-3.4e38) = -0.0` under IEEE. The row max
   below therefore sees `-0.0`. IDENTITY_PATHS row 13 asks how `-0.0` and
   `+0.0` are ordered; the answer is in `logsumexp_kernel`'s docstring,
   and `kde/checks/kde_check.mojo::check_kde_zero_sign_cannot_leak`
   proves it cannot reach the output.
3. **`log(0)` is avoided by `maximum(z, 1e-30)`** (`:62`, `:81`, `:92`),
   and the support test is applied AFTER the log as another bool product:
   `y = (x < h) * log(z); y += (x >= h) * FLOAT_MIN`. For `x >= h` that is
   `(+/-0.0) + FLOAT_MIN = FLOAT_MIN` exactly; for `x < h` it is `log(z) +
   (-0.0) = log(z)`. The branches below are that product, resolved.

The whole thing is FLOAT32 (DEVIATION 600, below): their numba
`logsumexp_kernel` accumulates `sum = 0.0` in float64 and writes a float64
`log_probabilities`; Metal has no float64 on the device, so this implementation is
float32 end to end and the Float64 host reference in
`kde/checks/kde_oracle.mojo` measures what that costs.

============ DEVIATION 600 (2026-08-23): FLOAT32 END TO END ============
THEIRS: `distances` is float32 (cuML casts the inputs, `fit:248`), the
log-kernels are float32, but `logsumexp_kernel` (`:144-156`) sums
`math.exp(float32)` into a float64 `sum`, `log_probabilities` is
`cp.zeros(n)` = float64, and the two normalizations (`:343`, `:356`) are
float64 host scalars subtracted from it.
OURS: float32 throughout -- the exp sum, the `log(sum) + max`, the two
subtractions -- because this library's device target set has no float64
(`mojolearn hardware limits`: Apple has none on device) and one source
serves every vendor. MEASURED: `check_kde_oracle_vs_float64` prints the
largest |float32 - float64| over the fixture per kernel and metric; the
gate is a tolerance, and the Float64 reference is sklearn's `-inf`
formulation, not theirs. Where their sentinel (`-3.4e38`) stands in for
`-inf` the Float64 reference is `-inf` and the comparison is skipped by
name.

============ DEVIATION 601 (2026-08-23): THE NORMALIZATION CONSTANT UNDER
============ IDENTICAL IS A FLOAT32 CONSTRUCTION, NOT A HOST libm CALL ====
THEIRS: `norm_log_probabilities` (`:112-141`) is host float64 through
`np.log` and `math.lgamma`.
OURS, FAST: the same, host float64 through `std.math.log`/`lgamma`, cast
to float32 once (`log_kernel_norm_fast`).
OURS, IDENTICAL: a host libm's `log` and `lgamma` are not one arithmetic
across hosts (IDENTITY_PATHS row 18's class: cross-vendor is cross-HOST),
and `checks/numerics.mojo` has no portable float64 log or lgamma. So
under IDENTICAL the constant is built from `identical_log` over float32:
`lgamma` at integers and half-integers is the ascending sum of logs of the
Gamma recurrence (`Gamma(k) = (k-1)!`, `Gamma(k+1/2) = sqrt(pi) prod
(i-1/2)`), and every `a*b+c` is `identical_mul_add`. Its precision is
float32-class and degrades with `d` (about `d/2` roundings of a sum that
grows like `d log d`); MEASURED by `check_kde_log_norm_closed_form`, which
prints |IDENTICAL - float64 closed form| per kernel and `d` and asserts a
tolerance. What is purchased is that the constant is the same bits on
every host. HAND-OFF: a `portable_log64`/`portable_lgamma64` in
`numerics.mojo` would let IDENTICAL keep float64 precision here; that file
is not this lane's.
"""

from std.math import lgamma, log, pi, sqrt
from std.memory import bitcast

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.identity_trace import IdentityTrace
from kde.impl.distance.distance import pairwise_distance
from kde.impl.distance.distance_ops import (
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
    PAIRWISE_ELEM_TPB,
)
from neighbors.impl.distance.detail.distance_ops import (
    cosine_zero_norm_row,
    cosine_zero_norm_row_ptr,
    validate_metric_arg,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    ftz,
    ftz_simd,
    identical_cos,
    identical_exp,
    identical_log,
    identical_mul_add,
    identical_mul_add_simd,
    identical_sqrt,
)

# ---------------------------------------------------------------------------
# `VALID_KERNELS` (`kernel_density.py:31-38`) with the 26.08 enum's values
# (`cuml/neighbors/kde.hpp:17-24`: Gaussian 0 ... Cosine 5).
# ---------------------------------------------------------------------------
comptime KDE_KERNEL_GAUSSIAN = 0
comptime KDE_KERNEL_TOPHAT = 1
comptime KDE_KERNEL_EPANECHNIKOV = 2
comptime KDE_KERNEL_EXPONENTIAL = 3
comptime KDE_KERNEL_LINEAR = 4
comptime KDE_KERNEL_COSINE = 5
comptime KDE_N_KERNELS = 6

#: `np.finfo(np.float32).min` = -3.4028234663852886e+38, bits 0xFF7FFFFF.
comptime KDE_FLOAT32_MIN_BITS: UInt32 = 0xFF7FFFFF
#: The `1e-30` floor under the log (`:62`, `:81`, `:92`), as float32.
comptime KDE_LOG_FLOOR = Float32(1e-30)

# ============ DEVIATION 604 (2026-08-23): INPUTS WHOSE FLOAT32 ARITHMETIC IS
# ============ NaN ARE REFUSED BY NAME BEFORE ANY LAUNCH ==================
# THEIRS: `fit`/`score_samples` (`:220-363`) validate `bandwidth > 0`, the
# kernel and metric names, `sample_weight.min() > 0` and its length; the
# data is `input_to_cuml_array` with no finiteness check, so a NaN or an
# infinity in X, a subnormal or infinite weight, a bandwidth whose square
# underflows, or a row norm that overflows in the expanded `sqeuclidean`
# identity all flow to the device and come back as NaN (or, for the
# subnormals, as a column-dependent value: `log(1e-40)` is -92 where
# denormals are kept and -inf where they flush).
# OURS: REFUSED BY NAME on the host before a buffer is uploaded. The
# reason is IDENTITY_PATHS row 39's FACT 2: every stage of this estimator
# is recorded on the card (`kde.dists` ... `kde.scores`) and a COMPUTED
# NaN carries the vendor's payload, so it can never be allowed to reach a
# recorded stage; refusing the input is the guard that costs no bit on
# any legal input. scikit-learn (the semantics oracle) refuses non-finite
# X the same way (`validate_data`: "Input X contains NaN" / "infinity").
# The four rules, each naming the offending parameter and position:
#   (1) `X` / `X_query` finite (no NaN, no +-inf);
#   (2) `metric='sqeuclidean'` only: every |x| < 2^63 / sqrt(n_features),
#       so `||x||^2`, `||y||^2`, their sum and `2 x.y` stay below 2^128
#       and the expanded identity cannot form `inf - inf` (the unexpanded
#       metrics saturate to +inf and never NaN, so they carry no bound);
#   (3) `bandwidth >= 2^-63`, so `h*h` and `2*h*h` are normal float32;
#   (4) `sample_weight` normal and finite (>= 2^-126, < inf).
# MEASURED: `check_kde_nan_cannot_reach_a_stage` drives each rule and the
# `check_kde_refusals` table gained them. DEVIATION 603 (in
# `logsumexp_kernel`) covers the one NaN a legal finite input can still
# compute after these: a row whose every log-kernel is -inf.
# ==========================================================================
comptime KDE_MIN_BANDWIDTH = Float32(1.0842021724855044e-19)  # 2^-63
comptime KDE_FLOAT32_MIN_NORMAL = Float32(1.1754943508222875e-38)  # 2^-126
comptime KDE_INF_BITS: UInt32 = 0x7F800000

#: SCHEDULING. `KDE_ELEM_TPB` is the one-thread-per-cell width of the
#: elementwise kernels, `KDE_LSE_TPB` the one-thread-per-ROW width of the
#: logsumexp. Neither moves a bit; `kde_check` varies both.
comptime KDE_ELEM_TPB = PAIRWISE_ELEM_TPB
comptime KDE_LSE_TPB = 128


def kde_float32_min() -> Float32:
    return bitcast[DType.float32](KDE_FLOAT32_MIN_BITS)


def kde_inf() -> Float32:
    return bitcast[DType.float32](KDE_INF_BITS)


def kde_validate_data(
    x: List[Float32], n_rows: Int, n_features: Int, metric: Int, what: String
) raises:
    """DEVIATION 604, rules (1) and (2): `x` (row-major `n_rows x
    n_features`, named `what` in the message) must be finite, and under
    `sqeuclidean` every |value| must be below `2^63 / sqrt(n_features)`.
    Host-only; no device bit depends on it. A NaN is found by `v != v`.

    DEVIATION 553 (2026-09-01), rule (5), COSINE ONLY: no row may be all
    zeros. `cosine.cuh:86` divides by `||x|| * ||y||` with no guard, so a
    zero row makes the whole of that row's distances NaN, and a NaN then
    enters a top-k or a log-kernel where nothing looks at it again. The
    argument and the two different wrong answers the two selectors give
    are in `neighbors/impl/distance/detail/distance_ops.mojo`'s module
    docstring. Refusing is the correct behaviour for an undefined input,
    which is the one case a refusal is still right."""
    if len(x) != n_rows * n_features:
        raise Error(
            "kde: " + what + " has " + String(len(x)) + " values, expected "
            + String(n_rows * n_features)
        )
    if metric == DIST_COSINE_EXPANDED:
        var zr = cosine_zero_norm_row(x, n_rows, n_features)
        if zr >= 0:
            raise Error(
                "kde: metric='cosine' but " + what + " row " + String(zr)
                + " is all zeros; cosine distance divides by ||x|| and is"
                " undefined at the origin (DEVIATION 553)"
            )
    var bound = Float32(0.0)
    var bounded = metric == DIST_L2_EXPANDED
    if bounded:
        # 2^63 / sqrt(d), formed in float64 on the host (a threshold, not
        # a certified value).
        bound = Float32(9.223372036854775808e18 / sqrt(Float64(n_features)))
    for i in range(n_rows * n_features):
        var v = x[i]
        if v != v or v == kde_inf() or v == -kde_inf():
            raise Error(
                "kde: " + what + " contains " + ("NaN" if v != v else "infinity")
                + " at row " + String(i // n_features) + ", column "
                + String(i % n_features) + " (DEVIATION 604)"
            )
        if bounded and abs(v) >= bound:
            raise Error(
                "kde: metric='sqeuclidean' needs |" + what + "| < 2^63/sqrt("
                + String(n_features) + ") so the expanded identity cannot form"
                " inf - inf; row " + String(i // n_features) + ", column "
                + String(i % n_features) + " is " + String(v) + " (DEVIATION 604)"
            )


# ---------------------------------------------------------------------------
# DEVIATION 2660 (2026-09-11): DEVIATION 604'S DATA RULES OVER THE CALLER'S
# MEMORY, BLOCK SCREENED
# ---------------------------------------------------------------------------
# THEIRS: cuML's `fit` and `score_samples` read the caller's array in place
# (`input_to_cuml_array`); no host copy.
# OURS BEFORE: `kde_score_samples_binding` copied X and the queries into two
# `List`s (36 to 54 ms at 100,000 x 220 on the H100), then
# `kde_validate_data` walked every value with an early-exit loop that does
# not vectorize (39 ms), then `_upload` copied the List again into pinned
# memory.
# OURS NOW: `kde_validate_data_ptr` takes the caller's pointer. Blocks of
# `KDE_VALIDATE_W` values are screened with one mask each: an exponent field
# of all ones (exactly the values where `v != v or v == inf or v == -inf`),
# and under sqeuclidean `abs(v) >= bound`. The first block that screens
# positive, and the tail, run `kde_validate_data`'s own per-value loop from
# that block's first index, so the refusal names the same first value, in
# the same words, as before. Host-only, no device bit depends on it; the
# score bits are unchanged because the uploaded values are the caller's.
# The cosine zero-row rule (DEVIATION 553) runs first, as before, through
# `cosine_zero_norm_row_ptr`. The length rule is the caller's: the pointer
# entry runs after `kde_fit_validate` and the `n_query` check, so both
# dimensions are positive and the span is `n_rows * n_features` by
# construction. Gate: `kde/checks/kde_check.mojo::check_kde_host_ptr_equals_list`.
comptime KDE_VALIDATE_W = 16


def kde_validate_data_ptr(
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    metric: Int,
    what: String,
) raises:
    """DEVIATION 2660: `kde_validate_data` over a host pointer, same rules,
    same first refusal, same message."""
    if n_rows <= 0 or n_features <= 0:
        raise Error(
            "kde: " + what + " must have positive dimensions, got "
            + String(n_rows) + " x " + String(n_features)
        )
    if metric == DIST_COSINE_EXPANDED:
        var zr = cosine_zero_norm_row_ptr(x, n_rows, n_features)
        if zr >= 0:
            raise Error(
                "kde: metric='cosine' but " + what + " row " + String(zr)
                + " is all zeros; cosine distance divides by ||x|| and is"
                " undefined at the origin (DEVIATION 553)"
            )
    var bound = Float32(0.0)
    var bounded = metric == DIST_L2_EXPANDED
    if bounded:
        bound = Float32(9.223372036854775808e18 / sqrt(Float64(n_features)))
    var n = n_rows * n_features
    var expm = SIMD[DType.uint32, KDE_VALIDATE_W](UInt32(0x7F800000))
    var one = SIMD[DType.uint32, KDE_VALIDATE_W](UInt32(1))
    var zero = SIMD[DType.uint32, KDE_VALIDATE_W](UInt32(0))
    var boundv = SIMD[DType.float32, KDE_VALIDATE_W](bound)
    var i = 0
    var body = n - n % KDE_VALIDATE_W
    while i < body:
        var v = x.unsafe_load[width=KDE_VALIDATE_W](i)
        var hit = (bitcast[DType.uint32](v) & expm).eq(expm).select(one, zero).reduce_max()
        if bounded:
            hit = hit | abs(v).ge(boundv).select(one, zero).reduce_max()
        if hit != UInt32(0):
            break
        i += KDE_VALIDATE_W
    # The screened-positive block (if any) and the tail, value by value in
    # `kde_validate_data`'s order and words.
    while i < n:
        var v = x.unsafe_load(i)
        if v != v or v == kde_inf() or v == -kde_inf():
            raise Error(
                "kde: " + what + " contains " + ("NaN" if v != v else "infinity")
                + " at row " + String(i // n_features) + ", column "
                + String(i % n_features) + " (DEVIATION 604)"
            )
        if bounded and abs(v) >= bound:
            raise Error(
                "kde: metric='sqeuclidean' needs |" + what + "| < 2^63/sqrt("
                + String(n_features) + ") so the expanded identity cannot form"
                " inf - inf; row " + String(i // n_features) + ", column "
                + String(i % n_features) + " is " + String(v) + " (DEVIATION 604)"
            )
        i += 1


def kernel_from_name(name: String) raises -> Int:
    """`VALID_KERNELS`; anything else RAISES with the name
    (`KernelDensity.__init__:211-212`)."""
    if name == "gaussian":
        return KDE_KERNEL_GAUSSIAN
    if name == "tophat":
        return KDE_KERNEL_TOPHAT
    if name == "epanechnikov":
        return KDE_KERNEL_EPANECHNIKOV
    if name == "exponential":
        return KDE_KERNEL_EXPONENTIAL
    if name == "linear":
        return KDE_KERNEL_LINEAR
    if name == "cosine":
        return KDE_KERNEL_COSINE
    raise Error("invalid kernel: '" + name + "'")


def kernel_name(kernel: Int) -> String:
    if kernel == KDE_KERNEL_GAUSSIAN:
        return String("gaussian")
    if kernel == KDE_KERNEL_TOPHAT:
        return String("tophat")
    if kernel == KDE_KERNEL_EPANECHNIKOV:
        return String("epanechnikov")
    if kernel == KDE_KERNEL_EXPONENTIAL:
        return String("exponential")
    if kernel == KDE_KERNEL_LINEAR:
        return String("linear")
    if kernel == KDE_KERNEL_COSINE:
        return String("cosine")
    return String("?")


def metric_from_name(name: String) raises -> Int:
    """`cuml.metrics.pairwise_distances`'s dense table
    (`metrics/pairwise_distances.pyx:68-86`), the SIX implemented rows; every
    other row of THEIR table is refused BY NAME so a caller learns it is
    unimplemented rather than unknown.

    `cosine` (`:70`) and `minkowski` (`:78`) joined the table on
    2026-09-01. They were refused here with the words "is in cuML's
    pairwise_distances table but is NOT IMPLEMENTED"; that sentence is now
    false for those two and is deleted rather than annotated.
    """
    if name == "euclidean" or name == "l2":
        return DIST_L2_SQRT_UNEXPANDED
    if name == "sqeuclidean":
        return DIST_L2_EXPANDED
    if name == "l1" or name == "cityblock" or name == "manhattan":
        return DIST_L1
    if name == "chebyshev":
        return DIST_LINF
    if name == "cosine":
        return DIST_COSINE_EXPANDED
    if name == "minkowski":
        return DIST_LP_UNEXPANDED
    if (
        name == "canberra"
        or name == "hellinger"
        or name == "correlation"
        or name == "jensenshannon"
        or name == "hamming"
        or name == "kldivergence"
        or name == "russellrao"
        or name == "nan_euclidean"
    ):
        raise Error(
            "kde: metric='"
            + name
            + "' is in cuML's pairwise_distances table but is NOT IMPLEMENTED"
            " (kde/NOT_IMPLEMENTED.tsv); implemented: euclidean, l2, sqeuclidean, l1,"
            " cityblock, manhattan, chebyshev, cosine, minkowski"
        )
    raise Error("Unknown metric: " + name)


def metric_name(metric: Int) -> String:
    if metric == DIST_L2_SQRT_UNEXPANDED:
        return String("euclidean")
    if metric == DIST_L2_EXPANDED:
        return String("sqeuclidean")
    if metric == DIST_L1:
        return String("l1")
    if metric == DIST_LINF:
        return String("chebyshev")
    if metric == DIST_COSINE_EXPANDED:
        return String("cosine")
    if metric == DIST_LP_UNEXPANDED:
        return String("minkowski")
    return String("?")


# ---------------------------------------------------------------------------
# The log-kernels, `kernel_density.py:43-99`, scalar, float32. Shared by the
# device kernel below and (as a SECOND SPELLING, not an import) by
# `kde/checks/kde_oracle.mojo`.
# ---------------------------------------------------------------------------


@always_inline
def gaussian_log_kernel(x: Float32, h: Float32) -> Float32:
    """`:44-45`: `-(x * x) / (2 * h * h)`."""
    var xx = ftz(x * x)
    var hh2 = ftz(ftz(Float32(2.0) * h) * h)
    return ftz((-xx) / hh2)


@always_inline
def tophat_log_kernel(x: Float32, h: Float32) -> Float32:
    """`:48-55`: `(x >= h) * FLOAT_MIN`. `0.0 * FLOAT_MIN` is `-0.0`."""
    if x >= h:
        return kde_float32_min()
    return Float32(-0.0)


@always_inline
def epanechnikov_log_kernel(x: Float32, h: Float32) -> Float32:
    """`:59-73`: `z = maximum(1 - x*x/h_squared, 1e-30); y = (x < h) *
    log(z); y += (x >= h) * FLOAT_MIN`, with `h_squared = h * h` formed
    outside the fused kernel (their cupy workaround, `:68-73`)."""
    if x >= h:
        return kde_float32_min()
    var h_squared = ftz(h * h)
    var z = ftz(Float32(1.0) - ftz(ftz(x * x) / h_squared))
    if z < KDE_LOG_FLOOR:
        z = KDE_LOG_FLOOR
    return ftz(identical_log(z))


@always_inline
def exponential_log_kernel(x: Float32, h: Float32) -> Float32:
    """`:76-78`: `-x / h`."""
    return ftz((-x) / h)


@always_inline
def linear_log_kernel(x: Float32, h: Float32) -> Float32:
    """`:81-88`: `z = maximum(1 - x/h, 1e-30)`, then the bool products."""
    if x >= h:
        return kde_float32_min()
    var z = ftz(Float32(1.0) - ftz(x / h))
    if z < KDE_LOG_FLOOR:
        z = KDE_LOG_FLOOR
    return ftz(identical_log(z))


@always_inline
def cosine_log_kernel(x: Float32, h: Float32) -> Float32:
    """`:91-99`: `z = maximum(cos(0.5 * pi * x / h), 1e-30)`, then the
    bool products. `0.5 * np.pi` is a host float64 (1.5707963267948966)
    that the fused kernel takes as float32: 0x3FC90FDB."""
    if x >= h:
        return kde_float32_min()
    var half_pi = Float32(1.5707963267948966)
    var arg = ftz(ftz(half_pi * x) / h)
    var z = ftz(identical_cos(arg))
    if z < KDE_LOG_FLOOR:
        z = KDE_LOG_FLOOR
    return ftz(identical_log(z))


@always_inline
def compute_log_kernel(x: Float32, h: Float32, kernel: Int) -> Float32:
    """`log_probability_kernels_[kernel](distances, h)` (`:102-109`),
    one cell. An unknown kernel value writes NaN (the host raises before
    any launch; this is the device's refusal)."""
    if kernel == KDE_KERNEL_GAUSSIAN:
        return gaussian_log_kernel(x, h)
    if kernel == KDE_KERNEL_TOPHAT:
        return tophat_log_kernel(x, h)
    if kernel == KDE_KERNEL_EPANECHNIKOV:
        return epanechnikov_log_kernel(x, h)
    if kernel == KDE_KERNEL_EXPONENTIAL:
        return exponential_log_kernel(x, h)
    if kernel == KDE_KERNEL_LINEAR:
        return linear_log_kernel(x, h)
    if kernel == KDE_KERNEL_COSINE:
        return cosine_log_kernel(x, h)
    return bitcast[DType.float32](UInt32(0x7FC00000))


# ---------------------------------------------------------------------------
# Kernel norms, `kernel_density.py:112-141`. Both arms return the scalar
# `factor + d * log(h)` that `norm_log_probabilities` SUBTRACTS (`:141`).
# ---------------------------------------------------------------------------


# ============ DEVIATION 602 (2026-08-23): THE COSINE KERNEL'S NORM IS WRONG
# ============ UPSTREAM FOR EVEN d, AND IS NOT IMPLEMENTED AS WRITTEN ============
# THEIRS (`kernel_density.py:131-137`, copied from scikit-learn
# `_binary_tree.pxi:465-470`):
#
#     factor = 0; tmp = 2/pi
#     for k in range(1, d + 1, 2):
#         factor += tmp
#         tmp *= -(d - k) * (d - k - 1) * (2/pi)**2
#     factor = log(factor) + logSn(d - 1)
#
# which is meant to be `log(S_{d-1} * I_{d-1})` with `I_n = int_0^1 r^n
# cos(pi r / 2) dr`, the volume under the cosine kernel in d dimensions.
# Integrating by parts, `I_n = 2/pi - n(n-1)(2/pi)^2 I_{n-2}` for n >= 2,
# `I_0 = 2/pi`, and `I_1 = 2/pi - (2/pi)^2`. Their loop unrolls that
# recurrence and stops when `-(d-k)(d-k-1)` reaches 0, which is the
# `I_0 = 2/pi` base -- CORRECT for even n (odd d). For odd n (EVEN d) the
# chain should end at `I_1`, whose second term `-(2/pi)^2` the loop never
# adds: at d = 2 it returns log 4 for a true volume of `4 - 8/pi` (log
# 0.374, theirs 1.386, a density 2.75x too small); at d = 4 the truncated
# sum is NEGATIVE (-0.911) and `log` of it is NaN; at d = 6 it returns
# 5.517 for 0.114. Odd d agree to 1e-13.
#
# MEASURED 2026-08-23 by Simpson quadrature of `S_{d-1} I_{d-1}` against
# their loop, d = 1..6 (the table is in kde/README.md), and gated by
# `check_kde_log_norm_closed_form` at d = 2 (`log(4 - 8/pi) + 2 log h`)
# and d = 4 (`log(2 pi^2 (2/pi - 6(2/pi)^3 + 6(2/pi)^4)) + 4 log h`).
#
# OURS: `I_{d-1}` by its power series (see `_cosine_radial_integral_fast`
# for why not the corrected recurrence). ASSUME-OUR-CODE-IS-BROKEN's corollary is "do not implement their
# BUGS": a `cosine` KDE in 2 or 4 dimensions would otherwise be
# misnormalized or NaN by construction, and scikit-learn -- the oracle
# for semantics -- has the same defect, so agreement with it would be
# agreement about the wrong number. Consequence stated plainly: for EVEN
# d, `metric='euclidean', kernel='cosine'` here does NOT match
# `sklearn.neighbors.KernelDensity`; the difference is their bug and is
# the subject of a report owed upstream (README, HAND-OFF).
# ==========================================================================


def _cosine_radial_integral_fast(n: Int) -> Float64:
    """`I_n = int_0^1 r^n cos(pi r/2) dr` as the series `sum_k (-1)^k a^k /
    ((2k)! (n + 2k + 1))`, `a = (pi/2)^2`, float64.

    THE SERIES AND NOT THE RECURRENCE, MEASURED: the by-parts recurrence
    `I_m = 2/pi - m(m-1)(2/pi)^2 I_{m-2}` is what the upstream loop
    unrolls, and it CANCELS -- each step subtracts two terms near 0.6 to
    leave a result near 0.02 -- so in float32 it was off by 2.8e-3 at
    d = 9 (`check_kde_log_norm_closed_form` under IDENTICAL, 2026-08-23:
    got 4.09675, float64 4.09951). The series' terms fall like `a^k /
    (2k)!` from the first, so it loses nothing in either width; both arms
    use it so the two modes compute ONE formula."""
    var a = (Float64(pi) / 2.0) * (Float64(pi) / 2.0)
    var pw = 1.0
    var acc = 1.0 / Float64(n + 1)
    var k = 1
    while k <= 40:
        pw = pw * a / Float64((2 * k - 1) * (2 * k))
        var term = pw / Float64(n + 2 * k + 1)
        if k % 2 == 1:
            acc -= term
        else:
            acc += term
        if term < 1e-20:
            break
        k += 1
    return acc


def _cosine_radial_integral_identical(n: Int) -> Float32:
    """The same series over float32 through `ftz`; division is IEEE-correct
    on every column measured (IDENTITY_PATHS row 10), so this is one
    arithmetic everywhere. Stops when a term is below 2^-30 (every later
    term is smaller; 20 terms at most)."""
    var half_pi = Float32(1.5707963267948966)
    var a = ftz(half_pi * half_pi)
    var pw = Float32(1.0)
    var acc = ftz(Float32(1.0) / Float32(n + 1))
    var k = 1
    while k <= 20:
        pw = ftz(ftz(pw * a) / Float32((2 * k - 1) * (2 * k)))
        var term = ftz(pw / Float32(n + 2 * k + 1))
        if k % 2 == 1:
            acc = ftz(acc - term)
        else:
            acc = ftz(acc + term)
        if term < Float32(9.313225746154785e-10):
            break
        k += 1
    return acc


def log_kernel_norm_fast(kernel: Int, h: Float64, d: Int) raises -> Float32:
    """Their float64 host arithmetic, `std.math` for `np.log`/`math.lgamma`,
    transcribed line for line; cast to float32 once at the end."""
    var dd = Float64(d)
    var factor: Float64
    if kernel == KDE_KERNEL_GAUSSIAN:
        factor = 0.5 * dd * log(2.0 * Float64(pi))
    elif kernel == KDE_KERNEL_TOPHAT:
        factor = _log_vn_fast(d)
    elif kernel == KDE_KERNEL_EPANECHNIKOV:
        factor = _log_vn_fast(d) + log(2.0 / (dd + 2.0))
    elif kernel == KDE_KERNEL_EXPONENTIAL:
        factor = _log_sn_fast(d - 1) + lgamma(dd)
    elif kernel == KDE_KERNEL_LINEAR:
        factor = _log_vn_fast(d) - log(dd + 1.0)
    elif kernel == KDE_KERNEL_COSINE:
        # DEVIATION 602: NOT their loop (`:131-136`), which is wrong for
        # even d. The radial integral by its series; see the block above
        # `_cosine_radial_integral_fast`.
        factor = log(_cosine_radial_integral_fast(d - 1)) + _log_sn_fast(d - 1)
    else:
        raise Error("Unsupported kernel.")
    return Float32(factor + dd * log(h))


def _log_vn_fast(n: Int) -> Float64:
    """`logVn(n) = 0.5 * n * log(pi) - lgamma(0.5 * n + 1)` (`:112-113`)."""
    return 0.5 * Float64(n) * log(Float64(pi)) - lgamma(0.5 * Float64(n) + 1.0)


def _log_sn_fast(n: Int) -> Float64:
    """`logSn(n) = log(2 * pi) + logVn(n - 1)` (`:116-117`)."""
    return log(2.0 * Float64(pi)) + _log_vn_fast(n - 1)


def _lgamma_half_identical(two_x: Int) -> Float32:
    """`lgamma(two_x / 2)` for `two_x >= 1`, as an ascending float32 sum of
    `identical_log` over the Gamma recurrence (DEVIATION 601):
    `Gamma(k) = (k-1)!`; `Gamma(k + 1/2) = sqrt(pi) * prod_{i=1..k}(i-1/2)`.
    """
    var acc = Float32(0.0)
    if two_x % 2 == 0:
        var k = two_x // 2
        var i = 2
        while i <= k - 1:
            acc = ftz(acc + identical_log(Float32(i)))
            i += 1
        return acc
    var k = (two_x - 1) // 2
    acc = ftz(Float32(0.5) * identical_log(Float32(pi)))
    var i = 1
    while i <= k:
        acc = ftz(acc + identical_log(ftz(Float32(i) - Float32(0.5))))
        i += 1
    return acc


def _log_vn_identical(n: Int) -> Float32:
    """`0.5 * n * log(pi) - lgamma(0.5 * n + 1)`: the product and the
    subtraction as ONE `identical_mul_add`; `lgamma` at `(n + 2) / 2`."""
    var lg = _lgamma_half_identical(n + 2)
    return ftz(
        identical_mul_add(
            Float32(0.5) * Float32(n), identical_log(Float32(pi)), -lg
        )
    )


def _log_sn_identical(n: Int) -> Float32:
    return ftz(
        identical_log(ftz(Float32(2.0) * Float32(pi))) + _log_vn_identical(n - 1)
    )


def log_kernel_norm_identical(kernel: Int, h: Float32, d: Int) raises -> Float32:
    """DEVIATION 601: the same formulas over float32 through the portable
    helpers, every `a*b+c` an `identical_mul_add`, the cosine loop's
    products in their order."""
    var d32 = Float32(d)
    var factor: Float32
    if kernel == KDE_KERNEL_GAUSSIAN:
        factor = ftz(
            ftz(Float32(0.5) * d32) * identical_log(ftz(Float32(2.0) * Float32(pi)))
        )
    elif kernel == KDE_KERNEL_TOPHAT:
        factor = _log_vn_identical(d)
    elif kernel == KDE_KERNEL_EPANECHNIKOV:
        factor = ftz(
            _log_vn_identical(d)
            + identical_log(ftz(Float32(2.0) / ftz(d32 + Float32(2.0))))
        )
    elif kernel == KDE_KERNEL_EXPONENTIAL:
        factor = ftz(_log_sn_identical(d - 1) + _lgamma_half_identical(2 * d))
    elif kernel == KDE_KERNEL_LINEAR:
        factor = ftz(
            _log_vn_identical(d) - identical_log(ftz(d32 + Float32(1.0)))
        )
    elif kernel == KDE_KERNEL_COSINE:
        # DEVIATION 602, the float32 arm of the same series.
        factor = ftz(
            identical_log(_cosine_radial_integral_identical(d - 1))
            + _log_sn_identical(d - 1)
        )
    else:
        raise Error("Unsupported kernel.")
    return ftz(identical_mul_add(d32, identical_log(h), factor))


def log_kernel_norm(kernel: Int, h: Float32, d: Int) raises -> Float32:
    """The mode dispatch: FAST is theirs (float64 host libm, cast once),
    IDENTICAL is DEVIATION 601's construction."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return log_kernel_norm_identical(kernel, h, d)
    return log_kernel_norm_fast(kernel, Float64(h), d)


# ---------------------------------------------------------------------------
# The device kernels of `score_samples`, `:264-363`, one per cupy/numba
# step they run.
# ---------------------------------------------------------------------------


def log_kernel_matrix_kernel(
    logk: MutPointer[Float32, MutAnyOrigin],
    dist: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
    h: Float32,
    kernel_in: Int32,
):
    """`distances = log_probability_kernels_[self.kernel](distances, h)`
    (`:331-334`): the fused elementwise kernel, one thread per cell, no
    cross-thread combination anywhere. `logk` is a separate buffer so the
    card can hold both the distances and their log-kernels (their `cp.fuse`
    returns a new array too)."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(cells_in):
        return
    logk.unsafe_store(
        idx, compute_log_kernel(ftz(dist.unsafe_load(idx)), h, Int(kernel_in))
    )


def log_weights_kernel(
    logw: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`cp.log(self.sample_weight_)` (`:338`), on the device as theirs is."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        logw.unsafe_store(i, ftz(identical_log(ftz(w.unsafe_load(i)))))


def add_log_weights_kernel(
    logk: MutPointer[Float32, MutAnyOrigin],
    logw: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
    n_train_in: Int32,
):
    """`distances += cp.log(self.sample_weight_)` (`:338`): the row-broadcast
    in-place add, one thread per cell."""
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(cells_in):
        return
    var j = idx % Int(n_train_in)
    logk.unsafe_store(
        idx, ftz(logk.unsafe_load(idx) + logw.unsafe_load(j))
    )


def logsumexp_kernel(
    logk: MutPointer[Float32, MutAnyOrigin],
    lse: MutPointer[Float32, MutAnyOrigin],
    rowmax: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    n_train_in: Int32,
):
    """`logsumexp_kernel` (`:144-156`), their numba kernel line for line:
    ONE THREAD PER QUERY ROW, the max then the sum, each a serial ascending
    walk over `j`.

    THIS IS THE FIXED-ORDER REDUCTION. The order is `j = 0, 1, ..., n_train
    - 1`, a pure function of `n_train` alone: not of the block size, the
    grid, the lane width, or which other queries share the launch. There is
    no block fold, no warp primitive and no atomic to pin because their
    kernel has none; the brief's alternative (a `pinned_block_sum` tree plus
    a cross-block fold) was not taken because it is a different summation
    order from the one upstream ships and COPY-DO-NOT-IMPROVE decides it.
    `rowmax` is this lane's addition for the card (`kde.rowmax`); their
    kernel keeps `max_exp` in a register.

    ROW 13, STATED: the max is their strict `>` from `distances[i, 0]`, so
    among EQUAL values the FIRST in ascending `j` survives; `-0.0` and
    `+0.0` compare equal, so which zero survives is decided by position,
    which is fixed, so it is deterministic -- and it CANNOT reach the
    output bits: `exp(v - max)` with `v`, `max` in {-0.0, +0.0} is
    `exp(+/-0.0) = 1.0` exactly, and `log(sum) + max` differs between the
    two zeros only if `log(sum)` is itself a zero, i.e. `sum == 1`, where
    `identical_log(1.0)` is `+0.0` and `+0.0 + (-0.0) = +0.0 + (+0.0) =
    +0.0`. `check_kde_zero_sign_cannot_leak` measures both facts.

    ROW 12: `exp` and `log` through `identical_exp`/`identical_log`.
    ROW 10: the partial sum and the result stored through `ftz`. `exp` of
    the `FLOAT_MIN - max` gap underflows to exactly `0.0`.

    ROW 39 (2026-08-23, the signed-zero audit): `rowmax` is a RECORDED
    stage, so the sign bit of a zero max IS certified, and the fold below
    is the one place in this lane a `+0.0` and a `-0.0` could be the two
    candidates. It is NOT a hardware `max` (whose answer on (+0, -0) is
    the vendor's: -0 on Apple, +0 on NVIDIA/AMD): it is their strict `>`
    from `j = 0`, so on a tie the LOWER index survives -- decided by
    position in a serial walk, the same answer on every vendor, and a NaN
    candidate (`>` false both ways) never displaces a non-NaN seed.
    Reachability: no LEGAL row mixes the two zeros at all. Unweighted,
    gaussian/exponential/tophat produce only `-0.0` (`-(+0)/h2`, `-0/h`,
    `0 * FLOAT_MIN`; a negative subnormal flushes to `-0.0`, never `+0.0`)
    and epanechnikov/linear/cosine only `+0.0` (`identical_log(1.0)`,
    gated `+0.0`; `log(z)` for `z < 1` is at least 2^-24 in magnitude, never
    a zero). Weighted, every cell is `logk + log(w)`: `-0.0 + (+0.0)` is
    `+0.0`, a nonzero `log(w)` is at least 2^-24 in magnitude, and a
    difference of two floats that large is zero or at least 2^-47 (never a
    subnormal), so a weighted row's zeros are all `+0.0`. The positional
    rule is therefore never exercised by real input; `kde/checks/
    kde_check.mojo::check_kde_row39_signed_zero_rowmax` PLANTS mixed rows
    into this kernel (both orders) and asserts the lower-index zero's bits.

    ============ DEVIATION 603 (2026-08-23): A ROW OF ALL -inf IS -inf, NOT
    ============ NaN ====================================================
    THEIRS (`logsumexp_kernel:144-156`): `max_exp = -inf` when every cell
    is `-inf` (gaussian/exponential with `x*x/(2 h^2)` or `x/h`
    overflowed -- legal finite input, e.g. h = 2^-62 and points 3 apart),
    then `math.exp(-inf - (-inf))` = `exp(NaN)` = NaN, so `log(sum) + max`
    is NaN and the score is NaN. scikit-learn (the semantics oracle)
    folds the same row with `logaddexp`, whose `(-inf, -inf)` is `-inf`.
    OURS: `if max_exp == -inf: lse = -inf` before the sum -- the value
    the mathematics and sklearn give. WHY: IDENTITY_PATHS row 39's FACT
    2 -- a COMPUTED NaN carries the vendor's payload (Apple 0x7fc00000,
    NVIDIA 0x7fffffff, AMD 0xffc00000) and can never sit in a certified
    stage; `kde.logsumexp` and `kde.scores` are certified. MEASURED:
    `check_kde_nan_cannot_reach_a_stage` plants the row and asserts
    `0xFF800000` at both stages on the device and the oracle, and the
    sabotage (guard dropped) is recorded in the README. The mixed row
    (some `-inf`, a finite max) needed no change: `exp(-inf - max)` is
    `+0.0`. Every other NaN a legal input could compute is REFUSED BY
    NAME before any launch (DEVIATION 604, `kde_fit_validate` /
    `kde_validate_data`).
    ======================================================================
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_query = Int(n_query_in)
    var n_train = Int(n_train_in)
    if i >= n_query:
        return
    var base = i * n_train
    var max_exp = logk.unsafe_load(base)
    for j in range(1, n_train):
        var v = logk.unsafe_load(base + j)
        # ROW 39: strict `>`, lower index wins a tie; NOT a hardware max.
        if v > max_exp:
            max_exp = v
    rowmax.unsafe_store(i, max_exp)
    # DEVIATION 603: every cell -inf -> the log-sum-exp is -inf.
    if max_exp == bitcast[DType.float32](UInt32(0xFF800000)):
        lse.unsafe_store(i, max_exp)
        return
    var s = Float32(0.0)
    for j in range(0, n_train):
        s = ftz(s + ftz(identical_exp(ftz(logk.unsafe_load(base + j) - max_exp))))
    lse.unsafe_store(i, ftz(identical_log(s) + max_exp))


def normalize_scores_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    lse: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    log_sum_weights: Float32,
    norm: Float32,
):
    """`log_probabilities -= np.log(sum_weights)` (`:343`) then
    `log_probabilities - (factor + d * np.log(h))` (`:141`): two cupy
    elementwise ops, two roundings, in that order, one thread per row."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_query_in):
        return
    var a = ftz(lse.unsafe_load(i) - log_sum_weights)
    scores.unsafe_store(i, ftz(a - norm))


# ---------------------------------------------------------------------------
# The host flow: `fit`'s validation and `score_samples`'s sequence.
# ---------------------------------------------------------------------------


def kde_fit_validate(
    n_train: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: Int,
    metric: Int,
    weights: List[Float32],
    has_weights: Bool,
) raises:
    """`KernelDensity.__init__:211-214` and `fit:240-252`: `bandwidth > 0`,
    a valid kernel, a valid metric, `sample_weight.min() > 0` when given
    and its length `n_train`. Every refusal names the parameter. Plus
    DEVIATION 604's rules (3) and (4): `bandwidth >= 2^-63`, weights
    normal and finite; the data rules (1) and (2) are `kde_validate_data`."""
    if not (bandwidth > Float32(0.0)):
        raise Error("bandwidth must be positive")
    # DEVIATION 604: below 2^-63 the kernels' `h*h` and `2*h*h` underflow
    # float32 (to zero on an FTZ column), and `-(0)/0` at a coincident
    # query is NaN.
    if bandwidth < KDE_MIN_BANDWIDTH:
        raise Error(
            "bandwidth=" + String(bandwidth)
            + " is below 2^-63: h*h underflows float32 (DEVIATION 604)"
        )
    if kernel < 0 or kernel >= KDE_N_KERNELS:
        raise Error("invalid kernel: value " + String(kernel))
    if (
        metric != DIST_L2_SQRT_UNEXPANDED
        and metric != DIST_L2_EXPANDED
        and metric != DIST_L1
        and metric != DIST_LINF
        and metric != DIST_COSINE_EXPANDED
        and metric != DIST_LP_UNEXPANDED
    ):
        raise Error("kde: metric value " + String(metric) + " is not implemented")
    if n_train <= 0:
        raise Error("kde: X must have at least one row (n_train)")
    if n_features <= 0:
        raise Error("kde: X must have at least one column (n_features)")
    if has_weights:
        if len(weights) != n_train:
            raise Error(
                "sample_weight: expected "
                + String(n_train)
                + " values, got "
                + String(len(weights))
            )
        for i in range(n_train):
            if not (weights[i] > Float32(0.0)):
                raise Error("sample_weight must have positive values")
            # DEVIATION 604: `log(w)` must be finite on every column. A
            # subnormal weight flushes to zero on an FTZ column (`log` is
            # -inf, a whole row can become -inf); an infinite one makes
            # `log(w) = inf` and `inf - inf = NaN` in the logsumexp.
            if weights[i] < KDE_FLOAT32_MIN_NORMAL or weights[i] == kde_inf():
                raise Error(
                    "sample_weight[" + String(i) + "]=" + String(weights[i])
                    + " is subnormal or infinite: log(w) is not finite in"
                    " float32 (DEVIATION 604)"
                )


def host_sum_weights(weights: List[Float32]) -> Float32:
    """`cp.sum(self.sample_weight_)` (`:348`): cupy's device tree, order
    cupy's. Ours is a SERIAL ASCENDING host fold through `ftz` so the order
    is a function of `n_train` alone (the same reason the logsumexp is
    serial); its `log` is taken by the caller through `identical_log`."""
    var s = Float32(0.0)
    for i in range(len(weights)):
        s = ftz(s + weights[i])
    return s



# ---------------------------------------------------------------------------
# DEVIATION 2490 (2026-09-10): THE FUSED FAST SCORE PASS
# ---------------------------------------------------------------------------
# The staged path above materializes TWO `n_query x n_train` float32
# matrices (the distances, then the log-kernel values) and walks the second
# one twice from global memory, one thread per query. That is 16 bytes of
# device traffic per cell for arithmetic that needs none, and it caps the
# problem at the matrices' size (2 x 4 bytes x cells: 16k x 16k already
# holds 2 GB, 100k x 100k cannot be allocated at all). It is the shape
# upstream ships (`kernel_density.py:332-342`, cupy matrices between numba
# kernels), and IDENTICAL keeps it because the card certifies each stage
# (`kde.dists`, `kde.logk`, `kde.rowmax`) and its serial ascending fold.
#
# FAST takes ONE kernel instead. Each thread owns one query row; the block
# streams the training set through shared memory in tiles; per cell the
# thread forms the distance, the log-kernel value, the log-weight and folds
# it into an ONLINE log-sum-exp (running max `m`, running sum `s` of
# `exp(v - m)`, rescaled by `exp(m_old - m_new)` when the max moves).
# Nothing is written per cell. Memory is O(n_query + n_train). The
# per-query fold still walks `j` ascending, so a query's score does not
# depend on which other queries share the launch (the launch-invariance
# gate's case D) -- but the fold ORDER is not the two-pass serial one and
# the bits differ from the staged path, which is exactly what FAST may do
# and IDENTICAL may not (`checks/numerics.mojo`). This arm is taken ONLY
# when `GLOBAL_NUMERIC_MODE == NUMERIC_FAST` AND no identity trace is
# recording (a trace asks for the staged card, so it gets the stages).
#
# Semantics kept from the staged path: `-inf` cells (an overflowed
# gaussian/exponential exponent) fold to `-inf` when every cell is `-inf`
# (DEVIATION 603) and contribute `exp(-inf) = 0` otherwise; tophat's
# FLOAT_MIN "log zero" behaves as in the two-pass fold (a row of all
# FLOAT_MIN scores FLOAT_MIN, since `log(n) + FLOAT_MIN == FLOAT_MIN` in
# float32); cosine's zero-norm refusal happens on the host before any
# launch, as before. Gaussian over euclidean skips the sqrt: the kernel
# wants `x * x`, which IS the summed square.
#
# Scheduling constants. 128 queries per block. The train tile is
# `KDE_FUSED_TILE_FLOATS` floats of shared memory (12 KB) holding
# `rows = min(KDE_FUSED_TILE_FLOATS // d, KDE_FUSED_TILE_ROWS_MAX)` rows,
# plus `KDE_FUSED_TILE_ROWS_MAX` floats of log-weights (4 KB): 16 KB,
# inside Apple's 32 KB threadgroup limit with room for the compiler. A
# query row with `d <= KDE_FUSED_QREG` lives in registers (the feature
# loop is unrolled at compile time and predicated on `d`); wider rows are
# re-read from global memory per cell, which is correct and slower. When
# `d > KDE_FUSED_TILE_FLOATS` the tile cannot hold one row and the staged
# path runs instead.
comptime KDE_FUSED_TPB = 128
comptime KDE_FUSED_TILE_FLOATS = 3072
comptime KDE_FUSED_TILE_ROWS_MAX = 1024
comptime KDE_FUSED_QREG = 64


@always_inline
def _kde_fused_cell_distance[DPAD: Int, qo: MutOrigin, to: MutOrigin](
    qreg: MutPointer[Float32, qo],
    query: MutPointer[Float32, MutAnyOrigin],
    tile: MutPointer[Float32, to, address_space = AddressSpace.SHARED],
    qbase: Int,
    tb: Int,
    d: Int,
    metric: Int,
    metric_arg: Float32,
    mut tnorm: Float32,
) -> Float32:
    """The per-cell distance accumulator: summed squares (both L2 spellings),
    summed |diff| (L1), max |diff| (Linf), the dot product with the train
    row's summed squares in `tnorm` (cosine), or summed |diff|^p (Lp).
    `DPAD > 0` walks a compile-time-unrolled, zero-padded register row;
    `DPAD == 0` is the wide-row fallback that re-reads the query from global
    memory over the runtime `d`."""
    from std.math import exp, log

    var acc = Float32(0.0)
    comptime if DPAD > 0:
        if metric == DIST_L2_SQRT_UNEXPANDED or metric == DIST_L2_EXPANDED:
            comptime for f in range(DPAD):
                var diff = qreg.unsafe_load(f) - tile.unsafe_load(tb + f)
                acc += diff * diff
        elif metric == DIST_L1:
            comptime for f in range(DPAD):
                acc += abs(qreg.unsafe_load(f) - tile.unsafe_load(tb + f))
        elif metric == DIST_LINF:
            comptime for f in range(DPAD):
                var a = abs(qreg.unsafe_load(f) - tile.unsafe_load(tb + f))
                if a > acc:
                    acc = a
        elif metric == DIST_COSINE_EXPANDED:
            comptime for f in range(DPAD):
                var t = tile.unsafe_load(tb + f)
                acc += qreg.unsafe_load(f) * t
                tnorm += t * t
        else:
            comptime for f in range(DPAD):
                var a = abs(qreg.unsafe_load(f) - tile.unsafe_load(tb + f))
                if a > Float32(0.0):
                    acc += exp(metric_arg * log(a))
    else:
        if metric == DIST_L2_SQRT_UNEXPANDED or metric == DIST_L2_EXPANDED:
            for f in range(d):
                var diff = query.unsafe_load(qbase + f) - tile.unsafe_load(tb + f)
                acc += diff * diff
        elif metric == DIST_L1:
            for f in range(d):
                acc += abs(query.unsafe_load(qbase + f) - tile.unsafe_load(tb + f))
        elif metric == DIST_LINF:
            for f in range(d):
                var a = abs(query.unsafe_load(qbase + f) - tile.unsafe_load(tb + f))
                if a > acc:
                    acc = a
        elif metric == DIST_COSINE_EXPANDED:
            for f in range(d):
                var t = tile.unsafe_load(tb + f)
                acc += query.unsafe_load(qbase + f) * t
                tnorm += t * t
        else:
            for f in range(d):
                var a = abs(query.unsafe_load(qbase + f) - tile.unsafe_load(tb + f))
                if a > Float32(0.0):
                    acc += exp(metric_arg * log(a))
    return acc


def kde_fused_logsumexp_kernel[DPAD: Int](
    lse: MutPointer[Float32, MutAnyOrigin],
    query: MutPointer[Float32, MutAnyOrigin],
    train: MutPointer[Float32, MutAnyOrigin],
    logw: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    n_train_in: Int32,
    d_in: Int32,
    has_weights_in: Int32,
    bandwidth: Float32,
    kernel_in: Int32,
    metric_in: Int32,
    metric_arg: Float32,
):
    """DEVIATION 2490: distance + log-kernel + log-weight + online
    log-sum-exp in one pass, one thread per query, train tiles in shared
    memory. FAST only; see the block comment above.

    `DPAD` is the feature count rounded up to a multiple of 4 when
    `d <= KDE_FUSED_QREG` (the query row sits in `DPAD` registers, the tile
    rows are stored at stride `DPAD` with zero padding, and the feature
    loop is unrolled with no predicate), or 0 for the wide-row fallback
    (tile stride `d`, runtime feature loop, query re-read from global
    memory). A zero pad lane contributes `0 - 0` to every metric, so it
    changes no distance."""
    from std.math import exp, log, sqrt
    comptime IN_REGS = DPAD > 0

    var n_query = Int(n_query_in)
    var n_train = Int(n_train_in)
    var d = Int(d_in)
    var has_weights = Int(has_weights_in) != 0
    var kernel = Int(kernel_in)
    var metric = Int(metric_in)
    var tid = Int(thread_idx.x)
    var q = Int(block_idx.x) * KDE_FUSED_TPB + tid
    var active = q < n_query

    var tile = stack_allocation[
        KDE_FUSED_TILE_FLOATS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tile_w = stack_allocation[
        KDE_FUSED_TILE_ROWS_MAX,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var qreg = stack_allocation[
        DPAD if IN_REGS else 1, Scalar[DType.float32]
    ]()

    # Tile row stride and rows per tile.
    var dp: Int
    comptime if IN_REGS:
        dp = DPAD
    else:
        dp = d
    var rows = KDE_FUSED_TILE_FLOATS // dp
    if rows > KDE_FUSED_TILE_ROWS_MAX:
        rows = KDE_FUSED_TILE_ROWS_MAX

    # The query row, once. Inactive threads hold zeros and never store.
    var qbase = q * d
    var qnorm = Float32(0.0)
    comptime if IN_REGS:
        comptime for f in range(DPAD):
            var v = Float32(0.0)
            if active and f < d:
                v = query.unsafe_load(qbase + f)
            qreg.unsafe_store(f, v)
            qnorm += v * v
    else:
        if active:
            for f in range(d):
                var v = query.unsafe_load(qbase + f)
                qnorm += v * v
    qnorm = sqrt(qnorm)

    var neg_inv_2h2 = Float32(-1.0) / (Float32(2.0) * bandwidth * bandwidth)
    var one_over_p = Float32(1.0) / metric_arg
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var m = neg_inf
    var s = Float32(0.0)

    var t0 = 0
    while t0 < n_train:
        var rows_here = n_train - t0
        if rows_here > rows:
            rows_here = rows
        barrier()
        var total = rows_here * dp
        var idx = tid
        while idx < total:
            var v = Float32(0.0)
            comptime if IN_REGS:
                var r = idx // DPAD
                var f = idx - r * DPAD
                if f < d:
                    v = train.unsafe_load((t0 + r) * d + f)
            else:
                v = train.unsafe_load(t0 * d + idx)
            tile.unsafe_store(idx, v)
            idx += KDE_FUSED_TPB
        if has_weights:
            idx = tid
            while idx < rows_here:
                tile_w.unsafe_store(idx, logw.unsafe_load(t0 + idx))
                idx += KDE_FUSED_TPB
        barrier()
        if active:
            for j in range(rows_here):
                var tnorm = Float32(0.0)
                var acc = _kde_fused_cell_distance[DPAD](
                    qreg, query, tile, qbase, j * dp, d, metric, metric_arg, tnorm
                )
                # The log-kernel value of this cell.
                var v: Float32
                if kernel == KDE_KERNEL_GAUSSIAN and metric == DIST_L2_SQRT_UNEXPANDED:
                    v = acc * neg_inv_2h2
                else:
                    var x: Float32
                    if metric == DIST_L2_SQRT_UNEXPANDED:
                        x = sqrt(acc)
                    elif metric == DIST_COSINE_EXPANDED:
                        x = Float32(1.0) - acc / (qnorm * sqrt(tnorm))
                    elif metric == DIST_LP_UNEXPANDED:
                        x = Float32(0.0)
                        if acc > Float32(0.0):
                            x = exp(one_over_p * log(acc))
                    else:
                        x = acc
                    v = compute_log_kernel(x, bandwidth, kernel)
                if has_weights:
                    v = v + tile_w.unsafe_load(j)

                # Online log-sum-exp. Natural base on purpose: a base-2
                # rescale of the FLOAT_MIN "log zero" overflows to -inf and
                # a row of all-out-of-range cells would score -inf instead of
                # the FLOAT_MIN sentinel the staged path and cuML give.
                if v > m:
                    s = s * exp(m - v) + Float32(1.0)
                    m = v
                elif v == m:
                    s += Float32(1.0)
                else:
                    s += exp(v - m)
        t0 += rows
    if active:
        if m == neg_inf:
            lse.unsafe_store(q, neg_inf)
        else:
            lse.unsafe_store(q, log(s) + m)



comptime KDE_FUSED_WIDE_ROWS = KDE_FUSED_TILE_FLOATS // (KDE_FUSED_QREG + 1)


def kde_fused_wide_kernel(
    lse: MutPointer[Float32, MutAnyOrigin],
    query: MutPointer[Float32, MutAnyOrigin],
    train: MutPointer[Float32, MutAnyOrigin],
    logw: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    n_train_in: Int32,
    d_in: Int32,
    has_weights_in: Int32,
    bandwidth: Float32,
    kernel_in: Int32,
    metric_in: Int32,
    metric_arg: Float32,
):
    """DEVIATION 2490's wide-row arm, `d > KDE_FUSED_QREG`. Same fold as
    `kde_fused_logsumexp_kernel`, but the feature axis is walked in chunks
    of `KDE_FUSED_QREG` registers OUTSIDE the row loop: one chunk of the
    query row is loaded, every tile row's partial distance is advanced in a
    per-row accumulator, then the next chunk. Every metric here is a fold
    over features (sum, max, or dot with a norm), so the chunk order
    changes only the association. Tile rows per pass are
    `KDE_FUSED_TILE_FLOATS // d` (at most `KDE_FUSED_WIDE_ROWS`), which is
    also the accumulator count."""
    from std.math import exp, log, sqrt
    comptime Q = KDE_FUSED_QREG

    var n_query = Int(n_query_in)
    var n_train = Int(n_train_in)
    var d = Int(d_in)
    var has_weights = Int(has_weights_in) != 0
    var kernel = Int(kernel_in)
    var metric = Int(metric_in)
    var tid = Int(thread_idx.x)
    var q = Int(block_idx.x) * KDE_FUSED_TPB + tid
    var active = q < n_query

    var tile = stack_allocation[
        KDE_FUSED_TILE_FLOATS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tile_w = stack_allocation[
        KDE_FUSED_WIDE_ROWS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var qreg = stack_allocation[Q, Scalar[DType.float32]]()
    var acc = stack_allocation[KDE_FUSED_WIDE_ROWS, Scalar[DType.float32]]()
    var tn = stack_allocation[KDE_FUSED_WIDE_ROWS, Scalar[DType.float32]]()

    var rows = KDE_FUSED_TILE_FLOATS // d
    if rows > KDE_FUSED_WIDE_ROWS:
        rows = KDE_FUSED_WIDE_ROWS
    var qbase = q * d
    var qnorm = Float32(0.0)
    if active:
        for f in range(d):
            var v = query.unsafe_load(qbase + f)
            qnorm += v * v
    qnorm = sqrt(qnorm)

    var neg_inv_2h2 = Float32(-1.0) / (Float32(2.0) * bandwidth * bandwidth)
    var one_over_p = Float32(1.0) / metric_arg
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var m = neg_inf
    var s = Float32(0.0)
    var is_l2 = metric == DIST_L2_SQRT_UNEXPANDED or metric == DIST_L2_EXPANDED

    var t0 = 0
    while t0 < n_train:
        var rows_here = n_train - t0
        if rows_here > rows:
            rows_here = rows
        barrier()
        var total = rows_here * d
        var idx = tid
        while idx < total:
            tile.unsafe_store(idx, train.unsafe_load(t0 * d + idx))
            idx += KDE_FUSED_TPB
        if has_weights:
            idx = tid
            while idx < rows_here:
                tile_w.unsafe_store(idx, logw.unsafe_load(t0 + idx))
                idx += KDE_FUSED_TPB
        barrier()
        if active:
            for j in range(rows_here):
                acc.unsafe_store(j, Float32(0.0))
                tn.unsafe_store(j, Float32(0.0))
            var c0 = 0
            while c0 < d:
                var width = d - c0
                if width > Q:
                    width = Q
                comptime for f in range(Q):
                    var v = Float32(0.0)
                    if f < width:
                        v = query.unsafe_load(qbase + c0 + f)
                    qreg.unsafe_store(f, v)
                for j in range(rows_here):
                    var tb = j * d + c0
                    var a = acc.unsafe_load(j)
                    if is_l2:
                        comptime for f in range(Q):
                            if f < width:
                                var diff = qreg.unsafe_load(f) - tile.unsafe_load(tb + f)
                                a += diff * diff
                    elif metric == DIST_L1:
                        comptime for f in range(Q):
                            if f < width:
                                a += abs(qreg.unsafe_load(f) - tile.unsafe_load(tb + f))
                    elif metric == DIST_LINF:
                        comptime for f in range(Q):
                            if f < width:
                                var x = abs(qreg.unsafe_load(f) - tile.unsafe_load(tb + f))
                                if x > a:
                                    a = x
                    elif metric == DIST_COSINE_EXPANDED:
                        var t2 = tn.unsafe_load(j)
                        comptime for f in range(Q):
                            if f < width:
                                var t = tile.unsafe_load(tb + f)
                                a += qreg.unsafe_load(f) * t
                                t2 += t * t
                        tn.unsafe_store(j, t2)
                    else:
                        comptime for f in range(Q):
                            if f < width:
                                var x = abs(qreg.unsafe_load(f) - tile.unsafe_load(tb + f))
                                if x > Float32(0.0):
                                    a += exp(metric_arg * log(x))
                    acc.unsafe_store(j, a)
                c0 += Q
            for j in range(rows_here):
                var a = acc.unsafe_load(j)
                var v: Float32
                if kernel == KDE_KERNEL_GAUSSIAN and metric == DIST_L2_SQRT_UNEXPANDED:
                    v = a * neg_inv_2h2
                else:
                    var x: Float32
                    if metric == DIST_L2_SQRT_UNEXPANDED:
                        x = sqrt(a)
                    elif metric == DIST_COSINE_EXPANDED:
                        x = Float32(1.0) - a / (qnorm * sqrt(tn.unsafe_load(j)))
                    elif metric == DIST_LP_UNEXPANDED:
                        x = Float32(0.0)
                        if a > Float32(0.0):
                            x = exp(one_over_p * log(a))
                    else:
                        x = a
                    v = compute_log_kernel(x, bandwidth, kernel)
                if has_weights:
                    v = v + tile_w.unsafe_load(j)
                if v > m:
                    s = s * exp(m - v) + Float32(1.0)
                    m = v
                elif v == m:
                    s += Float32(1.0)
                else:
                    s += exp(v - m)
        t0 += rows
    if active:
        if m == neg_inf:
            lse.unsafe_store(q, neg_inf)
        else:
            lse.unsafe_store(q, log(s) + m)


def _kde_fused_launch[DPAD: Int](
    ctx: DeviceContext,
    mut lse: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut train: DeviceBuffer[DType.float32],
    mut logw: DeviceBuffer[DType.float32],
    n_query: Int,
    n_train: Int,
    n_features: Int,
    has_weights: Bool,
    bandwidth: Float32,
    kernel: Int,
    metric: Int,
    metric_arg: Float32,
) raises:
    comptime if DPAD == 0:
        ctx.enqueue_function[kde_fused_wide_kernel](
            lse.unsafe_ptr(),
            query.unsafe_ptr(),
            train.unsafe_ptr(),
            logw.unsafe_ptr(),
            Int32(n_query),
            Int32(n_train),
            Int32(n_features),
            Int32(1 if has_weights else 0),
            bandwidth,
            Int32(kernel),
            Int32(metric),
            metric_arg,
            grid_dim=((n_query + KDE_FUSED_TPB - 1) // KDE_FUSED_TPB, 1, 1),
            block_dim=(KDE_FUSED_TPB, 1, 1),
        )
        return
    ctx.enqueue_function[kde_fused_logsumexp_kernel[DPAD]](
        lse.unsafe_ptr(),
        query.unsafe_ptr(),
        train.unsafe_ptr(),
        logw.unsafe_ptr(),
        Int32(n_query),
        Int32(n_train),
        Int32(n_features),
        Int32(1 if has_weights else 0),
        bandwidth,
        Int32(kernel),
        Int32(metric),
        metric_arg,
        grid_dim=((n_query + KDE_FUSED_TPB - 1) // KDE_FUSED_TPB, 1, 1),
        block_dim=(KDE_FUSED_TPB, 1, 1),
    )


def _kde_score_samples_fused(
    ctx: DeviceContext,
    mut train: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    sum_weights: Float32,
    n_train: Int,
    n_query: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: Int,
    metric: Int,
    metric_arg: Float32,
    mut scores: DeviceBuffer[DType.float32],
    elem_tpb: Int,
) raises:
    """DEVIATION 2490's host side: log-weights (if any), the fused pass,
    the normalization. Two or three launches, one drain."""
    if n_train <= 0 or n_features <= 0:
        raise Error(
            "kde: n_train and n_features must be positive, got "
            + String(n_train) + ", " + String(n_features)
        )
    # DEVIATION 552: Minkowski's p refused by value before any launch; the
    # staged path does this inside `pairwise_distance`, which the fused
    # pass does not call.
    validate_metric_arg(metric, metric_arg)
    var lse = ctx.enqueue_create_buffer[DType.float32](n_query)
    var logw: DeviceBuffer[DType.float32]
    if has_weights:
        logw = ctx.enqueue_create_buffer[DType.float32](n_train)
        ctx.enqueue_function[log_weights_kernel](
            logw.unsafe_ptr(),
            weights.unsafe_ptr(),
            Int32(n_train),
            grid_dim=((n_train + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    else:
        logw = ctx.enqueue_create_buffer[DType.float32](1)
    # The register row: d rounded up to a multiple of 4, up to KDE_FUSED_QREG.
    var dpad = ((n_features + 3) // 4) * 4
    if dpad == 4:
        _kde_fused_launch[4](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 8:
        _kde_fused_launch[8](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 12:
        _kde_fused_launch[12](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 16:
        _kde_fused_launch[16](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 20:
        _kde_fused_launch[20](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 24:
        _kde_fused_launch[24](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 28:
        _kde_fused_launch[28](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 32:
        _kde_fused_launch[32](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 36:
        _kde_fused_launch[36](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 40:
        _kde_fused_launch[40](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 44:
        _kde_fused_launch[44](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 48:
        _kde_fused_launch[48](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 52:
        _kde_fused_launch[52](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 56:
        _kde_fused_launch[56](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 60:
        _kde_fused_launch[60](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    elif dpad == 64:
        _kde_fused_launch[64](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    else:
        _kde_fused_launch[0](ctx, lse, query, train, logw, n_query, n_train, n_features, has_weights, bandwidth, kernel, metric, metric_arg)
    var log_sw = ftz(identical_log(sum_weights))
    var norm = log_kernel_norm(kernel, bandwidth, n_features)
    ctx.enqueue_function[normalize_scores_kernel](
        scores.unsafe_ptr(),
        lse.unsafe_ptr(),
        Int32(n_query),
        log_sw,
        norm,
        grid_dim=((n_query + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.synchronize()
    _ = lse^
    _ = logw^


# ---------------------------------------------------------------------------
# DEVIATION 2625 (2026-09-11): THE TILED IDENTICAL SCORE PASS, SAME BITS
# ---------------------------------------------------------------------------
# THEIRS (cuVS 26.08 `cpp/src/distance/kde.cu:335-441`, `kde_tiled_kernel`):
# one thread per query, the training rows cooperatively loaded into shared
# memory in tiles of `feat_tile x CELL_TILE` (64 x 64), a per-train-row
# accumulator in registers that persists across the feature tiles, and a 2D
# grid (query blocks x train chunks, `:594-603`) so a small query count
# still fills the GPU; the log-kernel is folded into an online log-sum-exp
# and a reduce kernel merges the per-chunk partials (`:447-473`).
#
# OURS BEFORE (the staged path below): the distance matrix one thread per
# CELL, each thread re-reading its query row and its train row from global
# memory (`2 x d` loads per cell: 440 per cell on Istella-S, 176 GB of
# global traffic for 100,000 x 2,000), then a second matrix for the
# log-kernel, then one thread per query ROW walking 100,000 cells twice with
# `identical_exp` in the inner loop, so the busiest stage ran on
# `n_query / lse_tpb` blocks (16 on the lane's shape).
#
# OURS NOW, under IDENTICAL with no identity trace recording and one of the
# three unexpanded metrics (euclidean, l1, chebyshev):
#   1. `kde_tiled_logk_kernel`: THEIR tile geometry and 2D grid, but every
#      cell's arithmetic is the staged path's, operation for operation:
#      `l2_unexp_core` / `l1_core` / `linf_core` over the features in
#      ascending order (the feature tiles are walked in order and a cell's
#      accumulator persists across them), `identical_sqrt` for euclidean,
#      `compute_log_kernel(ftz(dist))`, `ftz(logk + logw[j])` when weighted.
#      `ftz` of a training value moves from the use to the tile load; `ftz`
#      is idempotent, so no bit moves. The log-kernel is WRITTEN to the
#      `n_query x n_train` matrix (the distance matrix is never allocated)
#      and each (query, chunk) thread keeps the chunk's max by strict `>`
#      from `-inf`. DEVIATION 2626 (2026-09-11): the 64 accumulators of a
#      cell tile are one `SIMD[float32, 64]` value in registers, advanced
#      a whole tile row per feature (`ftz_simd`, `identical_mul_add_simd`,
#      lane-wise `abs` and a strict `>` select); each lane is its scalar
#      core operation for operation, so the matrix is unchanged.
#   2. `kde_rowmax_reduce_kernel`: the chunk maxima folded in ascending
#      chunk order by strict `>`. The staged fold is strict `>` from
#      `logk[0]` over ascending `j`; both return the value AND the bits of
#      the first index that attains the maximum (the only case where two
#      equal values have different bits is -0.0/+0.0, row 39, and the first
#      one still wins), and a seed of `-inf` is replaced by any cell that is
#      not `-inf`, so the two folds agree on every input, including the
#      all-`-inf` row of DEVIATION 603.
#   3. `kde_lse_terms_kernel`: `ftz(identical_exp(ftz(logk - rowmax)))` per
#      cell, in place, one thread per cell (the staged inner term, moved out
#      of the serial loop; it depends on nothing but its own cell and the
#      row's max).
#   4. `kde_lse_serial_sum_kernel`: `s = ftz(s + term)` over ascending `j`,
#      then `ftz(identical_log(s) + max)`: the staged sum in the staged
#      order. DEVIATION 603's `-inf` row is checked first, as before.
# Nothing here is a new summation order: the only sum is the serial one.
# The grid, the block width and the chunk length are SCHEDULING and move no
# bit; `kde/checks/kde_check.mojo::check_kde_tiled_equals_staged` asserts
# tiled == staged bit for bit over 6 kernels x 3 metrics x weighted and
# unweighted x shapes that straddle every tile and chunk edge, under two
# schedules. The staged path still runs whenever a trace records (the card
# certifies `kde.dists`), for cosine, sqeuclidean and minkowski, and under
# FAST (DEVIATION 2490) and DETERMINISTIC, whose codegen contraction is not
# pinned and so could differ between two loop shapes.
# MEASURED: `kde/checks/kde_stage_profile.mojo` and the lane's race on the
# H100; numbers in `kde/README.md` and `bench/OPPONENT_REFERENCE.md`.
comptime KDE_TILED_FEAT = 64
comptime KDE_TILED_CELL = 64
comptime KDE_TILED_TILE_FLOATS = KDE_TILED_FEAT * KDE_TILED_CELL
# DEVIATION 2626 also sets the tiled pass's DEFAULT SCHEDULE from the sweep
# in `kde/checks/kde_stage_profile.mojo` (H100, pod ur95zh3h9qbx2p, 3
# repetitions after a warm-up, 100,000 fit rows x 2,000 queries). Chunks of
# 256 train rows beat 1,024 on both shapes (d = 220: 36.5 against 41.1 ms;
# d = 11: 26.7 against 27.3 ms) and 4,096 loses on both; a 128-thread query
# block beats 256 at d = 220 (36.5 against 38.9 ms) and ties it at d = 11.
# More, shorter chunks give the 2D grid more blocks to fill the GPU with and
# keep each thread's `logk` writes closer together. A block width and a chunk
# length are SCHEDULING: the gate asserts the same bits across schedules
# (`check_kde_tiled_equals_staged` runs q_tpb 32 with 100-row chunks beside
# the defaults) and the profile hashed nine of them equal on both shapes.
comptime KDE_TILED_Q_TPB = 128
comptime KDE_TILED_CHUNK_ROWS = 256


def kde_identical_tiled_applies(metric: Int, trace_enabled: Bool) -> Bool:
    """DEVIATION 2625's dispatch rule, one definition for the entry and the
    gate that proves the entry reaches the tiled pass."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if trace_enabled:
            return False
        return (
            metric == DIST_L2_SQRT_UNEXPANDED
            or metric == DIST_L1
            or metric == DIST_LINF
        )
    return False


def kde_tiled_logk_kernel(
    logk: MutPointer[Float32, MutAnyOrigin],
    part_max: MutPointer[Float32, MutAnyOrigin],
    query: MutPointer[Float32, MutAnyOrigin],
    train: MutPointer[Float32, MutAnyOrigin],
    logw: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    n_train_in: Int32,
    d_in: Int32,
    n_chunks_in: Int32,
    chunk_rows_in: Int32,
    has_weights_in: Int32,
    bandwidth: Float32,
    kernel_in: Int32,
    metric_in: Int32,
):
    """DEVIATION 2625, step 1. Grid x: query blocks of `block_dim.x`
    threads; grid y: train chunks of `chunk_rows_in` rows. Writes
    `logk[q * n_train + j]` for the chunk's `j` and `part_max[q * n_chunks
    + chunk]`."""
    var n_query = Int(n_query_in)
    var n_train = Int(n_train_in)
    var d = Int(d_in)
    var n_chunks = Int(n_chunks_in)
    var chunk_rows = Int(chunk_rows_in)
    var has_weights = Int(has_weights_in) != 0
    var kernel = Int(kernel_in)
    var metric = Int(metric_in)
    var tpb = Int(block_dim.x)
    var tid = Int(thread_idx.x)
    var q = Int(block_idx.x) * tpb + tid
    var chunk = Int(block_idx.y)
    var valid = q < n_query and chunk < n_chunks
    var j_begin = chunk * chunk_rows
    var j_end = j_begin + chunk_rows
    if j_end > n_train:
        j_end = n_train

    var tile = stack_allocation[
        KDE_TILED_TILE_FLOATS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var m = neg_inf
    var qbase = q * d
    var obase = q * n_train

    var j_base = j_begin
    while j_base < j_end:
        var cells = j_end - j_base
        if cells > KDE_TILED_CELL:
            cells = KDE_TILED_CELL
        # DEVIATION 2626: the cell tile's accumulators as ONE SIMD value in
        # registers, lane c = cell c. Every lane runs the scalar core's
        # operations in the scalar core's order (see the block comment
        # above `KDE_TILED_FEAT`); lanes past `cells` fold the tile's zero
        # padding and are never read.
        var acc = SIMD[DType.float32, KDE_TILED_CELL](0.0)
        var f0 = 0
        while f0 < d:
            var feats = d - f0
            if feats > KDE_TILED_FEAT:
                feats = KDE_TILED_FEAT
            # Cooperative load, slot = feat * CELL + cell (their layout).
            barrier()
            var idx = tid
            while idx < KDE_TILED_TILE_FLOATS:
                var feat = idx // KDE_TILED_CELL
                var cell = idx - feat * KDE_TILED_CELL
                var v = Float32(0.0)
                if cell < cells and feat < feats:
                    v = ftz(train.unsafe_load((j_base + cell) * d + f0 + feat))
                tile.unsafe_store(idx, v)
                idx += tpb
            barrier()
            if valid:
                # DEVIATION 2626: one tile row (64 train values of feature
                # `f0 + feat`) against the query value, lane by lane:
                #   euclidean  `l2_unexp_core`: diff = ftz(q - t);
                #              acc = ftz(identical_mul_add(diff, diff, acc))
                #   l1         `l1_core`: acc = ftz(acc + abs(ftz(q - t)))
                #   chebyshev  `linf_core`: diff = abs(ftz(q - t));
                #              acc = diff > acc ? diff : acc (row 39, strict)
                # `ftz_simd` is `ftz` on each lane and
                # `identical_mul_add_simd` is one `fma` per lane, so each
                # lane is bit for bit the scalar core it replaces.
                if metric == DIST_L2_SQRT_UNEXPANDED:
                    for feat in range(feats):
                        var qv = SIMD[DType.float32, KDE_TILED_CELL](
                            ftz(query.unsafe_load(qbase + f0 + feat))
                        )
                        var row = tile.unsafe_load[width=KDE_TILED_CELL](feat * KDE_TILED_CELL)
                        var diff = ftz_simd[KDE_TILED_CELL](qv - row)
                        acc = ftz_simd[KDE_TILED_CELL](
                            identical_mul_add_simd[KDE_TILED_CELL](diff, diff, acc)
                        )
                elif metric == DIST_L1:
                    for feat in range(feats):
                        var qv = SIMD[DType.float32, KDE_TILED_CELL](
                            ftz(query.unsafe_load(qbase + f0 + feat))
                        )
                        var row = tile.unsafe_load[width=KDE_TILED_CELL](feat * KDE_TILED_CELL)
                        acc = ftz_simd[KDE_TILED_CELL](
                            acc + abs(ftz_simd[KDE_TILED_CELL](qv - row))
                        )
                else:
                    for feat in range(feats):
                        var qv = SIMD[DType.float32, KDE_TILED_CELL](
                            ftz(query.unsafe_load(qbase + f0 + feat))
                        )
                        var row = tile.unsafe_load[width=KDE_TILED_CELL](feat * KDE_TILED_CELL)
                        var diff = abs(ftz_simd[KDE_TILED_CELL](qv - row))
                        acc = diff.gt(acc).select(diff, acc)
            f0 += KDE_TILED_FEAT
        if valid:
            for c in range(cells):
                var dist = acc[c]
                if metric == DIST_L2_SQRT_UNEXPANDED:
                    # `pairwise_unexpanded_kernel`'s epilog.
                    dist = ftz(identical_sqrt(dist))
                # `log_kernel_matrix_kernel`'s cell.
                var v = compute_log_kernel(ftz(dist), bandwidth, kernel)
                if has_weights:
                    # `add_log_weights_kernel`'s cell.
                    v = ftz(v + logw.unsafe_load(j_base + c))
                logk.unsafe_store(obase + j_base + c, v)
                # Row 39: strict `>`, the first index attaining the max wins.
                if v > m:
                    m = v
        j_base += KDE_TILED_CELL
    if valid:
        part_max.unsafe_store(q * n_chunks + chunk, m)


def kde_rowmax_reduce_kernel(
    rowmax: MutPointer[Float32, MutAnyOrigin],
    part_max: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    n_chunks_in: Int32,
):
    """DEVIATION 2625, step 2: the chunk maxima in ascending chunk order,
    strict `>` (row 39), one thread per query."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_query_in):
        return
    var n_chunks = Int(n_chunks_in)
    var base = i * n_chunks
    var m = part_max.unsafe_load(base)
    for b in range(1, n_chunks):
        var v = part_max.unsafe_load(base + b)
        if v > m:
            m = v
    rowmax.unsafe_store(i, m)


def kde_lse_terms_kernel(
    logk: MutPointer[Float32, MutAnyOrigin],
    rowmax: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    n_train_in: Int32,
):
    """DEVIATION 2625, step 3: `logsumexp_kernel`'s term
    `ftz(identical_exp(ftz(logk - max)))`, in place, one thread per cell.
    A row whose max is `-inf` (DEVIATION 603) is left untouched: step 4
    never reads it."""
    var n_train = Int(n_train_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_query_in) * n_train:
        return
    var mx = rowmax.unsafe_load(idx // n_train)
    if mx == bitcast[DType.float32](UInt32(0xFF800000)):
        return
    logk.unsafe_store(idx, ftz(identical_exp(ftz(logk.unsafe_load(idx) - mx))))


def kde_lse_serial_sum_kernel(
    terms: MutPointer[Float32, MutAnyOrigin],
    rowmax: MutPointer[Float32, MutAnyOrigin],
    lse: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    n_train_in: Int32,
):
    """DEVIATION 2625, step 4: `logsumexp_kernel`'s serial ascending sum and
    its `log(sum) + max`, one thread per query, DEVIATION 603 first."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_query_in):
        return
    var n_train = Int(n_train_in)
    var mx = rowmax.unsafe_load(i)
    if mx == bitcast[DType.float32](UInt32(0xFF800000)):
        lse.unsafe_store(i, mx)
        return
    var base = i * n_train
    var s = Float32(0.0)
    for j in range(n_train):
        s = ftz(s + terms.unsafe_load(base + j))
    lse.unsafe_store(i, ftz(identical_log(s) + mx))


def kde_score_samples_tiled_identical(
    ctx: DeviceContext,
    mut train: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    sum_weights: Float32,
    n_train: Int,
    n_query: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: Int,
    metric: Int,
    mut scores: DeviceBuffer[DType.float32],
    elem_tpb: Int = KDE_ELEM_TPB,
    lse_tpb: Int = KDE_LSE_TPB,
    q_tpb: Int = KDE_TILED_Q_TPB,
    chunk_rows_in: Int = KDE_TILED_CHUNK_ROWS,
) raises:
    """DEVIATION 2625's host side: log-weights (if any), the tiled
    log-kernel matrix, the row max, the terms, the serial sum, the staged
    normalization. Five or six launches, one drain. `q_tpb` and
    `chunk_rows_in` are scheduling, here so the gate can vary them."""
    if n_query <= 0 or n_train <= 0 or n_features <= 0:
        raise Error(
            "kde: n_query, n_train and n_features must be positive, got "
            + String(n_query) + ", " + String(n_train) + ", " + String(n_features)
        )
    if (
        metric != DIST_L2_SQRT_UNEXPANDED
        and metric != DIST_L1
        and metric != DIST_LINF
    ):
        raise Error(
            "kde: the tiled pass (DEVIATION 2625) takes euclidean, l1 or"
            " chebyshev; metric value " + String(metric)
        )
    if elem_tpb <= 0 or lse_tpb <= 0 or q_tpb <= 0 or chunk_rows_in <= 0:
        raise Error("kde: block widths and the chunk length must be positive")
    if q_tpb > KDE_TILED_TILE_FLOATS:
        raise Error("kde: q_tpb must not exceed the tile (" + String(KDE_TILED_TILE_FLOATS) + ")")
    validate_metric_arg(metric, Float32(2.0))
    # Grid y stays inside every vendor's 65,535 limit.
    var chunk_rows = chunk_rows_in
    var min_rows = (n_train + 32767) // 32768
    if chunk_rows < min_rows:
        chunk_rows = min_rows
    var n_chunks = (n_train + chunk_rows - 1) // chunk_rows
    var cells = n_query * n_train

    var logw: DeviceBuffer[DType.float32]
    if has_weights:
        logw = ctx.enqueue_create_buffer[DType.float32](n_train)
        ctx.enqueue_function[log_weights_kernel](
            logw.unsafe_ptr(),
            weights.unsafe_ptr(),
            Int32(n_train),
            grid_dim=((n_train + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    else:
        logw = ctx.enqueue_create_buffer[DType.float32](1)
    var logk = ctx.enqueue_create_buffer[DType.float32](cells)
    var part = ctx.enqueue_create_buffer[DType.float32](n_query * n_chunks)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n_query)
    var lse = ctx.enqueue_create_buffer[DType.float32](n_query)
    ctx.enqueue_function[kde_tiled_logk_kernel](
        logk.unsafe_ptr(),
        part.unsafe_ptr(),
        query.unsafe_ptr(),
        train.unsafe_ptr(),
        logw.unsafe_ptr(),
        Int32(n_query),
        Int32(n_train),
        Int32(n_features),
        Int32(n_chunks),
        Int32(chunk_rows),
        Int32(1 if has_weights else 0),
        bandwidth,
        Int32(kernel),
        Int32(metric),
        grid_dim=((n_query + q_tpb - 1) // q_tpb, n_chunks, 1),
        block_dim=(q_tpb, 1, 1),
    )
    ctx.enqueue_function[kde_rowmax_reduce_kernel](
        rowmax.unsafe_ptr(),
        part.unsafe_ptr(),
        Int32(n_query),
        Int32(n_chunks),
        grid_dim=((n_query + lse_tpb - 1) // lse_tpb, 1, 1),
        block_dim=(lse_tpb, 1, 1),
    )
    ctx.enqueue_function[kde_lse_terms_kernel](
        logk.unsafe_ptr(),
        rowmax.unsafe_ptr(),
        Int32(n_query),
        Int32(n_train),
        grid_dim=((cells + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.enqueue_function[kde_lse_serial_sum_kernel](
        logk.unsafe_ptr(),
        rowmax.unsafe_ptr(),
        lse.unsafe_ptr(),
        Int32(n_query),
        Int32(n_train),
        grid_dim=((n_query + lse_tpb - 1) // lse_tpb, 1, 1),
        block_dim=(lse_tpb, 1, 1),
    )
    var log_sw = ftz(identical_log(sum_weights))
    var norm = log_kernel_norm(kernel, bandwidth, n_features)
    ctx.enqueue_function[normalize_scores_kernel](
        scores.unsafe_ptr(),
        lse.unsafe_ptr(),
        Int32(n_query),
        log_sw,
        norm,
        grid_dim=((n_query + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.synchronize()
    _ = logk^
    _ = part^
    _ = rowmax^
    _ = lse^
    _ = logw^


def kde_score_samples_device(
    ctx: DeviceContext,
    mut train: DeviceBuffer[DType.float32],
    mut query: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    has_weights: Bool,
    sum_weights: Float32,
    n_train: Int,
    n_query: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: Int,
    metric: Int,
    mut scores: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    elem_tpb: Int = KDE_ELEM_TPB,
    lse_tpb: Int = KDE_LSE_TPB,
    metric_arg: Float32 = Float32(2.0),
    staged_only: Bool = False,
) raises:
    """`score_samples` (`:264-363`), stage by stage, with the card.

    `weights` is read only when `has_weights`; pass any buffer otherwise.
    `sum_weights` is `cp.sum(self.sample_weight_)` or `n_train` (`:347-351`),
    computed by the caller (`host_sum_weights`). `scores` holds `n_query`
    float32 log-densities on return. `elem_tpb` / `lse_tpb` are scheduling
    widths (see the `KDE_*_TPB` note), here so the gates can vary them.
    `staged_only` forces the staged path (the gates' reference for the
    fused FAST pass and the tiled IDENTICAL pass).
    """
    if n_query <= 0:
        raise Error("kde: X must have at least one row (n_query)")
    if elem_tpb <= 0 or lse_tpb <= 0:
        raise Error("kde: block widths must be positive")
    # DEVIATION 2625: IDENTICAL with no trace recording takes the tiled
    # pass for the three unexpanded metrics; the same bits as below.
    if (not staged_only) and kde_identical_tiled_applies(metric, trace.enabled):
        kde_score_samples_tiled_identical(
            ctx, train, query, weights, has_weights, sum_weights,
            n_train, n_query, n_features, bandwidth, kernel, metric, scores,
            elem_tpb, lse_tpb,
        )
        return
    # DEVIATION 2490: FAST with no trace recording takes the fused pass.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST:
        if (not staged_only) and not trace.enabled and n_features <= KDE_FUSED_TILE_FLOATS:
            _kde_score_samples_fused(
                ctx, train, query, weights, has_weights, sum_weights,
                n_train, n_query, n_features, bandwidth, kernel, metric,
                metric_arg, scores, elem_tpb,
            )
            return
    var cells = n_query * n_train

    # distances = pairwise_distances(X, self.X_, metric=self.metric)  (:332-340)
    var dist = ctx.enqueue_create_buffer[DType.float32](cells)
    var logk = ctx.enqueue_create_buffer[DType.float32](cells)
    var lse = ctx.enqueue_create_buffer[DType.float32](n_query)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n_query)
    ctx.synchronize()
    pairwise_distance(
        ctx, dist, query, train, n_query, n_train, n_features, metric,
        metric_arg, elem_tpb,
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "kde.dists", dist, cells)

    # distances = log_probability_kernels_[self.kernel](distances, h)  (:331-334)
    var elem_grid = (cells + elem_tpb - 1) // elem_tpb
    ctx.enqueue_function[log_kernel_matrix_kernel](
        logk.unsafe_ptr(),
        dist.unsafe_ptr(),
        Int32(cells),
        bandwidth,
        Int32(kernel),
        grid_dim=(elem_grid, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    # if self.sample_weight_ is not None: distances += cp.log(self.sample_weight_)  (:337-338)
    if has_weights:
        var logw = ctx.enqueue_create_buffer[DType.float32](n_train)
        ctx.enqueue_function[log_weights_kernel](
            logw.unsafe_ptr(),
            weights.unsafe_ptr(),
            Int32(n_train),
            grid_dim=((n_train + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        ctx.enqueue_function[add_log_weights_kernel](
            logk.unsafe_ptr(),
            logw.unsafe_ptr(),
            Int32(cells),
            Int32(n_train),
            grid_dim=(elem_grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        ctx.synchronize()
        _ = logw^
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "kde.logk", logk, cells)

    # logsumexp_kernel.forall(log_probabilities.size)(distances, log_probabilities)  (:340-342)
    ctx.enqueue_function[logsumexp_kernel](
        logk.unsafe_ptr(),
        lse.unsafe_ptr(),
        rowmax.unsafe_ptr(),
        Int32(n_query),
        Int32(n_train),
        grid_dim=((n_query + lse_tpb - 1) // lse_tpb, 1, 1),
        block_dim=(lse_tpb, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "kde.rowmax", rowmax, n_query)
    trace.record_device[DType.float32](ctx, "kde.logsumexp", lse, n_query)

    # log_probabilities -= np.log(sum_weights)  (:343-351)
    # log_probabilities = norm_log_probabilities(..., self.kernel, h, dimension)  (:353-361)
    var log_sw = ftz(identical_log(sum_weights))
    var norm = log_kernel_norm(kernel, bandwidth, n_features)
    trace.record_scalar_f32("kde.logsw", log_sw)
    trace.record_scalar_f32("kde.lognorm", norm)
    ctx.enqueue_function[normalize_scores_kernel](
        scores.unsafe_ptr(),
        lse.unsafe_ptr(),
        Int32(n_query),
        log_sw,
        norm,
        grid_dim=((n_query + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    ctx.synchronize()
    trace.record_device[DType.float32](ctx, "kde.scores", scores, n_query)
    _ = dist^
    _ = logk^
    _ = lse^
    _ = rowmax^
