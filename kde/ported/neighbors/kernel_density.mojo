"""cuML's `KernelDensity`: the six log-kernels, their norms, the logsumexp.

PORT OF cuML `python/cuml/cuml/neighbors/kernel_density.py` at cuML
`00094f7` (the 25.08 Python layer: `*_log_kernel` at `:43-99`,
`logVn`/`logSn`/`norm_log_probabilities` at `:112-141`,
`logsumexp_kernel` at `:144-156`, `KernelDensity.fit` at `:220-262`,
`KernelDensity.score_samples` at `:264-363`). Do not improve.

WHY THE 25.08 PYTHON FILE AND NOT THE 26.08 C++
------------------------------------------------
The brief names cuML v26.08's `cpp/src/kde/kde.cu`. That file is 83 lines
and does ONE thing: it casts the enums and calls
`cuvs::distance::kde(...)` (`kde.cu:45-55`); the Cython
`kernel_density.pyx` (26.08) validates and forwards. The algorithm itself
lives in cuVS 26.08, and the cuVS checkout this tree is pinned to
(`PORTING_RULES.md` 0a: `upstream/cuvs` at `94c2819`, 25.08) predates it
and has no `kde`. The 25.08 cuML Python file above IS the algorithm the
26.08 fused kernel was written to reproduce -- the same six log-kernels
with the same `FLOAT_MIN` sentinel, the same normalization, the same
per-row logsumexp -- and is the version read symbol by symbol here.
`kde/ported/kde/kde.mojo` carries the 26.08 entry's shape (enum values,
signature, `sum_weights` passed in) over this algorithm. When cuVS 26.08
is cloned the fused kernel is the next port; `kde/UNPORTED.tsv` names it.

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
   and `kde/mojo_only/kde_check.mojo::check_kde_zero_sign_cannot_leak`
   proves it cannot reach the output.
3. **`log(0)` is avoided by `maximum(z, 1e-30)`** (`:62`, `:81`, `:92`),
   and the support test is applied AFTER the log as another bool product:
   `y = (x < h) * log(z); y += (x >= h) * FLOAT_MIN`. For `x >= h` that is
   `(+/-0.0) + FLOAT_MIN = FLOAT_MIN` exactly; for `x < h` it is `log(z) +
   (-0.0) = log(z)`. The branches below are that product, resolved.

The whole thing is FLOAT32 (DEVIATION 600, below): their numba
`logsumexp_kernel` accumulates `sum = 0.0` in float64 and writes a float64
`log_probabilities`; Metal has no float64 on the device, so this port is
float32 end to end and the Float64 host reference in
`kde/mojo_only/kde_oracle.mojo` measures what that costs.

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
and `mojo_only/numerics.mojo` has no portable float64 log or lgamma. So
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

from std.math import lgamma, log, pi
from std.memory import bitcast

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from kde.ported.distance.distance import pairwise_distance
from kde.ported.distance.distance_ops import (
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
    PAIRWISE_ELEM_TPB,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_cos,
    identical_exp,
    identical_log,
    identical_mul_add,
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

#: SCHEDULING. `KDE_ELEM_TPB` is the one-thread-per-cell width of the
#: elementwise kernels, `KDE_LSE_TPB` the one-thread-per-ROW width of the
#: logsumexp. Neither moves a bit; `kde_check` varies both.
comptime KDE_ELEM_TPB = PAIRWISE_ELEM_TPB
comptime KDE_LSE_TPB = 128


def kde_float32_min() -> Float32:
    return bitcast[DType.float32](KDE_FLOAT32_MIN_BITS)


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
    (`metrics/pairwise_distances.pyx:68-86`), the four ported rows; every
    other row of THEIR table is refused BY NAME so a caller learns it is
    unported rather than unknown."""
    if name == "euclidean" or name == "l2":
        return DIST_L2_SQRT_UNEXPANDED
    if name == "sqeuclidean":
        return DIST_L2_EXPANDED
    if name == "l1" or name == "cityblock" or name == "manhattan":
        return DIST_L1
    if name == "chebyshev":
        return DIST_LINF
    if (
        name == "cosine"
        or name == "canberra"
        or name == "minkowski"
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
            + "' is in cuML's pairwise_distances table but is NOT PORTED"
            " (kde/UNPORTED.tsv); ported: euclidean, l2, sqeuclidean, l1,"
            " cityblock, manhattan, chebyshev"
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
    return String("?")


# ---------------------------------------------------------------------------
# The log-kernels, `kernel_density.py:43-99`, scalar, float32. Shared by the
# device kernel below and (as a SECOND SPELLING, not an import) by
# `kde/mojo_only/kde_oracle.mojo`.
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
# ============ UPSTREAM FOR EVEN d, AND IS NOT PORTED AS WRITTEN ============
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
# for why not the corrected recurrence). ASSUME-OUR-CODE-IS-BROKEN's corollary is "do not port their
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
        if v > max_exp:
            max_exp = v
    var s = Float32(0.0)
    for j in range(0, n_train):
        s = ftz(s + ftz(identical_exp(ftz(logk.unsafe_load(base + j) - max_exp))))
    rowmax.unsafe_store(i, max_exp)
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
    and its length `n_train`. Every refusal names the parameter."""
    if not (bandwidth > Float32(0.0)):
        raise Error("bandwidth must be positive")
    if kernel < 0 or kernel >= KDE_N_KERNELS:
        raise Error("invalid kernel: value " + String(kernel))
    if (
        metric != DIST_L2_SQRT_UNEXPANDED
        and metric != DIST_L2_EXPANDED
        and metric != DIST_L1
        and metric != DIST_LINF
    ):
        raise Error("kde: metric value " + String(metric) + " is not ported")
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


def host_sum_weights(weights: List[Float32]) -> Float32:
    """`cp.sum(self.sample_weight_)` (`:348`): cupy's device tree, order
    cupy's. Ours is a SERIAL ASCENDING host fold through `ftz` so the order
    is a function of `n_train` alone (the same reason the logsumexp is
    serial); its `log` is taken by the caller through `identical_log`."""
    var s = Float32(0.0)
    for i in range(len(weights)):
        s = ftz(s + weights[i])
    return s


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
) raises:
    """`score_samples` (`:264-363`), stage by stage, with the card.

    `weights` is read only when `has_weights`; pass any buffer otherwise.
    `sum_weights` is `cp.sum(self.sample_weight_)` or `n_train` (`:347-351`),
    computed by the caller (`host_sum_weights`). `scores` holds `n_query`
    float32 log-densities on return. `elem_tpb` / `lse_tpb` are scheduling
    widths (see the `KDE_*_TPB` note), here so the gates can vary them.
    """
    if n_query <= 0:
        raise Error("kde: X must have at least one row (n_query)")
    if elem_tpb <= 0 or lse_tpb <= 0:
        raise Error("kde: block widths must be positive")
    var cells = n_query * n_train

    # distances = pairwise_distances(X, self.X_, metric=self.metric)  (:332-340)
    var dist = ctx.enqueue_create_buffer[DType.float32](cells)
    var logk = ctx.enqueue_create_buffer[DType.float32](cells)
    var lse = ctx.enqueue_create_buffer[DType.float32](n_query)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n_query)
    ctx.synchronize()
    pairwise_distance(
        ctx, dist, query, train, n_query, n_train, n_features, metric, elem_tpb
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
