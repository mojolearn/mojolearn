# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The six ported distances and the distance weights, gated and sabotaged.

Run: `pixi run check-metric` (and `check-metric-identical`).

WHAT THIS FILE GATES, AND WHY EACH ARM EXISTS
----------------------------------------------
The metric lane added two arithmetic seams to the k-NN path
(`CosineExpanded`, `LpUnexpanded`) and one to the vote (`weights=
'distance'`). Each of the three has a value that LOOKS RIGHT WHEN IT IS
WRONG, which is the class `mojotrees-verify-reach-not-output` names, so
each carries an arm that MUST FAIL if the seam is broken:

  seam                         looks right when wrong because           arm
  ---------------------------  ---------------------------------------  ---
  cosine's SQRT'd norm         a squared norm gives finite plausible    3
                               distances in [1-1/n, 1] on every row
  Minkowski's `metric_arg`     at the p=2 default, Lp and euclidean     4
                               agree to a few ulp, so a kernel that
                               ignored p entirely would pass a
                               euclidean cross-check
  the weight rule's ROW-LEVEL  a per-ELEMENT zero rule agrees with      8
  replacement                  sklearn on every fixture with no
                               duplicate point

PREDICTED AND OBSERVED. Every sabotage clause below states what it
PREDICTS when the seam is broken. The OBSERVED half is the orchestrator's
to fill in from the run; `neighbors/README.md` is where it goes.

THE THREE LEVELS OF TRUTH are in `neighbors/checks/metric_oracle.mojo`:
a float32 second spelling (bit-equal under IDENTICAL), a float64
mathematical reference computed a THIRD way (so a wrong formula cannot
hide behind a faithful second spelling), and scikit-learn's weight rule
transcribed from their file.
"""

from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    numeric_mode_name,
)
from neighbors.checks.metric_oracle import (
    oracle_distance_weights,
    oracle_metric_distance,
    oracle_row_norm,
    reference_metric_distance_f64,
)
from neighbors.estimator import (
    knn_classifier_predict,
    knn_metric_from_name,
    knn_regressor_predict,
    knn_search,
)
from neighbors.impl.distance.detail.distance_ops import (
    COSINE_NORM_TPB,
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_L2_UNEXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
    METRIC_ELEM_TPB,
    cosine_row_norm_kernel,
    metric_distance_kernel,
    metric_norm_takes_sqrt,
    metric_uses_norms,
    metric_value_name,
    validate_metric_arg,
)
from neighbors.impl.selection.distance_weights import (
    WEIGHTS_DISTANCE,
    WEIGHTS_UNIFORM,
    host_distance_weights,
    weights_from_name,
)
from core.row_norms import NORM_TPB, row_norm_kernel


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

comptime M_ROWS = 37
comptime M_COLS = 53
comptime M_FEAT = 11

#: The end-to-end k-NN fixture. Small enough to brute force in float64 on
#: the host, large enough that a neighbour set is not trivially unique.
comptime E_INDEX = 512
comptime E_QUERIES = 31
comptime E_FEAT = 9
comptime E_K = 7


def _mode_name() -> String:
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    """`kde/checks/kde_check.mojo:206-213`, the same spelling.

    `DIGITS[byte=nib]` and NOT `DIGITS[nib]`: a Mojo `String` is not
    indexable and `s[i]` is a compile error, which is a trap this tree has
    hit before."""
    comptime DIGITS = "0123456789abcdef"
    var b = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((b >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _coord(row: Int, feature: Int, salt: Int) -> Float32:
    """The splitmix64 mixer `neighbors/checks/knn_check.mojo::_coord` uses,
    for the reason that file's docstring gives at length: the affine
    generator it replaced was a LATTICE, and on a lattice the expanded
    identity cannot rank neighbours in float32 at all. Copied rather than
    imported so this file's fixture cannot be changed by an edit to
    another check's."""
    var z = (
        UInt64(row) * UInt64(0x9E3779B97F4A7C15)
        + UInt64(feature) * UInt64(0xBF58476D1CE4E5B9)
        + UInt64(salt) * UInt64(0x94D049BB133111EB)
    )
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> UInt64(31))
    return Float32(z >> UInt64(40)) * Float32(1.0 / 16777216.0)


def _fixture(n: Int, d: Int, salt: Int) -> List[Float32]:
    var out = List[Float32](capacity=n * d)
    for i in range(n):
        for f in range(d):
            out.append(_coord(i, f, salt))
    return out^


def _scaled_fixture(n: Int, d: Int, salt: Int) -> List[Float32]:
    """The same points with WIDELY DIFFERENT ROW NORMS, row `i` scaled by
    `2^(i mod 9 - 4)`.

    Cosine ignores magnitude and euclidean does not, so this is the
    fixture on which the two metrics MUST disagree about the neighbour
    set. On the unscaled fixture every row norm is within a factor of two
    and the two orderings largely coincide, which would make clause 6's
    reach test pass for the wrong reason."""
    var out = _fixture(n, d, salt)
    for i in range(n):
        var e = (i % 9) - 4
        var s = Float32(1.0)
        for _ in range(abs(e)):
            s = s * Float32(2.0)
        if e < 0:
            s = Float32(1.0) / s
        for f in range(d):
            out[i * d + f] = out[i * d + f] * s
    return out^


def _upload(
    ctx: DeviceContext, v: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(v)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, v[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _host_f32(
    ctx: DeviceContext, v: List[Float32]
) raises -> HostBuffer[DType.float32]:
    """A RUNTIME host buffer holding `v`.

    `knn_search`'s boundary is host POINTERS, and `UNWIRED.md:31` records
    that a pointer from `enqueue_create_host_buffer` is not interchangeable
    with an arbitrary host pointer on this stack -- silently. Every fixture
    this file hands to an estimator goes through here, which is what
    `neighbors/checks/estimator_check.mojo:160-180` already does."""
    var h = ctx.enqueue_create_host_buffer[DType.float32](len(v))
    ctx.synchronize()
    for i in range(len(v)):
        h.unsafe_ptr().unsafe_store(i, v[i])
    return h^


def _host_i32(
    ctx: DeviceContext, v: List[Int32]
) raises -> HostBuffer[DType.int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](len(v))
    ctx.synchronize()
    for i in range(len(v)):
        h.unsafe_ptr().unsafe_store(i, v[i])
    return h^


def _read(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def _all_metrics() -> List[Int]:
    """cuVS's enumerators, in their numeric order. `L2Unexpanded` and
    `L2SqrtUnexpanded` are here even though cuML's k-NN table never emits
    them (`_build_metric_type:522-525` sends euclidean to the EXPANDED
    pair), because `metric_distance_kernel` computes them and an arm that
    is computed and never checked is an arm that rots."""
    return [
        DIST_L2_EXPANDED,
        DIST_L2_SQRT_EXPANDED,
        DIST_COSINE_EXPANDED,
        DIST_L1,
        DIST_L2_UNEXPANDED,
        DIST_L2_SQRT_UNEXPANDED,
        DIST_LINF,
        DIST_LP_UNEXPANDED,
    ]


def _device_matrix(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    m: Int,
    n: Int,
    d: Int,
    metric: Int,
    metric_arg: Float32,
    swap_norm_flag: Bool = False,
) raises -> List[Float32]:
    """`metric_distance_kernel` over one whole matrix, with the norms
    computed the way `compute_norms_for_metric` computes them.

    `swap_norm_flag` is THE SABOTAGE HANDLE and nothing else uses it: when
    True, the two norm kernels are swapped -- cosine gets the SQUARED norm
    and the L2 pair gets the SQRT'd one. It is a parameter rather than an
    edit so the sabotage can be run and reverted inside one gate, which is
    the discipline `neighbors/checks/knn_check.mojo` already follows.
    """
    var dx = _upload(ctx, x)
    var dy = _upload(ctx, y)
    var dxn = ctx.enqueue_create_buffer[DType.float32](m)
    var dyn = ctx.enqueue_create_buffer[DType.float32](n)
    var dz = ctx.enqueue_create_buffer[DType.float32](m * n)
    ctx.synchronize()

    if metric_uses_norms(metric):
        var want_sqrt = metric_norm_takes_sqrt(metric)
        if swap_norm_flag:
            want_sqrt = not want_sqrt
        if want_sqrt:
            ctx.enqueue_function[cosine_row_norm_kernel](
                dxn.unsafe_ptr(), dx.unsafe_ptr(), Int32(d),
                grid_dim=(m, 1, 1), block_dim=(COSINE_NORM_TPB, 1, 1),
            )
            ctx.enqueue_function[cosine_row_norm_kernel](
                dyn.unsafe_ptr(), dy.unsafe_ptr(), Int32(d),
                grid_dim=(n, 1, 1), block_dim=(COSINE_NORM_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[row_norm_kernel](
                dxn.unsafe_ptr(), dx.unsafe_ptr(), Int32(d), Int32(0),
                grid_dim=(m, 1, 1), block_dim=(NORM_TPB, 1, 1),
            )
            ctx.enqueue_function[row_norm_kernel](
                dyn.unsafe_ptr(), dy.unsafe_ptr(), Int32(d), Int32(0),
                grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1),
            )
        ctx.synchronize()

    var cells = m * n
    ctx.enqueue_function[metric_distance_kernel](
        dz.unsafe_ptr(),
        dx.unsafe_ptr(),
        dy.unsafe_ptr(),
        dxn.unsafe_ptr(),
        dyn.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(d),
        Int32(metric),
        metric_arg,
        grid_dim=((cells + METRIC_ELEM_TPB - 1) // METRIC_ELEM_TPB, 1, 1),
        block_dim=(METRIC_ELEM_TPB, 1, 1),
    )
    ctx.synchronize()
    var out = _read(ctx, dz, cells)
    _ = dx^
    _ = dy^
    _ = dxn^
    _ = dyn^
    _ = dz^
    return out^


def _oracle_matrix(
    x: List[Float32],
    y: List[Float32],
    m: Int,
    n: Int,
    d: Int,
    metric: Int,
    metric_arg: Float32,
    swap_norm_flag: Bool = False,
) raises -> List[Float32]:
    var xn = List[Float32]()
    var yn = List[Float32]()
    if metric_uses_norms(metric):
        var want_sqrt = metric_norm_takes_sqrt(metric)
        if swap_norm_flag:
            want_sqrt = not want_sqrt
        for i in range(m):
            xn.append(oracle_row_norm(x, i, d, want_sqrt))
        for j in range(n):
            yn.append(oracle_row_norm(y, j, d, want_sqrt))
    else:
        for _ in range(m):
            xn.append(Float32(0.0))
        for _ in range(n):
            yn.append(Float32(0.0))
    var out = List[Float32](capacity=m * n)
    for i in range(m):
        for j in range(n):
            out.append(
                oracle_metric_distance(
                    x, y, i, j, d, metric, xn[i], yn[j], metric_arg
                )
            )
    return out^


# ===========================================================================
# 1. THE DEVICE KERNEL AGAINST THE FLOAT32 ORACLE, EVERY METRIC.
# ===========================================================================


def check_metric_device_equals_oracle() raises:
    var ctx = DeviceContext()
    var x = _scaled_fixture(M_ROWS, M_FEAT, 7)
    var y = _scaled_fixture(M_COLS, M_FEAT, 23)
    var cells = M_ROWS * M_COLS
    var n_equal = 0
    var n_diff = 0
    var report = String("")
    for metric in _all_metrics():
        var pa = Float32(2.0)
        if metric == DIST_LP_UNEXPANDED:
            pa = Float32(3.0)
        var dev = _device_matrix(ctx, x, y, M_ROWS, M_COLS, M_FEAT, metric, pa)
        var orc = _oracle_matrix(x, y, M_ROWS, M_COLS, M_FEAT, metric, pa)
        var diff = 0
        var first = String("")
        for i in range(cells):
            if bitcast[DType.uint32](dev[i]) != bitcast[DType.uint32](orc[i]):
                diff += 1
                if first == "":
                    first = (
                        "cell " + String(i) + " device " + _hex32(dev[i])
                        + " oracle " + _hex32(orc[i])
                    )
        if diff > 0:
            n_diff += diff
            var msg = (
                metric_value_name(metric) + ": " + String(diff) + " of "
                + String(cells) + " cells differ; " + first
            )
            comptime if IDENTICAL:
                raise Error(
                    "check_metric_device_equals_oracle FAILED " + msg
                )
            else:
                if report == "":
                    report = msg
                print("  report " + msg)
        else:
            n_equal += cells
    print(
        "check_metric_device_equals_oracle "
        + ("OK" if IDENTICAL else "REPORT") + " [" + _mode_name() + "]: 8"
        " DistanceType values x " + String(cells) + " cells, "
        + String(n_equal) + " bit-equal, " + String(n_diff) + " differ"
        + (
            ""
            if IDENTICAL
            else " (FAST: the vendor sqrt/exp/log/div spellings are free to"
            " differ)"
        )
    )
    _ = ctx^


# ===========================================================================
# 2. THE FLOAT64 REFERENCE. WHAT GENERAL p COSTS, AS A NUMBER.
# ===========================================================================


def check_metric_matches_float64_reference() raises:
    """Every metric within tolerance of the metric computed a THIRD way.

    This is the gate that prices `identical_pow`. `portable_powf` is
    `exp(p log z)` and is a few ulp where a native `powf` is one or two,
    so Minkowski's tolerance is looser than L1's and the NUMBER is printed
    rather than hidden inside a pass. Nothing here is refused for it: the
    lane's contract is SAMENESS across vendors, which the construction
    gives exactly, and accuracy is a measurement.
    """
    var ctx = DeviceContext()
    var x = _scaled_fixture(M_ROWS, M_FEAT, 7)
    var y = _scaled_fixture(M_COLS, M_FEAT, 23)
    var cells = M_ROWS * M_COLS
    var line = String("")
    for metric in _all_metrics():
        var ps = List[Float32]()
        if metric == DIST_LP_UNEXPANDED:
            ps.append(Float32(1.0))
            ps.append(Float32(1.5))
            ps.append(Float32(3.0))
            ps.append(Float32(4.0))
        else:
            ps.append(Float32(2.0))
        for pa in ps:
            var dev = _device_matrix(
                ctx, x, y, M_ROWS, M_COLS, M_FEAT, metric, pa
            )
            var worst = 0.0
            for i in range(M_ROWS):
                for j in range(M_COLS):
                    var r = reference_metric_distance_f64(
                        x, y, i, j, M_FEAT, metric, Float64(pa)
                    )
                    var g = Float64(dev[i * M_COLS + j])
                    var rel = abs(g - r) / (abs(r) + 1e-12)
                    if rel > worst:
                        worst = rel
            # THE TOLERANCE IS PER METRIC AND EACH VALUE HAS A REASON.
            # The reference is the metric as mathematics; the device is
            # the metric as cuVS computes it, and the gap between them is
            # the FORMULA's cost, which this gate exists to price rather
            # than to be surprised by.
            #
            #  - the two EXPANDED L2 metrics get 1e-2, because
            #    `||x||^2 + ||y||^2 - 2 x.y` CANCELS: for a close pair the
            #    result is a small difference of large numbers, and
            #    float32's ulp at the norms sets the floor.
            #    `neighbors/README.md`'s scar 2 is the full version of
            #    this, measured: on 4,096 COLLINEAR points the expansion
            #    cannot rank neighbours at all, at any scale. This fixture
            #    is not collinear, and the number this arm prints is what
            #    the expansion costs on it.
            #  - Minkowski gets 1e-3, the `exp(p log z)` composition's few
            #    ulp compounded over the sum and the final root.
            #  - the unexpanded arms get 1e-5, which is float32 rounding
            #    and nothing else: they subtract first and never cancel.
            var tol = 1e-5
            if (
                metric == DIST_L2_EXPANDED
                or metric == DIST_L2_SQRT_EXPANDED
                or metric == DIST_COSINE_EXPANDED
            ):
                tol = 1e-2
            elif metric == DIST_LP_UNEXPANDED:
                tol = 1e-3
            var pnote = String("")
            if metric == DIST_LP_UNEXPANDED:
                pnote = " (p=" + String(pa) + ")"
            if worst > tol:
                raise Error(
                    "check_metric_matches_float64_reference: "
                    + metric_value_name(metric)
                    + pnote
                    + " worst relative error " + String(worst)
                    + " against the float64 reference, tolerance "
                    + String(tol)
                    + ". Every cell was computed a THIRD way there (direct"
                    " dot and norms for cosine, host pow for Lp), so this"
                    " is a FORMULA difference, not a rounding one."
                )
            if metric == DIST_LP_UNEXPANDED:
                line += (
                    " Lp(p=" + String(pa) + ")=" + String(worst) + ";"
                )
            else:
                line += (
                    " " + metric_value_name(metric) + "="
                    + String(worst) + ";"
                )
    print(
        "check_metric_matches_float64_reference OK [" + _mode_name()
        + "]: worst relative error vs the float64 reference per metric:"
        + line
    )
    _ = ctx^


# ===========================================================================
# 3. SABOTAGE: COSINE'S NORM FLAG.
# ===========================================================================


def check_metric_cosine_norm_flag() raises:
    """`knn_brute_force.cuh:122`: "cosine needs the l2norm, where as l2
    distances needs the squared norm".

    SABOTAGE. `_device_matrix(..., swap_norm_flag=True)` feeds cosine the
    SQUARED norm and the L2 pair the SQRT'd one. Both still produce finite
    plausible numbers -- squared-norm "cosine" lands in a narrow band near
    1 and sqrt-norm "L2 expanded" is a small positive number -- which is
    exactly why an output check alone cannot see this and a reach arm is
    required.

    PREDICTED when the seam is broken (i.e. if `metric_norm_takes_sqrt`
    ever returns the wrong flag, or the cosine branch is pointed at
    `row_norm_kernel`'s `take_sqrt = 0` arm): the sabotaged matrix equals
    the honest one, `moved == 0`, and clause 1 raises. OBSERVED: the
    orchestrator's run fills this in.

    Clause 2 is the other half and is the one that catches a SWAPPED pair
    rather than a single wrong flag: the sabotaged run must still agree
    with the ORACLE run under the same swap, because if it did not, the
    difference would be somewhere other than the flag and clause 1's
    evidence would be worthless.
    """
    var ctx = DeviceContext()
    var x = _scaled_fixture(M_ROWS, M_FEAT, 7)
    var y = _scaled_fixture(M_COLS, M_FEAT, 23)
    var cells = M_ROWS * M_COLS
    var total_moved = 0
    for metric in [DIST_COSINE_EXPANDED, DIST_L2_SQRT_EXPANDED]:
        var honest = _device_matrix(
            ctx, x, y, M_ROWS, M_COLS, M_FEAT, metric, Float32(2.0)
        )
        var sab = _device_matrix(
            ctx, x, y, M_ROWS, M_COLS, M_FEAT, metric, Float32(2.0), True
        )
        var moved = 0
        for i in range(cells):
            if bitcast[DType.uint32](honest[i]) != bitcast[DType.uint32](
                sab[i]
            ):
                moved += 1
        if moved == 0:
            raise Error(
                "check_metric_cosine_norm_flag clause 1 SABOTAGE FAILED TO"
                " REGISTER for " + metric_value_name(metric) + ": swapping"
                " the sqrt flag on the norm changed NO cell of "
                + String(cells) + ". Either the norm is not read at all or"
                " the fixture has unit row norms; `_scaled_fixture` scales"
                " rows by 2^-4..2^4 precisely so it cannot."
            )
        total_moved += moved

        # clause 2: the sabotaged device still matches the sabotaged oracle
        var sab_orc = _oracle_matrix(
            x, y, M_ROWS, M_COLS, M_FEAT, metric, Float32(2.0), True
        )
        comptime if IDENTICAL:
            var d2 = 0
            for i in range(cells):
                if bitcast[DType.uint32](sab[i]) != bitcast[DType.uint32](
                    sab_orc[i]
                ):
                    d2 += 1
            if d2 > 0:
                raise Error(
                    "check_metric_cosine_norm_flag clause 2: under the SAME"
                    " swap the device and the oracle disagree on "
                    + String(d2) + " of " + String(cells) + " cells for "
                    + metric_value_name(metric) + ", so clause 1's movement"
                    " is not attributable to the flag alone"
                )
    print(
        "check_metric_cosine_norm_flag OK [" + _mode_name()
        + "]: swapping the norm's sqrt flag moved " + String(total_moved)
        + " of " + String(2 * cells) + " cells across cosine and"
        " L2SqrtExpanded, and both sabotaged runs still track their"
        " sabotaged oracle"
    )
    _ = ctx^


# ===========================================================================
# 4. SABOTAGE: IS `metric_arg` READ AT ALL?
# ===========================================================================


def check_metric_arg_is_reached() raises:
    """PREDICTED when the seam is broken: a kernel that never reads
    `metric_arg` -- passed and dropped, or read from the wrong argument
    slot -- returns the SAME matrix for every p, so clause 1 raises with
    `same == cells`. A kernel that reads p but applies `p` where `1/p`
    belongs still varies with p and passes clause 1; clause 2 is the one
    that catches THAT, because the p-norms are nested and a swapped
    exponent inverts the ordering.

    Clause 3 is the reason clause 1 is not redundant with
    `check_metric_matches_float64_reference`: at the DEFAULT p = 2 the Lp
    arm and the unexpanded euclidean arm agree to a few ulp, so a
    p-ignoring kernel would pass every tolerance check in this file.
    """
    var ctx = DeviceContext()
    var x = _scaled_fixture(M_ROWS, M_FEAT, 7)
    var y = _scaled_fixture(M_COLS, M_FEAT, 23)
    var cells = M_ROWS * M_COLS
    var ps: List[Float32] = [
        Float32(1.0), Float32(1.5), Float32(3.0), Float32(4.0)
    ]
    var mats = List[List[Float32]]()
    for pa in ps:
        mats.append(
            _device_matrix(
                ctx, x, y, M_ROWS, M_COLS, M_FEAT, DIST_LP_UNEXPANDED, pa
            )
        )

    # clause 1: pairwise distinct
    for a in range(len(ps)):
        for b in range(a + 1, len(ps)):
            var same = 0
            for i in range(cells):
                if bitcast[DType.uint32](mats[a][i]) == bitcast[
                    DType.uint32
                ](mats[b][i]):
                    same += 1
            if same == cells:
                raise Error(
                    "check_metric_arg_is_reached clause 1 SABOTAGE FAILED"
                    " TO REGISTER: p=" + String(ps[a]) + " and p="
                    + String(ps[b]) + " produced BIT-IDENTICAL matrices on"
                    " all " + String(cells) + " cells -- metric_arg is not"
                    " being read"
                )

    # clause 2: Lp is non-increasing in p
    var viol = 0
    for a in range(len(ps) - 1):
        for i in range(cells):
            if mats[a + 1][i] > mats[a][i] + Float32(1e-4):
                viol += 1
    if viol > 0:
        raise Error(
            "check_metric_arg_is_reached clause 2: Lp INCREASED with p on "
            + String(viol) + " comparisons. The p-norms are nested, so p"
            " and 1/p are swapped at the core or at the epilogue."
        )

    # clause 3: p = 2 tracks unexpanded euclidean (the reason clause 1 is
    # needed at all)
    var lp2 = _device_matrix(
        ctx, x, y, M_ROWS, M_COLS, M_FEAT, DIST_LP_UNEXPANDED, Float32(2.0)
    )
    var eu = _device_matrix(
        ctx, x, y, M_ROWS, M_COLS, M_FEAT, DIST_L2_SQRT_UNEXPANDED,
        Float32(2.0),
    )
    var worst = Float32(0.0)
    var bit_equal = 0
    for i in range(cells):
        var e = abs(lp2[i] - eu[i]) / (abs(eu[i]) + Float32(1e-6))
        if e > worst:
            worst = e
        if bitcast[DType.uint32](lp2[i]) == bitcast[DType.uint32](eu[i]):
            bit_equal += 1
    if worst > Float32(1e-3):
        raise Error(
            "check_metric_arg_is_reached clause 3: Lp at p=2 differs from"
            " unexpanded euclidean by " + String(worst) + " relative; the"
            " exp(p log) construction should be a few ulp, not this"
        )
    print(
        "check_metric_arg_is_reached OK [" + _mode_name() + "]: 4 p values"
        " pairwise distinct on every one of " + String(cells) + " cells, 0"
        " monotonicity violations, p=2 vs euclidean worst rel "
        + String(worst) + " and " + String(bit_equal) + " of "
        + String(cells) + " bit-equal (the exp(p log) cost, measured)"
    )
    _ = ctx^


# ===========================================================================
# 5. REFUSALS BY VALUE. DEVIATIONS 552 AND 553.
# ===========================================================================


def check_metric_refusals() raises:
    var n = 0
    var bad_p: List[Float32] = [
        Float32(0.0),
        Float32(-2.0),
        bitcast[DType.float32](UInt32(0x7F800000)),
        bitcast[DType.float32](UInt32(0x7FC00000)),
        Float32(1e-40),
    ]
    for pv in bad_p:
        var raised = False
        try:
            validate_metric_arg(DIST_LP_UNEXPANDED, pv)
        except e:
            raised = True
            n += 1
            print("  refused  minkowski p=" + String(pv) + ": " + String(e))
        if not raised:
            raise Error(
                "check_metric_refusals: minkowski p=" + String(pv)
                + " did NOT raise (DEVIATION 552)"
            )
        # the SAME p on a non-Lp metric must be ACCEPTED and discarded,
        # which is `distance.cuh:193`'s `DataT)  // unused`.
        validate_metric_arg(DIST_COSINE_EXPANDED, pv)

    # DEVIATION 553: a zero row under cosine, refused before any launch.
    var ctx = DeviceContext()
    var idx_l = _fixture(16, 4, 5)
    for f in range(4):
        idx_l[9 * 4 + f] = Float32(0.0)
    var qry_l = _fixture(3, 4, 6)
    var h_idx = _host_f32(ctx, idx_l)
    var h_qry = _host_f32(ctx, qry_l)
    var h_od = ctx.enqueue_create_host_buffer[DType.float32](9)
    var h_oi = ctx.enqueue_create_host_buffer[DType.uint32](9)
    ctx.synchronize()
    var zraised = False
    try:
        _ = knn_search(
            ctx,
            h_idx.unsafe_ptr(),
            16,
            h_qry.unsafe_ptr(),
            3,
            4,
            3,
            h_od.unsafe_ptr(),
            h_oi.unsafe_ptr(),
            True,
            256,
            2,
            DIST_COSINE_EXPANDED,
            Float32(2.0),
        )
    except e:
        zraised = True
        n += 1
        print("  refused  cosine zero index row: " + String(e))
    if not zraised:
        raise Error(
            "check_metric_refusals: an all-zero INDEX row under cosine did"
            " NOT raise (DEVIATION 553); its distances would have been"
            " NaN, and the radix selector sorts a NaN ABOVE every finite"
            " distance while the FAISS queue sorts it below -- two arms,"
            " two different wrong answers, no error either way"
        )
    # the same fixture is fine under euclidean: the refusal is cosine's.
    _ = knn_search(
        ctx,
        h_idx.unsafe_ptr(),
        16,
        h_qry.unsafe_ptr(),
        3,
        4,
        3,
        h_od.unsafe_ptr(),
        h_oi.unsafe_ptr(),
        True,
        256,
        2,
    )

    # the metric NAME table, cuML's `_build_metric_type`
    var names: List[String] = [
        "euclidean", "l2", "sqeuclidean", "l1", "cityblock", "manhattan",
        "taxicab", "chebyshev", "linf", "cosine", "minkowski", "lp",
    ]
    for nm in names:
        _ = knn_metric_from_name(nm)
    var unported: List[String] = [
        "canberra", "jensenshannon", "correlation", "inner_product",
        "haversine", "braycurtis",
    ]
    for nm in unported:
        var r = False
        try:
            _ = knn_metric_from_name(nm)
        except:
            r = True
            n += 1
        if not r:
            raise Error(
                "check_metric_refusals: metric='" + nm + "' is in cuML's"
                " VALID_METRICS['brute'] and is NOT ported here, so it must"
                " be refused BY NAME"
            )
    var unknown = False
    try:
        _ = knn_metric_from_name("not_a_metric")
    except:
        unknown = True
        n += 1
    if not unknown:
        raise Error("check_metric_refusals: an unknown metric did not raise")

    # weights
    _ = weights_from_name("uniform")
    _ = weights_from_name("distance")
    var wr = False
    try:
        _ = weights_from_name("callable")
    except:
        wr = True
        n += 1
    if not wr:
        raise Error(
            "check_metric_refusals: weights=<callable> must be refused by"
            " name (sklearn accepts one; a Python function cannot run"
            " inside a GPU kernel)"
        )
    _ = h_idx^
    _ = h_qry^
    _ = h_od^
    _ = h_oi^
    print(
        "check_metric_refusals OK [" + _mode_name() + "]: " + String(n)
        + " refusals by name/value, 12 metric names and 2 weightings"
        " resolve, a non-Lp metric discards every bad p"
    )
    _ = ctx^


# ===========================================================================
# 6. END TO END: `knn_search` PER METRIC, AGAINST A FLOAT64 BRUTE FORCE.
# ===========================================================================


def _host_true_neighbours(
    index: List[Float32],
    queries: List[Float32],
    n_index: Int,
    n_queries: Int,
    d: Int,
    k: Int,
    metric: Int,
    metric_arg: Float64,
) raises -> List[Int]:
    """The exact k nearest, in FLOAT64, by full scan with a stable
    (distance, index) key. Independent of every device path."""
    var out = List[Int](capacity=n_queries * k)
    for q in range(n_queries):
        var best_d = List[Float64]()
        var best_i = List[Int]()
        for j in range(n_index):
            var dv = reference_metric_distance_f64(
                queries, index, q, j, d, metric, metric_arg
            )
            var pos = len(best_d)
            while pos > 0 and (
                best_d[pos - 1] > dv
                or (best_d[pos - 1] == dv and best_i[pos - 1] > j)
            ):
                pos -= 1
            best_d.insert(pos, dv)
            best_i.insert(pos, j)
            if len(best_d) > k:
                _ = best_d.pop()
                _ = best_i.pop()
        for s in range(k):
            out.append(best_i[s])
    return out^


def check_metric_knn_end_to_end() raises:
    """Every metric through the whole `knn_search`, neighbour SET against
    the float64 truth.

    A SET comparison, not an ordered one, and the reason is
    `neighbors/estimator.mojo`'s own table: the tiled arm returns the right
    k in an unspecified order (RAFT's radix select does not sort) and the
    host sort then imposes a total (distance, index) order. What the
    float64 truth can adjudicate without re-deriving float32 tie-breaking
    is WHICH k came back.

    Clause 2 is the REACH arm: cosine's neighbour sets must differ from
    euclidean's. `_scaled_fixture` gives the rows norms spanning 2^-4 to
    2^4, so a metric that ignores magnitude and one that does not cannot
    agree; if they do, the metric parameter is not reaching the kernel.
    """
    var ctx = DeviceContext()
    var index = _scaled_fixture(E_INDEX, E_FEAT, 31)
    var queries = _scaled_fixture(E_QUERIES, E_FEAT, 97)
    var h_index = _host_f32(ctx, index)
    var h_query = _host_f32(ctx, queries)
    var h_od = ctx.enqueue_create_host_buffer[DType.float32](E_QUERIES * E_K)
    var h_oi = ctx.enqueue_create_host_buffer[DType.uint32](E_QUERIES * E_K)
    ctx.synchronize()
    var sets = List[List[Int]]()
    var checked = 0
    for metric in [
        DIST_L2_SQRT_EXPANDED,
        DIST_COSINE_EXPANDED,
        DIST_L1,
        DIST_LINF,
        DIST_LP_UNEXPANDED,
    ]:
        var pa = Float32(2.0)
        if metric == DIST_LP_UNEXPANDED:
            pa = Float32(3.0)
        _ = knn_search(
            ctx,
            h_index.unsafe_ptr(),
            E_INDEX,
            h_query.unsafe_ptr(),
            E_QUERIES,
            E_FEAT,
            E_K,
            h_od.unsafe_ptr(),
            h_oi.unsafe_ptr(),
            True,
            256,
            2,
            metric,
            pa,
        )
        var truth = _host_true_neighbours(
            index, queries, E_INDEX, E_QUERIES, E_FEAT, E_K, metric,
            Float64(pa),
        )
        var wrong = 0
        for q in range(E_QUERIES):
            for sl in range(E_K):
                var got = Int(h_oi.unsafe_ptr().unsafe_load(q * E_K + sl))
                var found = False
                for t in range(E_K):
                    if truth[q * E_K + t] == got:
                        found = True
                        break
                if not found:
                    wrong += 1
        checked += E_QUERIES * E_K
        if wrong > 0:
            raise Error(
                "check_metric_knn_end_to_end: " + metric_value_name(metric)
                + " returned " + String(wrong) + " of "
                + String(E_QUERIES * E_K) + " neighbours that are not in"
                " the float64 true set"
            )
        var flat = List[Int]()
        for i in range(E_QUERIES * E_K):
            flat.append(Int(h_oi.unsafe_ptr().unsafe_load(i)))
        sets.append(flat^)

    # clause 2: REACH. cosine (index 1) vs euclidean (index 0).
    var same = 0
    for i in range(E_QUERIES * E_K):
        if sets[0][i] == sets[1][i]:
            same += 1
    if same == E_QUERIES * E_K:
        raise Error(
            "check_metric_knn_end_to_end clause 2: cosine and euclidean"
            " returned the SAME neighbour in every one of "
            + String(E_QUERIES * E_K) + " slots on a fixture whose row"
            " norms span 2^-4 to 2^4 -- the metric parameter is not"
            " reaching the kernel"
        )
    print(
        "check_metric_knn_end_to_end OK [" + _mode_name() + "]: 5 metrics x "
        + String(E_QUERIES) + " queries x k=" + String(E_K) + ", "
        + String(checked) + " returned neighbours all in the float64 true"
        " set, cosine differs from euclidean in "
        + String(E_QUERIES * E_K - same) + " of "
        + String(E_QUERIES * E_K) + " slots"
    )
    _ = h_index^
    _ = h_query^
    _ = h_od^
    _ = h_oi^
    _ = ctx^


# ===========================================================================
# 7 & 8. THE DISTANCE WEIGHTS: the rule, and the row-level sabotage.
# ===========================================================================


def check_knn_distance_weights() raises:
    """`_get_weights(dist, 'distance')`, `_base.py:108-113`.

    Clause 1: `host_distance_weights` equals `oracle_distance_weights` --
    two spellings of their three sentences, one through `identical_div`
    and one through a plain `/`.

    Clause 2: THE PLANTED DUPLICATE. Row 3 of the distance fixture has an
    exact 0.0 in slot 0. sklearn's rule replaces THE WHOLE ROW by its
    infinity mask, so that row must come back `[1, 0, 0, ...]` and NOT
    `[huge, 1/d1, 1/d2, ...]`.

    Clause 3: THE SABOTAGE, and it is the reason clause 2 is written as a
    row and not as an element. A PER-ELEMENT rule -- "set the zero slot's
    weight to 1 and leave the rest alone" -- is spelled out below and must
    DIFFER from the shipped rule on this fixture. PREDICTED when the seam
    is broken (i.e. if `host_distance_weights` ever loses its row-level
    replacement): the two agree, `moved == 0`, and this clause raises.
    Every fixture with no duplicate point would pass either way, which is
    exactly the trap.
    """
    var nq = 6
    var k = 5
    var dist = List[Float32](capacity=nq * k)
    for i in range(nq):
        for j in range(k):
            dist.append(
                Float32(0.25) * Float32(j + 1) + Float32(0.03) * Float32(i)
            )
    # clause 2's plant: an exact zero in row 3, slot 0 (the sorted matrix
    # puts a duplicate point at slot 0 by construction).
    dist[3 * k + 0] = Float32(0.0)

    var h_w = host_distance_weights(dist, nq, k)
    var orc = oracle_distance_weights(dist, nq, k)

    var diff = 0
    for i in range(nq * k):
        if bitcast[DType.uint32](h_w[i]) != bitcast[DType.uint32](orc[i]):
            diff += 1
    if diff > 0:
        raise Error(
            "check_knn_distance_weights clause 1: the shipped weights and"
            " the scikit-learn transcription differ on " + String(diff)
            + " of " + String(nq * k) + " entries"
        )

    # clause 2
    if h_w[3 * k + 0] != Float32(1.0):
        raise Error(
            "check_knn_distance_weights clause 2: the exact-zero slot got"
            " weight " + _hex32(h_w[3 * k + 0]) + ", expected 1.0"
        )
    for j in range(1, k):
        if h_w[3 * k + j] != Float32(0.0):
            raise Error(
                "check_knn_distance_weights clause 2: slot " + String(j)
                + " of the row containing an exact zero got "
                + _hex32(h_w[3 * k + j])
                + ", expected 0.0. sklearn REPLACES THE WHOLE ROW by its"
                " infinity mask (_base.py:113), it does not merely clamp"
                " the zero element."
            )

    # clause 3: the per-element rule a careless port would write
    var per_elem = List[Float32](capacity=nq * k)
    for i in range(nq):
        for j in range(k):
            var d = dist[i * k + j]
            per_elem.append(
                Float32(1.0) if d == Float32(0.0)
                else Float32(1.0) / d
            )
    var moved = 0
    for i in range(nq * k):
        if bitcast[DType.uint32](per_elem[i]) != bitcast[DType.uint32](
            h_w[i]
        ):
            moved += 1
    if moved == 0:
        raise Error(
            "check_knn_distance_weights clause 3 SABOTAGE FAILED TO"
            " REGISTER: the per-ELEMENT zero rule and the shipped"
            " row-level rule agree on all " + String(nq * k) + " entries,"
            " so this fixture cannot tell them apart -- the planted"
            " duplicate at row 3 slot 0 is not doing its job"
        )
    print(
        "check_knn_distance_weights OK [" + _mode_name() + "]: "
        + String(nq * k) + " weights bit-equal to the scikit-learn"
        " transcription, the planted zero row is [1,0,0,0,0], and the"
        " per-element rule differs on " + String(moved) + " entries"
    )


def check_knn_weighted_vote_moves_a_prediction() raises:
    """REACH FOR THE WEIGHTED ARM, end to end.

    A fixture built so the two arms MUST disagree: `k = 5` neighbours of
    which THREE are far away and share class 1 and TWO are very close and
    share class 0. A uniform vote gives class 1 (3 votes to 2); a distance
    weight gives class 0, because `1/d` for the two near points dominates.

    PREDICTED when the seam is broken (`weights` dropped on the floor
    somewhere between the Python surface and `weighted_class_probs_kernel`,
    or the weight buffer never uploaded): both arms return class 1 and this
    raises. There is no tolerance here and no oracle needed -- the two
    answers are different INTEGERS.

    Clause 2 does the same for the regressor, where the difference is a
    float and the assertion is that the weighted mean is strictly nearer
    the near points' target.
    """
    var ctx = DeviceContext()
    var d = 2
    var n_index = 5
    var n_queries = 1
    # index rows: two at distance ~0.01 (class 0), three at ~1.0 (class 1)
    var index: List[Float32] = [
        Float32(0.01), Float32(0.0),
        Float32(0.0), Float32(0.01),
        Float32(1.0), Float32(0.0),
        Float32(0.0), Float32(1.0),
        Float32(0.7), Float32(0.7),
    ]
    var queries: List[Float32] = [Float32(0.0), Float32(0.0)]
    var y: List[Int32] = [Int32(0), Int32(0), Int32(1), Int32(1), Int32(1)]
    var yr: List[Float32] = [
        Float32(0.0), Float32(0.0), Float32(10.0), Float32(10.0),
        Float32(10.0),
    ]

    var n_classes: List[Int] = [2]
    var h_index = _host_f32(ctx, index)
    var h_query = _host_f32(ctx, queries)
    var h_y = _host_i32(ctx, y)
    var h_yr = _host_f32(ctx, yr)
    var h_out_u = ctx.enqueue_create_host_buffer[DType.int32](1)
    var h_out_w = ctx.enqueue_create_host_buffer[DType.int32](1)
    var h_proba = ctx.enqueue_create_host_buffer[DType.float32](2)
    var h_uniq = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.synchronize()

    _ = knn_classifier_predict(
        ctx, h_index.unsafe_ptr(), n_index, h_query.unsafe_ptr(), n_queries,
        d, 5, h_y.unsafe_ptr(), 1, n_classes, h_out_u.unsafe_ptr(),
        h_proba.unsafe_ptr(), h_uniq.unsafe_ptr(), False, 256,
        DIST_L2_SQRT_EXPANDED, Float32(2.0), WEIGHTS_UNIFORM,
    )
    _ = knn_classifier_predict(
        ctx, h_index.unsafe_ptr(), n_index, h_query.unsafe_ptr(), n_queries,
        d, 5, h_y.unsafe_ptr(), 1, n_classes, h_out_w.unsafe_ptr(),
        h_proba.unsafe_ptr(), h_uniq.unsafe_ptr(), False, 256,
        DIST_L2_SQRT_EXPANDED, Float32(2.0), WEIGHTS_DISTANCE,
    )
    var out_u = h_out_u.unsafe_ptr().unsafe_load(0)
    var out_w = h_out_w.unsafe_ptr().unsafe_load(0)
    if out_u != Int32(1):
        raise Error(
            "check_knn_weighted_vote_moves_a_prediction: the UNIFORM arm"
            " predicted " + String(out_u) + ", expected 1 (three of five"
            " neighbours carry class 1). The fixture is wrong, not the"
            " weighting."
        )
    if out_w != Int32(0):
        raise Error(
            "check_knn_weighted_vote_moves_a_prediction SABOTAGE ARM: the"
            " DISTANCE-weighted arm predicted " + String(out_w)
            + ", expected 0. The two near neighbours are at ~0.01 and the"
            " three far ones at ~1.0, so 1/d gives class 0 about 100x the"
            " weight. Getting 1 here means `weights` never reached"
            " `weighted_class_probs_kernel`."
        )

    var h_ru = ctx.enqueue_create_host_buffer[DType.float32](1)
    var h_rw = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.synchronize()
    _ = knn_regressor_predict(
        ctx, h_index.unsafe_ptr(), n_index, h_query.unsafe_ptr(), n_queries,
        d, 5, h_yr.unsafe_ptr(), 1, h_ru.unsafe_ptr(), 256,
        DIST_L2_SQRT_EXPANDED, Float32(2.0), WEIGHTS_UNIFORM,
    )
    _ = knn_regressor_predict(
        ctx, h_index.unsafe_ptr(), n_index, h_query.unsafe_ptr(), n_queries,
        d, 5, h_yr.unsafe_ptr(), 1, h_rw.unsafe_ptr(), 256,
        DIST_L2_SQRT_EXPANDED, Float32(2.0), WEIGHTS_DISTANCE,
    )
    var ru = h_ru.unsafe_ptr().unsafe_load(0)
    var rw = h_rw.unsafe_ptr().unsafe_load(0)
    if not (rw < ru):
        raise Error(
            "check_knn_weighted_vote_moves_a_prediction clause 2: the"
            " weighted regression gave " + String(rw) + " and the"
            " uniform one " + String(ru) + "; the weighted mean must be"
            " STRICTLY nearer the two near points' target of 0.0"
        )
    print(
        "check_knn_weighted_vote_moves_a_prediction OK [" + _mode_name()
        + "]: uniform predicts class " + String(out_u)
        + " and distance-weighted predicts class " + String(out_w)
        + "; regression " + String(ru) + " -> " + String(rw)
    )
    _ = h_index^
    _ = h_query^
    _ = h_y^
    _ = h_yr^
    _ = h_out_u^
    _ = h_out_w^
    _ = h_proba^
    _ = h_uniq^
    _ = h_ru^
    _ = h_rw^
    _ = ctx^


def main() raises:
    print("== neighbors/checks/metric_check.mojo [" + _mode_name() + "] ==")
    check_metric_refusals()
    check_metric_device_equals_oracle()
    check_metric_matches_float64_reference()
    check_metric_cosine_norm_flag()
    check_metric_arg_is_reached()
    check_metric_knn_end_to_end()
    check_knn_distance_weights()
    check_knn_weighted_vote_moves_a_prediction()
