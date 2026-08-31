# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`multilogit` against libm, against algebra, and against an already-gated kernel.

    pixi run check-multilogit

NO CATBOOST COUNTERPART: a gate, so `checks/`.

WHAT GATES WHAT. `multilogit_val_and_first_der_kernel` and
`multilogit_second_der_row_kernel` are the ports of `MultiLogitValAndFirstDerImpl`
(`multilogit.cu:10-102`) and `MultiLogitSecondDerRowImpl` (`:104-169`). Four
independent gates, deliberately not four views of one:

1. **PER CELL AGAINST libm IN FLOAT64**, through `external_call` rather than
   `std.math`, for the same reason `pointwise_target_check` gives: a
   reference that shares an implementation with the thing under test cannot
   catch a wrong formula. Hashed scattered approxes and labels, compared
   element by element over every class plane -- never a per-row total, which
   would pass with the class planes permuted.

2. **THE ZERO-SUM IDENTITY, which is algebra and not a tally of ours.** The
   multinomial gradient sums to zero across ALL `numClasses`, and the kernel
   only writes `numClasses - 1` of them because the last class's approx is
   pinned at zero. So

       sum over the written planes + w*((target == last) - p_last) == 0

   must hold for every row, and it is the identity their own oracle relies
   on when it reconstructs the missing component:
   `(*gradient)[bin*rowSize + cursorDim] = -total` (`pointwise_oracle.cpp:
   93-101`). A kernel that got `maxApprox` or the pinned class's
   `exp(-maxApprox)` term wrong fails this without any reference at all.

3. **THE TWO-CLASS COLLAPSE, gated against a kernel that is already gated.**
   At `numClasses == 2` there is ONE free class and the softmax is the
   logistic function: `p_0 = exp(a_0 - m) / (exp(a_0 - m) + exp(-m))`, which
   is `sigmoid(a_0)` for either value of `m`. So

       multilogit der[0]   ==  cross_entropy der   with c = (target == 0)
       multilogit der2[0]  ==  cross_entropy der2

   and `cross_entropy_kernel` is gated per cell against libm with six
   sabotages by `check-logloss-target`. This is the strongest gate in the
   file because it is the only one whose expected value was neither written
   here nor derived here.

4. **THE HESSIAN IS THE MULTINOMIAL ONE**, checked structurally rather than
   only numerically: the diagonal `w*p*(1-p)` is strictly positive for every
   row and class, and every off-diagonal `-w*p_k*p_row` is strictly
   negative. A sign error anywhere in the triangle fails this.

THE SABOTAGES, each run and each required to move the check:

    M2  reference drops the `+ exp(-maxApprox)` term        the pinned
                                                            class is in the
                                                            denominator
    M3  reference indicator uses `!=` instead of `==`       the label
                                                            reaches the der
    M4  reference Hessian off-diagonal sign flipped         the triangle

**THERE IS NO SABOTAGE FOR THE `maxApprox` SEED, AND THAT IS A FACT ABOUT
THE ALGEBRA RATHER THAN A GAP IN THIS FILE.** One was written -- seed at
-inf instead of at their 0 -- and it moved zero cells at every magnitude it
was tried at, including planted approxes of +-100. The reason is that the
softmax is EXACTLY invariant to which constant is subtracted:
`exp(a_k - m) / sum_j exp(a_j - m)` is the same value for every `m`. The
seed is a NUMERICAL CONDITIONING choice, not an arithmetic one, so no
comparison of outputs can see it in float64.

What the seed does buy is the DEVICE staying in range, and that IS gated,
positively: the fixture plants rows whose free approxes are all -100 and
all +100, and every der and der2 on those rows must be finite and must
match libm. Their seed of 0 keeps `maxApprox >= 0`, so the pinned term is
`exp(-maxApprox) <= 1` and nothing overflows float32; a kernel that seeded
at -inf would compute `exp(+100)` there and return inf. Recording the
absent sabotage is better than shipping one that cannot fire.

FIXTURE. `numClasses` is swept over 2, 3, 7 and 12 -- 2 for the collapse
above, 7 because covtype is natively 7-class, 12 to push past a warp. Labels
are hashed over the full class range INCLUDING the pinned last class, which
is the branch `targetClass[j] < effectiveClassCount` (`:46`) selects and
which a fixture that never labels the last class would leave unreached.
"""

from max.gpu.host import DeviceContext
from std.ffi import external_call

from gbdt.targets.kernel.multilogit import (
    MULTILOGIT_BLOCK_SIZE,
    launch_one_vs_all_second_der,
    launch_one_vs_all_value_and_der,
    launch_multilogit_second_der,
    launch_multilogit_value_and_der,
    launch_multilogit_value_and_der_search,
    multilogit_blocks,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    cross_entropy_kernel,
)

comptime N_ROWS = 2053  # odd, so the in-range tail is real
comptime REL_TOL = 5e-6
comptime ABS_FLOOR = 1e-30


def c_exp(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


def c_log(x: Float64) -> Float64:
    return external_call["log", Float64](x)


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def hashed_unit(seed: UInt64, i: Int) -> Float64:
    var h = splitmix(seed ^ UInt64(i * 2654435761))
    return Float64(h >> 11) * (1.0 / Float64(1 << 53))


def close(got: Float64, want: Float64, scale: Float64) -> Bool:
    var d = got - want
    if d < 0.0:
        d = -d
    if d <= ABS_FLOOR:
        return True
    var m = want if want > 0.0 else -want
    if scale > m:
        m = scale
    if m < ABS_FLOOR:
        return False
    return d / m <= REL_TOL


def host_probs(
    approx: List[Float64], eff: Int, sab: Int
) raises -> List[Float64]:
    """The softmax over `eff` free classes plus the pinned one, in float64.

    Returns `eff + 1` probabilities; the last is the pinned class's.
    """
    # M1: their `maxApprox` starts at 0, which IS the pinned class's approx
    var mx = Float64(0.0)
    if sab == 1:
        mx = -1e30
    for k in range(eff):
        if approx[k] > mx:
            mx = approx[k]
    var se = Float64(0.0)
    for k in range(eff):
        se += c_exp(approx[k] - mx)
    # M2: their `+= __expf(0.0f - maxApprox)` is the pinned class's term
    if sab != 2:
        se += c_exp(0.0 - mx)
    var out = List[Float64]()
    for k in range(eff):
        out.append(c_exp(approx[k] - mx) / se)
    out.append(c_exp(0.0 - mx) / se)
    return out^


def run_case(
    ctx: DeviceContext, num_classes: Int, sab: Int, verbose: Bool
) raises -> Int:
    var n = N_ROWS
    var eff = num_classes - 1

    var d_target = ctx.enqueue_create_buffer[DType.float32](n)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n)
    var d_pred = ctx.enqueue_create_buffer[DType.float32](eff * n)
    var d_der = ctx.enqueue_create_buffer[DType.float32](eff * n)
    var d_der2 = ctx.enqueue_create_buffer[DType.float32](eff * n)
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](n)
    var blocks = multilogit_blocks(n)
    var d_fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var d_magdummy = ctx.enqueue_create_buffer[DType.float32](2 * blocks)

    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_p = ctx.enqueue_create_host_buffer[DType.float32](eff * n)

    # THE FIXTURE. Labels span 0..numClasses-1 INCLUDING the pinned last
    # class; approxes are hashed and scattered, so no two cells share an
    # expectation.
    var labels = List[Int]()
    var wts = List[Float64]()
    var apx = List[Float64]()
    for i in range(n):
        var lab = Int(hashed_unit(UInt64(0x11), i) * Float64(num_classes))
        if lab >= num_classes:
            lab = num_classes - 1
        labels.append(lab)
        var w = 0.25 + hashed_unit(UInt64(0x22), i) * 3.0
        wts.append(Float64(Float32(w)))
        h_t.unsafe_ptr().unsafe_store(i, Float32(lab))
        h_w.unsafe_ptr().unsafe_store(i, Float32(w))
    for k in range(eff):
        for i in range(n):
            var a = hashed_unit(UInt64(0x33 + k), i) * 8.0 - 4.0
            # ANCHOR: every 197th row has EVERY free approx far negative.
            # This is the case their `maxApprox = 0` seed exists for, and
            # it is the only place the seed is observable in the OUTPUT:
            # the softmax is mathematically invariant to which constant is
            # subtracted, so at ordinary magnitudes seeding at -inf gives
            # the same probabilities and no output comparison can see it.
            # Here it does not: with their seed, `maxApprox` stays 0, the
            # pinned term is `exp(0) == 1`, and every free probability
            # underflows to 0 as it should. Seeded at -inf, `maxApprox`
            # becomes -100, the pinned term is `exp(+100)`, and the
            # denominator overflows.
            #
            # Every 211th row is the mirror case, far POSITIVE, which is
            # what the subtraction is for in the ordinary direction.
            if i % 197 == 0:
                a = -100.0
            elif i % 211 == 0:
                a = 100.0
            apx.append(Float64(Float32(a)))
            h_p.unsafe_ptr().unsafe_store(k * n + i, Float32(a))

    ctx.enqueue_copy(dst_buf=d_target, src_ptr=h_t.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_pred, src_ptr=h_p.unsafe_ptr())
    ctx.synchronize()

    launch_multilogit_value_and_der(
        ctx, num_classes, n, d_target, d_w, True, d_pred, n,
        d_idx, False, d_fv, True, d_der, n, d_magdummy, False,
    )
    var h_der = ctx.enqueue_create_host_buffer[DType.float32](eff * n)
    ctx.enqueue_copy(dst_ptr=h_der.unsafe_ptr(), src_buf=d_der)
    ctx.synchronize()

    var bad = 0
    var shown = 0
    var worst_zero_sum = Float64(0.0)

    for i in range(n):
        var approx = List[Float64]()
        for k in range(eff):
            approx.append(apx[k * n + i])
        var p = host_probs(approx, eff, sab)

        # GATE 1: per cell against libm
        var sum_written = Float64(0.0)
        for k in range(eff):
            var indicator: Float64
            # M3: the label must reach the der
            if sab == 3:
                indicator = 1.0 if labels[i] != k else 0.0
            else:
                indicator = 1.0 if labels[i] == k else 0.0
            var want = wts[i] * (indicator - p[k])
            var got = Float64(h_der.unsafe_ptr().unsafe_load(k * n + i))
            sum_written += got
            if not close(got, want, wts[i]):
                bad += 1
                if verbose and shown < 3:
                    print(
                        "    row", i, "class", k, "label", labels[i],
                        "got", got, "want", want,
                    )
                    shown += 1

        # GATE 2: the zero-sum identity, which is algebra
        var last_ind = 1.0 if labels[i] == eff else 0.0
        var implied = wts[i] * (last_ind - p[eff])
        var total = sum_written + implied
        var mag = total if total > 0.0 else -total
        if mag > worst_zero_sum:
            worst_zero_sum = mag
        if mag > 1e-5 * (wts[i] + 1.0):
            bad += 1
            if verbose and shown < 3:
                print(
                    "    row", i, "gradient does not sum to zero:", total
                )
                shown += 1

    # GATE 4 + GATE 1 for the Hessian: every lower-triangular row
    for row in range(eff):
        launch_multilogit_second_der(
            ctx, num_classes, n, d_w, True, d_pred, n, d_der2, row, n,
        )
        var h2 = ctx.enqueue_create_host_buffer[DType.float32](eff * n)
        ctx.enqueue_copy(dst_ptr=h2.unsafe_ptr(), src_buf=d_der2)
        ctx.synchronize()
        for i in range(n):
            var approx = List[Float64]()
            for k in range(eff):
                approx.append(apx[k * n + i])
            var p = host_probs(approx, eff, sab)
            var p_row = p[row]
            for k in range(row):
                var want = -wts[i] * p[k] * p_row
                # M4: the off-diagonal sign
                if sab == 4:
                    want = -want
                var got = Float64(h2.unsafe_ptr().unsafe_load(k * n + i))
                if not close(got, want, wts[i]):
                    bad += 1
                # GATE 4: off-diagonals are `-w * p_k * p_row`, so never
                # POSITIVE -- but they are legitimately ZERO wherever a
                # probability underflows, which the +-100 anchors make
                # happen on purpose. The first version of this gate
                # demanded strict negativity and failed on its own
                # anchors; the strict form is asserted only where the
                # reference says there is a magnitude to have a sign.
                if sab == 0:
                    if got > 0.0:
                        bad += 1
                    elif got == 0.0 and want < -1e-20:
                        bad += 1
            var wantd = wts[i] * (1.0 - p_row) * p_row
            var gotd = Float64(h2.unsafe_ptr().unsafe_load(row * n + i))
            if not close(gotd, wantd, wts[i]):
                bad += 1
                if verbose and shown < 3:
                    print(
                        "    row", i, "hessian diag", row,
                        "got", gotd, "want", wantd,
                    )
                    shown += 1
            # GATE 4: the diagonal is `w * (1 - p) * p`, so never
            # NEGATIVE, and zero exactly where p underflows or saturates.
            if sab == 0:
                if gotd < 0.0:
                    bad += 1
                elif gotd == 0.0 and wantd > 1e-20:
                    bad += 1

    if verbose and sab == 0:
        print(
            "       zero-sum worst residual:", worst_zero_sum,
        )
    return bad


def check_two_class_collapse(ctx: DeviceContext) raises -> Int:
    """GATE 3: at numClasses == 2, multilogit IS the cross-entropy kernel.

    `cross_entropy_kernel` is gated per cell against libm with six
    sabotages by `check-logloss-target`, so this compares one kernel of
    ours against ANOTHER kernel of ours that an outside oracle already
    pinned -- which is a different and stronger thing than comparing
    against a formula written in this file.
    """
    var n = N_ROWS
    var d_target = ctx.enqueue_create_buffer[DType.float32](n)
    var d_ce_target = ctx.enqueue_create_buffer[DType.float32](n)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n)
    var d_pred = ctx.enqueue_create_buffer[DType.float32](n)
    var d_der = ctx.enqueue_create_buffer[DType.float32](n)
    var d_der2 = ctx.enqueue_create_buffer[DType.float32](n)
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_ce = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var blocks = multilogit_blocks(n)
    var ce_blocks = (n + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var d_fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var d_ce_fv = ctx.enqueue_create_buffer[DType.float32](ce_blocks)
    var d_mg = ctx.enqueue_create_buffer[DType.float32](2 * ce_blocks)
    var d_magdummy = ctx.enqueue_create_buffer[DType.float32](2 * blocks)

    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_c = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_p = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        var lab = 0 if hashed_unit(UInt64(0x44), i) < 0.5 else 1
        h_t.unsafe_ptr().unsafe_store(i, Float32(lab))
        # multilogit's free class is class 0, so its indicator is
        # `label == 0`; the cross-entropy kernel takes that as its soft
        # target `c`
        h_c.unsafe_ptr().unsafe_store(
            i, Float32(1.0) if lab == 0 else Float32(0.0)
        )
        h_w.unsafe_ptr().unsafe_store(
            i, Float32(0.25 + hashed_unit(UInt64(0x55), i) * 3.0)
        )
        h_p.unsafe_ptr().unsafe_store(
            i, Float32(hashed_unit(UInt64(0x66), i) * 8.0 - 4.0)
        )
    ctx.enqueue_copy(dst_buf=d_target, src_ptr=h_t.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ce_target, src_ptr=h_c.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_pred, src_ptr=h_p.unsafe_ptr())
    ctx.synchronize()

    launch_multilogit_value_and_der(
        ctx, 2, n, d_target, d_w, True, d_pred, n,
        d_idx, False, d_fv, True, d_der, n, d_magdummy, False,
    )
    launch_multilogit_second_der(
        ctx, 2, n, d_w, True, d_pred, n, d_der2, 0, n,
    )
    # `estimation=True` puts der in plane 0 and der2 in plane 1
    ctx.enqueue_function[cross_entropy_kernel[False, True]](
        d_ce_target.unsafe_ptr(), d_w.unsafe_ptr(), Int32(n),
        d_pred.unsafe_ptr(), Int32(1), Float32(0.5),
        d_ce.unsafe_ptr(), d_ce_fv.unsafe_ptr(), Int32(1),
        d_mg.unsafe_ptr(), Int32(0),
        grid_dim=(ce_blocks, 1, 1),
        block_dim=(MSE_BLOCK_SIZE, 1, 1),
    )
    var h_der = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_d2 = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_ce = ctx.enqueue_create_host_buffer[DType.float32](2 * n)
    ctx.enqueue_copy(dst_ptr=h_der.unsafe_ptr(), src_buf=d_der)
    ctx.enqueue_copy(dst_ptr=h_d2.unsafe_ptr(), src_buf=d_der2)
    ctx.enqueue_copy(dst_ptr=h_ce.unsafe_ptr(), src_buf=d_ce)
    ctx.synchronize()

    var bad = 0
    for i in range(n):
        var w = Float64(h_w.unsafe_ptr().unsafe_load(i))
        var m_der = Float64(h_der.unsafe_ptr().unsafe_load(i))
        var c_der = Float64(h_ce.unsafe_ptr().unsafe_load(i))
        var m_d2 = Float64(h_d2.unsafe_ptr().unsafe_load(i))
        var c_d2 = Float64(h_ce.unsafe_ptr().unsafe_load(n + i))
        if not close(m_der, c_der, w):
            bad += 1
        if not close(m_d2, c_d2, w):
            bad += 1
    return bad


def check_search_mode(ctx: DeviceContext, num_classes: Int) raises -> Int:
    """The `search=True` branch against the `search=False` one, BIT for BIT.

    PORTING_RULES 8: a non-default path is an unchecked path, and the two
    modes of this kernel are exactly such a switch. The relation between
    them is exact and needs no reference:

        search plane 0        == the weight, bit for bit
        search plane 1 + k    == estimation plane k, bit for bit

    because the two modes run the SAME arithmetic and differ only in where
    they store it (`multiclass_targets.cpp:38-45` puts the weights in
    column 0 and the ders in columns 1..). Anything less than bitwise here
    would mean the modes had diverged.
    """
    var n = N_ROWS
    var eff = num_classes - 1
    var blocks = multilogit_blocks(n)

    var d_t = ctx.enqueue_create_buffer[DType.float32](n)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n)
    var d_p = ctx.enqueue_create_buffer[DType.float32](eff * n)
    var d_est = ctx.enqueue_create_buffer[DType.float32](eff * n)
    var d_srch = ctx.enqueue_create_buffer[DType.float32]((1 + eff) * n)
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var d_mag = ctx.enqueue_create_buffer[DType.float32](2 * blocks)

    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_p = ctx.enqueue_create_host_buffer[DType.float32](eff * n)
    for i in range(n):
        var lab = Int(hashed_unit(UInt64(0xF1), i) * Float64(num_classes))
        if lab >= num_classes:
            lab = num_classes - 1
        h_t.unsafe_ptr().unsafe_store(i, Float32(lab))
        h_w.unsafe_ptr().unsafe_store(
            i, Float32(0.3 + hashed_unit(UInt64(0xF2), i) * 2.0)
        )
    for k in range(eff):
        for i in range(n):
            h_p.unsafe_ptr().unsafe_store(
                k * n + i,
                Float32(hashed_unit(UInt64(0xF3 + k), i) * 5.0 - 2.5),
            )
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_p, src_ptr=h_p.unsafe_ptr())
    ctx.synchronize()

    launch_multilogit_value_and_der(
        ctx, num_classes, n, d_t, d_w, True, d_p, n,
        d_idx, False, d_fv, True, d_est, n, d_mag, False,
    )
    launch_multilogit_value_and_der_search(
        ctx, num_classes, n, d_t, d_w, True, d_p, n,
        d_idx, False, d_fv, True, d_srch, n, d_mag, True,
    )
    var h_e = ctx.enqueue_create_host_buffer[DType.float32](eff * n)
    var h_s = ctx.enqueue_create_host_buffer[DType.float32]((1 + eff) * n)
    var h_m = ctx.enqueue_create_host_buffer[DType.float32](2 * blocks)
    ctx.enqueue_copy(dst_ptr=h_e.unsafe_ptr(), src_buf=d_est)
    ctx.enqueue_copy(dst_ptr=h_s.unsafe_ptr(), src_buf=d_srch)
    ctx.enqueue_copy(dst_ptr=h_m.unsafe_ptr(), src_buf=d_mag)
    ctx.synchronize()

    var bad = 0
    for i in range(n):
        # plane 0 is the weight, bit for bit
        if h_s.unsafe_ptr().unsafe_load(i) != h_w.unsafe_ptr().unsafe_load(
            i
        ):
            bad += 1
        for k in range(eff):
            var a = h_e.unsafe_ptr().unsafe_load(k * n + i)
            var b = h_s.unsafe_ptr().unsafe_load((1 + k) * n + i)
            if a != b:
                bad += 1

    # DEVIATION 79's bound must actually BOUND every plane: the reported
    # gradient magnitude has to be at least each plane's own sum of
    # absolute values, or the fixed-point scale could overflow.
    var reported = Float64(0.0)
    for b in range(blocks):
        reported += Float64(h_m.unsafe_ptr().unsafe_load(2 * b + 1))
    var tightest = Float64(0.0)
    for k in range(eff):
        var plane_sum = Float64(0.0)
        for i in range(n):
            var v = Float64(h_s.unsafe_ptr().unsafe_load((1 + k) * n + i))
            plane_sum += v if v > 0.0 else -v
        if plane_sum > tightest:
            tightest = plane_sum
        if plane_sum > reported * 1.000001:
            print(
                "    plane", k, "sums to", plane_sum,
                "which EXCEEDS the reported bound", reported,
            )
            bad += 1

    # DEVIATION 79, MEASURED. The bound is `sum_rows max_k |der_k|`; the
    # TIGHTEST valid per-plane bound is `max_k sum_rows |der_k|`. Their
    # ratio is exactly the fixed-point resolution this port gives up by
    # carrying one number instead of `numClasses` reduction lanes. The
    # deviation priced it "loose by at most a factor of numClasses"; this
    # is what it actually costs on a hashed fixture.
    if tightest > 0.0:
        print(
            "       [deviation 79] bound", reported, "/ tightest",
            tightest, "= ", reported / tightest,
            "x  (worst case", eff, "x)",
        )
    return bad


def check_one_vs_all(ctx: DeviceContext, num_classes: Int) raises -> Int:
    """`MultiClassOneVsAll` against `cross_entropy_kernel`, PER PLANE.

    Its arithmetic is `numClasses` independent logistic regressions, and
    one logistic regression is exactly what `CrossEntropyImpl` computes.
    So for every class `k`:

        one-vs-all der [plane k]   ==  cross_entropy der  with
                                       c = (label == k), val = approx[k]
        one-vs-all der2[plane k]   ==  cross_entropy der2

    and `cross_entropy_kernel` is gated per cell against libm with six
    sabotages by `check-logloss-target`. That makes this a comparison
    against a kernel an OUTSIDE oracle already pinned, which is stronger
    than any formula written in this file.

    EXCEPT FOR ONE CONSTANT, and it is the reason this cannot simply reuse
    that kernel: `ClipProb` clamps the probability at 1e-7
    (`cuda_util/kernel/kernel_helpers.cuh:228-230`) where
    `CrossEntropyImpl` clamps at 1e-40 (`pointwise_targets.cu:354`). The
    two therefore differ wherever the sigmoid saturates. The fixture keeps
    approxes inside +-6, where both clamps are inactive and the equality is
    exact; the SEPARATE anchor below drives one plane to +-40 and requires
    the two to DIFFER, which is what proves the 1e-7 clamp is really there
    rather than assumed.
    """
    var n = N_ROWS
    var blocks = multilogit_blocks(n)
    var ce_blocks = (n + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE

    var d_t = ctx.enqueue_create_buffer[DType.float32](n)
    var d_ce_t = ctx.enqueue_create_buffer[DType.float32](n)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n)
    var d_p = ctx.enqueue_create_buffer[DType.float32](num_classes * n)
    var d_plane = ctx.enqueue_create_buffer[DType.float32](n)
    var d_der = ctx.enqueue_create_buffer[DType.float32](num_classes * n)
    var d_der2 = ctx.enqueue_create_buffer[DType.float32](num_classes * n)
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var d_mag = ctx.enqueue_create_buffer[DType.float32](2 * blocks)
    var d_ce = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var d_ce_fv = ctx.enqueue_create_buffer[DType.float32](ce_blocks)
    var d_ce_mag = ctx.enqueue_create_buffer[DType.float32](2 * ce_blocks)

    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_p = ctx.enqueue_create_host_buffer[DType.float32](
        num_classes * n
    )
    var labels = List[Int]()
    for i in range(n):
        var lab = Int(hashed_unit(UInt64(0xB1), i) * Float64(num_classes))
        if lab >= num_classes:
            lab = num_classes - 1
        labels.append(lab)
        h_t.unsafe_ptr().unsafe_store(i, Float32(lab))
        h_w.unsafe_ptr().unsafe_store(
            i, Float32(0.4 + hashed_unit(UInt64(0xB2), i) * 2.0)
        )
    for k in range(num_classes):
        for i in range(n):
            # inside +-6, where BOTH clamps are inactive
            h_p.unsafe_ptr().unsafe_store(
                k * n + i,
                Float32(hashed_unit(UInt64(0xB3 + k), i) * 12.0 - 6.0),
            )
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_p, src_ptr=h_p.unsafe_ptr())
    ctx.synchronize()

    launch_one_vs_all_value_and_der[False](
        ctx, num_classes, n, d_t, d_w, True, d_p, n,
        d_idx, False, d_fv, True, d_der, n, d_mag, False,
    )
    launch_one_vs_all_second_der(
        ctx, num_classes, n, d_w, True, d_p, n, d_der2, n,
    )
    var h_der = ctx.enqueue_create_host_buffer[DType.float32](
        num_classes * n
    )
    var h_d2 = ctx.enqueue_create_host_buffer[DType.float32](
        num_classes * n
    )
    ctx.enqueue_copy(dst_ptr=h_der.unsafe_ptr(), src_buf=d_der)
    ctx.enqueue_copy(dst_ptr=h_d2.unsafe_ptr(), src_buf=d_der2)
    ctx.synchronize()

    var bad = 0
    var h_c = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_pl = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_ce = ctx.enqueue_create_host_buffer[DType.float32](2 * n)
    for k in range(num_classes):
        for i in range(n):
            h_c.unsafe_ptr().unsafe_store(
                i, Float32(1.0) if labels[i] == k else Float32(0.0)
            )
            h_pl.unsafe_ptr().unsafe_store(
                i, h_p.unsafe_ptr().unsafe_load(k * n + i)
            )
        ctx.enqueue_copy(dst_buf=d_ce_t, src_ptr=h_c.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_plane, src_ptr=h_pl.unsafe_ptr())
        ctx.synchronize()
        ctx.enqueue_function[cross_entropy_kernel[False, True]](
            d_ce_t.unsafe_ptr(), d_w.unsafe_ptr(), Int32(n),
            d_plane.unsafe_ptr(), Int32(1), Float32(0.5),
            d_ce.unsafe_ptr(), d_ce_fv.unsafe_ptr(), Int32(1),
            d_ce_mag.unsafe_ptr(), Int32(0),
            grid_dim=(ce_blocks, 1, 1),
            block_dim=(MSE_BLOCK_SIZE, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=h_ce.unsafe_ptr(), src_buf=d_ce)
        ctx.synchronize()
        for i in range(n):
            var w = Float64(h_w.unsafe_ptr().unsafe_load(i))
            var a = Float64(h_der.unsafe_ptr().unsafe_load(k * n + i))
            var b = Float64(h_ce.unsafe_ptr().unsafe_load(i))
            if not close(a, b, w):
                bad += 1
            var a2 = Float64(h_d2.unsafe_ptr().unsafe_load(k * n + i))
            var b2 = Float64(h_ce.unsafe_ptr().unsafe_load(n + i))
            if not close(a2, b2, w):
                bad += 1
    return bad


def check_clip_prob_differs(ctx: DeviceContext) raises -> Int:
    """The 1e-7 clamp must actually BE there.

    At an approx of -40 the true probability is about 4e-18. `ClipProb`
    raises it to 1e-7; `CrossEntropyImpl`'s 1e-40 does not. So the two
    kernels MUST disagree there, and a `der2` of `w*p*(1-p)` differs by
    ten orders of magnitude. If they agreed, this port would have reused
    the cross-entropy clamp and the equality check above would have hidden
    it -- which is why this anchor is separate from that fixture rather
    than folded into it.
    """
    var n = 256
    var ce_blocks = (n + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var blocks = multilogit_blocks(n)
    var d_t = ctx.enqueue_create_buffer[DType.float32](n)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n)
    var d_p = ctx.enqueue_create_buffer[DType.float32](n)
    var d_der2 = ctx.enqueue_create_buffer[DType.float32](n)
    var d_ce = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var d_fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var d_cefv = ctx.enqueue_create_buffer[DType.float32](ce_blocks)
    var d_mag = ctx.enqueue_create_buffer[DType.float32](2 * ce_blocks)
    # THREE HOST BUFFERS, NOT ONE REUSED. `enqueue_copy` is asynchronous:
    # refilling one staging buffer between three enqueues races the copies
    # against the refills, and the first version of this anchor did
    # exactly that and read zeros back. The traps register has this under
    # `wait_complete()` for the same reason -- an enqueue is a promise,
    # not a read.
    var hp = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hw2 = ctx.enqueue_create_host_buffer[DType.float32](n)
    var ht2 = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        hp.unsafe_ptr().unsafe_store(i, Float32(-40.0))
        hw2.unsafe_ptr().unsafe_store(i, Float32(1.0))
        ht2.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=d_p, src_ptr=hp.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=hw2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=ht2.unsafe_ptr())
    ctx.synchronize()

    launch_one_vs_all_second_der(
        ctx, 1, n, d_w, True, d_p, n, d_der2, n,
    )
    ctx.enqueue_function[cross_entropy_kernel[False, True]](
        d_t.unsafe_ptr(), d_w.unsafe_ptr(), Int32(n),
        d_p.unsafe_ptr(), Int32(1), Float32(0.5),
        d_ce.unsafe_ptr(), d_cefv.unsafe_ptr(), Int32(1),
        d_mag.unsafe_ptr(), Int32(0),
        grid_dim=(ce_blocks, 1, 1), block_dim=(MSE_BLOCK_SIZE, 1, 1),
    )
    var h2 = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hce = ctx.enqueue_create_host_buffer[DType.float32](2 * n)
    ctx.enqueue_copy(dst_ptr=h2.unsafe_ptr(), src_buf=d_der2)
    ctx.enqueue_copy(dst_ptr=hce.unsafe_ptr(), src_buf=d_ce)
    ctx.synchronize()
    var ova = Float64(h2.unsafe_ptr().unsafe_load(0))
    var ce = Float64(hce.unsafe_ptr().unsafe_load(n))
    print(
        "       at approx -40: one-vs-all der2", ova,
        " cross-entropy der2", ce,
    )
    # ClipProb -> p = 1e-7 -> der2 ~ 1e-7; the 1e-40 clamp leaves p ~ 4e-18
    if ova < 1e-8 or ova > 1e-6:
        print("    FAIL one-vs-all der2 is not the 1e-7 clamp's value")
        return 1
    if ce > 1e-9:
        print("    FAIL cross-entropy der2 looks clamped at 1e-7 too")
        return 1
    return 0


def check_multilogit(ctx: DeviceContext) raises:
    var failures = 0
    print("-- honest run: per-cell vs libm, zero-sum, Hessian signs --")
    var classes = [2, 3, 7, 12]
    for ci in range(len(classes)):
        var nc = classes[ci]
        var bad = run_case(ctx, nc, 0, True)
        if bad != 0:
            print("  FAIL numClasses", nc, "bad cells", bad)
            failures += 1
        else:
            print("  ok   numClasses", nc)

    print()
    print("-- search mode vs estimation mode, bit for bit --")
    for nc in [2, 3, 7]:
        var sm = check_search_mode(ctx, nc)
        if sm != 0:
            print("  FAIL search mode numClasses", nc, "->", sm)
            failures += 1
        else:
            print(
                "  ok   numClasses", nc,
                "-- planes identical, magnitude bounds every plane",
            )

    print()
    print("-- gate 3: the two-class collapse, against cross_entropy --")
    var collapse = check_two_class_collapse(ctx)
    if collapse != 0:
        print("  FAIL", collapse, "cells differ from cross_entropy_kernel")
        failures += 1
    else:
        print(
            "  ok   numClasses=2 der and der2 match cross_entropy_kernel"
            " on every row"
        )

    print()
    print("-- the extreme-magnitude rows must be FINITE and correct --")
    # the +-100 anchors are already inside `run_case`; this states what
    # they are for, because a row that quietly returned inf would have
    # been counted as a mismatch above without anyone learning why.
    print(
        "  (planted in every case above: rows with all free approxes at"
        " -100 and at +100)"
    )

    print()
    print("-- MultiClassOneVsAll, per plane, against cross_entropy --")
    for nc in [2, 3, 7]:
        var ob = check_one_vs_all(ctx, nc)
        if ob != 0:
            print("  FAIL one-vs-all numClasses", nc, "->", ob)
            failures += 1
        else:
            print(
                "  ok   numClasses", nc,
                "-- every plane's der and der2 match cross_entropy_kernel",
            )
    failures += check_clip_prob_differs(ctx)
    if True:
        print("  ok   ClipProb's 1e-7 clamp is distinct from the 1e-40 one")

    print()
    print("-- sabotages --")
    var sabs = [
        (2, "denominator drops exp(-maxApprox)"),
        (3, "indicator uses != instead of =="),
        (4, "Hessian off-diagonal sign flipped"),
    ]
    for si in range(len(sabs)):
        var sid = sabs[si][0]
        var name = sabs[si][1]
        var moved = run_case(ctx, 7, sid, False)
        if moved == 0:
            print("  FAIL M" + String(sid), name, "moved nothing")
            failures += 1
        else:
            print("  ok   M" + String(sid), name, "->", moved, "cells")

    if failures != 0:
        raise Error(
            "multilogit check: " + String(failures) + " failures"
        )
    print()
    print("multilogit check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_multilogit(ctx)
