# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`bins.mojo` and `objectives.mojo` against arithmetic nobody in this repo wrote.

    tools/with_build_lock.sh pixi run mojo run -I . \
        ensemble/checks/objectives_check.mojo

NO CUML COUNTERPART. cuML gates these functions with C++ unit tests that
compare against cuML.

WHAT THIS GATES AGAINST, and it is never our own tally
-------------------------------------------------------
1. **HAND-COMPUTED RATIONALS.** The Gini and MSE fixtures below are small
   enough to do on paper, and the expected values are pasted as exact
   rationals: Gini at bin 1 is 49/810, at bin 2 is 1/567; MSE at bin 1 is
   1/32, at bin 2 is 1/96. MSE at bin 1 is asserted with EXACT EQUALITY,
   not a tolerance, because every operand in that cell is a dyadic
   rational and float32 has to land on it bit for bit.
2. **AN INDEPENDENT INTEGER TALLY** for the histogram arms, computed on
   the host by a loop that shares no code with the kernel.
3. **CPython float64 anchors** for Entropy, Poisson and Gamma, which
   contain a transcendental. Those are compared loosely on purpose;
   DEVIATION 113 says the device `log` is not cuML's and the tolerance is
   the size of that admission, not a hiding place.

THE SCAR THIS FILE IS BUILT AROUND
-----------------------------------
STANDING_ORDERS rule 8. A histogram kernel in this repository once read
`0 wrong of 512` with uniform bins and `490 wrong of 512` with hashed
bins -- the same kernel, the same parameters -- and two earlier checks had
already reported it correct at exactly the failing configuration.

So arm A does not merely plant hashed values, it RUNS BOTH and reports
both. `nclasses` and `n_bins` are both 4, which makes the histogram
square, which makes a TRANSPOSED write (`b * nclasses + label` instead of
their `label * n_bins + b`, `bins.cuh:26`) produce an identical uniform
tally. The uniform arm therefore PASSES the transposed kernel, and the
check says so out loud before failing it on the hashed one. A gate that
cannot see the difference is not evidence.

THE SABOTAGES, one per mechanism
---------------------------------
S1 (histogram placement): a transposed-offset kernel. Must pass uniform
   and fail hashed.
S2 (fixed-point struct layout): an `AtomicAdd` that writes the label sum
   into the count's slot and vice versa. Must fail per cell.
S3 (three-field layout): an `AtomicAdd` that rotates the three slots of
   `WeightedRegressionBin`. Must fail per cell.
S4 (gain arithmetic and its tolerance): one histogram cell perturbed by
   ONE COUNT. The gain must move further than the tolerance the passing
   assertions use -- otherwise those tolerances are blind.
S5 (the quantizer's rounding rule): the same measured plane scored
   against a FLOORING expectation instead of a truncating one. Must
   mismatch, or arm B is not testing the rule DEVIATION 101b chose.
Each sabotage is a FAILURE of this check if it does not move the result.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace

from ensemble.decisiontree.batched_levelalgo.bins import (
    BinScales,
    ClassificationBin,
    RegressionBin,
    WeightedRegressionBin,
)
from ensemble.decisiontree.batched_levelalgo.objectives import (
    CRITERION_ENTROPY,
    CRITERION_GAMMA,
    CRITERION_GINI,
    CRITERION_INVERSE_GAUSSIAN,
    CRITERION_MSE,
    CRITERION_POISSON,
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)

comptime CObj = ClassificationObjectiveFunction[
    DType.float32, DType.int32, ClassificationBin
]
comptime RObj = RegressionObjectiveFunction[
    DType.float32, DType.float32, RegressionBin
]

# --- arm A geometry. SQUARE ON PURPOSE: see the module docstring. -------
comptime A_NCLASSES = 4
comptime A_NBINS = 4
comptime A_CELLS = A_NCLASSES * A_NBINS
comptime A_BLOCK = 256
comptime A_BLOCKS = 16
comptime A_N = A_BLOCK * A_BLOCKS

# --- arm B/C geometry ---------------------------------------------------
comptime B_NBINS = 8
comptime B_BLOCK = 256
comptime B_BLOCKS = 16
comptime B_N = B_BLOCK * B_BLOCKS
#: Powers of two, exactly as `checks/fixed_point.mojo::choose_scale`
#: returns them: the dequantize division is then exact.
#:
#: THESE ARE DELIBERATELY TOO COARSE TO MAKE THE QUANTIZATION EXACT. With
#: labels in units of 1/4 and a scale of 16 every scaled value would land
#: on an integer, and then `_quantize`'s truncation-toward-zero would be
#: indistinguishable from rounding -- the arms below would pass a bin file
#: that rounded. At scale 2 an odd unit lands on a half, so truncation is
#: observable, and it is observable in BOTH SIGNS because the label units
#: run from -20 to +20.
comptime B_LABEL_SCALE = Float32(2.0)
comptime C_LABEL_SCALE = Float32(4.0)
comptime C_WEIGHT_SCALE = Float32(2.0)
#: Arm E/F/H hold their fixture at a scale that IS exact, because there
#: the point is the gain arithmetic and not the quantizer.
comptime MSE_LABEL_SCALE = Float32(16.0)


@always_inline
def _hash(x: UInt64) -> UInt64:
    """One multiply-shift. Shared by the kernel and the host tally on
    purpose -- what must NOT be shared is the placement arithmetic."""
    var h = x * 0x9E3779B97F4A7C15
    return h ^ (h >> 29)


@always_inline
def _a_label(tid: Int, hashed: Bool) -> Int32:
    if hashed:
        return Int32(Int((_hash(UInt64(tid)) >> 13) % UInt64(A_NCLASSES)))
    return Int32((tid // A_NBINS) % A_NCLASSES)


@always_inline
def _a_bin(tid: Int, hashed: Bool) -> Int32:
    if hashed:
        return Int32(Int((_hash(UInt64(tid) + 7919) >> 17) % UInt64(A_NBINS)))
    return Int32(tid % A_NBINS)


@always_inline
def _b_bin(tid: Int) -> Int32:
    return Int32(Int((_hash(UInt64(tid) + 104729) >> 19) % UInt64(B_NBINS)))


@always_inline
def _b_label_units(tid: Int) -> Int:
    """The label in units of 1/4. Runs from -20 to +20, so the signed
    integer atomic and truncation-toward-zero on NEGATIVE values are both
    exercised rather than assumed."""
    return Int((_hash(UInt64(tid) + 15485863) >> 23) % 41) - 20


@always_inline
def _trunc_div(n: Int, d: Int) -> Int:
    """`Int32(x)`'s truncation toward zero, written in integers so that
    the expected value owes nothing to any float in the kernel. Mojo's
    `//` FLOORS, which is a different answer for a negative numerator, and
    that difference is the whole point of this helper."""
    if n >= 0:
        return n // d
    return -((-n) // d)


@always_inline
def _c_weight_units(tid: Int) -> Int:
    """The weight in units of 1/4, always positive."""
    return Int((_hash(UInt64(tid) + 32452843) >> 11) % 8) + 1


# ============================================================== kernels ==
def _a_kernel(
    hist: MutPointer[ClassificationBin, MutAnyOrigin],
    n: Int32,
    hashed: Int32,
    transposed: Int32,
):
    """Arm A. `transposed == 1` is SABOTAGE S1: it writes
    `b * nclasses + label` where `bins.cuh:26` writes
    `label * n_bins + b`."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= Int(n):
        return
    var label = _a_label(tid, hashed == 1)
    var b = _a_bin(tid, hashed == 1)
    if transposed == 1:
        ClassificationBin.AtomicAdd(
            hist.unsafe_offset(Int(b) * A_NCLASSES + Int(label)),
            ClassificationBin(1),
        )
    else:
        ClassificationBin.IncrementHistogram(
            hist, Int32(A_NBINS), b, label, Float32(0.0), BinScales.unit()
        )


def _b_kernel(
    hist: MutPointer[RegressionBin, MutAnyOrigin],
    n: Int32,
    swapped: Int32,
):
    """Arm B. `swapped == 1` is SABOTAGE S2: label sum into the count's
    slot and the count into the label sum's."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= Int(n):
        return
    var b = _b_bin(tid)
    var label = Float32(_b_label_units(tid)) * 0.25
    if swapped == 1:
        var q = Int32(label * B_LABEL_SCALE)
        RegressionBin.AtomicAdd(
            hist.unsafe_offset(Int(b)), RegressionBin(1, UInt32(Int(q)))
        )
    else:
        RegressionBin.IncrementHistogram(
            hist,
            Int32(B_NBINS),
            b,
            label,
            Float32(0.0),
            BinScales(B_LABEL_SCALE, 1.0),
        )


def _c_kernel(
    hist: MutPointer[WeightedRegressionBin, MutAnyOrigin],
    n: Int32,
    rotated: Int32,
):
    """Arm C. `rotated == 1` is SABOTAGE S3: the three slots rotated by
    one, which a two-field check could never see."""
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= Int(n):
        return
    var b = _b_bin(tid)
    var label = Float32(_b_label_units(tid)) * 0.25
    var weight = Float32(_c_weight_units(tid)) * 0.25
    if rotated == 1:
        var ql = Int32(label * weight * C_LABEL_SCALE)
        var qw = Int32(weight * C_WEIGHT_SCALE)
        WeightedRegressionBin.AtomicAdd(
            hist.unsafe_offset(Int(b)),
            WeightedRegressionBin(qw, UInt32(Int(ql)), 1),
        )
    else:
        WeightedRegressionBin.IncrementHistogram(
            hist,
            Int32(B_NBINS),
            b,
            label,
            weight,
            BinScales(C_LABEL_SCALE, C_WEIGHT_SCALE),
        )


def _gain_kernel(
    hist: MutPointer[ClassificationBin, MutAnyOrigin],
    quantiles: MutPointer[Float32, MutAnyOrigin],
    out_gain: MutPointer[Float32, MutAnyOrigin],
    out_colid: MutPointer[Int32, MutAnyOrigin],
    out_nleft: MutPointer[Int64, MutAnyOrigin],
    out_ques: MutPointer[Float32, MutAnyOrigin],
    nclasses: Int32,
    n_bins: Int32,
    col: Int32,
    n_rows: Int32,
):
    """`ClassificationObjectiveFunction::Gain`, `objectives.cuh:163-177`,
    with `blockDim.x == n_bins` so thread `t` owns bin `t` exactly. Each
    thread's own `Split` is written out, so the gate is per THREAD and per
    FIELD rather than on a reduced winner."""
    var obj = CObj(
        nclasses, Int32(1), CRITERION_GINI, Float32(-1.0), BinScales.unit()
    )
    var sp = obj.Gain(hist, quantiles, col, Int64(Int(n_rows)), n_bins)
    var t = Int(thread_idx.x)
    out_gain[unsafe_offset=t] = sp.best_metric_val
    out_colid[unsafe_offset=t] = sp.colid
    out_nleft[unsafe_offset=t] = sp.global_nLeft
    out_ques[unsafe_offset=t] = sp.quesval


# ================================================================ fixtures ==
def _gini_fixture() -> List[ClassificationBin]:
    """nclasses = 2, n_bins = 4, CUMULATIVE, `hist[n_bins * j + i]`.

    Per-bin counts are class 0: 3,1,4,2 and class 1: 1,5,0,2. Prefix-summed
    along the bins, because that is the state `objectives.cuh` reads
    (see the module docstring of `objectives.mojo`).
    """
    var h = List[ClassificationBin]()
    for c in [3, 4, 8, 10, 1, 6, 6, 8]:
        h.append(ClassificationBin(UInt32(c)))
    return h^


def _mse_fixture() -> List[RegressionBin]:
    """n_bins = 4, CUMULATIVE. Counts 2,4,6,8; label sums 1.0, 3.0, 2.5,
    4.0, held at `MSE_LABEL_SCALE` = 16 so every raw slot is an exact
    integer -- here the subject is the gain arithmetic, not the
    quantizer."""
    var h = List[RegressionBin]()
    var counts = [2, 4, 6, 8]
    var raws = [16, 48, 40, 64]
    for i in range(4):
        h.append(RegressionBin(Int32(raws[i]), UInt32(counts[i])))
    return h^


# ================================================================== main ==
def main() raises:
    var failures = 0
    var ctx = DeviceContext()

    # ------------------------------------------------------------ arm A --
    # Histogram placement, on the GPU, uniform AND hashed, correct AND
    # transposed. Four runs, because the point is the DIFFERENCE.
    print("--- arm A: ClassificationBin::IncrementHistogram placement ---")

    var a_names = [
        "uniform  + correct  ",
        "uniform  + TRANSPOSED (S1)",
        "hashed   + correct  ",
        "hashed   + TRANSPOSED (S1)",
    ]
    var a_hashed = [0, 0, 1, 1]
    var a_trans = [0, 1, 0, 1]
    var a_wrong = [0, 0, 0, 0]

    for arm in range(4):
        # THE INDEPENDENT TALLY: a host loop that shares no placement code
        # with the kernel.
        var want = List[Int]()
        for _ in range(A_CELLS):
            want.append(0)
        for tid in range(A_N):
            var label = Int(_a_label(tid, a_hashed[arm] == 1))
            var b = Int(_a_bin(tid, a_hashed[arm] == 1))
            want[label * A_NBINS + b] += 1

        var dev = ctx.enqueue_create_buffer[DType.uint32](A_CELLS)
        ctx.enqueue_memset(dev, UInt32(0))
        ctx.enqueue_function[_a_kernel](
            dev.unsafe_ptr().unsafe_bitcast[ClassificationBin](),
            Int32(A_N),
            Int32(a_hashed[arm]),
            Int32(a_trans[arm]),
            grid_dim=A_BLOCKS,
            block_dim=A_BLOCK,
        )
        var host = ctx.enqueue_create_host_buffer[DType.uint32](A_CELLS)
        ctx.enqueue_copy(dst_buf=host, src_buf=dev)
        ctx.synchronize()

        var wrong = 0
        for c in range(A_CELLS):
            if Int(host.unsafe_ptr().unsafe_load(c)) != want[c]:
                wrong += 1
        a_wrong[arm] = wrong
        print(
            "   ", a_names[arm], ":", wrong, "wrong of", A_CELLS,
        )

    # The four verdicts, and what each one has to be.
    if a_wrong[0] != 0:
        failures += 1
        print("    FAIL: the honest uniform arm does not match its tally")
    if a_wrong[2] != 0:
        failures += 1
        print("    FAIL: the honest hashed arm does not match its tally")
    if a_wrong[1] != 0:
        failures += 1
        print(
            "    FAIL: the uniform arm was expected to be BLIND to the"
            " transposed kernel (square histogram); it was not, so the"
            " demonstration below is not the one this file claims"
        )
    else:
        print(
            "    S1 demonstration: the transposed kernel passes UNIFORM"
            " planting with 0 wrong -- a uniform gate would have shipped it"
        )
    if a_wrong[3] == 0:
        failures += 1
        print(
            "    FAIL: SABOTAGE S1 did not move the hashed comparison, so"
            " that comparison cannot see a wrong offset"
        )
    else:
        print(
            "    S1 caught by hashed planting:",
            a_wrong[3],
            "wrong of",
            A_CELLS,
        )

    # ------------------------------------------------------------ arm B --
    print("--- arm B: RegressionBin fixed-point slots ---")
    var b_want_raw = List[Int]()
    var b_want_cnt = List[Int]()
    for _ in range(B_NBINS):
        b_want_raw.append(0)
        b_want_cnt.append(0)
    for tid in range(B_N):
        var b = Int(_b_bin(tid))
        # label = units/4 and the scale is 2, so the exact scaled value is
        # units/2 and the stored slot is trunc(units/2) TOWARD ZERO.
        # Computed here in INTEGERS, so this expectation owes nothing to
        # any float in the kernel -- and it separates truncation from
        # rounding, which a scale of 16 would not.
        b_want_raw[b] += _trunc_div(_b_label_units(tid), 2)
        b_want_cnt[b] += 1

    for arm in range(2):
        var dev = ctx.enqueue_create_buffer[DType.int32](B_NBINS * 2)
        ctx.enqueue_memset(dev, Int32(0))
        ctx.enqueue_function[_b_kernel](
            dev.unsafe_ptr().unsafe_bitcast[RegressionBin](),
            Int32(B_N),
            Int32(arm),
            grid_dim=B_BLOCKS,
            block_dim=B_BLOCK,
        )
        var host = ctx.enqueue_create_host_buffer[DType.int32](B_NBINS * 2)
        ctx.enqueue_copy(dst_buf=host, src_buf=dev)
        ctx.synchronize()
        var wrong = 0
        for b in range(B_NBINS):
            var got_raw = Int(host.unsafe_ptr().unsafe_load(b * 2))
            var got_cnt = Int(host.unsafe_ptr().unsafe_load(b * 2 + 1))
            if got_raw != b_want_raw[b] or got_cnt != b_want_cnt[b]:
                wrong += 1
        if arm == 0:
            print("    correct AtomicAdd:", wrong, "wrong of", B_NBINS)
            if wrong != 0:
                failures += 1
                print("    FAIL: arm B does not match its integer tally")
            # SABOTAGE S5: the SAME data scored against a FLOORING
            # quantizer. `_quantize` truncates toward zero (`bins.mojo`,
            # DEVIATION 101b) and the two answers differ on every negative
            # odd unit. If this does not move, the arm above cannot tell
            # truncation from rounding and the deviation is unverified.
            var floor_wrong = 0
            for b in range(B_NBINS):
                var want_floor = 0
                for tid in range(B_N):
                    if Int(_b_bin(tid)) == b:
                        want_floor += _b_label_units(tid) // 2
                if Int(host.unsafe_ptr().unsafe_load(b * 2)) != want_floor:
                    floor_wrong += 1
            print(
                "    SABOTAGE S5 (floor instead of truncate):",
                floor_wrong,
                "wrong of",
                B_NBINS,
            )
            if floor_wrong == 0:
                failures += 1
                print(
                    "    FAIL: S5 did not move arm B, so arm B cannot see a"
                    " quantizer that rounds the wrong way"
                )
        else:
            print("    SABOTAGE S2 (slots swapped):", wrong, "wrong of", B_NBINS)
            if wrong == 0:
                failures += 1
                print(
                    "    FAIL: S2 did not move arm B, so arm B cannot see a"
                    " wrong field offset"
                )

    # ------------------------------------------------------------ arm C --
    print("--- arm C: WeightedRegressionBin, three slots ---")
    var c_want_l = List[Int]()
    var c_want_c = List[Int]()
    var c_want_w = List[Int]()
    for _ in range(B_NBINS):
        c_want_l.append(0)
        c_want_c.append(0)
        c_want_w.append(0)
    for tid in range(B_N):
        var b = Int(_b_bin(tid))
        # label = lu/4, weight = wu/4, so label*weight = lu*wu/16 and at
        # scale 4 the exact scaled value is lu*wu/4, stored truncated
        # toward zero. The weight plane is wu/2, likewise truncated.
        var lu = _b_label_units(tid)
        var wu = _c_weight_units(tid)
        c_want_l[b] += _trunc_div(lu * wu, 4)
        c_want_c[b] += 1
        c_want_w[b] += _trunc_div(wu, 2)
    for arm in range(2):
        var dev = ctx.enqueue_create_buffer[DType.int32](B_NBINS * 3)
        ctx.enqueue_memset(dev, Int32(0))
        ctx.enqueue_function[_c_kernel](
            dev.unsafe_ptr().unsafe_bitcast[WeightedRegressionBin](),
            Int32(B_N),
            Int32(arm),
            grid_dim=B_BLOCKS,
            block_dim=B_BLOCK,
        )
        var host = ctx.enqueue_create_host_buffer[DType.int32](B_NBINS * 3)
        ctx.enqueue_copy(dst_buf=host, src_buf=dev)
        ctx.synchronize()
        var wrong = 0
        for b in range(B_NBINS):
            var gl = Int(host.unsafe_ptr().unsafe_load(b * 3))
            var gc = Int(host.unsafe_ptr().unsafe_load(b * 3 + 1))
            var gw = Int(host.unsafe_ptr().unsafe_load(b * 3 + 2))
            if gl != c_want_l[b] or gc != c_want_c[b] or gw != c_want_w[b]:
                wrong += 1
        if arm == 0:
            print("    correct AtomicAdd:", wrong, "wrong of", B_NBINS)
            if wrong != 0:
                failures += 1
                print("    FAIL: arm C does not match its integer tally")
        else:
            print("    SABOTAGE S3 (slots rotated):", wrong, "wrong of", B_NBINS)
            if wrong == 0:
                failures += 1
                print("    FAIL: S3 did not move arm C")

    # ------------------------------------------------------------ arm D --
    # Gini, against rationals done on paper.
    print("--- arm D: GiniGain against 49/810 and 1/567 ---")
    var gh = _gini_fixture()
    var cobj = CObj(
        Int32(2), Int32(1), CRITERION_GINI, Float32(0.0), BinScales.unit()
    )
    comptime GINI_1 = Float64(49.0) / Float64(810.0)
    comptime GINI_2 = Float64(1.0) / Float64(567.0)
    comptime GINI_TOL = Float64(2.0e-7)

    var g1 = Float64(
        cobj.GiniGain(gh.unsafe_ptr(), Int32(1), Int32(4), 18, 10, 8)
    )
    var g2 = Float64(
        cobj.GiniGain(gh.unsafe_ptr(), Int32(2), Int32(4), 18, 14, 4)
    )
    var g3 = Float64(
        cobj.GiniGain(gh.unsafe_ptr(), Int32(3), Int32(4), 18, 18, 0)
    )
    print("    bin 1:", g1, " want", GINI_1)
    print("    bin 2:", g2, " want", GINI_2)
    if abs(g1 - GINI_1) > GINI_TOL:
        failures += 1
        print("    FAIL: Gini bin 1")
    if abs(g2 - GINI_2) > GINI_TOL:
        failures += 1
        print("    FAIL: Gini bin 2")
    # `:52-53` -- right_weight == 0 must return the exact sentinel that
    # `split.cuh:150` compares against, not merely something small.
    if Float32(g3) != -Scalar[DType.float32].MAX_FINITE:
        failures += 1
        print("    FAIL: Gini bin 3 did not return -max(), got", g3)
    else:
        print("    bin 3: -max() sentinel, exact")

    # SABOTAGE S4: one count, one cell.
    var gh_sab = _gini_fixture()
    gh_sab[5] = ClassificationBin(gh_sab[5].count + 1)
    var g1_sab = Float64(
        cobj.GiniGain(gh_sab.unsafe_ptr(), Int32(1), Int32(4), 18, 10, 8)
    )
    if abs(g1_sab - g1) <= GINI_TOL:
        failures += 1
        print(
            "    FAIL: SABOTAGE S4 moved Gini by",
            abs(g1_sab - g1),
            "which is inside the tolerance, so the tolerance is blind",
        )
    else:
        print(
            "    S4 (one cell +1) moved Gini by",
            abs(g1_sab - g1),
            "against tolerance",
            GINI_TOL,
        )

    # ------------------------------------------------------------ arm E --
    print("--- arm E: MSEGain against 1/32 EXACTLY and 1/96 ---")
    var mh = _mse_fixture()
    var robj = RObj(
        Int32(1),
        Int32(1),
        CRITERION_MSE,
        Float32(0.0),
        BinScales(MSE_LABEL_SCALE, 1.0),
    )
    var m1 = robj.MSEGain(mh.unsafe_ptr(), Int32(1), Int32(4), 8, 4, 4)
    var m2 = Float64(
        robj.MSEGain(mh.unsafe_ptr(), Int32(2), Int32(4), 8, 6, 2)
    )
    print("    bin 1:", m1, " want exactly 0.03125")
    print("    bin 2:", m2, " want", Float64(1.0) / Float64(96.0))
    # EXACT. Every operand in this cell is a dyadic rational: parent weight
    # 8, invLen 0.125, label sums 4.0 / 3.0 / 1.0, left and right weights 4.
    if m1 != Float32(0.03125):
        failures += 1
        print("    FAIL: MSE bin 1 is not bit-exact 1/32")
    if abs(m2 - Float64(1.0) / Float64(96.0)) > 2.0e-7:
        failures += 1
        print("    FAIL: MSE bin 2")
    var m3 = robj.MSEGain(mh.unsafe_ptr(), Int32(3), Int32(4), 8, 8, 0)
    if m3 != -Scalar[DType.float32].MAX_FINITE:
        failures += 1
        print("    FAIL: MSE bin 3 did not return -max(), got", m3)

    # SABOTAGE S4b: one count in the cumulative count plane.
    var mh_sab = _mse_fixture()
    mh_sab[1] = RegressionBin(mh_sab[1].label_sum, mh_sab[1].count + 1)
    var m1_sab = robj.MSEGain(mh_sab.unsafe_ptr(), Int32(1), Int32(4), 8, 5, 3)
    if m1_sab == m1:
        failures += 1
        print("    FAIL: SABOTAGE S4b did not move MSE at all")
    else:
        print("    S4b (one count +1) moved MSE from", m1, "to", m1_sab)

    # ------------------------------------------------------------ arm F --
    # The transcendental criteria, against CPython float64 anchors pasted
    # from outside this repository. LOOSE ON PURPOSE: DEVIATION 113.
    print("--- arm F: Entropy / Poisson / Gamma / InverseGaussian ---")
    comptime LOG_TOL = Float64(1.0e-5)
    var eobj = CObj(
        Int32(2), Int32(1), CRITERION_ENTROPY, Float32(0.0), BinScales.unit()
    )
    var e1 = Float64(
        eobj.EntropyGain(gh.unsafe_ptr(), Int32(1), Int32(4), 18, 10, 8)
    )
    var e2 = Float64(
        eobj.EntropyGain(gh.unsafe_ptr(), Int32(2), Int32(4), 18, 14, 4)
    )
    print("    entropy bin 1:", e1, " anchor 0.0910910076037918")
    print("    entropy bin 2:", e2, " anchor 0.0025652873671375698")
    if abs(e1 - 0.0910910076037918) > LOG_TOL:
        failures += 1
        print("    FAIL: entropy bin 1")
    if abs(e2 - 0.0025652873671375698) > LOG_TOL:
        failures += 1
        print("    FAIL: entropy bin 2")

    var pobj = RObj(
        Int32(1),
        Int32(1),
        CRITERION_POISSON,
        Float32(0.0),
        BinScales(MSE_LABEL_SCALE, 1.0),
    )
    var p1 = Float64(
        pobj.PoissonGain(mh.unsafe_ptr(), Int32(1), Int32(4), 8, 4, 4)
    )
    var ga1 = Float64(
        pobj.GammaGain(mh.unsafe_ptr(), Int32(1), Int32(4), 8, 4, 4)
    )
    var ig1 = Float64(
        pobj.InverseGaussianGain(mh.unsafe_ptr(), Int32(1), Int32(4), 8, 4, 4)
    )
    print("    poisson bin 1:", p1, " anchor 0.0654060179705685")
    print("    gamma   bin 1:", ga1, " anchor 0.1438410362258905")
    print("    invgaus bin 1:", ig1, " anchor 1/3")
    if abs(p1 - 0.0654060179705685) > LOG_TOL:
        failures += 1
        print("    FAIL: poisson bin 1")
    if abs(ga1 - 0.1438410362258905) > LOG_TOL:
        failures += 1
        print("    FAIL: gamma bin 1")
    # No transcendental in InverseGaussian, so this one is held tight.
    if abs(ig1 - Float64(1.0) / Float64(3.0)) > 2.0e-7:
        failures += 1
        print("    FAIL: inverse gaussian bin 1")

    # ------------------------------------------------------------ arm G --
    print("--- arm G: GainPerSplit gating and dispatch ---")
    # `:128-130` -- min_samples_leaf rejects before any arithmetic runs.
    var strict = CObj(
        Int32(2), Int32(11), CRITERION_GINI, Float32(0.0), BinScales.unit()
    )
    var gated = strict.GainPerSplit(gh.unsafe_ptr(), Int32(1), Int32(4), 18, 10, 8)
    if gated != -Scalar[DType.float32].MAX_FINITE:
        failures += 1
        print("    FAIL: min_samples_leaf=11 did not reject nLeft=10")
    else:
        print("    min_samples_leaf gate: rejected, exact sentinel")
    # The dispatch must reach the same number the direct call reached.
    var via = Float64(
        cobj.GainPerSplit(gh.unsafe_ptr(), Int32(1), Int32(4), 18, 10, 8)
    )
    if via != g1:
        failures += 1
        print("    FAIL: GINI dispatch did not reach GiniGain")
    var bad = CObj(
        Int32(2), Int32(1), CRITERION_MSE, Float32(0.0), BinScales.unit()
    )
    if (
        bad.GainPerSplit(gh.unsafe_ptr(), Int32(1), Int32(4), 18, 10, 8)
        != -Scalar[DType.float32].MAX_FINITE
    ):
        failures += 1
        print("    FAIL: a regression criterion did not fall to default:")
    else:
        print("    default: arm reached for a regression criterion")

    # ------------------------------------------------------------ arm H --
    print("--- arm H: SetLeafVector ---")
    # Classification: `:179-195`. The leaf reads shist[0..nclasses), which
    # in the node kernel is the LAST bin's cumulative row. Plant 3 and 5.
    var leaf_in = List[ClassificationBin]()
    leaf_in.append(ClassificationBin(3))
    leaf_in.append(ClassificationBin(5))
    var leaf_out = List[Float32]()
    leaf_out.append(0.0)
    leaf_out.append(0.0)
    CObj.SetLeafVector(
        leaf_in.unsafe_ptr(), Int32(2), leaf_out.unsafe_ptr(), BinScales.unit()
    )
    if (
        abs(Float64(leaf_out[0]) - 0.375) > 1.0e-7
        or abs(Float64(leaf_out[1]) - 0.625) > 1.0e-7
    ):
        failures += 1
        print("    FAIL: classification leaf", leaf_out[0], leaf_out[1])
    else:
        print("    classification leaf: 3/8, 5/8 exact")
    # The `total <= 0` arm at `:186-191`.
    var zero_in = List[ClassificationBin]()
    zero_in.append(ClassificationBin(0))
    zero_in.append(ClassificationBin(0))
    leaf_out[0] = 9.0
    leaf_out[1] = 9.0
    CObj.SetLeafVector(
        zero_in.unsafe_ptr(), Int32(2), leaf_out.unsafe_ptr(), BinScales.unit()
    )
    if leaf_out[0] != 0.0 or leaf_out[1] != 0.0:
        failures += 1
        print("    FAIL: empty classification leaf was not zeroed")
    else:
        print("    empty classification leaf: zeroed")
    # Regression: `:380-386`. label_sum 40/16 = 2.5 over 5 rows -> 0.5.
    var rleaf_in = List[RegressionBin]()
    rleaf_in.append(RegressionBin(Int32(40), UInt32(5)))
    rleaf_in.append(RegressionBin(Int32(0), UInt32(0)))
    var rleaf_out = List[Float32]()
    rleaf_out.append(9.0)
    rleaf_out.append(9.0)
    RObj.SetLeafVector(
        rleaf_in.unsafe_ptr(),
        Int32(2),
        rleaf_out.unsafe_ptr(),
        BinScales(MSE_LABEL_SCALE, 1.0),
    )
    if rleaf_out[0] != Float32(0.5) or rleaf_out[1] != Float32(0.0):
        failures += 1
        print("    FAIL: regression leaf", rleaf_out[0], rleaf_out[1])
    else:
        print("    regression leaf: 0.5 exact, empty slot 0")

    # ------------------------------------------------------------ arm I --
    # `Gain` on the DEVICE, per thread, which is the only place
    # `_count_left` and `Split::update_bin` are reached together.
    print("--- arm I: Gain on the device, per thread ---")
    var dh = ctx.enqueue_create_buffer[DType.uint32](8)
    var hh = ctx.enqueue_create_host_buffer[DType.uint32](8)
    for i in range(8):
        hh.unsafe_ptr().unsafe_store(i, gh[i].count)
    ctx.enqueue_copy(dst_buf=dh, src_buf=hh)

    var dq = ctx.enqueue_create_buffer[DType.float32](4)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](4)
    for i in range(4):
        hq.unsafe_ptr().unsafe_store(i, Float32(i) + 0.5)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)

    var d_gain = ctx.enqueue_create_buffer[DType.float32](4)
    var d_col = ctx.enqueue_create_buffer[DType.int32](4)
    var d_nl = ctx.enqueue_create_buffer[DType.int64](4)
    var d_qu = ctx.enqueue_create_buffer[DType.float32](4)
    ctx.enqueue_function[_gain_kernel](
        dh.unsafe_ptr().unsafe_bitcast[ClassificationBin](),
        dq.unsafe_ptr(),
        d_gain.unsafe_ptr(),
        d_col.unsafe_ptr(),
        d_nl.unsafe_ptr(),
        d_qu.unsafe_ptr(),
        Int32(2),
        Int32(4),
        Int32(7),
        Int32(18),
        grid_dim=1,
        block_dim=4,
    )
    var h_gain = ctx.enqueue_create_host_buffer[DType.float32](4)
    var h_col = ctx.enqueue_create_host_buffer[DType.int32](4)
    var h_nl = ctx.enqueue_create_host_buffer[DType.int64](4)
    var h_qu = ctx.enqueue_create_host_buffer[DType.float32](4)
    ctx.enqueue_copy(dst_buf=h_gain, src_buf=d_gain)
    ctx.enqueue_copy(dst_buf=h_col, src_buf=d_col)
    ctx.enqueue_copy(dst_buf=h_nl, src_buf=d_nl)
    ctx.enqueue_copy(dst_buf=h_qu, src_buf=d_qu)
    ctx.synchronize()
    # Device form of the freed-at-enqueue UAF (perf-lane find,
    # 2026-08-22): `dh`/`dq` died at their `.unsafe_ptr()` in the kernel
    # argument list. Keep-alives AFTER the sync.
    _ = dh^
    _ = dq^

    # Analytic, per thread. nLeft = 4, 10, 14, 18 and nRight = 14, 8, 4, 0,
    # so thread 3 is skipped entirely by `:170-171` and must come back
    # INVALID -- a different outcome per thread, not one repeated value.
    var want_gain = [0.021604938271604868, GINI_1, GINI_2, 0.0]
    var want_nl = [4, 10, 14, 0]
    for t in range(4):
        var gg = Float64(h_gain.unsafe_ptr().unsafe_load(t))
        var cc = Int(h_col.unsafe_ptr().unsafe_load(t))
        var nn = Int(h_nl.unsafe_ptr().unsafe_load(t))
        var qq = Float64(h_qu.unsafe_ptr().unsafe_load(t))
        if t == 3:
            if cc != -1 or nn != 0:
                failures += 1
                print("    FAIL: thread 3 should hold an invalid Split, got",
                      cc, nn)
            else:
                print("    thread 3: invalid Split, as nRight = 0 requires")
            continue
        var ok = (
            abs(gg - want_gain[t]) <= GINI_TOL
            and cc == 7
            and nn == want_nl[t]
            and qq == Float64(t) + 0.5
        )
        if not ok:
            failures += 1
            print(
                "    FAIL thread", t, ": gain", gg, "want", want_gain[t],
                " colid", cc, " nLeft", nn, "want", want_nl[t],
                " quesval", qq,
            )
        else:
            print(
                "    thread", t, ": gain", gg, " nLeft", nn, " quesval", qq,
            )

    # ------------------------------------------------------------------
    if failures == 0:
        print("objectives_check: ALL OK")
    else:
        raise Error(
            "objectives_check: " + String(failures) + " failure(s)"
        )
