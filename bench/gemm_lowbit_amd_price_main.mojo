# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The low-bit timing harness WITH THE AMD PLANS: every arm of
`bench/gemm_lowbit_price_main.mojo` and, beside them in the same run, the
plans of `gemm/checks/gemm_int8_mma_amd.mojo`.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_lowbit_amd_price_main.mojo

Lane lane/lowbit-amd-tuned, 2026-09-29. This file ADDS ARMS and copies
none: an arm of the harness it extends is enqueued, named, digested and
noted by that harness's own functions, imported; its shapes, its operands,
its poisons, its digests, its sabotage arm and its `LOWBIT` line are that
harness's, so `tools/lowbit_units/table.py` and
`tools/lowbit_mma_speed/table.py` read this file's output as they read its.
It is a second file, and not an edit of the first, because
lane/lowbit-mma-speed is still adding arms to the first and the arms here
launch on one column only. It certifies nothing: the gate is
`gemm/checks/gemm_int8_mma_amd_check.mojo`.

THE ARMS IT ADDS, every one AMD only (on any other column each prints a
LOWBIT-NOT-RUN line):

    probe.amd.launch-floor          one block of one thread that stores one
                                    cell: the time of ONE LAUNCH AND ONE
                                    WAIT on this box. Not a product. At the
                                    decode rows a product's time is a small
                                    multiple of it, and a ratio of two
                                    times that both hold it says less than
                                    it seems to.
    int8i32.v1.mma.amd.*            one arm per one-product plan of the AMD
                                    file; the name is the plan's. Its digest
                                    must equal the flat plan's.
    pieces.int8.mma.amd.*           one arm per four-product plan of the AMD
                                    file. Three Int32 sums per cell; the
                                    rate is over `4 m n k`. Its digest must
                                    equal `pieces.int8.flat`'s.
    inference.int8i32.v1.amd        parallel quantize A, one product on the
    training.int8i32.v1.amd         AMD launcher's plan; with B quantized
                                    too.
    inference.4x.int8i32.v1.amd     the same with FOUR products, four
    training.4x.int8i32.v1.amd      launches.
    inference.pieces.int8.amd       THE COMPLETE OPERATION on the AMD
    training.pieces.int8.amd        launcher's plan: parallel quantize A
                                    (and B), four products in one launch,
                                    then the extended harness's second
                                    launch, the fifteen-bit seam
                                    (`int15_store_cell`) one thread a
                                    cell. One wait.
    inference.pieces.int8.plan.*    the same complete operation on EVERY
    training.pieces.int8.plan.*     four-product plan, the tuned file's
                                    (`plan.staged...`) and the AMD file's
                                    (`plan.amd...`), so the plan a row's
                                    complete operation takes the least time
                                    on is read from one run and not
                                    inferred from the product alone.
    inference.pieces.int8.amd.fused THE FUSED FORM: quantize A, then the
    inference.pieces.int8.fused.plan.*  four products AND the seam in ONE
                                    launch, on the AMD launcher's plan and
                                    on every plan of both files.

THE COMPLETE OPERATION'S LIMITS are the extended harness's and are repeated
because the number is the one that is quoted: the quantizer is the int8
one, standing in for the fifteen-bit one, and its codes are not the planes
the products read (those are the fixture's, made before anything is timed).
The seam is the fifteen-bit profile's own (`int15_store_cell`). The WORK is
a complete call's; the digest is compared among these arms and with nothing
else. lane/lowbit-int15's harness times the profile's own complete call.

THE BASES. Per shape a `BASES` line prints each operand buffer's address
modulo 16 and modulo 4096 as the launch is about to pass it: a load states
its alignment only where the base is a multiple of the load's bytes, so the
line says whether the stated path or the byte path was timed.

ENVIRONMENT: the extended harness's (`MOJOLEARN_LOWBIT_PRICE_*`). An arm
name of `MOJOLEARN_LOWBIT_PRICE_ARMS` may be one of this file's.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, thread_idx
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from bench.gemm_lowbit_price_main import (
    ARM_BF16_WIDEN_B,
    ARM_COUNT,
    ARM_DECODE_BASE,
    ARM_INF_PIECES,
    ARM_INF_PIECES_FUSED,
    ARM_INT8_DEQUANT_B,
    ARM_INT8_FLAT,
    ARM_INT8_MMA,
    ARM_INT8_PACK_B_PAR,
    ARM_PIECES_BASE,
    ARM_PIECES_FLAT,
    ARM_QUANTIZE_A_PAR,
    ARM_TUNED_BASE,
    DEFAULT_MAC_BUDGET,
    DEFAULT_REPEATS,
    PIECE_PRODUCTS,
    ShapeBuffers,
    _arm_asked,
    _arm_digest,
    _arm_is_probe,
    _arm_is_product,
    _arm_is_sums,
    _arm_name,
    _arm_note,
    _arm_runs,
    _digest_f32,
    _digest_sums,
    _enqueue_arm,
    _median_ms,
    _min_ms,
    _must_agree,
    _pieces_fixture,
    _pieces_recombine_probe,
    _poison_codes,
    _poison_f32,
    _poison_sums,
    _rate,
    _sabotage_f32,
    _wanted,
    _whole,
    price_sabotage_name,
)
from bench.gemm_price_main import _capped, _dev_fill, _fixed, _mode, _report_device
from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT,
    gemm_shape_k,
    gemm_shape_m,
    gemm_shape_n,
    gemm_shape_name,
    gemm_shape_op,
)
from bench.gemm_shapes import OP_NT as TBL_OP_NT
from checks.kernel_matrix import TARGET_COLUMN, column_name
from gemm.checks.gemm_identical import choose_gemm_plan, gemm_plan_name
from gemm.checks.gemm_int8_mma_amd import (
    INT8_AMD_AVAILABLE,
    INT8_AMD_PIECES_PLAN_COUNT,
    INT8_AMD_PLAN_COUNT,
    identical_gemm_int8_mma_amd_into,
    identical_gemm_int8_mma_amd_with_plan,
    identical_gemm_int8_pieces_amd_fused_into,
    identical_gemm_int8_pieces_amd_fused_with_plan,
    identical_gemm_int8_pieces_amd_into,
    identical_gemm_int8_pieces_amd_with_plan,
    int8_amd_dispatch,
    int8_amd_pieces_dispatch,
    int8_amd_pieces_plan_name,
    int8_amd_plan_name,
    int8_amd_sabotage_name,
)
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_PIECES_MAX_K,
    INT8_PIECES_PLAN_COUNT,
    identical_gemm_int8_pieces_tuned_fused_with_plan,
    identical_gemm_int8_pieces_tuned_with_plan,
    int8_pieces_plan_name,
)
from gemm.checks.gemm_lowbit import bf16_narrow, quantize_rows_int8_device
from gemm.checks.quantize_int8_par import quantize_rows_int8_par_device

#: Every four-product plan the box has: the tuned file's, then the AMD
#: file's. The complete operation has one arm per plan of it.
comptime ALL_PIECES_PLANS = INT8_PIECES_PLAN_COUNT + INT8_AMD_PIECES_PLAN_COUNT

#: This file's arms, numbered after the extended harness's.
comptime AMD_ARM_FLOOR = ARM_COUNT
comptime AMD_ARM_PLAN_BASE = AMD_ARM_FLOOR + 1
comptime AMD_ARM_PIECES_BASE = AMD_ARM_PLAN_BASE + INT8_AMD_PLAN_COUNT
comptime AMD_ARM_INF = AMD_ARM_PIECES_BASE + INT8_AMD_PIECES_PLAN_COUNT
comptime AMD_ARM_TRAIN = AMD_ARM_INF + 1
comptime AMD_ARM_INF_4X = AMD_ARM_INF + 2
comptime AMD_ARM_TRAIN_4X = AMD_ARM_INF + 3
comptime AMD_ARM_INF_PIECES = AMD_ARM_INF + 4
comptime AMD_ARM_TRAIN_PIECES = AMD_ARM_INF + 5
comptime AMD_ARM_INF_PLAN_BASE = AMD_ARM_INF + 6
comptime AMD_ARM_TRAIN_PLAN_BASE = AMD_ARM_INF_PLAN_BASE + ALL_PIECES_PLANS
#: THE FUSED FORM of the complete inference operation: the four products and
#: the fifteen-bit seam (`int15_store_cell`) in ONE launch, on the AMD
#: launcher's plan and on every plan of both files.
comptime AMD_ARM_INF_FUSED = AMD_ARM_TRAIN_PLAN_BASE + ALL_PIECES_PLANS
comptime AMD_ARM_INF_FUSED_PLAN_BASE = AMD_ARM_INF_FUSED + 1
comptime ALL_ARM_COUNT = AMD_ARM_INF_FUSED_PLAN_BASE + ALL_PIECES_PLANS


def _all_pieces_plan_name(u: Int) -> String:
    """A four-product plan of either file by its number here. No spaces."""
    if u < INT8_PIECES_PLAN_COUNT:
        return int8_pieces_plan_name(u)
    return String("amd.") + int8_amd_pieces_plan_name(u - INT8_PIECES_PLAN_COUNT)


def _name(arm: Int) -> String:
    if arm < ARM_COUNT:
        return _arm_name(arm)
    if arm == AMD_ARM_FLOOR:
        return String("probe.amd.launch-floor")
    if arm < AMD_ARM_PIECES_BASE:
        return String("int8i32.v1.mma.amd.") + int8_amd_plan_name(arm - AMD_ARM_PLAN_BASE)
    if arm < AMD_ARM_INF:
        return String("pieces.int8.mma.amd.") + int8_amd_pieces_plan_name(
            arm - AMD_ARM_PIECES_BASE
        )
    if arm == AMD_ARM_INF:
        return String("inference.int8i32.v1.amd")
    if arm == AMD_ARM_TRAIN:
        return String("training.int8i32.v1.amd")
    if arm == AMD_ARM_INF_4X:
        return String("inference.4x.int8i32.v1.amd")
    if arm == AMD_ARM_TRAIN_4X:
        return String("training.4x.int8i32.v1.amd")
    if arm == AMD_ARM_INF_PIECES:
        return String("inference.pieces.int8.amd")
    if arm == AMD_ARM_TRAIN_PIECES:
        return String("training.pieces.int8.amd")
    if arm < AMD_ARM_TRAIN_PLAN_BASE:
        return String("inference.pieces.int8.plan.") + _all_pieces_plan_name(
            arm - AMD_ARM_INF_PLAN_BASE
        )
    if arm < AMD_ARM_INF_FUSED:
        return String("training.pieces.int8.plan.") + _all_pieces_plan_name(
            arm - AMD_ARM_TRAIN_PLAN_BASE
        )
    if arm == AMD_ARM_INF_FUSED:
        return String("inference.pieces.int8.amd.fused")
    return String("inference.pieces.int8.fused.plan.") + _all_pieces_plan_name(
        arm - AMD_ARM_INF_FUSED_PLAN_BASE
    )


def _is_sums(arm: Int) -> Bool:
    """Whether the arm's output is the three Int32 sums per cell."""
    if arm < ARM_COUNT:
        return _arm_is_sums(arm)
    return arm >= AMD_ARM_PIECES_BASE and arm < AMD_ARM_INF


def _is_floor(arm: Int) -> Bool:
    return arm == AMD_ARM_FLOOR


def _is_product(arm: Int) -> Bool:
    """Whether the arm's output is the product `C`."""
    if arm < ARM_COUNT:
        return _arm_is_product(arm)
    return not _is_floor(arm) and not _is_sums(arm)


def _needs_pieces(arm: Int) -> Bool:
    """Whether the arm reads the four planes, so that the four-product
    kernel's bound on `k` binds it and its buffers must exist."""
    if arm < ARM_COUNT:
        return arm >= ARM_PIECES_FLAT and arm <= ARM_INF_PIECES_FUSED
    if arm >= AMD_ARM_PIECES_BASE and arm < AMD_ARM_INF:
        return True
    return arm >= AMD_ARM_INF_PIECES


def _runs(arm: Int) -> Bool:
    """Whether this build runs the arm."""
    if arm < ARM_COUNT:
        return _arm_runs(arm)
    return INT8_AMD_AVAILABLE


def _launch_floor_kernel(cell: MutPointer[Float32, MutAnyOrigin]):
    """One cell, stored by one thread of one block."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    cell.unsafe_store(0, Float32(1.0))


def _enqueue(
    ctx: DeviceContext,
    mut sb: ShapeBuffers,
    mut floor: DeviceBuffer[DType.float32],
    arm: Int,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """Enqueue one arm's work and return; the caller waits."""
    if arm < ARM_COUNT:
        _enqueue_arm(ctx, sb, arm, m, n, k)
        return
    comptime if INT8_AMD_AVAILABLE:
        if arm == AMD_ARM_FLOOR:
            ctx.enqueue_function[_launch_floor_kernel](
                floor.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(1, 1, 1)
            )
        elif arm < AMD_ARM_PIECES_BASE:
            identical_gemm_int8_mma_amd_with_plan(
                ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k, arm - AMD_ARM_PLAN_BASE
            )
        elif arm < AMD_ARM_INF:
            identical_gemm_int8_pieces_amd_with_plan(
                ctx, sb.ps, sb.pah, sb.pal, sb.pbh, sb.pbl, m, n, k, arm - AMD_ARM_PIECES_BASE
            )
        elif arm < AMD_ARM_INF_PIECES:
            quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            if arm == AMD_ARM_TRAIN or arm == AMD_ARM_TRAIN_4X:
                quantize_rows_int8_par_device(ctx, sb.qb, sb.eb, sb.b, n, k)
            var products = 1
            if arm == AMD_ARM_INF_4X or arm == AMD_ARM_TRAIN_4X:
                products = PIECE_PRODUCTS
            for _ in range(products):
                identical_gemm_int8_mma_amd_into(
                    ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k
                )
        elif arm >= AMD_ARM_INF_FUSED:
            # THE FUSED COMPLETE OPERATION: quantize A, then the four
            # products and the seam in one launch.
            quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            if arm == AMD_ARM_INF_FUSED:
                identical_gemm_int8_pieces_amd_fused_into(
                    ctx, sb.c, sb.pah, sb.pal, sb.ea, sb.pbh, sb.pbl, sb.eb, m, n, k
                )
            else:
                var u = arm - AMD_ARM_INF_FUSED_PLAN_BASE
                if u < INT8_PIECES_PLAN_COUNT:
                    identical_gemm_int8_pieces_tuned_fused_with_plan(
                        ctx, sb.c, sb.pah, sb.pal, sb.ea, sb.pbh, sb.pbl, sb.eb, m, n, k, u
                    )
                else:
                    identical_gemm_int8_pieces_amd_fused_with_plan(
                        ctx, sb.c, sb.pah, sb.pal, sb.ea, sb.pbh, sb.pbl, sb.eb, m, n, k,
                        u - INT8_PIECES_PLAN_COUNT,
                    )
        else:
            # THE COMPLETE OPERATION: the conversions, four products in one
            # launch, the second launch (the seam, one thread a cell).
            var training = arm == AMD_ARM_TRAIN_PIECES or arm >= AMD_ARM_TRAIN_PLAN_BASE
            quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            if training:
                quantize_rows_int8_par_device(ctx, sb.qb, sb.eb, sb.b, n, k)
            if arm == AMD_ARM_INF_PIECES or arm == AMD_ARM_TRAIN_PIECES:
                identical_gemm_int8_pieces_amd_into(
                    ctx, sb.ps, sb.pah, sb.pal, sb.pbh, sb.pbl, m, n, k
                )
            else:
                var u = arm - AMD_ARM_INF_PLAN_BASE
                if training:
                    u = arm - AMD_ARM_TRAIN_PLAN_BASE
                if u < INT8_PIECES_PLAN_COUNT:
                    identical_gemm_int8_pieces_tuned_with_plan(
                        ctx, sb.ps, sb.pah, sb.pal, sb.pbh, sb.pbl, m, n, k, u
                    )
                else:
                    identical_gemm_int8_pieces_amd_with_plan(
                        ctx, sb.ps, sb.pah, sb.pal, sb.pbh, sb.pbl, m, n, k,
                        u - INT8_PIECES_PLAN_COUNT,
                    )
            _pieces_recombine_probe(ctx, sb.c, sb.ps, sb.ea, sb.eb, m, n)


def _digest(
    ctx: DeviceContext,
    mut sb: ShapeBuffers,
    mut floor: DeviceBuffer[DType.float32],
    arm: Int,
    m: Int,
    n: Int,
    k: Int,
    tag: String,
) raises -> UInt64:
    """The digest of what the arm just wrote (after the sabotage arm's one
    bit, in a build that names it)."""
    if arm < ARM_COUNT:
        return _arm_digest(ctx, sb, arm, m, n, k, tag)
    if _is_floor(arm):
        _sabotage_f32(ctx, floor, 1)
        return _digest_f32(ctx, floor, 1, tag)
    if _is_sums(arm):
        return _digest_sums(ctx, sb.ps, 3 * m * n, tag)
    _sabotage_f32(ctx, sb.c, m * n)
    return _digest_f32(ctx, sb.c, m * n, tag)


def _note(arm: Int, m: Int, n: Int, k: Int) -> String:
    """What a reader needs beside the arm's number. No spaces."""
    if arm < ARM_COUNT:
        return _arm_note(arm, m, n, k)
    if _is_floor(arm):
        return String("one-launch-and-one-wait,no-product")
    if arm < AMD_ARM_PIECES_BASE:
        if arm - AMD_ARM_PLAN_BASE == int8_amd_dispatch(m, n, k):
            return String("the-amd-launcher's-plan")
        return String("plan")
    if arm < AMD_ARM_INF:
        if arm - AMD_ARM_PIECES_BASE == int8_amd_pieces_dispatch(m, n, k):
            return String("the-amd-launcher's-plan,rate-over-4mnk")
        return String("plan,rate-over-4mnk")
    var one = int8_amd_plan_name(int8_amd_dispatch(m, n, k))
    if arm == AMD_ARM_INF:
        return String("quantize.a.par+") + one
    if arm == AMD_ARM_TRAIN:
        return String("quantize.a.par+pack.b.par+") + one
    if arm == AMD_ARM_INF_4X:
        return String("quantize.a.par+4x.") + one
    if arm == AMD_ARM_TRAIN_4X:
        return String("quantize.a.par+pack.b.par+4x.") + one
    var four = String("amd.") + int8_amd_pieces_plan_name(int8_amd_pieces_dispatch(m, n, k))
    if arm == AMD_ARM_INF_FUSED:
        return String("quantize.a.par+four-products-and-seam-FUSED.") + four
    if arm > AMD_ARM_INF_FUSED:
        return String("quantize.a.par+four-products-and-seam-FUSED.") + _all_pieces_plan_name(
            arm - AMD_ARM_INF_FUSED_PLAN_BASE
        )
    var lead = String("quantize.a.par+four-products.")
    if arm == AMD_ARM_TRAIN_PIECES or arm >= AMD_ARM_TRAIN_PLAN_BASE:
        lead = String("quantize.a.par+pack.b.par+four-products.")
    if arm >= AMD_ARM_TRAIN_PLAN_BASE:
        four = _all_pieces_plan_name(arm - AMD_ARM_TRAIN_PLAN_BASE)
    elif arm >= AMD_ARM_INF_PLAN_BASE:
        four = _all_pieces_plan_name(arm - AMD_ARM_INF_PLAN_BASE)
    return lead + four + "+seam-launch"


def _agree(
    a_arm: Int, b_arm: Int, dig: List[UInt64], ran: List[Bool], shape: String
) -> String:
    """Empty when the two arms' digests agree or either did not run; the
    disagreement otherwise."""
    if not ran[a_arm] or not ran[b_arm]:
        return String("")
    if dig[a_arm] == dig[b_arm]:
        return String("")
    return (
        String("PLANS DISAGREE at ") + shape + ": " + _name(a_arm) + " "
        + hex(dig[a_arm]) + " vs " + _name(b_arm) + " " + hex(dig[b_arm]) + "\n"
    )


def _base_mod(address: Int) -> String:
    return String(address & 15) + "/" + String(address & 4095)


def _time_shape(
    ctx: DeviceContext,
    idx: Int,
    name: String,
    m: Int,
    n: Int,
    k: Int,
    capped: Bool,
    repeats: Int,
    identity_only: Bool,
    arms: String,
) raises -> String:
    """Every arm at one shape: the extended harness's `_time_shape`, over
    its arms and this file's. Returns the plan disagreements found."""
    var cap_word = String("CAPPED") if capped else String("FULL")
    print()
    print(
        "== " + name + "  m=" + String(m) + " n=" + String(n) + " k=" + String(k)
        + "  " + cap_word + "  fp32 plan: " + gemm_plan_name(choose_gemm_plan(m, n, k))
    )
    var dig = List[UInt64]()
    var ran = List[Bool]()
    var samples = List[List[Int]]()
    var pieces = False
    for arm in range(ALL_ARM_COUNT):
        dig.append(UInt64(0))
        var runs = _runs(arm) and _arm_asked(arms, _name(arm))
        # The four-product kernels refuse a `k` above their bound.
        if _needs_pieces(arm) and k > INT8_PIECES_MAX_K:
            runs = False
        if _needs_pieces(arm) and runs:
            pieces = True
        ran.append(runs)
        samples.append(List[Int]())
    var sb = ShapeBuffers(ctx, m, n, k, pieces)
    var floor = ctx.enqueue_create_buffer[DType.float32](1)
    # The extended harness's operands, salts and seams, in its order.
    _whole(len(sb.a), m * k, String("the left operand"))
    _whole(len(sb.b), n * k, String("the right operand"))
    _dev_fill(ctx, sb.a, m * k, 11 + idx)
    _dev_fill(ctx, sb.b, n * k, 22 + idx)
    bf16_narrow(ctx, sb.bh, sb.b, n * k)
    quantize_rows_int8_device(ctx, sb.qa, sb.ea, sb.a, m, k)
    quantize_rows_int8_device(ctx, sb.qb, sb.eb, sb.b, n, k)
    if pieces:
        _pieces_fixture(ctx, sb.pah, sb.pal, sb.qa, m * k)
        _pieces_fixture(ctx, sb.pbh, sb.pbl, sb.qb, n * k)
    ctx.synchronize()
    print(
        "BASES", name, "address-mod-16/mod-4096",
        "qa", _base_mod(Int(sb.qa.unsafe_ptr())),
        "qb", _base_mod(Int(sb.qb.unsafe_ptr())),
        "a.high", _base_mod(Int(sb.pah.unsafe_ptr())),
        "a.low", _base_mod(Int(sb.pal.unsafe_ptr())),
        "b.high", _base_mod(Int(sb.pbh.unsafe_ptr())),
        "b.low", _base_mod(Int(sb.pbl.unsafe_ptr())),
    )

    # Untimed warm-up of every arm, its output poisoned first and read back
    # after, so an arm that launches without writing cannot turn in a time.
    for arm in range(ALL_ARM_COUNT):
        if not ran[arm]:
            continue
        var tag = String("lowbit.") + name + "." + _name(arm)
        if _is_floor(arm):
            _poison_f32(ctx, floor, 1, tag)
        elif _is_product(arm):
            _poison_f32(ctx, sb.c, m * n, tag)
        elif arm == ARM_BF16_WIDEN_B or arm == ARM_INT8_DEQUANT_B:
            _poison_f32(ctx, sb.work.wide, n * k, tag)
        elif arm == ARM_QUANTIZE_A_PAR:
            _poison_codes(ctx, sb.qsa, sb.esa, m * k, m)
        elif arm == ARM_INT8_PACK_B_PAR:
            _poison_codes(ctx, sb.qsb, sb.esb, n * k, n)
        elif _is_sums(arm):
            _poison_sums(ctx, sb.ps, 3 * m * n)
        _enqueue(ctx, sb, floor, arm, m, n, k)
        ctx.synchronize()
        dig[arm] = _digest(ctx, sb, floor, arm, m, n, k, tag)

    for _ in range(0 if identity_only else repeats):
        for arm in range(ALL_ARM_COUNT):
            if not ran[arm]:
                continue
            var t0 = perf_counter_ns()
            _enqueue(ctx, sb, floor, arm, m, n, k)
            ctx.synchronize()
            samples[arm].append(Int(perf_counter_ns() - t0))

    var macs = Float64(m) * Float64(n) * Float64(k)
    for arm in range(ALL_ARM_COUNT):
        var arm_name = _name(arm)
        if not ran[arm]:
            if _runs(arm):
                print(
                    "LOWBIT-NOT-RUN", column_name(TARGET_COLUMN), name, arm_name,
                    "not asked for (MOJOLEARN_LOWBIT_PRICE_ARMS), or k is above the four-product kernel's bound",
                )
            else:
                print(
                    "LOWBIT-NOT-RUN", column_name(TARGET_COLUMN), name, arm_name,
                    "this column cannot run the arm (it has no such unit, a block of the plan is above its thread limit, or the plan is another column's)",
                )
            continue
        var unit = String("GMAC/s")
        if not (_is_product(arm) or _is_sums(arm) or _is_floor(arm)):
            unit = String("Gelem/s")
        if identity_only:
            print(
                "LOWBIT", column_name(TARGET_COLUMN), name, m, n, k, cap_word, arm_name,
                "not-timed", "not-timed", "not-timed", unit,
                hex(dig[arm]), _note(arm, m, n, k),
            )
            continue
        var med = _median_ms(samples[arm])
        var best = _min_ms(samples[arm])
        var count = macs
        if _is_floor(arm):
            # One launch multiplies nothing.
            count = 0.0
        elif _is_sums(arm):
            count = Float64(PIECE_PRODUCTS) * macs
        elif not _is_product(arm):
            count = Float64(n) * Float64(k)
            if arm < ARM_COUNT and _arm_name(arm).find("quantize.a") >= 0:
                count = Float64(m) * Float64(k)
        _report_device(String("lowbit.") + name + "." + arm_name + ".device", med)
        print(
            "LOWBIT", column_name(TARGET_COLUMN), name, m, n, k, cap_word, arm_name,
            _fixed(med, 4), _fixed(best, 4), _fixed(_rate(count, med), 4), unit,
            hex(dig[arm]), _note(arm, m, n, k),
        )

    var bad = String("")
    # The extended harness's unit plans: each is the profile's product. A
    # probe is compared with nothing.
    bad += _must_agree(ARM_INT8_FLAT, ARM_INT8_MMA, dig, ran, name)
    for arm in range(ARM_TUNED_BASE, ARM_PIECES_FLAT):
        if not _arm_is_probe(arm):
            bad += _must_agree(ARM_INT8_FLAT, arm, dig, ran, name)
    for arm in range(ARM_PIECES_BASE, ARM_INF_PIECES):
        bad += _must_agree(ARM_PIECES_FLAT, arm, dig, ran, name)
    # This file's: every one-product plan and every operation that ends in
    # one is the profile's product; every four-product plan's sums are the
    # reference device plan's; every complete operation ends in the same
    # cells, the extended harness's included.
    for arm in range(AMD_ARM_PLAN_BASE, AMD_ARM_PIECES_BASE):
        bad += _agree(ARM_INT8_FLAT, arm, dig, ran, name)
    for arm in range(AMD_ARM_PIECES_BASE, AMD_ARM_INF):
        bad += _agree(ARM_PIECES_FLAT, arm, dig, ran, name)
    for arm in range(AMD_ARM_INF, AMD_ARM_INF_PIECES):
        bad += _agree(ARM_INT8_FLAT, arm, dig, ran, name)
    # The extended harness's own later arms: the fused operation ends in the
    # two-launch operation's cells; a decode plan is the profile's product.
    bad += _agree(ARM_INF_PIECES, ARM_INF_PIECES_FUSED, dig, ran, name)
    for arm in range(ARM_DECODE_BASE, ARM_COUNT):
        bad += _agree(ARM_INT8_FLAT, arm, dig, ran, name)
    # The first complete operation that ran is what the others are held to.
    var first_op = ARM_INF_PIECES
    if not ran[first_op]:
        for arm in range(AMD_ARM_INF_PIECES, ALL_ARM_COUNT):
            if ran[arm]:
                first_op = arm
                break
    for arm in range(AMD_ARM_INF_PIECES, ALL_ARM_COUNT):
        if arm != first_op:
            bad += _agree(first_op, arm, dig, ran, name)
    if bad.byte_length() > 0:
        print(bad)
    _ = floor
    _ = sb^
    return bad


def main() raises:
    comptime if not has_accelerator():
        raise Error("bench/gemm_lowbit_amd_price_main: no accelerator; every arm here is a device arm")
    else:
        var repeats = DEFAULT_REPEATS
        var rs = String(getenv("MOJOLEARN_LOWBIT_PRICE_REPEATS"))
        if rs != "":
            repeats = Int(atol(rs))
        var budget = DEFAULT_MAC_BUDGET
        var bs = String(getenv("MOJOLEARN_LOWBIT_PRICE_MAC_BUDGET"))
        if bs != "":
            budget = Int(atol(bs))
        var only = String(getenv("MOJOLEARN_LOWBIT_PRICE_ONLY"))
        var identity_only = String(getenv("MOJOLEARN_LOWBIT_PRICE_IDENTITY_ONLY")) == "1"
        var arms = String(getenv("MOJOLEARN_LOWBIT_PRICE_ARMS"))

        print("== bench/gemm_lowbit_amd_price_main.mojo [" + _mode() + "] ==")
        print("column", column_name(TARGET_COLUMN))
        print("profiles fp32.v1, int8i32.v1; the AMD plans of gemm/checks/gemm_int8_mma_amd.mojo")
        print(
            "sabotage: price", price_sabotage_name(), " amd and tuned unit",
            int8_amd_sabotage_name(),
        )
        print("repeats", repeats, " mac budget", budget, " only", only, " arms", arms)
        comptime if not INT8_AMD_AVAILABLE:
            print(
                "THE AMD ARMS DO NOT RUN on column", column_name(TARGET_COLUMN),
                "which is not a pass and not a time.",
            )
        if identity_only:
            print(
                "IDENTITY ONLY: every arm runs once for its digest and NOTHING",
                "IS TIMED on this box.",
            )
        else:
            print(
                "EVERY NUMBER BELOW IS ONE BOX'S TIME ON ONE RUN. One call and one",
                "synchronize per sample; the median of the timed calls is reported",
                "and the minimum beside it.",
            )

        var bad = String("")
        var shapes = 0
        with DeviceContext() as ctx:
            for i in range(GEMM_SHAPE_COUNT):
                if gemm_shape_op(i) != TBL_OP_NT:
                    continue
                var name = gemm_shape_name(i)
                if name.find("llama8b") < 0:
                    continue
                if not _wanted(only, name):
                    continue
                var m = gemm_shape_m(i)
                var n = gemm_shape_n(i)
                var k = gemm_shape_k(i)
                var dm = m
                var dn = n
                if budget > 0:
                    var cap = _capped(m, n, k, budget)
                    dm = cap[0]
                    dn = cap[1]
                bad += _time_shape(
                    ctx, i, name, dm, dn, k, dm != m or dn != n, repeats, identity_only, arms
                )
                shapes += 1

        print()
        print("== done [" + _mode() + "]: " + String(shapes) + " shapes ==")
        if shapes == 0:
            raise Error(
                "bench/gemm_lowbit_amd_price_main: no shape matched MOJOLEARN_LOWBIT_PRICE_ONLY="
                + only
            )
        if bad.byte_length() > 0:
            if _mode() == "IDENTICAL":
                raise Error(
                    "plans of one profile disagree (above). Under IDENTICAL that"
                    " is a contract violation, not a measurement."
                )
            print("NOTE (not IDENTICAL, not a failure): plans disagree, above.")
