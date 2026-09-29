# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What the TUNED Apple float-unit plans of the fifteen-bit GEMM cost
beside `fp32.v1`: one lever a variant, the product alone and the complete
operation.

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_int15_apple_tuned_price_main.mojo -o <binary>
    MOJOLEARN_TUNED_PRICE_IDENTITY_ONLY=1 <binary>      every arm once, nothing timed (the warm-up)
    <binary>                                            the timed run

Lane lane/lowbit-apple-tuned, 2026-09-29. Kernels
`gemm/checks/gemm_int15_apple_tuned.mojo`. It certifies nothing: the gate
is `gemm/checks/gemm_int15_apple_tuned_check.mojo`. It follows
`bench/gemm_int15_price_main.mojo` (lane/lowbit-int15): THE SAME ROWS
(`bench/gemm_shapes.mojo`'s twelve `llama8b` OP_NT rows and the two
backward rows derived from each training row), THE SAME OPERANDS (the same
fills and salts), the same budget and the same cap, so a digest printed
here is the digest that harness prints at the same row on any box, the
H100 and the MI325X included, and `fp32.v1` is the same measurement.
Apple only.

THE ARMS of one row, in the order the timed loop alternates them:

    fp32.v1                         `identical_gemm_into`, the shipped plan
                                    (`PLAN_APPLE_MMA` where its dispatcher
                                    picks it), at the row's own orientation
    int15i64.v1.apple.four          lane/lowbit-int15's form FOUR as it
                                    stands, the product alone: the start
                                    line, measured in THIS run
    convert.int15.planes.a.parallel float32 straight to planes, the left
    convert.int15.planes.b.parallel operand and the right, each alone
    tuned.<variant>                 the product alone on planes
    inference.tuned.<variant>       A straight to planes, then the product
                                    (forward rows only)
    training.tuned.<variant>        A and B straight to planes, then the
                                    product: THE COMPLETE OPERATION
    int15i64.v1.flat                identity only and only under
                                    MOJOLEARN_TUNED_PRICE_FLAT=1: the flat
                                    kernel, one thread per cell, never timed
                                    here (one launch of it takes seconds at
                                    the training rows on the M2 Pro, and
                                    macOS aborts a launch that long)

Each complete operation enqueues EVERY step of one call on the one in-order
context and waits once at its end (a launch in slices waits between its
slices as well), so its time is measured, not a sum of medians. The
recombination, the Int64 seam and the dequantization are the epilogue of
the product's kernel and are inside every product time.

NO TIME IS REPORTED FOR AN OUTPUT THAT WAS NOT CHECKED (the brief, "macOS
ABORTS A LONG METAL LAUNCH SILENTLY"). Before EVERY call of a product arm,
the warm-up and each timed one, the output is poisoned; after it the whole
output is read back, a surviving poison is refused, and the digest must be
the warm-up's. The poison and the read-back are outside the clock. An arm
that fails either prints `TUNED-FAILED` with the reason and no time.

THE LINES.
    TUNED <column> <row> <m> <n> <k> <FULL|CAPPED> <arm> <median ms>
          <min ms> <rate> <unit> <digest> <median over fp32.v1's>
    TUNED-FAILED <column> <row> <arm> <reason>
    TUNED-NOT-RUN <column> <row> <arm> <reason>
    DIGEST price-<row> <arm> <hex>      (fp32-<row> for `fp32.v1`)

ENVIRONMENT.
    MOJOLEARN_TUNED_PRICE_IDENTITY_ONLY  1: digests only, nothing timed
    MOJOLEARN_TUNED_PRICE_REPEATS        timed calls per arm (default 5)
    MOJOLEARN_TUNED_PRICE_MAC_BUDGET     the cap (default 2^35; 0 = none)
    MOJOLEARN_TUNED_PRICE_ONLY           comma separated substrings of row
                                         names; unset runs every row
    MOJOLEARN_TUNED_PRICE_VARIANTS       comma separated substrings of
                                         variant names; unset runs every one
    MOJOLEARN_TUNED_PRICE_SLICE_MACS     the slice of a tuned launch
                                         (default 2^32)
    MOJOLEARN_TUNED_PRICE_FLAT           1: the flat kernel's digest too

WHAT MAY NOT BE CONCLUDED. One call and one wait per sample, so a decode
row's time is mostly launch and wait. No arm is an inference engine or a
training step. A time printed here is one box's time on one run.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from bench.gemm_price_main import (
    _capped,
    _dev_digest,
    _dev_fill,
    _dev_poison,
    _fixed,
    _mode,
)
from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT,
    gemm_shape_k,
    gemm_shape_m,
    gemm_shape_n,
    gemm_shape_name,
    gemm_shape_op,
)
from bench.gemm_shapes import OP_NT as TBL_OP_NT
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, column_name
from gemm.checks.gemm_identical import (
    choose_gemm_plan,
    gemm_plan_name,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_int15 import (
    Int15QuantWorkspace,
    identical_gemm_int15_flat_into,
    int15_quant_chunks,
    quantize_int15_parallel_device,
    quantize_planes_int15_parallel_device,
)
from gemm.checks.gemm_int15_apple import (
    INT15_APPLE_FORM_FOUR,
    identical_gemm_int15_apple_into,
)
from gemm.checks.gemm_int15_apple_tuned import (
    INT15_TUNED_SLICE_MACS,
    TUNED_VARIANT_COUNT,
    identical_gemm_int15_apple_tuned_into,
    int15_apple_tuned_sabotage_name,
    int15_apple_tuned_variant_name,
)
from gemm.host.gemm_int15_oracle import INT15_MAX_K
from gemm.host.gemm_oracle import OP_NN, OP_NT, OP_TN

#: Timed calls per arm and row unless the environment says otherwise.
comptime DEFAULT_REPEATS = 5

#: The default multiply-accumulate budget per row, 2^35, lane/lowbit-units'.
comptime DEFAULT_MAC_BUDGET = 34_359_738_368

#: The arms that are not a tuned variant's.
comptime ARM_FP32 = 0
comptime ARM_C_FOUR = 1
comptime ARM_PLANES_A = 2
comptime ARM_PLANES_B = 3
comptime ARM_FLAT = 4
comptime ARM_FIRST_TUNED = 5

#: The three arms of a variant: `ARM_FIRST_TUNED + 3 * variant + part`.
comptime PART_PRODUCT = 0
comptime PART_INFERENCE = 1
comptime PART_TRAINING = 2
comptime ARM_COUNT = ARM_FIRST_TUNED + 3 * TUNED_VARIANT_COUNT

#: The kind of a row.
comptime ROW_FORWARD = 0
comptime ROW_BWD_DX = 1
comptime ROW_BWD_DW = 2

comptime IS_APPLE = TARGET_COLUMN == COLUMN_APPLE


def _arm_variant(arm: Int) -> Int:
    return (arm - ARM_FIRST_TUNED) // 3


def _arm_part(arm: Int) -> Int:
    return (arm - ARM_FIRST_TUNED) % 3


def _arm_name(arm: Int) -> String:
    if arm == ARM_FP32:
        return String("fp32.v1")
    if arm == ARM_C_FOUR:
        return String("int15i64.v1.apple.four")
    if arm == ARM_PLANES_A:
        return String("convert.int15.planes.a.parallel")
    if arm == ARM_PLANES_B:
        return String("convert.int15.planes.b.parallel")
    if arm == ARM_FLAT:
        return String("int15i64.v1.flat")
    var vn = int15_apple_tuned_variant_name(_arm_variant(arm))
    if _arm_part(arm) == PART_PRODUCT:
        return String("tuned.") + vn
    if _arm_part(arm) == PART_INFERENCE:
        return String("inference.tuned.") + vn
    return String("training.tuned.") + vn


def _arm_at(i: Int) -> Int:
    """The `i`-th arm of the alternation: `fp32.v1`, the start line, the
    conversions, every variant's product, every variant's inference call,
    every variant's training call, and the flat kernel last."""
    if i < ARM_FLAT:
        return i
    var j = i - ARM_FLAT
    if j < 3 * TUNED_VARIANT_COUNT:
        return ARM_FIRST_TUNED + 3 * (j % TUNED_VARIANT_COUNT) + j // TUNED_VARIANT_COUNT
    return ARM_FLAT


def _arm_is_product(arm: Int) -> Bool:
    """Whether the arm's output is the product `C`."""
    return arm != ARM_PLANES_A and arm != ARM_PLANES_B


def _wanted(only: String, name: String) -> Bool:
    if only == "":
        return True
    for part in only.split(","):
        if part.byte_length() > 0 and name.find(String(part)) >= 0:
            return True
    return False


def _why_not(arm: Int, kind: Int, k: Int, variants: String, flat: Bool) -> String:
    """Empty when this run takes the arm at this row; the reason
    otherwise, printed on a line of its own."""
    if arm == ARM_FP32:
        return String("")
    if k > INT15_MAX_K:
        return (
            String("REFUSED-BY-THE-PROFILE:k=") + String(k) + ">INT15_MAX_K="
            + String(INT15_MAX_K)
        )
    if arm == ARM_FLAT and not flat:
        return String("not-asked-for:MOJOLEARN_TUNED_PRICE_FLAT")
    if arm >= ARM_FIRST_TUNED:
        if not _wanted(variants, int15_apple_tuned_variant_name(_arm_variant(arm))):
            return String("not-asked-for:MOJOLEARN_TUNED_PRICE_VARIANTS")
        if _arm_part(arm) == PART_INFERENCE and kind != ROW_FORWARD:
            return String("inference-has-no-backward-product")
    return String("")


def _median_ms(samples: List[Int]) -> Float64:
    """The median of the timed calls, in milliseconds. An even count takes
    the mean of the middle pair."""
    var s = samples.copy()
    var n = len(s)
    for i in range(1, n):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return Float64(s[n // 2]) / 1.0e6
    return (Float64(s[n // 2 - 1]) + Float64(s[n // 2])) / 2.0e6


def _min_ms(samples: List[Int]) -> Float64:
    if len(samples) == 0:
        return 0.0
    var best = samples[0]
    for i in range(1, len(samples)):
        if samples[i] < best:
            best = samples[i]
    return Float64(best) / 1.0e6


def _rate(count: Float64, ms: Float64) -> Float64:
    """`count` per second, in units of 1e9."""
    if ms <= 0.0:
        return 0.0
    return count / (ms * 1.0e6)


struct TunedRowBuffers(Movable):
    """Every device buffer one row's arms read or write, allocated once per
    row. The conversion arms write scratch planes of their own, so timing a
    conversion never rewrites an operand a product arm reads."""

    var a: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var c: DeviceBuffer[DType.float32]
    var ws: DeviceBuffer[DType.float32]
    var qa: DeviceBuffer[DType.int16]
    var qb: DeviceBuffer[DType.int16]
    var ea: DeviceBuffer[DType.int32]
    var eb: DeviceBuffer[DType.int32]
    var ah: DeviceBuffer[DType.int8]
    var al: DeviceBuffer[DType.int8]
    var bh: DeviceBuffer[DType.int8]
    var bl: DeviceBuffer[DType.int8]
    var esa: DeviceBuffer[DType.int32]
    var esb: DeviceBuffer[DType.int32]
    var sah: DeviceBuffer[DType.int8]
    var sal: DeviceBuffer[DType.int8]
    var sbh: DeviceBuffer[DType.int8]
    var sbl: DeviceBuffer[DType.int8]
    var quant: Int15QuantWorkspace

    def __init__(out self, ctx: DeviceContext, m: Int, n: Int, k: Int, codes: Bool) raises:
        var nws = identical_gemm_workspace_max_floats(m, n, k)
        if nws < 1:
            nws = 1
        self.a = ctx.enqueue_create_buffer[DType.float32](m * k)
        self.b = ctx.enqueue_create_buffer[DType.float32](n * k)
        self.c = ctx.enqueue_create_buffer[DType.float32](m * n)
        self.ws = ctx.enqueue_create_buffer[DType.float32](nws)
        # The codes are read by the flat kernel only.
        self.qa = ctx.enqueue_create_buffer[DType.int16](m * k if codes else 1)
        self.qb = ctx.enqueue_create_buffer[DType.int16](n * k if codes else 1)
        self.ea = ctx.enqueue_create_buffer[DType.int32](m)
        self.eb = ctx.enqueue_create_buffer[DType.int32](n)
        self.ah = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.al = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.bh = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.bl = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.esa = ctx.enqueue_create_buffer[DType.int32](m)
        self.esb = ctx.enqueue_create_buffer[DType.int32](n)
        self.sah = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.sal = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.sbh = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.sbl = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.quant = Int15QuantWorkspace(ctx)
        self.quant.ensure(ctx, (m if m > n else n) * int15_quant_chunks(k))
        ctx.synchronize()


def _fp32_op(kind: Int) -> Int:
    """The orientation `fp32.v1` runs the row at."""
    if kind == ROW_BWD_DX:
        return OP_NN
    if kind == ROW_BWD_DW:
        return OP_TN
    return OP_NT


def _a_transposed(kind: Int) -> Bool:
    """The left operand is stored `k x m` on `bwd_dw` only."""
    return kind == ROW_BWD_DW


def _b_transposed(kind: Int) -> Bool:
    """The right operand is stored `k x n` on both backward rows."""
    return kind != ROW_FORWARD


def _enqueue_arm(
    ctx: DeviceContext,
    mut rb: TunedRowBuffers,
    arm: Int,
    kind: Int,
    m: Int,
    n: Int,
    k: Int,
    slice_macs: Int,
) raises:
    """Enqueue one arm's work and return; the caller waits. The timed
    region is this call and the wait, nothing else."""
    if arm == ARM_FP32:
        identical_gemm_into(ctx, rb.c, rb.a, rb.b, rb.ws, m, n, k, _fp32_op(kind))
    elif arm == ARM_C_FOUR:
        comptime if IS_APPLE:
            identical_gemm_int15_apple_into(
                ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k, INT15_APPLE_FORM_FOUR
            )
    elif arm == ARM_PLANES_A:
        quantize_planes_int15_parallel_device(
            ctx, rb.sah, rb.sal, rb.esa, rb.a, rb.quant, m, k, _a_transposed(kind)
        )
    elif arm == ARM_PLANES_B:
        quantize_planes_int15_parallel_device(
            ctx, rb.sbh, rb.sbl, rb.esb, rb.b, rb.quant, n, k, _b_transposed(kind)
        )
    elif arm == ARM_FLAT:
        identical_gemm_int15_flat_into(ctx, rb.c, rb.qa, rb.ea, rb.qb, rb.eb, m, n, k)
    else:
        comptime if IS_APPLE:
            var part = _arm_part(arm)
            if part != PART_PRODUCT:
                quantize_planes_int15_parallel_device(
                    ctx, rb.ah, rb.al, rb.ea, rb.a, rb.quant, m, k, _a_transposed(kind)
                )
            if part == PART_TRAINING:
                quantize_planes_int15_parallel_device(
                    ctx, rb.bh, rb.bl, rb.eb, rb.b, rb.quant, n, k, _b_transposed(kind)
                )
            identical_gemm_int15_apple_tuned_into(
                ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k,
                _arm_variant(arm), slice_macs,
            )


def _checked_call(
    ctx: DeviceContext,
    mut rb: TunedRowBuffers,
    arm: Int,
    kind: Int,
    m: Int,
    n: Int,
    k: Int,
    slice_macs: Int,
    tag: String,
    mut digest: UInt64,
) raises -> Int:
    """One call of one arm: the output poisoned, the call and its wait
    inside the clock, the whole output read back and its digest taken. A
    surviving poison raises. Returns the nanoseconds."""
    if _arm_is_product(arm):
        _dev_poison(ctx, rb.c, m * n)
    var t0 = perf_counter_ns()
    _enqueue_arm(ctx, rb, arm, kind, m, n, k, slice_macs)
    ctx.synchronize()
    var ns = Int(perf_counter_ns() - t0)
    if _arm_is_product(arm):
        digest = _dev_digest(ctx, rb.c, m * n, tag)
    return ns


def _time_row(
    ctx: DeviceContext,
    salt: Int,
    name: String,
    kind: Int,
    m: Int,
    n: Int,
    k: Int,
    capped: Bool,
    repeats: Int,
    identity_only: Bool,
    variants: String,
    flat: Bool,
    slice_macs: Int,
) raises -> String:
    """Every arm at one row. Returns what went wrong (empty when nothing
    did); `main` raises on it after every row has printed."""
    var cap_word = String("CAPPED") if capped else String("FULL")
    print()
    print(
        "== " + name + "  m=" + String(m) + " n=" + String(n) + " k=" + String(k)
        + "  " + cap_word + "  fp32 plan: " + gemm_plan_name(choose_gemm_plan(m, n, k))
    )
    var admitted = k <= INT15_MAX_K
    var rb = TunedRowBuffers(ctx, m, n, k, flat and admitted)
    # lane/lowbit-int15's salts, so every digest here is that harness's.
    _dev_fill(ctx, rb.a, m * k, 11 + salt)
    _dev_fill(ctx, rb.b, n * k, 22 + salt)
    if admitted:
        # The fifteen-bit operands, by the profile's own conversions on the
        # device, before anything is timed.
        quantize_planes_int15_parallel_device(
            ctx, rb.ah, rb.al, rb.ea, rb.a, rb.quant, m, k, _a_transposed(kind)
        )
        quantize_planes_int15_parallel_device(
            ctx, rb.bh, rb.bl, rb.eb, rb.b, rb.quant, n, k, _b_transposed(kind)
        )
        if flat:
            quantize_int15_parallel_device(
                ctx, rb.qa, rb.esa, rb.a, rb.quant, m, k, _a_transposed(kind)
            )
            quantize_int15_parallel_device(
                ctx, rb.qb, rb.esb, rb.b, rb.quant, n, k, _b_transposed(kind)
            )
    ctx.synchronize()

    var dig = List[UInt64]()
    var why = List[String]()
    var failed = List[String]()
    var samples = List[List[Int]]()
    for arm in range(ARM_COUNT):
        dig.append(UInt64(0))
        why.append(_why_not(arm, kind, k, variants, flat))
        failed.append(String(""))
        samples.append(List[Int]())

    # The untimed warm-up of every arm, its output poisoned first and read
    # back whole after.
    for at_ in range(ARM_COUNT):
        var arm = _arm_at(at_)
        if why[arm].byte_length() > 0:
            continue
        var tag = String("tuned.") + name + "." + _arm_name(arm)
        try:
            var d = UInt64(0)
            _ = _checked_call(ctx, rb, arm, kind, m, n, k, slice_macs, tag, d)
            dig[arm] = d
        except e:
            failed[arm] = String("warm-up:") + String(e)

    for _ in range(0 if identity_only else repeats):
        for at_ in range(ARM_COUNT):
            var arm = _arm_at(at_)
            if why[arm].byte_length() > 0 or failed[arm].byte_length() > 0:
                continue
            if arm == ARM_FLAT:
                continue
            var tag = String("tuned.") + name + "." + _arm_name(arm)
            try:
                var d = UInt64(0)
                var ns = _checked_call(ctx, rb, arm, kind, m, n, k, slice_macs, tag, d)
                if _arm_is_product(arm) and d != dig[arm]:
                    failed[arm] = (
                        String("a-timed-call-printed-") + hex(d) + "-the-warm-up-" + hex(dig[arm])
                    )
                else:
                    samples[arm].append(ns)
            except e:
                failed[arm] = String("timed:") + String(e)

    var macs = Float64(m) * Float64(n) * Float64(k)
    var fp32_med = _median_ms(samples[ARM_FP32])
    var bad = String("")
    for arm in range(ARM_COUNT):
        var arm_name = _arm_name(arm)
        if why[arm].byte_length() > 0:
            print("TUNED-NOT-RUN", column_name(TARGET_COLUMN), name, arm_name, why[arm])
            continue
        if failed[arm].byte_length() > 0:
            print("TUNED-FAILED", column_name(TARGET_COLUMN), name, arm_name, failed[arm])
            bad += String("FAILED at ") + name + ": " + arm_name + ": " + failed[arm] + "\n"
            continue
        var unit = String("GMAC/s")
        var count = macs
        if arm == ARM_PLANES_A:
            unit = String("Gelem/s")
            count = Float64(m) * Float64(k)
        elif arm == ARM_PLANES_B:
            unit = String("Gelem/s")
            count = Float64(n) * Float64(k)
        if arm == ARM_FP32:
            print("DIGEST fp32-" + name + " " + arm_name + " " + hex(dig[arm]))
        elif _arm_is_product(arm):
            print("DIGEST price-" + name + " " + arm_name + " " + hex(dig[arm]))
        if identity_only or arm == ARM_FLAT:
            print(
                "TUNED", column_name(TARGET_COLUMN), name, m, n, k, cap_word, arm_name,
                "not-timed", "not-timed", "not-timed", unit, hex(dig[arm]), "not-timed",
            )
            continue
        var med = _median_ms(samples[arm])
        var best = _min_ms(samples[arm])
        var over = String("no-fp32-time")
        if fp32_med > 0.0:
            over = _fixed(med / fp32_med, 3)
        print(
            "TUNED", column_name(TARGET_COLUMN), name, m, n, k, cap_word, arm_name,
            _fixed(med, 4), _fixed(best, 4), _fixed(_rate(count, med), 4), unit,
            hex(dig[arm]), over,
        )

    # Every fifteen-bit product and every complete operation of a row ends
    # in the profile's product of the same operands: one digest. The
    # reference is the flat kernel's where it ran, the start line's
    # elsewhere.
    var ref_arm = ARM_C_FOUR
    if flat and why[ARM_FLAT].byte_length() == 0 and failed[ARM_FLAT].byte_length() == 0:
        ref_arm = ARM_FLAT
    if why[ref_arm].byte_length() == 0 and failed[ref_arm].byte_length() == 0:
        for arm in range(ARM_COUNT):
            if arm == ARM_FP32 or arm == ref_arm or not _arm_is_product(arm):
                continue
            if why[arm].byte_length() > 0 or failed[arm].byte_length() > 0:
                continue
            if dig[arm] != dig[ref_arm]:
                bad += (
                    String("PLANS DISAGREE at ") + name + ": " + _arm_name(ref_arm) + " "
                    + hex(dig[ref_arm]) + " vs " + _arm_name(arm) + " " + hex(dig[arm]) + "\n"
                )
    if bad.byte_length() > 0:
        print(bad)
    _ = rb^
    return bad


def _run_row(
    ctx: DeviceContext,
    salt: Int,
    name: String,
    kind: Int,
    m: Int,
    n: Int,
    k: Int,
    budget: Int,
    only: String,
    repeats: Int,
    identity_only: Bool,
    variants: String,
    flat: Bool,
    slice_macs: Int,
    mut rows: Int,
) raises -> String:
    if not _wanted(only, name):
        return String("")
    var dm = m
    var dn = n
    if budget > 0:
        var cap = _capped(m, n, k, budget)
        dm = cap[0]
        dn = cap[1]
    rows += 1
    return _time_row(
        ctx, salt, name, kind, dm, dn, k, dm != m or dn != n, repeats, identity_only,
        variants, flat, slice_macs,
    )


def main() raises:
    comptime if not has_accelerator():
        raise Error("bench/gemm_int15_apple_tuned_price_main: no accelerator; every arm here is a device arm")
    else:
        comptime if not IS_APPLE:
            raise Error("bench/gemm_int15_apple_tuned_price_main: this column is not Apple")
        else:
            var repeats = DEFAULT_REPEATS
            var rs = String(getenv("MOJOLEARN_TUNED_PRICE_REPEATS"))
            if rs != "":
                repeats = Int(atol(rs))
            var budget = DEFAULT_MAC_BUDGET
            var bs = String(getenv("MOJOLEARN_TUNED_PRICE_MAC_BUDGET"))
            if bs != "":
                budget = Int(atol(bs))
            var slice_macs = INT15_TUNED_SLICE_MACS
            var ss = String(getenv("MOJOLEARN_TUNED_PRICE_SLICE_MACS"))
            if ss != "":
                slice_macs = Int(atol(ss))
            var only = String(getenv("MOJOLEARN_TUNED_PRICE_ONLY"))
            var variants = String(getenv("MOJOLEARN_TUNED_PRICE_VARIANTS"))
            var identity_only = String(getenv("MOJOLEARN_TUNED_PRICE_IDENTITY_ONLY")) == "1"
            var flat = String(getenv("MOJOLEARN_TUNED_PRICE_FLAT")) == "1"

            print("== bench/gemm_int15_apple_tuned_price_main.mojo [" + _mode() + "] ==")
            print("column", column_name(TARGET_COLUMN))
            print("profiles fp32.v1, int15i64.v1")
            print("sabotage:", int15_apple_tuned_sabotage_name())
            print(
                "repeats", repeats, " mac budget", budget, " slice", slice_macs,
                " only", only, " variants", variants, " flat", flat,
            )
            if identity_only:
                print(
                    "IDENTITY ONLY: every arm runs once for its digest and NOTHING",
                    "IS TIMED in this run.",
                )
            else:
                print(
                    "EVERY NUMBER BELOW IS ONE BOX'S TIME ON ONE RUN. One call per",
                    "sample; the median of the timed calls is reported and the",
                    "minimum beside it. Every timed output was poisoned before the",
                    "call and read back whole after it.",
                )

            var bad = String("")
            var rows = 0
            with DeviceContext() as ctx:
                for i in range(GEMM_SHAPE_COUNT):
                    if gemm_shape_op(i) != TBL_OP_NT:
                        continue
                    var name = gemm_shape_name(i)
                    if name.find("llama8b") < 0:
                        continue
                    var m = gemm_shape_m(i)
                    var n = gemm_shape_n(i)
                    var k = gemm_shape_k(i)
                    bad += _run_row(
                        ctx, i, name, ROW_FORWARD, m, n, k, budget, only, repeats,
                        identity_only, variants, flat, slice_macs, rows,
                    )
                    if name.find(".t512") < 0:
                        continue
                    # The two backward rows of a training row, derived from it.
                    bad += _run_row(
                        ctx, 100 + i, name + ".bwd_dx", ROW_BWD_DX, m, k, n, budget, only,
                        repeats, identity_only, variants, flat, slice_macs, rows,
                    )
                    bad += _run_row(
                        ctx, 200 + i, name + ".bwd_dw", ROW_BWD_DW, n, k, m, budget, only,
                        repeats, identity_only, variants, flat, slice_macs, rows,
                    )

            print()
            print("== done [" + _mode() + "]: " + String(rows) + " rows ==")
            if rows == 0:
                raise Error("bench/gemm_int15_apple_tuned_price_main: no row matched MOJOLEARN_TUNED_PRICE_ONLY=" + only)
            if bad.byte_length() > 0:
                raise Error(
                    "an arm failed or plans of one profile disagree (above). Under"
                    " IDENTICAL that is a contract violation, not a measurement."
                )
