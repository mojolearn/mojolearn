# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What the FIFTEEN-BIT GEMM costs beside `fp32.v1`: the complete operation.

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_int15_price_main.mojo -o <binary>
    MOJOLEARN_INT15_PRICE_IDENTITY_ONLY=1 <binary>      every arm once, nothing timed (the warm-up)
    <binary>                                            the timed run

Lane lane/lowbit-int15, 2026-09-29. Profile
`mojolearn.identical.gemm.int15i64.v1`, contract
`gemm/IDENTICAL_LOWBIT_CONTRACT.md` section 6. It certifies nothing: the
gates are `gemm/checks/gemm_int15_check.mojo` and
`gemm/checks/gemm_int15_sim_check.mojo`, and a time printed here is one
box's time on one run. It follows `bench/gemm_lowbit_price_main.mojo` of
lane/lowbit-units (the same rows, the same operands, the same budget, the
same alternation), so the two lanes' `fp32.v1` columns are the same
measurement.

THE ROWS ARE `bench/gemm_shapes.mojo`'s. The FORWARD rows are its twelve
OP_NT transformer rows (`llama8b.*`: decode at t1 and t8, training at
t512). The BACKWARD rows are not in that table, and they are not invented
either: each is DERIVED from a forward training row `(m, n, k)` by the
rule a linear layer's gradient follows, and carries the row's name:

    <row>.bwd_dx   dX = dY . W       m x k  from  [m x n] . [n x k]
                   fp32.v1: OP_NN, A = dY [m x n], B = W [n x k]
                   the contraction is over the OUTPUT width, n
    <row>.bwd_dw   dW = dY^T . X     n x k  from  [m x n]^T . [m x k]
                   fp32.v1: OP_TN, A = dY [m x n], B = X [m x k]
                   the contraction is over the TOKEN count, m

The fifteen-bit profile is OP_NT only. A backward product runs under it as
an OP_NT product of operands quantized along the contracted extent, so an
operand that is stored the other way is quantized by the TRANSPOSING
quantizer (`quantize_cols_int15_device`), and that cost is counted.

`k` IS NEVER CAPPED. `m` and `n` are reduced by `bench/gemm_price_main.mojo`'s
`_capped` rule where a row's multiply-accumulate count is above
`MOJOLEARN_INT15_PRICE_MAC_BUDGET` (default 2^35). A row whose contracted
extent is above `INT15_MAX_K` is REFUSED by the profile; its fifteen-bit
arms print a refusal line and `fp32.v1` is still timed.

THE ARMS. Every arm of one row lives in this one binary and the timed loop
ALTERNATES them call by call, in `_arm_at`'s order, IN TWO BLOCKS:

    the first block    `fp32.v1`, the unit products, the parallel
                       conversions and the splits, the complete operations
    the slow block     the reference kernels that are one thread per cell
                       or one thread per row (flat, pieces, the row
                       quantizer and the two complete operations built on
                       it), timed after the first block has finished

A LAUNCH OF HUNDREDS OF MILLISECONDS DISTURBS THE ARM TIMED AFTER IT, and
this harness found that twice. Runs 1 to 3 (H100): the unit product was
timed directly after the pieces kernel (300 ms at the training rows) and
read ABOVE the complete call that contains it. Run 4 moved the slow kernels
to the end of one alternation, which put them directly before `fp32.v1` of
the next pass: `fp32.v1` read 5.6 to 6.1 ms at four rows where three runs
had read 3.5 to 3.9. In both cases the ratio moved without any kernel
changing. With two blocks no arm of the first block follows a slow one.

    fp32.v1                    `identical_gemm_into`, the shipped plan, at
                               the row's own orientation
    int15i64.v1.flat           the product alone, on codes
    int15i64.v1.pieces         the product alone, on planes, no unit
    int15i64.v1.mma            the product alone, on planes, on the integer
                               matrix unit (NVIDIA, AMD): FOUR unit products
    int15i64.v1.tuned          the product alone, on planes, on the integer
                               matrix unit, lane/lowbit-mma-speed's four
                               products with ONE staging and then this
                               profile's epilogue as a launch of its own
                               (contract W-13). Not dispatched.
    int15i64.v1.apple.two      the product alone, on planes, on Apple's
                               FLOAT matrix unit: TWO unit products, the
                               left operand whole, carried into integers
                               after every step of the unit
    int15i64.v1.apple.four     the same unit, FOUR unit products, carried
                               into integers every 512 steps
    convert.int15.quantize.a            float32 to codes, the left
                                        operand, the REFERENCE schedule:
                                        one thread per row
    convert.int15.quantize.a.parallel   the same codes, the PARALLEL
                                        schedule (three launches)
    convert.int15.planes.a.parallel     float32 straight to planes, the
                                        parallel schedule
    convert.int15.split.a               codes to planes
    convert.int15.*.b                   the same four, the right operand

The recombination (clause W-5) and the dequantization (W-6, W-7) are the
epilogue of every fifteen-bit kernel, one per output cell, and are inside
each product arm's time; they have no launch of their own to time.

THE COMPLETE OPERATION, TWICE. Each arm below enqueues EVERY step of one
call on the one in-order context and waits once, so its time is measured,
not a sum of medians.

    INFERENCE (forward rows): the weights were quantized and split ONCE and
    are reused. One call pays for its activations and the product.
        inference.int15i64.v1.planes   A straight to planes (parallel), the
                                       product on planes (the unit where
                                       the column has one, the pieces
                                       kernel elsewhere)
        inference.int15i64.v1.codes    A to codes (parallel), the entry
                                       point for codes: on a column with
                                       the unit it splits BOTH operands
                                       every call, elsewhere it runs the
                                       flat kernel
        inference.int15i64.v1.planes.rowquant
                                       the planes arm with the REFERENCE
                                       quantizer and the split: what the
                                       parallel schedule bought

    TRAINING (forward and backward rows): the weights change every step and
    a gradient is new every step, so BOTH operands are converted per call.
        training.int15i64.v1.planes    A and B straight to planes
                                       (parallel), the product on planes
        training.int15i64.v1.codes     A and B to codes (parallel), the
                                       entry point for codes
        training.int15i64.v1.planes.rowquant
                                       the planes arm with the reference
                                       quantizer and the splits

    ON A COLUMN WITH THE INTEGER UNIT each complete operation is measured
    on the tuned plan too:
        inference.int15i64.v1.tuned    A straight to planes, the tuned plan
        training.int15i64.v1.tuned     A and B straight to planes, the
                                       tuned plan

    ON APPLE each complete operation is measured on the float unit too:
        inference.int15i64.v1.apple.two, .apple.four
        training.int15i64.v1.apple.two, .apple.four
    (operands straight to planes by the parallel quantizer, then the unit).

A training STEP of one layer is its forward row and its two backward rows;
`tools/lowbit_int15/table.py` adds the three measured operations and says
that it added them.

ON APPLE EVERY TIME IS THE WHOLE KERNEL'S. `fp32.v1` there is
`PLAN_APPLE_MMA` where its dispatcher picks it: the staging, the admission
test of every window and any window that fell back to the exact step are
inside its time. The fifteen-bit plans on Apple admit every shape, so they
have no admission test and no fallback; the float-unit plans' staging, the
conversion of every code to a float and every carry into integers are
inside their times.

THE DIGESTS. After its untimed warm-up every arm's output is read back, a
surviving poison is refused, and the FNV-1a digest of the output bits is
printed on the arm's line and, for the cross-box comparison, as a
`DIGEST price-<row> <arm> <hex>` line (`tools/lowbit_int15/digests.py`).
Inside one run every fifteen-bit product and every complete operation of a
row must print ONE digest; a disagreement under IDENTICAL raises after
every line is printed. A build carrying `-D MOJOLEARN_LOWBIT_SABOTAGE=1`
must print other digests than a clean one at every fifteen-bit arm.

THE LINES.
    PRICE <mode> int15.<row>.<arm>.device <median ms>
    INT15 <column> <row> <m> <n> <k> <FULL|CAPPED> <arm> <median ms>
          <min ms> <rate> <unit> <digest> <note>
    INT15-NOT-RUN <column> <row> <arm> <reason>
    DIGEST price-<row> <arm> <hex>

ENVIRONMENT.
    MOJOLEARN_INT15_PRICE_IDENTITY_ONLY  1: digests only, nothing timed
    MOJOLEARN_INT15_PRICE_REPEATS        timed calls per arm (default 5)
    MOJOLEARN_INT15_PRICE_MAC_BUDGET     see above (default 2^35; 0 = no cap)
    MOJOLEARN_INT15_PRICE_ONLY           comma separated substrings of row
                                         names; unset runs every row

WHAT MAY NOT BE CONCLUDED. One call and one synchronize per sample, so a
decode row's time is mostly launch and wait. No arm is an inference engine
or a training step. The flat and pieces kernels are one thread per cell
with no tiling and the unit kernel stages nothing in shared memory, so an
arm against `fp32.v1` compares two levels of kernel engineering as well as
two arithmetics.
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
    _report_device,
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
from checks.kernel_matrix import (
    COLUMN_APPLE,
    TARGET_COLUMN,
    column_name,
    lib_int8_matrix_unit_for,
)
from gemm.checks.gemm_identical import (
    choose_gemm_plan,
    gemm_plan_name,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_int15 import (
    Int15QuantWorkspace,
    Int15Workspace,
    identical_gemm_int15_flat_into,
    identical_gemm_int15_into,
    identical_gemm_int15_mma_into,
    identical_gemm_int15_pieces_into,
    identical_gemm_int15_planes_into,
    int15_plan_dispatch_name,
    int15_sabotage_name,
    int15_quant_chunks,
    quantize_cols_int15_device,
    quantize_int15_parallel_device,
    quantize_planes_int15_parallel_device,
    quantize_rows_int15_device,
    split_int15_device,
)
from gemm.checks.gemm_int15_apple import (
    INT15_APPLE_FORM_FOUR,
    INT15_APPLE_FORM_TWO,
    identical_gemm_int15_apple_into,
)
from gemm.checks.gemm_int15_tuned import (
    Int15SumsWorkspace,
    identical_gemm_int15_tuned_into,
)
from gemm.host.gemm_int15_oracle import INT15_MAX_K
from gemm.host.gemm_oracle import OP_NN, OP_NT, OP_TN

#: Timed calls per arm and row unless the environment says otherwise.
comptime DEFAULT_REPEATS = 5

#: The default multiply-accumulate budget per row, 2^35, lane/lowbit-units'.
comptime DEFAULT_MAC_BUDGET = 34_359_738_368

#: The arms, in the order the timed loop alternates them.
comptime ARM_FP32 = 0
comptime ARM_FLAT = 1
comptime ARM_PIECES = 2
comptime ARM_MMA = 3
comptime ARM_QUANTIZE_A = 4
comptime ARM_QUANTIZE_A_PAR = 5
comptime ARM_PLANES_A_PAR = 6
comptime ARM_SPLIT_A = 7
comptime ARM_QUANTIZE_B = 8
comptime ARM_QUANTIZE_B_PAR = 9
comptime ARM_PLANES_B_PAR = 10
comptime ARM_SPLIT_B = 11
comptime ARM_INF_PLANES = 12
comptime ARM_INF_CODES = 13
comptime ARM_INF_ROWQUANT = 14
comptime ARM_TRAIN_PLANES = 15
comptime ARM_TRAIN_CODES = 16
comptime ARM_TRAIN_ROWQUANT = 17
comptime ARM_APPLE2 = 18
comptime ARM_APPLE4 = 19
comptime ARM_INF_APPLE2 = 20
comptime ARM_INF_APPLE4 = 21
comptime ARM_TRAIN_APPLE2 = 22
comptime ARM_TRAIN_APPLE4 = 23
comptime ARM_TUNED = 24
comptime ARM_INF_TUNED = 25
comptime ARM_TRAIN_TUNED = 26
comptime ARM_COUNT = 27

#: The kinds of row.
comptime ROW_FORWARD = 0
comptime ROW_BWD_DX = 1
comptime ROW_BWD_DW = 2

comptime HAS_UNIT = lib_int8_matrix_unit_for[TARGET_COLUMN]()
comptime IS_APPLE = TARGET_COLUMN == COLUMN_APPLE


def _arm_name(arm: Int) -> String:
    if arm == ARM_FP32:
        return String("fp32.v1")
    if arm == ARM_FLAT:
        return String("int15i64.v1.flat")
    if arm == ARM_PIECES:
        return String("int15i64.v1.pieces")
    if arm == ARM_MMA:
        return String("int15i64.v1.mma")
    if arm == ARM_QUANTIZE_A:
        return String("convert.int15.quantize.a")
    if arm == ARM_QUANTIZE_A_PAR:
        return String("convert.int15.quantize.a.parallel")
    if arm == ARM_PLANES_A_PAR:
        return String("convert.int15.planes.a.parallel")
    if arm == ARM_SPLIT_A:
        return String("convert.int15.split.a")
    if arm == ARM_QUANTIZE_B:
        return String("convert.int15.quantize.b")
    if arm == ARM_QUANTIZE_B_PAR:
        return String("convert.int15.quantize.b.parallel")
    if arm == ARM_PLANES_B_PAR:
        return String("convert.int15.planes.b.parallel")
    if arm == ARM_SPLIT_B:
        return String("convert.int15.split.b")
    if arm == ARM_INF_PLANES:
        return String("inference.int15i64.v1.planes")
    if arm == ARM_INF_CODES:
        return String("inference.int15i64.v1.codes")
    if arm == ARM_INF_ROWQUANT:
        return String("inference.int15i64.v1.planes.rowquant")
    if arm == ARM_TRAIN_PLANES:
        return String("training.int15i64.v1.planes")
    if arm == ARM_TRAIN_CODES:
        return String("training.int15i64.v1.codes")
    if arm == ARM_TRAIN_ROWQUANT:
        return String("training.int15i64.v1.planes.rowquant")
    if arm == ARM_APPLE2:
        return String("int15i64.v1.apple.two")
    if arm == ARM_APPLE4:
        return String("int15i64.v1.apple.four")
    if arm == ARM_INF_APPLE2:
        return String("inference.int15i64.v1.apple.two")
    if arm == ARM_INF_APPLE4:
        return String("inference.int15i64.v1.apple.four")
    if arm == ARM_TRAIN_APPLE2:
        return String("training.int15i64.v1.apple.two")
    if arm == ARM_TRAIN_APPLE4:
        return String("training.int15i64.v1.apple.four")
    if arm == ARM_TUNED:
        return String("int15i64.v1.tuned")
    if arm == ARM_INF_TUNED:
        return String("inference.int15i64.v1.tuned")
    return String("training.int15i64.v1.tuned")


def _arm_is_left(arm: Int) -> Bool:
    """Whether a conversion arm converts the left operand."""
    return arm >= ARM_QUANTIZE_A and arm <= ARM_SPLIT_A


def _arm_is_inference(arm: Int) -> Bool:
    return (
        arm == ARM_INF_PLANES or arm == ARM_INF_CODES or arm == ARM_INF_ROWQUANT
        or arm == ARM_INF_APPLE2 or arm == ARM_INF_APPLE4 or arm == ARM_INF_TUNED
    )


def _arm_is_apple_unit(arm: Int) -> Bool:
    return arm >= ARM_APPLE2 and arm <= ARM_TRAIN_APPLE4


def _arm_is_tuned(arm: Int) -> Bool:
    return arm >= ARM_TUNED


#: Arms of the first block of the alternation; the rest are the slow block.
comptime ARM_FIRST_BLOCK = 21


def _arm_at(i: Int) -> Int:
    """The `i`-th arm of the alternation (see THE ARMS): the first
    `ARM_FIRST_BLOCK` are the first block, the rest the slow block."""
    var order: List[Int] = [
        ARM_FP32, ARM_MMA, ARM_TUNED, ARM_APPLE2, ARM_APPLE4,
        ARM_QUANTIZE_A_PAR, ARM_PLANES_A_PAR, ARM_SPLIT_A,
        ARM_QUANTIZE_B_PAR, ARM_PLANES_B_PAR, ARM_SPLIT_B,
        ARM_INF_TUNED, ARM_TRAIN_TUNED,
        ARM_INF_APPLE2, ARM_INF_APPLE4, ARM_TRAIN_APPLE2, ARM_TRAIN_APPLE4,
        ARM_INF_PLANES, ARM_INF_CODES,
        ARM_TRAIN_PLANES, ARM_TRAIN_CODES,
        ARM_QUANTIZE_A, ARM_QUANTIZE_B,
        ARM_INF_ROWQUANT, ARM_TRAIN_ROWQUANT,
        ARM_FLAT, ARM_PIECES,
    ]
    return order[i]


def _arm_is_product(arm: Int) -> Bool:
    """Whether the arm's output is the product `C`."""
    return arm <= ARM_MMA or arm >= ARM_INF_PLANES


def _arm_is_int15_product(arm: Int) -> Bool:
    return _arm_is_product(arm) and arm != ARM_FP32


def _why_not(arm: Int, kind: Int, k: Int) -> String:
    """Empty when this build runs the arm at this row; the reason
    otherwise. An arm that does not run prints the reason on a line of its
    own: a missing line is never an agreeing one."""
    if arm == ARM_FP32:
        return String("")
    if k > INT15_MAX_K:
        return (
            String("REFUSED-BY-THE-PROFILE:k=") + String(k) + ">INT15_MAX_K="
            + String(INT15_MAX_K)
        )
    if (arm == ARM_MMA or _arm_is_tuned(arm)) and not HAS_UNIT:
        return String("this-column-has-no-integer-matrix-unit")
    if _arm_is_apple_unit(arm) and not IS_APPLE:
        return String("this-column-is-not-apple")
    if _arm_is_inference(arm) and kind != ROW_FORWARD:
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
    """`count` per second, in units of 1e9. Zero when the time is below the
    clock's resolution."""
    if ms <= 0.0:
        return 0.0
    return count / (ms * 1.0e6)


def _fnv(d: UInt64, word: UInt64) -> UInt64:
    return (d ^ word) * UInt64(0x100000001B3)


def _whole(have: Int, count: Int, what: String) raises:
    """A read-back copies the WHOLE device buffer into a host buffer of
    `count` elements. A device buffer of any other length is refused by
    name (lane/lowbit-units found what happens otherwise, on the M2 Pro)."""
    if have != count:
        raise Error(
            "bench/gemm_int15_price_main: " + what + " holds " + String(have)
            + " elements and " + String(count) + " were asked for; the copy is"
            " of the whole buffer, so the two must be equal"
        )


def _digest_codes(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int16],
    mut e: DeviceBuffer[DType.int32],
    rows: Int,
    cols: Int,
) raises -> UInt64:
    """The digest of a store of codes: every code, then every row exponent."""
    _whole(len(q), rows * cols, String("a code buffer"))
    _whole(len(e), rows, String("an exponent buffer"))
    var hq = ctx.enqueue_create_host_buffer[DType.int16](rows * cols)
    var he = ctx.enqueue_create_host_buffer[DType.int32](rows)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hq.unsafe_ptr(), src_buf=q)
    ctx.enqueue_copy(dst_ptr=he.unsafe_ptr(), src_buf=e)
    ctx.synchronize()
    var d = UInt64(0xCBF29CE484222325)
    for i in range(rows * cols):
        d = _fnv(d, UInt64(Int(hq.unsafe_ptr().unsafe_load(i)) & 0xFFFF))
    for i in range(rows):
        d = _fnv(d, UInt64(Int(he.unsafe_ptr().unsafe_load(i)) & 0xFFFFFFFF))
    _ = hq
    _ = he
    return d


def _digest_planes(
    ctx: DeviceContext,
    mut hi: DeviceBuffer[DType.int8],
    mut lo: DeviceBuffer[DType.int8],
    count: Int,
) raises -> UInt64:
    """The digest of two planes: every high piece, then every low piece."""
    _whole(len(hi), count, String("a high plane"))
    _whole(len(lo), count, String("a low plane"))
    var hh = ctx.enqueue_create_host_buffer[DType.int8](count)
    var hl = ctx.enqueue_create_host_buffer[DType.int8](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hh.unsafe_ptr(), src_buf=hi)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=lo)
    ctx.synchronize()
    var d = UInt64(0xCBF29CE484222325)
    for i in range(count):
        d = _fnv(d, UInt64(Int(hh.unsafe_ptr().unsafe_load(i)) & 0xFF))
    for i in range(count):
        d = _fnv(d, UInt64(Int(hl.unsafe_ptr().unsafe_load(i)) & 0xFF))
    _ = hh
    _ = hl
    return d


struct RowBuffers(Movable):
    """Every device buffer one row's arms read or write, allocated once per
    row and shared by the arms, so no arm's time holds an allocation. The
    conversion arms write scratch stores of their own, each EXACTLY the
    size of what is written to it, so timing a conversion never rewrites an
    operand a product arm reads."""

    var a: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var c: DeviceBuffer[DType.float32]
    var ws: DeviceBuffer[DType.float32]
    var qa: DeviceBuffer[DType.int16]
    var ea: DeviceBuffer[DType.int32]
    var qb: DeviceBuffer[DType.int16]
    var eb: DeviceBuffer[DType.int32]
    var ah: DeviceBuffer[DType.int8]
    var al: DeviceBuffer[DType.int8]
    var bh: DeviceBuffer[DType.int8]
    var bl: DeviceBuffer[DType.int8]
    var qsa: DeviceBuffer[DType.int16]
    var esa: DeviceBuffer[DType.int32]
    var qsb: DeviceBuffer[DType.int16]
    var esb: DeviceBuffer[DType.int32]
    var sah: DeviceBuffer[DType.int8]
    var sal: DeviceBuffer[DType.int8]
    var sbh: DeviceBuffer[DType.int8]
    var sbl: DeviceBuffer[DType.int8]
    var work: Int15Workspace
    var quant: Int15QuantWorkspace
    var sums: Int15SumsWorkspace

    def __init__(out self, ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
        var nws = identical_gemm_workspace_max_floats(m, n, k)
        if nws < 1:
            nws = 1
        self.a = ctx.enqueue_create_buffer[DType.float32](m * k)
        self.b = ctx.enqueue_create_buffer[DType.float32](n * k)
        self.c = ctx.enqueue_create_buffer[DType.float32](m * n)
        self.ws = ctx.enqueue_create_buffer[DType.float32](nws)
        self.qa = ctx.enqueue_create_buffer[DType.int16](m * k)
        self.ea = ctx.enqueue_create_buffer[DType.int32](m)
        self.qb = ctx.enqueue_create_buffer[DType.int16](n * k)
        self.eb = ctx.enqueue_create_buffer[DType.int32](n)
        self.ah = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.al = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.bh = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.bl = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.qsa = ctx.enqueue_create_buffer[DType.int16](m * k)
        self.esa = ctx.enqueue_create_buffer[DType.int32](m)
        self.qsb = ctx.enqueue_create_buffer[DType.int16](n * k)
        self.esb = ctx.enqueue_create_buffer[DType.int32](n)
        self.sah = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.sal = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.sbh = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.sbl = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.work = Int15Workspace(ctx)
        self.work.ensure(ctx, m * k, n * k)
        self.sums = Int15SumsWorkspace(ctx)
        comptime if HAS_UNIT:
            self.sums.ensure(ctx, m * n)
        self.quant = Int15QuantWorkspace(ctx)
        var chunks = int15_quant_chunks(k)
        self.quant.ensure(ctx, (m if m > n else n) * chunks)
        ctx.synchronize()


def _fp32_op(kind: Int) -> Int:
    """The orientation `fp32.v1` runs the row at."""
    if kind == ROW_BWD_DX:
        return OP_NN
    if kind == ROW_BWD_DW:
        return OP_TN
    return OP_NT


def _quantize_a(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int16],
    mut e: DeviceBuffer[DType.int32],
    mut a: DeviceBuffer[DType.float32],
    kind: Int,
    m: Int,
    k: Int,
) raises:
    """The left operand to `m x k` codes. It is stored `m x k` on a forward
    row and on `bwd_dx`, and `k x m` on `bwd_dw` (OP_TN's left operand),
    where the transposing quantizer reads it."""
    if kind == ROW_BWD_DW:
        quantize_cols_int15_device(ctx, q, e, a, k, m)
    else:
        quantize_rows_int15_device(ctx, q, e, a, m, k)


def _quantize_b(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int16],
    mut e: DeviceBuffer[DType.int32],
    mut b: DeviceBuffer[DType.float32],
    kind: Int,
    n: Int,
    k: Int,
) raises:
    """The right operand to `n x k` codes. It is stored `n x k` on a
    forward row and `k x n` on both backward rows (OP_NN's and OP_TN's
    right operand)."""
    if kind == ROW_FORWARD:
        quantize_rows_int15_device(ctx, q, e, b, n, k)
    else:
        quantize_cols_int15_device(ctx, q, e, b, k, n)


def _quantize_par(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int16],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    mut quant: Int15QuantWorkspace,
    transposed: Bool,
    rows: Int,
    k: Int,
) raises:
    quantize_int15_parallel_device(ctx, q, e, x, quant, rows, k, transposed)


def _planes_par(
    ctx: DeviceContext,
    mut hi: DeviceBuffer[DType.int8],
    mut lo: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    mut quant: Int15QuantWorkspace,
    transposed: Bool,
    rows: Int,
    k: Int,
) raises:
    quantize_planes_int15_parallel_device(ctx, hi, lo, e, x, quant, rows, k, transposed)


def _a_transposed(kind: Int) -> Bool:
    """The left operand is stored `k x m` on `bwd_dw` only."""
    return kind == ROW_BWD_DW


def _b_transposed(kind: Int) -> Bool:
    """The right operand is stored `k x n` on both backward rows."""
    return kind != ROW_FORWARD


def _enqueue_arm(
    ctx: DeviceContext, mut rb: RowBuffers, arm: Int, kind: Int, m: Int, n: Int, k: Int
) raises:
    """Enqueue one arm's work and return; the caller waits. The timed
    region is this call and the wait, nothing else."""
    if arm == ARM_FP32:
        identical_gemm_into(ctx, rb.c, rb.a, rb.b, rb.ws, m, n, k, _fp32_op(kind))
    elif arm == ARM_FLAT:
        identical_gemm_int15_flat_into(ctx, rb.c, rb.qa, rb.ea, rb.qb, rb.eb, m, n, k)
    elif arm == ARM_PIECES:
        identical_gemm_int15_pieces_into(ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k)
    elif arm == ARM_MMA:
        comptime if HAS_UNIT:
            identical_gemm_int15_mma_into(ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k)
    elif arm == ARM_QUANTIZE_A:
        _quantize_a(ctx, rb.qsa, rb.esa, rb.a, kind, m, k)
    elif arm == ARM_QUANTIZE_A_PAR:
        _quantize_par(ctx, rb.qsa, rb.esa, rb.a, rb.quant, _a_transposed(kind), m, k)
    elif arm == ARM_PLANES_A_PAR:
        _planes_par(ctx, rb.sah, rb.sal, rb.esa, rb.a, rb.quant, _a_transposed(kind), m, k)
    elif arm == ARM_SPLIT_A:
        split_int15_device(ctx, rb.sah, rb.sal, rb.qa, m * k)
    elif arm == ARM_QUANTIZE_B:
        _quantize_b(ctx, rb.qsb, rb.esb, rb.b, kind, n, k)
    elif arm == ARM_QUANTIZE_B_PAR:
        _quantize_par(ctx, rb.qsb, rb.esb, rb.b, rb.quant, _b_transposed(kind), n, k)
    elif arm == ARM_PLANES_B_PAR:
        _planes_par(ctx, rb.sbh, rb.sbl, rb.esb, rb.b, rb.quant, _b_transposed(kind), n, k)
    elif arm == ARM_SPLIT_B:
        split_int15_device(ctx, rb.sbh, rb.sbl, rb.qb, n * k)
    elif arm == ARM_INF_PLANES:
        _planes_par(ctx, rb.ah, rb.al, rb.ea, rb.a, rb.quant, _a_transposed(kind), m, k)
        identical_gemm_int15_planes_into(ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k)
    elif arm == ARM_INF_CODES:
        _quantize_par(ctx, rb.qa, rb.ea, rb.a, rb.quant, _a_transposed(kind), m, k)
        identical_gemm_int15_into(ctx, rb.c, rb.qa, rb.ea, rb.qb, rb.eb, rb.work, m, n, k)
    elif arm == ARM_INF_ROWQUANT:
        _quantize_a(ctx, rb.qa, rb.ea, rb.a, kind, m, k)
        split_int15_device(ctx, rb.ah, rb.al, rb.qa, m * k)
        identical_gemm_int15_planes_into(ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k)
    elif arm == ARM_TRAIN_PLANES:
        _planes_par(ctx, rb.ah, rb.al, rb.ea, rb.a, rb.quant, _a_transposed(kind), m, k)
        _planes_par(ctx, rb.bh, rb.bl, rb.eb, rb.b, rb.quant, _b_transposed(kind), n, k)
        identical_gemm_int15_planes_into(ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k)
    elif arm == ARM_TRAIN_CODES:
        _quantize_par(ctx, rb.qa, rb.ea, rb.a, rb.quant, _a_transposed(kind), m, k)
        _quantize_par(ctx, rb.qb, rb.eb, rb.b, rb.quant, _b_transposed(kind), n, k)
        identical_gemm_int15_into(ctx, rb.c, rb.qa, rb.ea, rb.qb, rb.eb, rb.work, m, n, k)
    elif arm == ARM_TRAIN_ROWQUANT:
        _quantize_a(ctx, rb.qa, rb.ea, rb.a, kind, m, k)
        _quantize_b(ctx, rb.qb, rb.eb, rb.b, kind, n, k)
        split_int15_device(ctx, rb.ah, rb.al, rb.qa, m * k)
        split_int15_device(ctx, rb.bh, rb.bl, rb.qb, n * k)
        identical_gemm_int15_planes_into(ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k)
    elif _arm_is_tuned(arm):
        comptime if HAS_UNIT:
            if arm != ARM_TUNED:
                _planes_par(ctx, rb.ah, rb.al, rb.ea, rb.a, rb.quant, _a_transposed(kind), m, k)
            if arm == ARM_TRAIN_TUNED:
                _planes_par(ctx, rb.bh, rb.bl, rb.eb, rb.b, rb.quant, _b_transposed(kind), n, k)
            identical_gemm_int15_tuned_into(
                ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, rb.sums, m, n, k
            )
    else:
        comptime if IS_APPLE:
            var form = INT15_APPLE_FORM_TWO
            if arm == ARM_APPLE4 or arm == ARM_INF_APPLE4 or arm == ARM_TRAIN_APPLE4:
                form = INT15_APPLE_FORM_FOUR
            if arm != ARM_APPLE2 and arm != ARM_APPLE4:
                _planes_par(ctx, rb.ah, rb.al, rb.ea, rb.a, rb.quant, _a_transposed(kind), m, k)
            if arm == ARM_TRAIN_APPLE2 or arm == ARM_TRAIN_APPLE4:
                _planes_par(ctx, rb.bh, rb.bl, rb.eb, rb.b, rb.quant, _b_transposed(kind), n, k)
            identical_gemm_int15_apple_into(
                ctx, rb.c, rb.ah, rb.al, rb.ea, rb.bh, rb.bl, rb.eb, m, n, k, form
            )


def _arm_digest(
    ctx: DeviceContext, mut rb: RowBuffers, arm: Int, m: Int, n: Int, k: Int, tag: String
) raises -> UInt64:
    """The digest of what the arm just wrote."""
    if _arm_is_product(arm):
        _whole(len(rb.c), m * n, tag)
        return _dev_digest(ctx, rb.c, m * n, tag)
    if arm == ARM_QUANTIZE_A or arm == ARM_QUANTIZE_A_PAR:
        return _digest_codes(ctx, rb.qsa, rb.esa, m, k)
    if arm == ARM_QUANTIZE_B or arm == ARM_QUANTIZE_B_PAR:
        return _digest_codes(ctx, rb.qsb, rb.esb, n, k)
    if arm == ARM_SPLIT_A or arm == ARM_PLANES_A_PAR:
        return _digest_planes(ctx, rb.sah, rb.sal, m * k)
    return _digest_planes(ctx, rb.sbh, rb.sbl, n * k)


def _arm_note(arm: Int, kind: Int, m: Int, n: Int, k: Int) -> String:
    """What a reader needs beside the arm's number. No spaces."""
    if arm == ARM_FP32:
        var op = String("OP_NT")
        if kind == ROW_BWD_DX:
            op = String("OP_NN")
        elif kind == ROW_BWD_DW:
            op = String("OP_TN")
        return op + ",plan=" + String(choose_gemm_plan(m, n, k))
    if arm == ARM_FLAT:
        comptime if HAS_UNIT:
            return String("not-dispatched(the-column-has-the-unit)")
        else:
            return String("dispatched-for-codes")
    if arm == ARM_PIECES:
        comptime if HAS_UNIT:
            return String("not-dispatched(the-column-has-the-unit)")
        else:
            return String("dispatched-for-planes")
    if arm == ARM_MMA:
        return String("dispatched,four-unit-products")
    if arm == ARM_QUANTIZE_A or arm == ARM_QUANTIZE_A_PAR or arm == ARM_PLANES_A_PAR:
        if kind == ROW_BWD_DW:
            return String("per-call,transposing")
        return String("per-call")
    if arm == ARM_QUANTIZE_B or arm == ARM_QUANTIZE_B_PAR or arm == ARM_PLANES_B_PAR:
        if kind == ROW_FORWARD:
            return String("once-per-weight-at-inference,per-step-at-training")
        return String("per-step,transposing")
    if arm == ARM_SPLIT_A:
        return String("per-call")
    if arm == ARM_SPLIT_B:
        return String("once-per-weight-at-inference,per-step-at-training")
    if arm == ARM_INF_PLANES:
        return String("planes.a.parallel+product")
    if arm == ARM_INF_CODES:
        comptime if HAS_UNIT:
            return String("quantize.a.parallel+split.a+split.b+product")
        else:
            return String("quantize.a.parallel+flat")
    if arm == ARM_INF_ROWQUANT:
        return String("quantize.a+split.a+product")
    if arm == ARM_TRAIN_PLANES:
        return String("planes.a.parallel+planes.b.parallel+product")
    if arm == ARM_TRAIN_CODES:
        comptime if HAS_UNIT:
            return String("quantize.a.parallel+quantize.b.parallel+split.a+split.b+product")
        else:
            return String("quantize.a.parallel+quantize.b.parallel+flat")
    if arm == ARM_TRAIN_ROWQUANT:
        return String("quantize.a+quantize.b+split.a+split.b+product")
    if arm == ARM_APPLE2:
        return String("not-dispatched,float-unit,two-products,carry-every-8-steps")
    if arm == ARM_APPLE4:
        return String("not-dispatched,float-unit,four-products,carry-every-512-steps")
    if arm == ARM_INF_APPLE2 or arm == ARM_INF_APPLE4:
        return String("planes.a.parallel+float-unit-product")
    if arm == ARM_TRAIN_APPLE2 or arm == ARM_TRAIN_APPLE4:
        return String("planes.a.parallel+planes.b.parallel+float-unit-product")
    if arm == ARM_TUNED:
        return String("not-dispatched,four-products-one-staging+epilogue")
    if arm == ARM_INF_TUNED:
        return String("planes.a.parallel+four-products-one-staging+epilogue")
    return String("planes.a.parallel+planes.b.parallel+four-products-one-staging+epilogue")


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
) raises -> String:
    """Every arm at one row. Returns the plan disagreements found (empty
    when there are none); `main` raises on them after every row has
    printed."""
    var cap_word = String("CAPPED") if capped else String("FULL")
    print()
    print(
        "== " + name + "  m=" + String(m) + " n=" + String(n) + " k=" + String(k)
        + "  " + cap_word + "  fp32 plan: " + gemm_plan_name(choose_gemm_plan(m, n, k))
    )
    var rb = RowBuffers(ctx, m, n, k)
    _whole(len(rb.a), m * k, String("the left operand"))
    _whole(len(rb.b), n * k, String("the right operand"))
    # lane/lowbit-units' salts at the forward rows, so the two lanes'
    # `fp32.v1` arms read the same operands there.
    _dev_fill(ctx, rb.a, m * k, 11 + salt)
    _dev_fill(ctx, rb.b, n * k, 22 + salt)
    var admitted = k <= INT15_MAX_K
    if admitted:
        # The fifteen-bit operands, by the profile's own conversions on the
        # device, before anything is timed.
        _quantize_a(ctx, rb.qa, rb.ea, rb.a, kind, m, k)
        _quantize_b(ctx, rb.qb, rb.eb, rb.b, kind, n, k)
        split_int15_device(ctx, rb.ah, rb.al, rb.qa, m * k)
        split_int15_device(ctx, rb.bh, rb.bl, rb.qb, n * k)
    ctx.synchronize()

    var dig = List[UInt64]()
    var why = List[String]()
    var samples = List[List[Int]]()
    for arm in range(ARM_COUNT):
        dig.append(UInt64(0))
        why.append(_why_not(arm, kind, k))
        samples.append(List[Int]())

    # Untimed warm-up of every arm, its output poisoned first where the
    # output is float32 and read back after, so an arm that launches
    # without writing cannot turn in a time.
    for at_ in range(ARM_COUNT):
        var arm = _arm_at(at_)
        if why[arm].byte_length() > 0:
            continue
        var tag = String("int15.") + name + "." + _arm_name(arm)
        if _arm_is_product(arm):
            _whole(len(rb.c), m * n, tag)
            _dev_poison(ctx, rb.c, m * n)
        _enqueue_arm(ctx, rb, arm, kind, m, n, k)
        ctx.synchronize()
        dig[arm] = _arm_digest(ctx, rb, arm, m, n, k, tag)

    comptime block_count = 2
    for block in range(block_count):
        var first = 0 if block == 0 else ARM_FIRST_BLOCK
        var last = ARM_FIRST_BLOCK if block == 0 else ARM_COUNT
        # One untimed call of the block's first arm that runs, so the first
        # timed call of the block does not follow whatever ran before it
        # (the read-back of the warm-up, or the first block).
        for at_ in range(first, last):
            var arm = _arm_at(at_)
            if why[arm].byte_length() > 0 or identity_only:
                continue
            _enqueue_arm(ctx, rb, arm, kind, m, n, k)
            ctx.synchronize()
            break
        for _ in range(0 if identity_only else repeats):
            for at_ in range(first, last):
                var arm = _arm_at(at_)
                if why[arm].byte_length() > 0:
                    continue
                var t0 = perf_counter_ns()
                _enqueue_arm(ctx, rb, arm, kind, m, n, k)
                ctx.synchronize()
                samples[arm].append(Int(perf_counter_ns() - t0))

    var macs = Float64(m) * Float64(n) * Float64(k)
    for arm in range(ARM_COUNT):
        var arm_name = _arm_name(arm)
        if why[arm].byte_length() > 0:
            print("INT15-NOT-RUN", column_name(TARGET_COLUMN), name, arm_name, why[arm])
            continue
        var unit = String("GMAC/s")
        var count = macs
        if not _arm_is_product(arm):
            unit = String("Gelem/s")
            count = Float64(n) * Float64(k)
            if _arm_is_left(arm):
                count = Float64(m) * Float64(k)
        if arm == ARM_FP32:
            print("DIGEST fp32-" + name + " " + arm_name + " " + hex(dig[arm]))
        elif _arm_is_product(arm):
            print("DIGEST price-" + name + " " + arm_name + " " + hex(dig[arm]))
        else:
            print("DIGEST conv-" + name + " " + arm_name + " " + hex(dig[arm]))
        if identity_only:
            print(
                "INT15", column_name(TARGET_COLUMN), name, m, n, k, cap_word, arm_name,
                "not-timed", "not-timed", "not-timed", unit,
                hex(dig[arm]), _arm_note(arm, kind, m, n, k),
            )
            continue
        var med = _median_ms(samples[arm])
        var best = _min_ms(samples[arm])
        _report_device(String("int15.") + name + "." + arm_name + ".device", med)
        print(
            "INT15", column_name(TARGET_COLUMN), name, m, n, k, cap_word, arm_name,
            _fixed(med, 4), _fixed(best, 4), _fixed(_rate(count, med), 4), unit,
            hex(dig[arm]), _arm_note(arm, kind, m, n, k),
        )

    # Every fifteen-bit product and every complete operation of a row ends
    # in the profile's product of the same operands: one digest.
    var bad = String("")
    for arm in range(ARM_COUNT):
        if not _arm_is_int15_product(arm) or arm == ARM_FLAT:
            continue
        if why[arm].byte_length() > 0 or why[ARM_FLAT].byte_length() > 0:
            continue
        if dig[arm] != dig[ARM_FLAT]:
            bad += (
                String("PLANS DISAGREE at ") + name + ": " + _arm_name(ARM_FLAT) + " "
                + hex(dig[ARM_FLAT]) + " vs " + _arm_name(arm) + " " + hex(dig[arm]) + "\n"
            )
    # The two schedules of the quantizer write one store of codes, and the
    # split of those codes is the planes the parallel schedule writes.
    if admitted:
        if dig[ARM_QUANTIZE_A] != dig[ARM_QUANTIZE_A_PAR] or dig[ARM_QUANTIZE_B] != dig[ARM_QUANTIZE_B_PAR]:
            bad += String("QUANTIZER SCHEDULES DISAGREE at ") + name + " (codes)\n"
        if dig[ARM_SPLIT_A] != dig[ARM_PLANES_A_PAR] or dig[ARM_SPLIT_B] != dig[ARM_PLANES_B_PAR]:
            bad += String("QUANTIZER SCHEDULES DISAGREE at ") + name + " (planes)\n"
    if bad.byte_length() > 0:
        print(bad)
    _ = rb^
    return bad


def _wanted(only: String, name: String) -> Bool:
    if only == "":
        return True
    for part in only.split(","):
        if part.byte_length() > 0 and name.find(String(part)) >= 0:
            return True
    return False


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
    return _time_row(ctx, salt, name, kind, dm, dn, k, dm != m or dn != n, repeats, identity_only)


def main() raises:
    comptime if not has_accelerator():
        raise Error("bench/gemm_int15_price_main: no accelerator; every arm here is a device arm")
    else:
        var repeats = DEFAULT_REPEATS
        var rs = String(getenv("MOJOLEARN_INT15_PRICE_REPEATS"))
        if rs != "":
            repeats = Int(atol(rs))
        var budget = DEFAULT_MAC_BUDGET
        var bs = String(getenv("MOJOLEARN_INT15_PRICE_MAC_BUDGET"))
        if bs != "":
            budget = Int(atol(bs))
        var only = String(getenv("MOJOLEARN_INT15_PRICE_ONLY"))
        var identity_only = String(getenv("MOJOLEARN_INT15_PRICE_IDENTITY_ONLY")) == "1"

        print("== bench/gemm_int15_price_main.mojo [" + _mode() + "] ==")
        print("column", column_name(TARGET_COLUMN))
        print("profiles fp32.v1, int15i64.v1")
        print("int15 dispatch:", int15_plan_dispatch_name())
        print("sabotage:", int15_sabotage_name())
        print("repeats", repeats, " mac budget", budget, " only", only)
        if identity_only:
            print(
                "IDENTITY ONLY: every arm runs once for its digest and NOTHING",
                "IS TIMED in this run.",
            )
        else:
            print(
                "EVERY NUMBER BELOW IS ONE BOX'S TIME ON ONE RUN. One call and one",
                "synchronize per sample; the median of the timed calls is reported",
                "and the minimum beside it.",
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
                bad += _run_row(ctx, i, name, ROW_FORWARD, m, n, k, budget, only, repeats, identity_only, rows)
                if name.find(".t512") < 0:
                    continue
                # The two backward rows of a training row, derived from it.
                bad += _run_row(
                    ctx, 100 + i, name + ".bwd_dx", ROW_BWD_DX, m, k, n, budget, only, repeats, identity_only, rows
                )
                bad += _run_row(
                    ctx, 200 + i, name + ".bwd_dw", ROW_BWD_DW, n, k, m, budget, only, repeats, identity_only, rows
                )

        print()
        print("== done [" + _mode() + "]: " + String(rows) + " rows ==")
        if rows == 0:
            raise Error("bench/gemm_int15_price_main: no row matched MOJOLEARN_INT15_PRICE_ONLY=" + only)
        if bad.byte_length() > 0:
            if _mode() == "IDENTICAL":
                raise Error(
                    "plans of one profile disagree (above). Under IDENTICAL that"
                    " is a contract violation, not a measurement."
                )
            print("NOTE (not IDENTICAL, not a failure): plans disagree, above.")
