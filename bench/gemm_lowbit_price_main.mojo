# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What the LOW-BIT GEMM plans cost beside `fp32.v1`: one timing harness.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_lowbit_price_main.mojo

Lane lane/lowbit-units, 2026-09-29 (`docs/lanes/LOWBIT_UNITS_PLAN.md`, order
of work 2). Contract `gemm/IDENTICAL_LOWBIT_CONTRACT.md` section 3 says of
every low-bit plan that none has been timed against the fp32 plans. This
file is that timing. It certifies nothing: the gates are
`gemm/checks/gemm_lowbit_check.mojo` and
`gemm/checks/gemm_int8_apple_chunk_check.mojo`, and a time printed here is
one box's time on one run.

THE SHAPES ARE `bench/gemm_shapes.mojo`'s AND NONE ARE INVENTED HERE. The
rows read are the OP_NT transformer rows (the int8 profile is OP_NT only,
contract section 0), and their token count `m` is what sorts them:

    decode     t1 and t8     one token, or a small batch of them, per call
    training   t512          512 tokens per call, the shape of a training
                             step's forward product and of a prefill chunk

`k` IS NEVER CAPPED. `m` and `n` are reduced by `bench/gemm_price_main.mojo`'s
own `_capped` rule only where a row's multiply-accumulate count is above
`MOJOLEARN_LOWBIT_PRICE_MAC_BUDGET` (default 2^35, which caps exactly one
row, the head at t512; 0 runs every row whole). A capped row says CAPPED on
its line and prints the extents it ran at. THE SAME BUDGET MUST RUN ON EVERY
BOX whose hashes are to be compared: a hash is a hash of the extents run.

THE ARMS. Every arm of one shape lives in this one binary and the timed loop
ALTERNATES them call by call (`bench/gemm_price_main.mojo`'s discipline: a
block of one arm and then a block of another measures the drift).

    fp32.v1                  `identical_gemm_into`, the shipped plan
    bf16f32.v1.fused         `identical_gemm_bf16w_fused_into`
    bf16f32.v1.widen         `identical_gemm_bf16w_widen_into` (the widening
                             of the right operand is INSIDE this time)
    int8i32.v1.flat          `identical_gemm_int8_flat_into`
    int8i32.v1.mma           `identical_gemm_int8_mma_into`, on a column that
                             has an integer matrix unit (NVIDIA, AMD)
    int8i32.v1.applechunk    `identical_gemm_int8_apple_chunk_into`, the
                             exact-chunk PROBE on Apple's float matrix unit

lane/lowbit-mma-speed (2026-09-29) adds the PARALLEL QUANTIZER
(`gemm/checks/quantize_int8_par.mojo`) as arms beside the reference
quantizer's, each with the digest of its codes and exponents, which must
equal the reference's:

    convert.int8.quantize.a.par     the activations, a block of threads per row
    convert.int8.pack.b.par         the weights, the same
    inference.int8i32.v1.parq       parallel quantize A, the dispatched product
    training.int8i32.v1.parq        parallel quantize A and B, the product
    inference.int8i32.v1.applechunk.parq   the same two with the Apple probe
    training.int8i32.v1.applechunk.parq    as the product (Apple only)

and the TUNED UNIT PLANS (`gemm/checks/gemm_int8_mma_tuned.mojo`), each a
schedule of the same product, so each one's digest must equal the flat
plan's:

    int8i32.v1.mma.staged.*         one arm per staged plan; the name is the
                                    plan's (warp tile, block tile, k steps
                                    per window, bytes per staging load)
    int8i32.v1.mma.direct.*         NVIDIA: the reference's schedule with one
                                    thing changed
    probe.int8.mma.*                NVIDIA: a part of the reference's kernel
                                    timed alone. ITS PRODUCT IS WRONG ON
                                    PURPOSE; its digest is printed and
                                    compared with nothing
    inference.int8i32.v1.tuned      parallel quantize A, the tuned product
    training.int8i32.v1.tuned       parallel quantize A and B, the same
    inference.4x.int8i32.v1.tuned   parallel quantize A, then FOUR tuned
    training.4x.int8i32.v1.tuned    products (the same one, four times): the
                                    count the 15-bit profile needs, measured
                                    as one operation with one wait. Its
                                    split and its recombination are
                                    lane/lowbit-int15's and are not in it.

and FOUR PRODUCTS, ONE STAGING (`identical_gemm_int8_pieces_tuned_kernel`):
two int8 planes of each operand in, three Int32 sums per cell out (HH,
HL + LH, LL). The planes here are made from the row's int8 codes by integer
arithmetic (`_pieces_fixture_kernel`), the same on every box:

    pieces.int8.flat                the reference device plan, one thread
                                    per cell: the digest the plans must equal
    pieces.int8.mma.staged.*        one arm per staged plan (pipe2.*: two
    pieces.int8.mma.pipe2.*         pages, the next window staged by
                                    cp.async). The rate is over `4 m n k`
                                    multiply-accumulates
    inference.pieces.int8.tuned     parallel quantize A (the int8
    training.pieces.int8.tuned      quantizer, standing in for the
                                    fifteen-bit one, which is
                                    lane/lowbit-int15's), the four products
                                    in one launch, and a second launch
                                    (`_pieces_recombine_probe_kernel`) that
                                    reads the sums back and calls
                                    lane/lowbit-int15's `int15_store_cell`
                                    (the recombination, the pinned seam, the
                                    scale) per cell. The planes are a
                                    fixture, not the profile's split, so the
                                    digest is compared between these arms
                                    and with nothing else.
    inference.pieces.int8.fused     the inference arm's operation in ONE
                                    launch: lane/lowbit-int15's epilogue
                                    (`int15_store_cell`, the same function
                                    the second launch calls) fused into the
                                    four-product kernel's last step. Its
                                    digest must equal the inference arm's.

THE CONVERSIONS ARE THEIR OWN ROWS, because a low-bit product's operands do
not arrive low-bit for free:

    convert.int8.quantize.a    float32 activations to codes, `m x k`. Paid on
                               EVERY call: activations are new each call.
    convert.int8.pack.b        float32 weights to codes, `n x k`. Paid once,
                               when the weights are packed.
    convert.bf16.pack.b        float32 weights to bf16, `n x k`. Paid once.
    convert.bf16.widen.b       bf16 weights back to float32, `n x k`: the
                               WIDEN plan's first step, and what a block
                               that materializes its weights pays.
    convert.int8.dequantize.b  int8 weights back to float32, `n x k`: what a
                               block that materializes its weights pays.

The dequantization of the PRODUCT (`dequant_int8_pinned`, one per output
cell) is the epilogue of every int8 kernel and is inside each int8 arm's
time; it has no launch of its own to time.

THE COMPLETE OPERATION, TWICE (the brief's review point 3). A product's time
alone is not what a caller pays. Each arm below enqueues EVERY step of one
call on the one in-order context and waits once, so its time is measured,
not a sum of medians, and each is read against `fp32.v1` at the same shape
on the same box:

    INFERENCE: the weights were packed once and are reused. One call pays
    the conversion of its activations, the product on the plan the
    profile's dispatcher picks, and the epilogue.
        inference.bf16f32.v1              the product (activations stay
                                          float32 in this profile)
        inference.int8i32.v1              quantize A, the product
        inference.int8i32.v1.applechunk   quantize A, the Apple probe

    TRAINING: the weights change every step, so their conversion is paid
    every step as well.
        training.bf16f32.v1               pack B to bf16, the product
        training.int8i32.v1               quantize A, quantize B, the product
        training.int8i32.v1.applechunk    quantize A, quantize B, the probe

    ONLY THE FORWARD PRODUCT. A training step's backward products are OP_TN
    and OP_NN; `int8i32.v1` is OP_NT only and no low-bit backward kernel
    exists, so nothing here times a gradient.

These arms convert INTO the operands the product arms read. The conversion
is a function of the float32 operand, which nothing here writes, so the
operands hold the same bits after as before.

ON APPLE EVERY TIME IS THE WHOLE KERNEL'S. `fp32.v1` there is
`PLAN_APPLE_MMA` where its dispatcher picks it: the staging, the admission
test of every window and any window that fell back to the exact step are
inside the time, as the probe's staging, int8 to float conversion and chunk
carries are inside the probe's. No arm here forces the fallback path, so
its cost alone is not measured.

THE HASH. After its untimed warm-up every arm's output is read back, a
surviving poison is refused (`_dev_digest`), and the FNV-1a digest of the
output bits is printed on the arm's `LOWBIT` line. Identity rides along:
`tools/lowbit_units/table.py` compares the digests of one arm and shape
across boxes. Inside one run the plans that must agree are compared here
(fused against widen; flat against mma and against the Apple probe) and a
disagreement under IDENTICAL raises AFTER every line is printed.

A HASH THAT CANNOT DIFFER IS NOT A CHECK. A build carrying
`-D MOJOLEARN_LOWBIT_PRICE_SABOTAGE=1` flips ONE BIT (the lowest) of ONE
CELL (the last) of what every arm wrote, after the arm ran and before the
digest is taken: on the device for a float32 output, in the words read back
for a bf16 or an int8 output. `tools/lowbit_units/table.py --expect-disagree`
requires every digest of that run to differ from a clean run's at the same
box, arm and shape. It is an arm of the DIGEST AND THE COMPARISON, which are
what this file adds; the kernels' own value arms
(`-D MOJOLEARN_LOWBIT_SABOTAGE=1`) belong to the gates and are run there.

THE LINES.
    PRICE <mode> lowbit.<shape>.<arm>.device <median ms>
        the shared median table's format (`_report_device`).
    LOWBIT <column> <shape> <m> <n> <k> <FULL|CAPPED> <arm> <median ms>
           <min ms> <rate> <unit> <digest> <note>
        one line per arm and shape, for `tools/lowbit_units/table.py`.
        The rate is G MAC/s (multiply-accumulates, `m n k`) for a product and
        G elem/s for a conversion.

IDENTITY ONLY. `MOJOLEARN_LOWBIT_PRICE_IDENTITY_ONLY=1` runs
every arm ONCE, for its digest, and times nothing: no timed loop, no `PRICE`
line, and the three number fields of every `LOWBIT` line read `not-timed`.
The in-run plan comparisons still run. It is also how a box is WARMED before
it is timed: the same binary (the mode is read from the environment, not
compiled in), every kernel launched once.

ENVIRONMENT.
    MOJOLEARN_LOWBIT_PRICE_IDENTITY_ONLY  1: digests only, nothing timed
    MOJOLEARN_LOWBIT_PRICE_REPEATS     timed calls per arm (default 5)
    MOJOLEARN_LOWBIT_PRICE_MAC_BUDGET  see above (default 2^35; 0 = no cap)
    MOJOLEARN_LOWBIT_PRICE_ONLY        comma separated substrings of shape
                                       names; unset runs every OP_NT
                                       transformer row
    MOJOLEARN_LOWBIT_PRICE_ARMS        comma separated arm names, whole
                                       names and not substrings (a name
                                       ending in * is a prefix); unset runs
                                       every arm the column has. An arm left
                                       out prints a LOWBIT-NOT-RUN line. The
                                       in-run comparisons need both of their
                                       arms and say nothing when one is out.

WHAT MAY NOT BE CONCLUDED. One call and one synchronize per sample, so a
decode row's time is mostly launch and wait, not arithmetic. No arm here is
an inference engine or a training step. The flat kernels are one thread per
cell with no tiling, so `flat` against `fp32.v1` compares two levels of
kernel engineering as well as two arithmetics.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_idx, thread_idx
from std.memory import bitcast
from std.os import getenv
from std.sys import has_accelerator, is_defined
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
from gemm.checks.gemm_int15_epilogue import int15_store_cell
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
from gemm.checks.gemm_int8_apple_chunk import (
    identical_gemm_int8_apple_chunk_into,
    int8_apple_chunk_geometry,
    int8_apple_chunk_sabotage_name,
)
from gemm.checks.gemm_int8_mma import identical_gemm_int8_mma_into
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_DECODE_AVAILABLE,
    INT8_DECODE_MAX_M,
    INT8_DECODE_PLAN_COUNT,
    INT8_DIRECT_AVAILABLE,
    INT8_DIRECT_COUNT,
    identical_gemm_int8_mma_decode_quant_with_plan,
    identical_gemm_int8_mma_decode_with_plan,
    int8_decode_dispatch,
    int8_decode_plan_name,
    INT8_PIECES_MAX_K,
    INT8_PIECES_PLAN_COUNT,
    INT8_TUNED_PLAN_COUNT,
    identical_gemm_int8_pieces_flat_into,
    identical_gemm_int8_pieces_tuned_fused_into,
    identical_gemm_int8_pieces_tuned_into,
    identical_gemm_int8_pieces_tuned_with_plan,
    int8_pieces_dispatch,
    int8_pieces_plan_admits,
    int8_pieces_plan_name,
    identical_gemm_int8_mma_direct_into,
    identical_gemm_int8_mma_tuned_into,
    identical_gemm_int8_mma_tuned_with_plan,
    int8_direct_is_probe,
    int8_direct_name,
    int8_tuned_dispatch,
    int8_tuned_plan_available,
    int8_tuned_plan_name,
    int8_tuned_sabotage_name,
)
from gemm.checks.quantize_int8_par import (
    quantize_par_block,
    quantize_par_sabotage_name,
    quantize_rows_int8_par_device,
)
from gemm.checks.gemm_lowbit import (
    BF16W_FUSED_MAX_CELLS,
    LowbitWorkspace,
    bf16_narrow,
    bf16_widen,
    dequantize_rows_int8_device,
    identical_gemm_bf16w_fused_into,
    identical_gemm_bf16w_into,
    identical_gemm_bf16w_widen_into,
    identical_gemm_int8_flat_into,
    identical_gemm_int8_into,
    int8_plan_dispatch_name,
    lowbit_sabotage_name,
    quantize_rows_int8_device,
)
from gemm.host.gemm_oracle import OP_NT

#: Timed calls per arm and shape unless the environment says otherwise.
comptime DEFAULT_REPEATS = 5

#: The default multiply-accumulate budget per row, 2^35. Of the twelve rows
#: only `llama8b.lm_head.t512` (2.7e11) is above it.
comptime DEFAULT_MAC_BUDGET = 34_359_738_368

#: The arms, in the order the timed loop alternates them.
comptime ARM_FP32 = 0
comptime ARM_BF16_FUSED = 1
comptime ARM_BF16_WIDEN = 2
comptime ARM_INT8_FLAT = 3
comptime ARM_INT8_MMA = 4
comptime ARM_INT8_APPLE_CHUNK = 5
comptime ARM_QUANTIZE_A = 6
comptime ARM_INT8_PACK_B = 7
comptime ARM_BF16_PACK_B = 8
comptime ARM_BF16_WIDEN_B = 9
comptime ARM_INT8_DEQUANT_B = 10
comptime ARM_INF_BF16 = 11
comptime ARM_INF_INT8 = 12
comptime ARM_INF_INT8_APPLE_CHUNK = 13
comptime ARM_TRAIN_BF16 = 14
comptime ARM_TRAIN_INT8 = 15
comptime ARM_TRAIN_INT8_APPLE_CHUNK = 16
#: lane/lowbit-mma-speed: the parallel quantizer, alone and in the complete
#: operations.
comptime ARM_QUANTIZE_A_PAR = 17
comptime ARM_INT8_PACK_B_PAR = 18
comptime ARM_INF_INT8_PARQ = 19
comptime ARM_TRAIN_INT8_PARQ = 20
comptime ARM_INF_INT8_APPLE_CHUNK_PARQ = 21
comptime ARM_TRAIN_INT8_APPLE_CHUNK_PARQ = 22
#: lane/lowbit-mma-speed: one arm per staged plan, one per instantiation of
#: the direct kernel, and the complete operations on the tuned product.
comptime ARM_TUNED_BASE = 23
comptime ARM_DIRECT_BASE = ARM_TUNED_BASE + INT8_TUNED_PLAN_COUNT
comptime ARM_INF_INT8_TUNED = ARM_DIRECT_BASE + INT8_DIRECT_COUNT
comptime ARM_TRAIN_INT8_TUNED = ARM_INF_INT8_TUNED + 1
comptime ARM_INF_4X_TUNED = ARM_INF_INT8_TUNED + 2
comptime ARM_TRAIN_4X_TUNED = ARM_INF_INT8_TUNED + 3
#: FOUR PRODUCTS, ONE STAGING.
comptime ARM_PIECES_FLAT = ARM_INF_INT8_TUNED + 4
comptime ARM_PIECES_BASE = ARM_PIECES_FLAT + 1
comptime ARM_INF_PIECES = ARM_PIECES_BASE + INT8_PIECES_PLAN_COUNT
comptime ARM_TRAIN_PIECES = ARM_INF_PIECES + 1
#: The same operation as ARM_INF_PIECES in ONE launch: the four products and
#: lane/lowbit-int15's epilogue fused into their last step.
comptime ARM_INF_PIECES_FUSED = ARM_TRAIN_PIECES + 1
#: THE DECODE KERNEL (m <= 16, NVIDIA): one arm per plan on the codes, and
#: one per plan with the quantizer in the product's launch (float32 in).
comptime ARM_DECODE_BASE = ARM_INF_PIECES_FUSED + 1
comptime ARM_DECODE_QUANT_BASE = ARM_DECODE_BASE + INT8_DECODE_PLAN_COUNT
comptime ARM_COUNT = ARM_DECODE_QUANT_BASE + INT8_DECODE_PLAN_COUNT

#: What no sum of the four-product kernel is (its largest is 2147450880).
comptime SUM_POISON = Int32(2147483647)

#: Products per GEMM of the 15-bit profile (the brief: HH, HL, LH, LL).
comptime PIECE_PRODUCTS = 4

#: The arm of the digest and the comparison: one bit of one cell of every
#: arm's output is flipped before the digest. Off in every build that does
#: not name it.
comptime PRICE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_PRICE_SABOTAGE"]()

#: Whether this build has each column-bound arm.
comptime HAS_INT8_MMA = lib_int8_matrix_unit_for[TARGET_COLUMN]()
comptime HAS_APPLE_CHUNK = TARGET_COLUMN == COLUMN_APPLE


def _arm_name(arm: Int) -> String:
    if arm == ARM_FP32:
        return String("fp32.v1")
    if arm == ARM_BF16_FUSED:
        return String("bf16f32.v1.fused")
    if arm == ARM_BF16_WIDEN:
        return String("bf16f32.v1.widen")
    if arm == ARM_INT8_FLAT:
        return String("int8i32.v1.flat")
    if arm == ARM_INT8_MMA:
        return String("int8i32.v1.mma")
    if arm == ARM_INT8_APPLE_CHUNK:
        return String("int8i32.v1.applechunk")
    if arm == ARM_QUANTIZE_A:
        return String("convert.int8.quantize.a")
    if arm == ARM_INT8_PACK_B:
        return String("convert.int8.pack.b")
    if arm == ARM_BF16_PACK_B:
        return String("convert.bf16.pack.b")
    if arm == ARM_BF16_WIDEN_B:
        return String("convert.bf16.widen.b")
    if arm == ARM_INT8_DEQUANT_B:
        return String("convert.int8.dequantize.b")
    if arm == ARM_INF_BF16:
        return String("inference.bf16f32.v1")
    if arm == ARM_INF_INT8:
        return String("inference.int8i32.v1")
    if arm == ARM_INF_INT8_APPLE_CHUNK:
        return String("inference.int8i32.v1.applechunk")
    if arm == ARM_TRAIN_BF16:
        return String("training.bf16f32.v1")
    if arm == ARM_TRAIN_INT8:
        return String("training.int8i32.v1")
    if arm == ARM_TRAIN_INT8_APPLE_CHUNK:
        return String("training.int8i32.v1.applechunk")
    if arm == ARM_QUANTIZE_A_PAR:
        return String("convert.int8.quantize.a.par")
    if arm == ARM_INT8_PACK_B_PAR:
        return String("convert.int8.pack.b.par")
    if arm == ARM_INF_INT8_PARQ:
        return String("inference.int8i32.v1.parq")
    if arm == ARM_TRAIN_INT8_PARQ:
        return String("training.int8i32.v1.parq")
    if arm == ARM_INF_INT8_APPLE_CHUNK_PARQ:
        return String("inference.int8i32.v1.applechunk.parq")
    if arm == ARM_TRAIN_INT8_APPLE_CHUNK_PARQ:
        return String("training.int8i32.v1.applechunk.parq")
    if arm < ARM_DIRECT_BASE:
        return String("int8i32.v1.mma.") + int8_tuned_plan_name(arm - ARM_TUNED_BASE)
    if arm < ARM_INF_INT8_TUNED:
        if int8_direct_is_probe(arm - ARM_DIRECT_BASE):
            return String("probe.int8.mma.") + int8_direct_name(arm - ARM_DIRECT_BASE)
        return String("int8i32.v1.mma.") + int8_direct_name(arm - ARM_DIRECT_BASE)
    if arm == ARM_INF_INT8_TUNED:
        return String("inference.int8i32.v1.tuned")
    if arm == ARM_TRAIN_INT8_TUNED:
        return String("training.int8i32.v1.tuned")
    if arm == ARM_INF_4X_TUNED:
        return String("inference.4x.int8i32.v1.tuned")
    if arm == ARM_TRAIN_4X_TUNED:
        return String("training.4x.int8i32.v1.tuned")
    if arm == ARM_PIECES_FLAT:
        return String("pieces.int8.flat")
    if arm < ARM_INF_PIECES:
        return String("pieces.int8.mma.") + int8_pieces_plan_name(arm - ARM_PIECES_BASE)
    if arm == ARM_INF_PIECES:
        return String("inference.pieces.int8.tuned")
    if arm == ARM_TRAIN_PIECES:
        return String("training.pieces.int8.tuned")
    if arm == ARM_INF_PIECES_FUSED:
        return String("inference.pieces.int8.fused")
    if arm < ARM_DECODE_QUANT_BASE:
        return String("int8i32.v1.mma.") + int8_decode_plan_name(arm - ARM_DECODE_BASE)
    return String("inference.int8i32.v1.quant-in-launch.") + int8_decode_plan_name(
        arm - ARM_DECODE_QUANT_BASE
    )


def _arm_is_sums(arm: Int) -> Bool:
    """Whether the arm's output is the three Int32 sums per cell."""
    return arm >= ARM_PIECES_FLAT and arm < ARM_INF_PIECES


def _arm_is_probe(arm: Int) -> Bool:
    """Whether the arm computes a wrong product on purpose: a time and no
    identity."""
    if arm < ARM_DIRECT_BASE or arm >= ARM_INF_INT8_TUNED:
        return False
    return int8_direct_is_probe(arm - ARM_DIRECT_BASE)


def _arm_is_product(arm: Int) -> Bool:
    """Whether the arm's output is the product `C` (a plan alone, or a
    complete operation that ends in one)."""
    if arm == ARM_QUANTIZE_A_PAR or arm == ARM_INT8_PACK_B_PAR:
        return False
    if _arm_is_sums(arm):
        return False
    return arm <= ARM_INT8_APPLE_CHUNK or arm >= ARM_INF_BF16


def _arm_runs(arm: Int) -> Bool:
    """Whether this build runs the arm. An arm that does not run prints a
    NOT RUN line with the reason; a missing line is never an agreeing one."""
    if arm == ARM_INT8_MMA:
        return HAS_INT8_MMA
    if (
        arm == ARM_INT8_APPLE_CHUNK
        or arm == ARM_INF_INT8_APPLE_CHUNK
        or arm == ARM_TRAIN_INT8_APPLE_CHUNK
        or arm == ARM_INF_INT8_APPLE_CHUNK_PARQ
        or arm == ARM_TRAIN_INT8_APPLE_CHUNK_PARQ
    ):
        return HAS_APPLE_CHUNK
    if arm >= ARM_DIRECT_BASE and arm < ARM_INF_INT8_TUNED:
        return INT8_DIRECT_AVAILABLE
    if arm >= ARM_DECODE_BASE:
        return INT8_DECODE_AVAILABLE
    if arm == ARM_PIECES_FLAT:
        return True
    if arm >= ARM_TUNED_BASE and arm < ARM_DIRECT_BASE:
        return HAS_INT8_MMA and int8_tuned_plan_available(arm - ARM_TUNED_BASE)
    if arm >= ARM_TUNED_BASE:
        return HAS_INT8_MMA
    return True


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
    clock's resolution, which is a fact about the clock and not a rate."""
    if ms <= 0.0:
        return 0.0
    return count / (ms * 1.0e6)


def _fnv(d: UInt64, word: UInt64) -> UInt64:
    return (d ^ word) * UInt64(0x100000001B3)


def price_sabotage_name() -> String:
    comptime if PRICE_SABOTAGE:
        return String("ONE_BIT_OF_THE_LAST_CELL")
    else:
        return String("none")


def _sabotage_flip_kernel(buf: MutPointer[Float32, MutAnyOrigin], at_in: Int32):
    """The sabotage arm on a float32 output: the lowest bit of cell `at_in`,
    flipped in place by one thread."""
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var at_ = Int(at_in)
    var bits = bitcast[DType.uint32](buf.unsafe_load(at_)) ^ UInt32(1)
    buf.unsafe_store(at_, bitcast[DType.float32](bits))


def _sabotage_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], count: Int
) raises:
    comptime if PRICE_SABOTAGE:
        ctx.enqueue_function[_sabotage_flip_kernel](
            buf.unsafe_ptr(), Int32(count - 1), grid_dim=(1, 1, 1), block_dim=(1, 1, 1)
        )
        ctx.synchronize()


def _sabotage_word(word: UInt64, i: Int, count: Int) -> UInt64:
    """The sabotage arm on a bf16 or an int8 output: the lowest bit of the
    last word read back."""
    comptime if PRICE_SABOTAGE:
        if i == count - 1:
            return word ^ UInt64(1)
    return word


def _whole(have: Int, count: Int, what: String) raises:
    """A read-back copies the WHOLE device buffer into a host buffer of
    `count` elements, and a poison copies `len(buffer)` elements out of one.
    A device buffer of any other length is refused by name."""
    if have != count:
        raise Error(
            "bench/gemm_lowbit_price_main: " + what + " holds " + String(have)
            + " elements and " + String(count) + " were asked for; the copy is"
            " of the whole buffer, so the two must be equal"
        )


def _digest_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], count: Int, tag: String
) raises -> UInt64:
    """`_dev_digest`, refused on a buffer that is not exactly `count` long."""
    _whole(len(buf), count, tag)
    return _dev_digest(ctx, buf, count, tag)


def _poison_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], count: Int, tag: String
) raises:
    """`_dev_poison`, refused on a buffer that is not exactly `count` long."""
    _whole(len(buf), count, tag)
    _dev_poison(ctx, buf, count)


def _digest_u16(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.uint16], count: Int
) raises -> UInt64:
    """`_dev_digest`'s digest over a bf16 buffer's bits."""
    _whole(len(buf), count, String("a bf16 buffer"))
    var h = ctx.enqueue_create_host_buffer[DType.uint16](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var d = UInt64(0xCBF29CE484222325)
    for i in range(count):
        d = _fnv(d, _sabotage_word(UInt64(Int(h.unsafe_ptr().unsafe_load(i))), i, count))
    _ = h
    return d


def _poison_codes(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    codes: Int,
    rows: Int,
) raises:
    """An int8 store filled with what no quantizer writes: the code -128
    (contract L-4 clamps to [-127, 127]) and an exponent no float32 row
    takes. `_digest_codes` refuses a code that is still -128."""
    _whole(len(q), codes, String("an int8 code buffer"))
    _whole(len(e), rows, String("an int8 exponent buffer"))
    var hq = ctx.enqueue_create_host_buffer[DType.int8](codes)
    var he = ctx.enqueue_create_host_buffer[DType.int32](rows)
    ctx.synchronize()
    for i in range(codes):
        hq.unsafe_ptr().unsafe_store(i, Int8(-128))
    for i in range(rows):
        he.unsafe_ptr().unsafe_store(i, Int32(-987654))
    ctx.enqueue_copy(dst_buf=q, src_ptr=hq.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=e, src_ptr=he.unsafe_ptr())
    ctx.synchronize()
    _ = hq
    _ = he


def _digest_codes(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    rows: Int,
    cols: Int,
) raises -> UInt64:
    """The digest of an int8 store: every code, then every row exponent.
    Contract section 2 promises the codes and the exponents across vendors,
    so this is the quantizer's identity riding along."""
    _whole(len(q), rows * cols, String("an int8 code buffer"))
    _whole(len(e), rows, String("an int8 exponent buffer"))
    var hq = ctx.enqueue_create_host_buffer[DType.int8](rows * cols)
    var he = ctx.enqueue_create_host_buffer[DType.int32](rows)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hq.unsafe_ptr(), src_buf=q)
    ctx.enqueue_copy(dst_ptr=he.unsafe_ptr(), src_buf=e)
    ctx.synchronize()
    var d = UInt64(0xCBF29CE484222325)
    for i in range(rows * cols):
        if hq.unsafe_ptr().unsafe_load(i) == Int8(-128):
            raise Error(
                "bench/gemm_lowbit_price_main: POISON SURVIVED at code " + String(i)
                + ": -128 is no code of the profile (contract L-4)"
            )
        d = _fnv(
            d,
            _sabotage_word(UInt64(Int(hq.unsafe_ptr().unsafe_load(i)) & 0xFF), i, rows * cols),
        )
    for i in range(rows):
        d = _fnv(d, UInt64(Int(he.unsafe_ptr().unsafe_load(i)) & 0xFFFFFFFF))
    _ = hq
    _ = he
    return d


def _pieces_fixture_kernel(
    hi: MutPointer[Int8, MutAnyOrigin],
    lo: MutPointer[Int8, MutAnyOrigin],
    q: MutPointer[Int8, MutAnyOrigin],
    count_in: Int32,
):
    """The two planes the four-product arms read, made from a row's int8
    codes by integer arithmetic, one element per thread: the high plane is
    the code, the low plane `(37 q + 11) mod 128`, in [0, 127]. A FIXTURE,
    not the fifteen-bit profile's split: what matters here is that the four
    planes differ and are the same on every box."""
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i >= Int(count_in):
        return
    var code = Int(q.unsafe_load(i))
    hi.unsafe_store(i, Int8(code))
    lo.unsafe_store(i, Int8((37 * code + 11) & 127))


def _pieces_fixture(
    ctx: DeviceContext,
    mut hi: DeviceBuffer[DType.int8],
    mut lo: DeviceBuffer[DType.int8],
    mut q: DeviceBuffer[DType.int8],
    count: Int,
) raises:
    ctx.enqueue_function[_pieces_fixture_kernel](
        hi.unsafe_ptr(),
        lo.unsafe_ptr(),
        q.unsafe_ptr(),
        Int32(count),
        grid_dim=((count + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )


def _pieces_recombine_probe_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    s: MutPointer[Int32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
):
    """THE SECOND LAUNCH of the two-launch form, one thread per cell: the
    three sums read back and handed to lane/lowbit-int15's
    `int15_store_cell` (the recombination in Int64, the pinned seam, the
    scale), the function the fused form calls at its last step."""
    var m = Int(m_in)
    var n = Int(n_in)
    var cell = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    int15_store_cell(
        c, ea, eb, s.unsafe_load(3 * cell), s.unsafe_load(3 * cell + 1), s.unsafe_load(3 * cell + 2), i, j, m, n
    )


def _pieces_recombine_probe(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut s: DeviceBuffer[DType.int32],
    mut ea: DeviceBuffer[DType.int32],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
) raises:
    ctx.enqueue_function[_pieces_recombine_probe_kernel](
        c.unsafe_ptr(),
        s.unsafe_ptr(),
        ea.unsafe_ptr(),
        eb.unsafe_ptr(),
        Int32(m),
        Int32(n),
        grid_dim=((m * n + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )


def _poison_sums(ctx: DeviceContext, mut s: DeviceBuffer[DType.int32], count: Int) raises:
    """The three sums per cell, filled with what no sum is."""
    _whole(len(s), count, String("a sums buffer"))
    var h = ctx.enqueue_create_host_buffer[DType.int32](count)
    ctx.synchronize()
    for i in range(count):
        h.unsafe_ptr().unsafe_store(i, SUM_POISON)
    ctx.enqueue_copy(dst_buf=s, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h


def _digest_sums(
    ctx: DeviceContext, mut s: DeviceBuffer[DType.int32], count: Int, tag: String
) raises -> UInt64:
    """`_dev_digest`'s digest over the Int32 sums, a surviving poison
    refused."""
    _whole(len(s), count, String("a sums buffer"))
    var h = ctx.enqueue_create_host_buffer[DType.int32](count)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=s)
    ctx.synchronize()
    var d = UInt64(0xCBF29CE484222325)
    for i in range(count):
        var v = h.unsafe_ptr().unsafe_load(i)
        if v == SUM_POISON:
            raise Error(
                "bench/gemm_lowbit_price_main: POISON SURVIVED at sum " + String(i)
                + " of " + tag
            )
        d = _fnv(d, _sabotage_word(UInt64(Int(v) & 0xFFFFFFFF), i, count))
    _ = h
    return d


struct ShapeBuffers(Movable):
    """Every device buffer one shape's arms read or write, allocated once
    per shape and shared by the arms, so no arm's time holds an allocation."""

    var a: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var bh: DeviceBuffer[DType.uint16]
    var qa: DeviceBuffer[DType.int8]
    var ea: DeviceBuffer[DType.int32]
    var qb: DeviceBuffer[DType.int8]
    var eb: DeviceBuffer[DType.int32]
    var c: DeviceBuffer[DType.float32]
    var ws: DeviceBuffer[DType.float32]
    #: The conversion arms' scratch outputs: a second int8 store for each
    #: operand and a second bf16 image, so timing a conversion never
    #: rewrites an operand a product arm reads. EACH IS EXACTLY THE SIZE OF
    #: WHAT IS WRITTEN TO IT: a read-back copies the whole device buffer, so
    #: a scratch larger than the host buffer it is read into writes past
    #: the host buffer's end (m2pro, 2026-09-29: one `max(m, n) * k` scratch
    #: for both operands gave conversion digests that differed from the
    #: other boxes' at the decode rows, where `m` is far below `n`).
    var qsa: DeviceBuffer[DType.int8]
    var esa: DeviceBuffer[DType.int32]
    var qsb: DeviceBuffer[DType.int8]
    var esb: DeviceBuffer[DType.int32]
    var hs: DeviceBuffer[DType.uint16]
    #: lane/lowbit-mma-speed, FOUR PRODUCTS: the two planes of each operand
    #: and the three sums per cell. One element each where the row's `k` is
    #: above the four-product kernel's bound (no row of the table is).
    var pah: DeviceBuffer[DType.int8]
    var pal: DeviceBuffer[DType.int8]
    var pbh: DeviceBuffer[DType.int8]
    var pbl: DeviceBuffer[DType.int8]
    var ps: DeviceBuffer[DType.int32]
    var work: LowbitWorkspace

    def __init__(
        out self, ctx: DeviceContext, m: Int, n: Int, k: Int, pieces: Bool
    ) raises:
        var nws = identical_gemm_workspace_max_floats(m, n, k)
        if nws < 1:
            nws = 1
        self.a = ctx.enqueue_create_buffer[DType.float32](m * k)
        self.b = ctx.enqueue_create_buffer[DType.float32](n * k)
        self.bh = ctx.enqueue_create_buffer[DType.uint16](n * k)
        self.qa = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.ea = ctx.enqueue_create_buffer[DType.int32](m)
        self.qb = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.eb = ctx.enqueue_create_buffer[DType.int32](n)
        self.c = ctx.enqueue_create_buffer[DType.float32](m * n)
        self.ws = ctx.enqueue_create_buffer[DType.float32](nws)
        self.qsa = ctx.enqueue_create_buffer[DType.int8](m * k)
        self.esa = ctx.enqueue_create_buffer[DType.int32](m)
        self.qsb = ctx.enqueue_create_buffer[DType.int8](n * k)
        self.esb = ctx.enqueue_create_buffer[DType.int32](n)
        self.hs = ctx.enqueue_create_buffer[DType.uint16](n * k)
        # The four-product arms' buffers only where one of them runs: a
        # box that runs none (Apple) does not pay their memory.
        self.pah = ctx.enqueue_create_buffer[DType.int8](m * k if pieces else 1)
        self.pal = ctx.enqueue_create_buffer[DType.int8](m * k if pieces else 1)
        self.pbh = ctx.enqueue_create_buffer[DType.int8](n * k if pieces else 1)
        self.pbl = ctx.enqueue_create_buffer[DType.int8](n * k if pieces else 1)
        self.ps = ctx.enqueue_create_buffer[DType.int32](3 * m * n if pieces else 1)
        self.work = LowbitWorkspace(ctx)
        self.work.ensure(ctx, nws, n * k)
        ctx.synchronize()


def _enqueue_arm(
    ctx: DeviceContext, mut sb: ShapeBuffers, arm: Int, m: Int, n: Int, k: Int
) raises:
    """Enqueue one arm's work and return; the caller waits. The timed region
    is this call and the wait, nothing else."""
    if arm == ARM_FP32:
        identical_gemm_into(ctx, sb.c, sb.a, sb.b, sb.ws, m, n, k, OP_NT)
    elif arm == ARM_BF16_FUSED:
        identical_gemm_bf16w_fused_into(ctx, sb.c, sb.a, sb.bh, m, n, k, OP_NT)
    elif arm == ARM_BF16_WIDEN:
        identical_gemm_bf16w_widen_into(ctx, sb.c, sb.a, sb.bh, sb.work, m, n, k, OP_NT)
    elif arm == ARM_INT8_FLAT:
        identical_gemm_int8_flat_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_INT8_MMA:
        comptime if HAS_INT8_MMA:
            identical_gemm_int8_mma_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_INT8_APPLE_CHUNK:
        comptime if HAS_APPLE_CHUNK:
            identical_gemm_int8_apple_chunk_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_QUANTIZE_A:
        quantize_rows_int8_device(ctx, sb.qsa, sb.esa, sb.a, m, k)
    elif arm == ARM_INT8_PACK_B:
        quantize_rows_int8_device(ctx, sb.qsb, sb.esb, sb.b, n, k)
    elif arm == ARM_BF16_PACK_B:
        bf16_narrow(ctx, sb.hs, sb.b, n * k)
    elif arm == ARM_BF16_WIDEN_B:
        bf16_widen(ctx, sb.work.wide, sb.bh, n * k)
    elif arm == ARM_INT8_DEQUANT_B:
        dequantize_rows_int8_device(ctx, sb.work.wide, sb.qb, sb.eb, n, k)
    elif arm == ARM_INF_BF16:
        identical_gemm_bf16w_into(ctx, sb.c, sb.a, sb.bh, sb.work, m, n, k, OP_NT)
    elif arm == ARM_INF_INT8:
        quantize_rows_int8_device(ctx, sb.qa, sb.ea, sb.a, m, k)
        identical_gemm_int8_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_INF_INT8_APPLE_CHUNK:
        comptime if HAS_APPLE_CHUNK:
            quantize_rows_int8_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            identical_gemm_int8_apple_chunk_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_TRAIN_BF16:
        bf16_narrow(ctx, sb.bh, sb.b, n * k)
        identical_gemm_bf16w_into(ctx, sb.c, sb.a, sb.bh, sb.work, m, n, k, OP_NT)
    elif arm == ARM_TRAIN_INT8:
        quantize_rows_int8_device(ctx, sb.qa, sb.ea, sb.a, m, k)
        quantize_rows_int8_device(ctx, sb.qb, sb.eb, sb.b, n, k)
        identical_gemm_int8_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_TRAIN_INT8_APPLE_CHUNK:
        comptime if HAS_APPLE_CHUNK:
            quantize_rows_int8_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            quantize_rows_int8_device(ctx, sb.qb, sb.eb, sb.b, n, k)
            identical_gemm_int8_apple_chunk_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_QUANTIZE_A_PAR:
        quantize_rows_int8_par_device(ctx, sb.qsa, sb.esa, sb.a, m, k)
    elif arm == ARM_INT8_PACK_B_PAR:
        quantize_rows_int8_par_device(ctx, sb.qsb, sb.esb, sb.b, n, k)
    elif arm == ARM_INF_INT8_PARQ:
        quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
        identical_gemm_int8_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_TRAIN_INT8_PARQ:
        quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
        quantize_rows_int8_par_device(ctx, sb.qb, sb.eb, sb.b, n, k)
        identical_gemm_int8_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_INF_INT8_APPLE_CHUNK_PARQ:
        comptime if HAS_APPLE_CHUNK:
            quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            identical_gemm_int8_apple_chunk_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm == ARM_TRAIN_INT8_APPLE_CHUNK_PARQ:
        comptime if HAS_APPLE_CHUNK:
            quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            quantize_rows_int8_par_device(ctx, sb.qb, sb.eb, sb.b, n, k)
            identical_gemm_int8_apple_chunk_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)
    elif arm < ARM_DIRECT_BASE:
        comptime if HAS_INT8_MMA:
            identical_gemm_int8_mma_tuned_with_plan(
                ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k, arm - ARM_TUNED_BASE
            )
    elif arm < ARM_INF_INT8_TUNED:
        comptime if INT8_DIRECT_AVAILABLE:
            identical_gemm_int8_mma_direct_into(
                ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k, arm - ARM_DIRECT_BASE
            )
    elif arm == ARM_PIECES_FLAT:
        identical_gemm_int8_pieces_flat_into(ctx, sb.ps, sb.pah, sb.pal, sb.pbh, sb.pbl, m, n, k)
    elif arm >= ARM_PIECES_BASE and arm < ARM_INF_PIECES:
        comptime if HAS_INT8_MMA:
            identical_gemm_int8_pieces_tuned_with_plan(
                ctx, sb.ps, sb.pah, sb.pal, sb.pbh, sb.pbl, m, n, k, arm - ARM_PIECES_BASE
            )
    elif arm == ARM_INF_PIECES or arm == ARM_TRAIN_PIECES:
        comptime if HAS_INT8_MMA:
            quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            if arm == ARM_TRAIN_PIECES:
                quantize_rows_int8_par_device(ctx, sb.qb, sb.eb, sb.b, n, k)
            identical_gemm_int8_pieces_tuned_into(
                ctx, sb.ps, sb.pah, sb.pal, sb.pbh, sb.pbl, m, n, k
            )
            _pieces_recombine_probe(ctx, sb.c, sb.ps, sb.ea, sb.eb, m, n)
    elif arm >= ARM_DECODE_BASE and arm < ARM_DECODE_QUANT_BASE:
        comptime if INT8_DECODE_AVAILABLE:
            identical_gemm_int8_mma_decode_with_plan(
                ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k, arm - ARM_DECODE_BASE
            )
    elif arm >= ARM_DECODE_QUANT_BASE:
        comptime if INT8_DECODE_AVAILABLE:
            identical_gemm_int8_mma_decode_quant_with_plan(
                ctx, sb.c, sb.a, sb.ea, sb.qb, sb.eb, m, n, k, arm - ARM_DECODE_QUANT_BASE
            )
    elif arm == ARM_INF_PIECES_FUSED:
        comptime if HAS_INT8_MMA:
            quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            identical_gemm_int8_pieces_tuned_fused_into(
                ctx, sb.c, sb.pah, sb.pal, sb.ea, sb.pbh, sb.pbl, sb.eb, m, n, k
            )
    else:
        comptime if HAS_INT8_MMA:
            quantize_rows_int8_par_device(ctx, sb.qa, sb.ea, sb.a, m, k)
            if arm == ARM_TRAIN_INT8_TUNED or arm == ARM_TRAIN_4X_TUNED:
                quantize_rows_int8_par_device(ctx, sb.qb, sb.eb, sb.b, n, k)
            var products = 1
            if arm == ARM_INF_4X_TUNED or arm == ARM_TRAIN_4X_TUNED:
                products = PIECE_PRODUCTS
            for _ in range(products):
                identical_gemm_int8_mma_tuned_into(ctx, sb.c, sb.qa, sb.ea, sb.qb, sb.eb, m, n, k)


def _arm_digest(
    ctx: DeviceContext,
    mut sb: ShapeBuffers,
    arm: Int,
    m: Int,
    n: Int,
    k: Int,
    tag: String,
) raises -> UInt64:
    """The digest of what the arm just wrote (after the sabotage arm's one
    bit, in a build that names it)."""
    if _arm_is_product(arm):
        _sabotage_f32(ctx, sb.c, m * n)
        return _digest_f32(ctx, sb.c, m * n, tag)
    if _arm_is_sums(arm):
        return _digest_sums(ctx, sb.ps, 3 * m * n, tag)
    if arm == ARM_QUANTIZE_A or arm == ARM_QUANTIZE_A_PAR:
        return _digest_codes(ctx, sb.qsa, sb.esa, m, k)
    if arm == ARM_INT8_PACK_B or arm == ARM_INT8_PACK_B_PAR:
        return _digest_codes(ctx, sb.qsb, sb.esb, n, k)
    if arm == ARM_BF16_PACK_B:
        return _digest_u16(ctx, sb.hs, n * k)
    _sabotage_f32(ctx, sb.work.wide, n * k)
    return _digest_f32(ctx, sb.work.wide, n * k, tag)


def _arm_note(arm: Int, m: Int, n: Int, k: Int) -> String:
    """What a reader needs beside the arm's number: the plan the dispatcher
    of its profile would have picked at this shape. No spaces."""
    if arm == ARM_FP32:
        return String("plan=") + String(choose_gemm_plan(m, n, k))
    if arm == ARM_BF16_FUSED:
        if m * n <= BF16W_FUSED_MAX_CELLS:
            return String("dispatched")
        return String("not-dispatched(cells>") + String(BF16W_FUSED_MAX_CELLS) + ")"
    if arm == ARM_BF16_WIDEN:
        if m * n <= BF16W_FUSED_MAX_CELLS:
            return String("not-dispatched(cells<=") + String(BF16W_FUSED_MAX_CELLS) + ")"
        return String("dispatched")
    if arm == ARM_INT8_FLAT:
        comptime if HAS_INT8_MMA:
            return String("not-dispatched(the-column-has-the-unit)")
        else:
            return String("dispatched")
    if arm == ARM_INT8_MMA:
        return String("dispatched")
    if arm == ARM_INT8_APPLE_CHUNK:
        return String("probe,geometry=") + String(int8_apple_chunk_geometry(m))
    if arm == ARM_QUANTIZE_A:
        return String("per-call")
    if arm == ARM_INT8_PACK_B or arm == ARM_BF16_PACK_B:
        return String("once-per-weight")
    if arm == ARM_BF16_WIDEN_B or arm == ARM_INT8_DEQUANT_B:
        return String("per-call-when-materialized")
    if arm == ARM_INF_BF16:
        return String("product")
    if arm == ARM_INF_INT8:
        return String("quantize.a+product")
    if arm == ARM_INF_INT8_APPLE_CHUNK:
        return String("quantize.a+probe")
    if arm == ARM_TRAIN_BF16:
        return String("pack.b+product")
    if arm == ARM_TRAIN_INT8:
        return String("quantize.a+pack.b+product")
    if arm == ARM_TRAIN_INT8_APPLE_CHUNK:
        return String("quantize.a+pack.b+probe")
    if arm == ARM_QUANTIZE_A_PAR:
        return String("per-call,block=") + String(quantize_par_block(k))
    if arm == ARM_INT8_PACK_B_PAR:
        return String("once-per-weight,block=") + String(quantize_par_block(k))
    if arm == ARM_INF_INT8_PARQ:
        return String("quantize.a.par+product")
    if arm == ARM_TRAIN_INT8_PARQ:
        return String("quantize.a.par+pack.b.par+product")
    if arm == ARM_INF_INT8_APPLE_CHUNK_PARQ:
        return String("quantize.a.par+probe")
    if arm == ARM_TRAIN_INT8_APPLE_CHUNK_PARQ:
        return String("quantize.a.par+pack.b.par+probe")
    if arm < ARM_DIRECT_BASE:
        if arm - ARM_TUNED_BASE == int8_tuned_dispatch(m, n, k):
            return String("the-tuned-launcher's-plan")
        return String("plan")
    if arm < ARM_INF_INT8_TUNED:
        if _arm_is_probe(arm):
            return String("WRONG-ON-PURPOSE,a-time-and-no-identity")
        return String("the-reference's-schedule,one-thing-changed")
    if arm == ARM_PIECES_FLAT:
        return String("the-reference-device-plan,three-sums-a-cell")
    if arm >= ARM_PIECES_BASE and arm < ARM_INF_PIECES:
        if arm - ARM_PIECES_BASE == int8_pieces_dispatch(m, n, k):
            return String("the-launcher's-plan,rate-over-4mnk")
        return String("plan,rate-over-4mnk")
    if arm == ARM_INF_PIECES:
        return (
            String("quantize.a.par+four-products-one-staging.")
            + int8_pieces_plan_name(int8_pieces_dispatch(m, n, k)) + "+int15_store_cell,second-launch"
        )
    if arm == ARM_TRAIN_PIECES:
        return (
            String("quantize.a.par+pack.b.par+four-products-one-staging.")
            + int8_pieces_plan_name(int8_pieces_dispatch(m, n, k)) + "+int15_store_cell,second-launch"
        )
    if arm >= ARM_DECODE_BASE and arm < ARM_DECODE_QUANT_BASE:
        if arm - ARM_DECODE_BASE == int8_decode_dispatch(m, n, k):
            return String("decode-kernel,the-launcher's-plan")
        return String("decode-kernel")
    if arm >= ARM_DECODE_QUANT_BASE:
        return String("quantizer-in-the-launch+decode-kernel,ONE-launch")
    if arm == ARM_INF_PIECES_FUSED:
        return (
            String("quantize.a.par+four-products-one-staging.")
            + int8_pieces_plan_name(int8_pieces_dispatch(m, n, k))
            + "+int15_store_cell-FUSED,one-launch"
        )
    var plan_name = int8_tuned_plan_name(int8_tuned_dispatch(m, n, k))
    if arm == ARM_INF_INT8_TUNED:
        return String("quantize.a.par+") + plan_name
    if arm == ARM_TRAIN_INT8_TUNED:
        return String("quantize.a.par+pack.b.par+") + plan_name
    if arm == ARM_INF_4X_TUNED:
        return String("quantize.a.par+4x.") + plan_name
    return String("quantize.a.par+pack.b.par+4x.") + plan_name


def _must_agree(
    a_arm: Int, b_arm: Int, dig: List[UInt64], ran: List[Bool], name: String
) -> String:
    """Empty when the two arms' digests agree or either did not run; the
    disagreement otherwise."""
    if not ran[a_arm] or not ran[b_arm]:
        return String("")
    if dig[a_arm] == dig[b_arm]:
        return String("")
    return (
        String("PLANS DISAGREE at ") + name + ": " + _arm_name(a_arm) + " "
        + hex(dig[a_arm]) + " vs " + _arm_name(b_arm) + " " + hex(dig[b_arm]) + "\n"
    )


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
    """Every arm at one shape. Returns the plan disagreements found (empty
    when there are none); `main` raises on them after every shape has
    printed."""
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
    for arm in range(ARM_COUNT):
        dig.append(UInt64(0))
        var runs = _arm_runs(arm) and _arm_asked(arms, _arm_name(arm))
        # The four-product kernel refuses a `k` above its own bound.
        var is_pieces = arm >= ARM_PIECES_FLAT and arm <= ARM_INF_PIECES_FUSED
        if is_pieces and k > INT8_PIECES_MAX_K:
            runs = False
        if is_pieces and runs:
            pieces = True
        # The decode kernel takes at most sixteen rows.
        if arm >= ARM_DECODE_BASE and m > INT8_DECODE_MAX_M:
            runs = False
        if arm >= ARM_PIECES_BASE and arm < ARM_INF_PIECES:
            if not int8_pieces_plan_admits(arm - ARM_PIECES_BASE, m, n, k):
                runs = False
        ran.append(runs)
        samples.append(List[Int]())
    var sb = ShapeBuffers(ctx, m, n, k, pieces)
    # `bench/gemm_price_main.mojo`'s salts at this shape index, so the fp32
    # operands are the ones its device arm reads at the same extents.
    _whole(len(sb.a), m * k, String("the left operand"))
    _whole(len(sb.b), n * k, String("the right operand"))
    _dev_fill(ctx, sb.a, m * k, 11 + idx)
    _dev_fill(ctx, sb.b, n * k, 22 + idx)
    # The low-bit operands, by the contract's own seams on the device, before
    # anything is timed: bf16 weights (L-2), int8 codes of both (L-3, L-4).
    bf16_narrow(ctx, sb.bh, sb.b, n * k)
    quantize_rows_int8_device(ctx, sb.qa, sb.ea, sb.a, m, k)
    quantize_rows_int8_device(ctx, sb.qb, sb.eb, sb.b, n, k)
    # The four-product arms' planes, from the codes.
    if pieces:
        _pieces_fixture(ctx, sb.pah, sb.pal, sb.qa, m * k)
        _pieces_fixture(ctx, sb.pbh, sb.pbl, sb.qb, n * k)
    ctx.synchronize()

    # Untimed warm-up of every arm, its output poisoned first where the
    # output is float32 and read back after, so an arm that launches without
    # writing cannot turn in a time.
    for arm in range(ARM_COUNT):
        if not ran[arm]:
            continue
        var tag = String("lowbit.") + name + "." + _arm_name(arm)
        if _arm_is_product(arm):
            _poison_f32(ctx, sb.c, m * n, tag)
        elif arm == ARM_BF16_WIDEN_B or arm == ARM_INT8_DEQUANT_B:
            _poison_f32(ctx, sb.work.wide, n * k, tag)
        elif arm == ARM_QUANTIZE_A_PAR:
            # The reference quantizer's arm ran before this one and left
            # its codes in the same scratch: without a poison a parallel
            # quantizer that wrote nothing would turn in the right digest.
            _poison_codes(ctx, sb.qsa, sb.esa, m * k, m)
        elif arm == ARM_INT8_PACK_B_PAR:
            _poison_codes(ctx, sb.qsb, sb.esb, n * k, n)
        elif _arm_is_sums(arm):
            _poison_sums(ctx, sb.ps, 3 * m * n)
        _enqueue_arm(ctx, sb, arm, m, n, k)
        ctx.synchronize()
        dig[arm] = _arm_digest(ctx, sb, arm, m, n, k, tag)

    for _ in range(0 if identity_only else repeats):
        for arm in range(ARM_COUNT):
            if not ran[arm]:
                continue
            var t0 = perf_counter_ns()
            _enqueue_arm(ctx, sb, arm, m, n, k)
            ctx.synchronize()
            samples[arm].append(Int(perf_counter_ns() - t0))

    var macs = Float64(m) * Float64(n) * Float64(k)
    for arm in range(ARM_COUNT):
        var arm_name = _arm_name(arm)
        if not ran[arm]:
            if _arm_runs(arm):
                print(
                    "LOWBIT-NOT-RUN", column_name(TARGET_COLUMN), name, arm_name,
                    "not asked for (MOJOLEARN_LOWBIT_PRICE_ARMS)",
                )
            else:
                print(
                    "LOWBIT-NOT-RUN", column_name(TARGET_COLUMN), name, arm_name,
                    "this column cannot run the arm (it has no such unit, or a block of the plan is above its thread limit)",
                )
            continue
        if identity_only:
            print(
                "LOWBIT", column_name(TARGET_COLUMN), name, m, n, k, cap_word, arm_name,
                "not-timed", "not-timed", "not-timed",
                "GMAC/s" if (_arm_is_product(arm) or _arm_is_sums(arm)) else "Gelem/s",
                hex(dig[arm]), _arm_note(arm, m, n, k),
            )
            continue
        var med = _median_ms(samples[arm])
        var best = _min_ms(samples[arm])
        var count = macs
        var unit = String("GMAC/s")
        if _arm_is_sums(arm):
            count = Float64(PIECE_PRODUCTS) * macs
        elif not _arm_is_product(arm):
            unit = String("Gelem/s")
            count = Float64(n) * Float64(k)
            if arm == ARM_QUANTIZE_A or arm == ARM_QUANTIZE_A_PAR:
                count = Float64(m) * Float64(k)
        _report_device(String("lowbit.") + name + "." + arm_name + ".device", med)
        print(
            "LOWBIT", column_name(TARGET_COLUMN), name, m, n, k, cap_word, arm_name,
            _fixed(med, 4), _fixed(best, 4), _fixed(_rate(count, med), 4), unit,
            hex(dig[arm]), _arm_note(arm, m, n, k),
        )

    var bad = String("")
    bad += _must_agree(ARM_BF16_FUSED, ARM_BF16_WIDEN, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_INT8_MMA, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_INT8_APPLE_CHUNK, dig, ran, name)
    # A complete operation ends in its profile's product, so its digest is
    # the profile's.
    bad += _must_agree(ARM_BF16_WIDEN, ARM_INF_BF16, dig, ran, name)
    bad += _must_agree(ARM_BF16_WIDEN, ARM_TRAIN_BF16, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_INF_INT8, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_TRAIN_INT8, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_INF_INT8_APPLE_CHUNK, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_TRAIN_INT8_APPLE_CHUNK, dig, ran, name)
    # lane/lowbit-mma-speed: the parallel quantizer's codes and exponents
    # are the reference quantizer's, and the operations that use it end in
    # the profile's product.
    bad += _must_agree(ARM_QUANTIZE_A, ARM_QUANTIZE_A_PAR, dig, ran, name)
    bad += _must_agree(ARM_INT8_PACK_B, ARM_INT8_PACK_B_PAR, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_INF_INT8_PARQ, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_TRAIN_INT8_PARQ, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_INF_INT8_APPLE_CHUNK_PARQ, dig, ran, name)
    bad += _must_agree(ARM_INT8_FLAT, ARM_TRAIN_INT8_APPLE_CHUNK_PARQ, dig, ran, name)
    # Every tuned plan, every direct plan and every operation that ends in
    # one is the profile's product. A probe is compared with nothing.
    for arm in range(ARM_TUNED_BASE, ARM_PIECES_FLAT):
        if not _arm_is_probe(arm):
            bad += _must_agree(ARM_INT8_FLAT, arm, dig, ran, name)
    # FOUR PRODUCTS: every staged plan's three sums are the reference device
    # plan's, and the two complete operations end in the same cells.
    for arm in range(ARM_PIECES_BASE, ARM_INF_PIECES):
        bad += _must_agree(ARM_PIECES_FLAT, arm, dig, ran, name)
    bad += _must_agree(ARM_INF_PIECES, ARM_TRAIN_PIECES, dig, ran, name)
    # The fused launch computes the two-launch operation's cells, bit for bit.
    bad += _must_agree(ARM_INF_PIECES, ARM_INF_PIECES_FUSED, dig, ran, name)
    # The decode kernel, on codes and from floats, is the profile's product.
    for arm in range(ARM_DECODE_BASE, ARM_COUNT):
        bad += _must_agree(ARM_INT8_FLAT, arm, dig, ran, name)
    if bad.byte_length() > 0:
        print(bad)
    _ = sb^
    return bad


def _arm_asked(arms: String, name: String) -> Bool:
    """Whether `MOJOLEARN_LOWBIT_PRICE_ARMS` names the arm: whole names,
    since one arm's name is the start of another's; a part that ends in `*`
    names every arm that starts with what is before it."""
    if arms == "":
        return True
    for part in arms.split(","):
        var want = String(part)
        if want.endswith("*"):
            if name.startswith(String(want.removesuffix("*"))):
                return True
        elif want == name:
            return True
    return False


def _wanted(only: String, name: String) -> Bool:
    if only == "":
        return True
    for part in only.split(","):
        if part.byte_length() > 0 and name.find(String(part)) >= 0:
            return True
    return False


def main() raises:
    comptime if not has_accelerator():
        raise Error("bench/gemm_lowbit_price_main: no accelerator; every arm here is a device arm")
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

        print("== bench/gemm_lowbit_price_main.mojo [" + _mode() + "] ==")
        print("column", column_name(TARGET_COLUMN))
        print("profiles fp32.v1, bf16f32.v1, int8i32.v1")
        print("int8 dispatch:", int8_plan_dispatch_name())
        print(
            "sabotage: price", price_sabotage_name(), " lowbit", lowbit_sabotage_name(),
            " apple chunk", int8_apple_chunk_sabotage_name(),
            " parallel quantizer", quantize_par_sabotage_name(),
            " tuned unit", int8_tuned_sabotage_name(),
        )
        print("repeats", repeats, " mac budget", budget, " only", only, " arms", arms)
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
            raise Error("bench/gemm_lowbit_price_main: no shape matched MOJOLEARN_LOWBIT_PRICE_ONLY=" + only)
        if bad.byte_length() > 0:
            if _mode() == "IDENTICAL":
                raise Error(
                    "plans of one profile disagree (above). Under IDENTICAL that"
                    " is a contract violation, not a measurement."
                )
            print("NOTE (not IDENTICAL, not a failure): plans disagree, above.")
