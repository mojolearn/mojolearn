# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BF16-UNITS RELAXATION OF `fast`: NEURAL PRODUCTS ON THE HALF-PRECISION MATRIX UNIT
(lane fx-fast-bf16, 2026-11-20; the kernel of mode-policy row BF16_UNITS, checks/mode_policy.mojo).

EXPERIMENT `-D MOJOLEARN_GEMM_FAST_BF16_UNITS=<kind>`, default OFF, A/B owed with quality beside time
(docs/apple-fast/EXPERIMENTS.md; arm `fast-bf16-units` of bench/results/paper_compare/owed-pre0867.json).

  kind 1 = bf16 operands, FP32 accumulation, FP32 OUTPUT: one call of MAX `linalg.matmul` over two bf16 operand
           tensors and the caller's FP32 output tensor (the mixed-dtype call instantiates on sm_89 under mojo
           1.2.0 / max 26.7.0: GitHub run 44733416797). The only value. A second form that had MAX store bf16
           and widened it (torch autocast's linear) was written as a fallback in case the mixed call did not
           compile; it compiled too (run 44733420954) and was deleted unmeasured: one more launch and an 8-bit
           output for nothing. Recoverable at lane/fx-fast-bf16@ef53d91d70.

WHERE IT IS LEGAL, AND ON WHICH VENDOR. Only in a `fast` build (`-D MOJOLEARN_MODE_FAST_ROWS`); in a build that
promises identical bits the define is a compile error (checks/mode_policy.mojo `_row`). Vendor-neutral source,
taken on NVIDIA (`mma.sync m16n8k16` bf16, FP32 accumulators) and AMD (MFMA 16x16x16 / 32x32x8 bf16) through the
same MAX entry. Off on Apple: MAX has bf16 products on M5 only, and the Apple set already has its bf16 simdgroup
route (gemm/afn_apple_fast.mojo, AFN_GEMM_BF16_MMA), which the policy row points at. The host column has no
matrix unit and stays exact.

WHAT IS RELAXED. Each OPERAND of a routed product is rounded to bfloat16 (a sign, 8 exponent bits, 7 explicit
mantissa bits: 8 significant bits) by round to nearest even with subnormals flushed
(checks/numerics.mojo::f32_to_bf16_bits_rne, the rounding of contract L-2), so its relative error is at most
2^-9. The unit multiplies the rounded operands exactly and accumulates in FP32. For one cell c = sum_k a_k b_k:

    |c_bf16 - c_fp32|  <=  (2 * 2^-9 + 2^-18) * sum_k |a_k b_k|  +  the FP32 accumulation difference
                       ~=  3.9e-3 * sum_k |a_k b_k|

relative to the ABSOLUTE sum (times the condition number where terms cancel). Round to nearest has zero mean
error, so the typical error is 2^-9 sqrt(2 / 3) / sqrt(K) of the absolute sum for K terms of similar size:
1.0e-4 at K = 256, 2.5e-5 at K = 4096. `bf16_units_cell_bound` is the bound as code.

WHAT STAYS FP32, and why.
  - ACCUMULATION: every cell is summed in FP32 inside the unit. A bf16 accumulator would lose a term below
    2^-8 of the running sum; no vendor's unit accumulates in bf16.
  - THE OUTPUT: activations, dInput and dWeight are stored FP32, so the next product rounds a
    full-precision value once instead of compounding rounded values through the layers.
  - THE MASTER WEIGHTS, the optimizer moments and the update: FP32. The bf16 image of an operand is made per
    product in scratch and never written back: an update of 1e-3 relative is below bf16's half-ulp (2e-3) and
    would be lost on a bf16 master (the reason mixed-precision training keeps FP32 masters).
  - THE LOSS and its gradient, softmax, the normalisations (their divisions amplify operand error by
    1 / variance), gradient clipping norms, the bias and activation epilogues (run in FP32 on the stored
    words, as after a TF32 product).
  - Every product under the size rule; every caller that closed the vendor route (`allow_vendor=False`: Llama
    stage projections, CNN linear, MoE grouped); the bias sums and one-hot gathers (an edge of 1); the
    classical callers (PCA, OLS, Gram, GP: a Gram feeds a solve, error = operand error x condition number).

HOW. A is narrowed into scratch as m x k bf16 (one launch; a transposed A is narrowed AND transposed by the same
launch, so OP_TN costs no extra pass), B is narrowed as stored (one launch; `transpose_b` handles n x k), then
one `linalg.matmul`. Asynchronous, stream-ordered, no wait, no host step. The narrow launches read (m + n) k
floats against the product's m n k multiply-adds: at the rule's smallest edge (64) that is 1 / 32 of the work.
Scratch is `bf16_units_workspace_floats` FP32 words of the caller's workspace, reinterpreted as 16-bit words.

THE SIZE RULE (`bf16_units_takes`): sizes and the hardware unit only, no data shape.
  1. K >= 64: four `m16n8k16` instructions of k (the instruction folds 16 terms).
  2. M >= 64 and N >= 64: as the TF32 rule (gemm/fast_matrix_units.mojo): a 16 x 8 fragment per warp.
  3. M N K >= 2^27 multiply-adds: about one launch of arithmetic at the scalar kernel's rate, and the route
     adds two narrow launches, so nothing under it can gain.
  The constants are reasoned, not measured: the owed shapes row times both sides of each.

COST PER CALL under the arm: the integer rule, then three launches; no lookup, allocation,
lock or wait here (the sequence executor's site takes its scratch from its own allocator). With the define off
nothing here is compiled into a launch.
"""
from std.sys.defines import get_defined_int
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.mode_policy import ROUTE_GEMM_BF16_UNITS
from checks.numerics import f32_to_bf16_bits_rne
from gemm.contract import OP_NN, OP_NT, OP_TN

#: The define's values.
comptime BF16_UNITS_KIND_OFF = 0
comptime BF16_UNITS_KIND_F32_OUT = 1
comptime BF16_UNITS_KIND = get_defined_int["MOJOLEARN_GEMM_FAST_BF16_UNITS", 0]()


def _bf16_units_checked() -> Bool:
    """The compile-time refusal of an unknown kind. True when the arm is on: the policy table's route constant,
    which is True only in a fast-rows build on the NVIDIA or AMD column with the define passed."""
    comptime assert (
        BF16_UNITS_KIND == BF16_UNITS_KIND_OFF or BF16_UNITS_KIND == BF16_UNITS_KIND_F32_OUT
    ), "MOJOLEARN_GEMM_FAST_BF16_UNITS takes 1 (bf16 operands, fp32 accumulation and output)"
    return ROUTE_GEMM_BF16_UNITS


#: The arm: bf16 matrix-unit products for the neural FP32 GEMMs of a `fast` build.
comptime GEMM_FAST_BF16_UNITS = _bf16_units_checked()

#: rule 1: four `m16n8k16` instructions of k
comptime BF16_UNITS_MIN_K = 64
#: rule 2: four 16-row fragments, eight 8-column fragments
comptime BF16_UNITS_MIN_EDGE = 64
#: rule 3: 2^27 multiply-adds
comptime BF16_UNITS_MIN_WORK = 134217728
#: threads a block of the narrow launches
comptime BF16_UNITS_TPB = 256

#: Operand relative error of one bf16 operand: round to nearest at 8 significant bits.
comptime BF16_UNITS_OPERAND_REL = Float64(1.0) / Float64(512.0)

comptime _BF16 = Scalar[DType.bfloat16]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]
comptime _U16P = MutPointer[UInt16, MutAnyOrigin]


@always_inline
def bf16_units_takes(m: Int, n: Int, k: Int) -> Bool:
    """Whether an m x n x k product is large enough for the bf16 unit (the size rule of the module docstring).
    Always False without the arm."""
    comptime if not GEMM_FAST_BF16_UNITS:
        return False
    if k < BF16_UNITS_MIN_K or m < BF16_UNITS_MIN_EDGE or n < BF16_UNITS_MIN_EDGE:
        return False
    return m * n * k >= BF16_UNITS_MIN_WORK


@always_inline
def _round8(x: Int) -> Int:
    """x rounded up to 8 sixteen-bit words, so each bf16 image starts on a 16-byte boundary of the scratch."""
    return (x + 7) // 8 * 8


@always_inline
def bf16_units_workspace_floats(m: Int, n: Int, k: Int) -> Int:
    """FP32 words of scratch a taken m x n x k product needs (any op): the bf16 images of A (m k) and B (n k),
    each started on a 16-byte boundary. 0 without the arm or under the rule."""
    comptime if not GEMM_FAST_BF16_UNITS:
        return 0
    if not bf16_units_takes(m, n, k):
        return 0
    return (_round8(m * k) + _round8(n * k) + 1) // 2


def bf16_units_cell_bound(abs_sum: Float64, k: Int) -> Float64:
    """The bound on |c_bf16 - c_fp32| for one cell whose absolute sum of terms is `abs_sum` (module docstring):
    both operands of every term rounded to bf16, and the FP32 accumulation of K terms in either order."""
    var operand = Float64(2.0) * BF16_UNITS_OPERAND_REL + BF16_UNITS_OPERAND_REL * BF16_UNITS_OPERAND_REL
    var fold = Float64(2.0) * Float64(k) / Float64(16777216.0)
    return (operand + fold) * abs_sum


def bf16_units_narrow_kernel(dst: _U16P, src: _F32P, count: Int64):
    """dst[i] = bf16(src[i]), round to nearest even, one element a thread."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(count):
        return
    dst.unsafe_store(i, f32_to_bf16_bits_rne(src.unsafe_load(i)))


def bf16_units_narrow_t_kernel(dst: _U16P, src: _F32P, m: Int64, k: Int64):
    """dst[m x k] = bf16(src[k x m]^T): the narrow and the transpose of a stored-transposed A in one launch, one
    element of dst a thread (coalesced store, strided load)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(m) * Int(k):
        return
    var r = i // Int(k)
    var col = i - r * Int(k)
    dst.unsafe_store(i, f32_to_bf16_bits_rne(src.unsafe_load(col * Int(m) + r)))


def bf16_units_product_ptr(
    ctx: DeviceContext,
    pc: _F32P,
    pa: _F32P,
    pb: _F32P,
    ps: _F32P,
    m: Int,
    n: Int,
    k: Int,
    a_transposed: Bool,
    b_transposed: Bool,
) raises:
    """The one bf16 matrix-unit product of the tree, with no arm and no size test: C[m x n] (FP32, dense) =
    A . B with A stored m x k (or k x m when `a_transposed`) and B stored k x n (or n x k when
    `b_transposed`), both FP32. `ps` is `bf16_units_workspace_floats(m, n, k)` FP32 words of scratch that
    overlap nothing. Three asynchronous launches, no wait: every pointer stays live to the
    caller's next synchronize. Instantiated only where the arm is on."""
    comptime if GEMM_FAST_BF16_UNITS:
        var a_cells = m * k
        var b_cells = n * k
        var sa = ps.bitcast[UInt16]().unsafe_origin_cast[MutAnyOrigin]()
        var sb = sa + _round8(a_cells)
        if a_transposed:
            ctx.enqueue_function[bf16_units_narrow_t_kernel](
                sa,
                pa,
                Int64(m),
                Int64(k),
                grid_dim=((a_cells + BF16_UNITS_TPB - 1) // BF16_UNITS_TPB, 1, 1),
                block_dim=(BF16_UNITS_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[bf16_units_narrow_kernel](
                sa,
                pa,
                Int64(a_cells),
                grid_dim=((a_cells + BF16_UNITS_TPB - 1) // BF16_UNITS_TPB, 1, 1),
                block_dim=(BF16_UNITS_TPB, 1, 1),
            )
        ctx.enqueue_function[bf16_units_narrow_kernel](
            sb,
            pb,
            Int64(b_cells),
            grid_dim=((b_cells + BF16_UNITS_TPB - 1) // BF16_UNITS_TPB, 1, 1),
            block_dim=(BF16_UNITS_TPB, 1, 1),
        )
        var ha = sa.bitcast[_BF16]().unsafe_origin_cast[MutAnyOrigin]()
        var hb = sb.bitcast[_BF16]().unsafe_origin_cast[MutAnyOrigin]()
        var ta = TileTensor(ptr=ha, layout=row_major(m, k))
        var tc = TileTensor(ptr=pc, layout=row_major(m, n))
        if b_transposed:
            var tbt = TileTensor(ptr=hb, layout=row_major(n, k))
            matmul[transpose_b=True, target="gpu"](tc, ta, tbt, ctx)
        else:
            var tb = TileTensor(ptr=hb, layout=row_major(k, n))
            matmul[transpose_b=False, target="gpu"](tc, ta, tb, ctx)


def bf16_units_gemm_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C = op(A) . op(B)` on the bf16 unit. True if it took the product; False leaves every buffer untouched
    and the caller runs its FP32 route (the arm off, a product under the size rule, an unknown op, or a `ws`
    shorter than `bf16_units_workspace_floats`). ASYNCHRONOUS like the route it replaces: no wait; the caller
    keeps c, a, b and ws alive past its own synchronize, and has validated the operands (lengths, overlap)."""
    comptime if not GEMM_FAST_BF16_UNITS:
        return False
    if not bf16_units_takes(m, n, k):
        return False
    if op != OP_NN and op != OP_NT and op != OP_TN:
        return False
    if len(ws) < bf16_units_workspace_floats(m, n, k):
        return False
    bf16_units_product_ptr(
        ctx,
        c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        ws.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        m,
        n,
        k,
        op == OP_TN,
        op == OP_NT,
    )
    return True
