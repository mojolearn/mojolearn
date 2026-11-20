# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE MATRIX-UNITS RELAXATION OF `fast` ON NVIDIA: NEURAL FP32 PRODUCTS ON THE TENSOR CORES
(lane fx-fast-matrix-units, 2026-11-19; stage 2 of briefs-20261118/ideas-mode-policy-RESULT.md).

EXPERIMENT `-D MOJOLEARN_GEMM_FAST_MATRIX_UNITS=<kind>`, default OFF, A/B owed with quality beside time
(docs/apple-fast/EXPERIMENTS.md). Unverified: written without a compile.

  kind 1 = tf32   the vendor entry `linalg.matmul` (MAX stdlib): TF32 operands, FP32 accumulation. This is the
                  only floating tensor-core product the tree can reach today (the inventory is in the lane's
                  result file): it is the kernel `_fast_vendor_gemm` (gemm/checks/gemm_identical.mojo) calls in
                  the old non-identical builds, and its precision class is recorded in
                  checks/kernel_matrix.mojo::column_vendor_fp32_matmul_is_tf32.
  kind 2 = bf16   REFUSED at compile time: the bf16 matrix-unit route is its own define and policy row,
                  `-D MOJOLEARN_GEMM_FAST_BF16_UNITS=1` (gemm/fast_bf16_units.mojo, row BF16_UNITS).
  kind 3 = tf32, every neural-dispatch caller: kind 1, and also the callers that closed the vendor route
                  (`allow_vendor=False` through gemm/neural_dispatch.mojo: the Llama stage projections, the
                  CNN linear products, the MoE grouped products). They closed it against the old unbounded
                  vendor route of the non-identical builds; whether they can take the bounded, size-ruled
                  product is a quality measurement, so it is a second arm value with its own owed row and
                  never the default of kind 1.

WHERE IT IS LEGAL, AND ON WHICH VENDOR. Only in a `fast` build (`-D MOJOLEARN_MODE_FAST_ROWS`: the identical
kernels with the contract off); in a build that promises identical bits the define is a compile error
(checks/mode_policy.mojo `_row`, and `_matrix_units_checked`). The route is vendor-neutral source, one call of MAX
`linalg.matmul`, and is taken on NVIDIA (TF32 operands, the bound below) and on AMD (fp32 MFMA 16x16x4: operands
stay full fp32, so what is relaxed there is the vendor's fold order and K grouping; the bound below holds with
room). On Apple the products already have their matrix-unit route (gemm/afn_apple_fast.mojo), so the arm is off
there and the policy row points at that route; the host column has no matrix unit and stays exact. Measured on
NVIDIA first; AMD rides along untimed.

WHAT IS RELAXED. A routed product rounds each OPERAND to TF32 (a sign, 8 exponent bits, 10 explicit mantissa
bits; 11 significant bits) inside the matrix unit and accumulates in FP32. Per operand the relative error is at
most 2^-11 by round to nearest; the unit's own rounding of operands is not documented as nearest, so the checks
take the truncation bound 2^-10. For one cell c = sum_k a_k b_k:

    |c_tf32 - c_fp32|  <=  (2 * 2^-10 + 2^-20) * sum_k |a_k b_k|  +  the FP32 accumulation difference
                       ~=  1.95e-3 * sum_k |a_k b_k|

The bound is relative to the ABSOLUTE sum, not to c: where the terms cancel, the relative error of c is that
bound times the condition number sum|a b| / |sum a b|. With independent operand errors the typical error is
about 2^-11 sqrt(2 / 3) sqrt(sum (a_k b_k)^2), which for K terms of similar size is 4e-4 / sqrt(K) of the
absolute sum: 1.4e-5 at K = 768, 6e-6 at K = 4096. The mean of the errors is not zero when the unit truncates,
which is why a long product is checked against the bound and not against the typical value.
`matrix_units_cell_bound` is this bound as code; the native check (checks/native_tests/gemm_matrix_units.mojo)
holds every routed cell to it.

WHAT STAYS FP32, and why.
  - Every product under the size rule below: it cannot gain.
  - Under kind 1, every caller that closed the vendor route (`allow_vendor=False`): the Llama stage
    projections, the CNN linear products and the MoE grouped products. Kind 3 routes them.
  - The workspace-less entry `gemm/neural_dispatch.mojo::identical_gemm`: its callers are the bias column
    sums, the embedding gradient (a one-hot product: a gather, exact in FP32) and the Mamba projections.
  - The bias gradient (a 1 x n x m or m x 1 x n column sum against ones): M or N is 1, under the rule; it is a
    small fan-in sum whose terms do not cancel in pairs, so operand rounding would bias it.
  - Everything that is not a GEMM of the neural dispatch: the loss and its gradient, softmax, the
    normalisations (their divisions amplify an operand error by 1 / variance), the optimizer update and the
    moments (the update is a small difference accumulated over thousands of steps: a 1e-3 relative error per
    step there is a different learning rate), gradient clipping norms, the store epilogues (bias, activation).
  - The classical callers (PCA, OLS, Gram, GP): they import gemm/checks/gemm_identical.mojo directly and are not
    this arm.
  - OP_TN whose transpose does not fit the caller's workspace (`matrix_units_workspace_floats`), and every
    transposed-A product of the sequence executor (sequence/gemm_matrix_units_route.mojo): `linalg.matmul` has
    no transpose_a.

THE SIZE RULE (`matrix_units_takes`): sizes and the hardware unit only, no data shape.
  1. K >= 32. The TF32 instruction (`mma.sync m16n8k8`) folds 8 terms of k; under four instructions of k the
     cell's chain is at most 31 scalar multiply-adds, the scalar kernel is not bound by its chain there, and the
     unit's operand staging is not amortized.
  2. M >= 64 and N >= 64. One instruction writes a 16 x 8 fragment for one warp; with either edge under 64 a
     block of four warps is mostly cells past the matrix. This also keeps every vector product (the bias
     sums, a matrix-vector head) on FP32.
  3. M N K >= 2^27 multiply-adds. The identical scalar kernel runs 19.5 to 25 TFLOP/s on the L40S (about 1e13
     multiply-adds a second), so 2^27 of them are about 13 microseconds of arithmetic, the order of one launch.
     Under that the call is bound by its launch and a faster unit cannot show, so it keeps its bits' worth of
     precision.
  The constants are reasoned, not measured: the owed A/B times shapes on both sides of each
  (docs/apple-fast/EXPERIMENTS.md row).

COST PER CALL under the arm: three integer compares and one multiply chain in `matrix_units_takes`; no lookup,
allocation, lock or wait. With the define off nothing here is compiled into a launch.
"""
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.mode_policy import (
    CONTRACT_IDENTICAL, POLICY_BUILD, POLICY_BUILD_FAST_ROWS, ROUTE_GEMM_MATRIX_UNITS,
)
from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel
from gemm.contract import OP_NN, OP_NT, OP_TN

#: The define's values.
comptime MATRIX_UNITS_KIND_OFF = 0
comptime MATRIX_UNITS_KIND_TF32 = 1
comptime MATRIX_UNITS_KIND_BF16 = 2
comptime MATRIX_UNITS_KIND_TF32_ALL = 3
comptime MATRIX_UNITS_KIND = get_defined_int["MOJOLEARN_GEMM_FAST_MATRIX_UNITS", 0]()


def _matrix_units_checked() -> Bool:
    """The compile-time refusals. True when the TF32 arm is on."""
    comptime assert (
        MATRIX_UNITS_KIND == MATRIX_UNITS_KIND_OFF
        or MATRIX_UNITS_KIND == MATRIX_UNITS_KIND_TF32
        or MATRIX_UNITS_KIND == MATRIX_UNITS_KIND_BF16
        or MATRIX_UNITS_KIND == MATRIX_UNITS_KIND_TF32_ALL
    ), "MOJOLEARN_GEMM_FAST_MATRIX_UNITS takes 1 (tf32), 2 (bf16) or 3 (tf32, closed callers too)"
    comptime assert MATRIX_UNITS_KIND != MATRIX_UNITS_KIND_BF16, (
        "MOJOLEARN_GEMM_FAST_MATRIX_UNITS=2 (bf16): the bf16 matrix-unit route is its own define,"
        " -D MOJOLEARN_GEMM_FAST_BF16_UNITS=1 (gemm/fast_bf16_units.mojo)"
    )
    # The mode policy table is the single source (checks/mode_policy.mojo): the arm is a term of its
    # matrix-units row, which that module resolves from this same define and refuses in an identical build. The
    # arm compiles only in the fast-rows build kind with the row on, so MODE_POLICY_MASK carries the row's bit.
    comptime assert MATRIX_UNITS_KIND == MATRIX_UNITS_KIND_OFF or (
        POLICY_BUILD == POLICY_BUILD_FAST_ROWS and not CONTRACT_IDENTICAL
    ), (
        "MOJOLEARN_GEMM_FAST_MATRIX_UNITS is a relaxation of `fast`: it needs -D MOJOLEARN_MODE_FAST_ROWS and is"
        " refused in a build that promises identical bits"
    )
    # VENDOR-NEUTRAL (2026-11-19, lane fx-fast-nvidia-routes): the route is one call of MAX `linalg.matmul`, which
    # has NVIDIA (TF32 operands), AMD CDNA (fp32 MFMA 16x16x4: full fp32 operands, the vendor's K grouping) and
    # Apple backends. It is taken on NVIDIA and AMD. On Apple the neural products already have their matrix-unit
    # route (gemm/afn_apple_fast.mojo, AFN_GEMM_FP32_MMA), so the policy row points at that and this arm is off
    # there; the host column has no matrix unit. `ROUTE_GEMM_MATRIX_UNITS` (checks/mode_policy.mojo) is that
    # column rule and the row's bit, in one place.
    return ROUTE_GEMM_MATRIX_UNITS and (
        MATRIX_UNITS_KIND == MATRIX_UNITS_KIND_TF32 or MATRIX_UNITS_KIND == MATRIX_UNITS_KIND_TF32_ALL
    )


#: The arm: TF32 tensor-core products for the neural FP32 GEMMs of an NVIDIA `fast` build.
comptime GEMM_FAST_MATRIX_UNITS = _matrix_units_checked()
#: Kind 3: the arm also takes the neural-dispatch callers that closed the vendor route.
comptime GEMM_FAST_MATRIX_UNITS_ALL = GEMM_FAST_MATRIX_UNITS and MATRIX_UNITS_KIND == MATRIX_UNITS_KIND_TF32_ALL

#: rule 1: four `m16n8k8` instructions of k
comptime MATRIX_UNITS_MIN_K = 32
#: rule 2: four 16-row fragments, eight 8-column fragments
comptime MATRIX_UNITS_MIN_EDGE = 64
#: rule 3: 2^27 multiply-adds, about one launch of arithmetic at the scalar kernel's rate
comptime MATRIX_UNITS_MIN_WORK = 134217728

#: Operand relative error of one TF32 operand, taken as truncation to 11 significant bits.
comptime MATRIX_UNITS_OPERAND_REL = Float64(1.0) / Float64(1024.0)


@always_inline
def vendor_matmul_size_rule(m: Int, n: Int, k: Int) -> Bool:
    """The size rule of the module docstring, with no arm test: what a matrix unit can gain on. The family
    routes of the matrix-units row (decomposition, neighbors) share it and add their own terms."""
    if k < MATRIX_UNITS_MIN_K or m < MATRIX_UNITS_MIN_EDGE or n < MATRIX_UNITS_MIN_EDGE:
        return False
    return m * n * k >= MATRIX_UNITS_MIN_WORK


@always_inline
def matrix_units_takes(m: Int, n: Int, k: Int) -> Bool:
    """Whether an m x n x k product is large enough for the matrix unit (the size rule of the module
    docstring). Always False without the arm."""
    comptime if not GEMM_FAST_MATRIX_UNITS:
        return False
    return vendor_matmul_size_rule(m, n, k)


@always_inline
def matrix_units_workspace_floats(m: Int, n: Int, k: Int, op: Int) -> Int:
    """Floats of caller workspace the arm needs for this product: `m k` for an OP_TN it takes (the transpose of
    A), else 0. `GemmWorkspace.run` grows its buffer to this; a caller-owned workspace that is shorter keeps
    the product on FP32."""
    comptime if not GEMM_FAST_MATRIX_UNITS:
        return 0
    if op == OP_TN and matrix_units_takes(m, n, k):
        return m * k
    return 0


def matrix_units_cell_bound(abs_sum: Float64, k: Int) -> Float64:
    """The bound on |c_tf32 - c_fp32| for one cell whose absolute sum of terms is `abs_sum` (module docstring):
    both operands of every term truncated to TF32, plus the FP32 accumulation of K terms in either order
    (K 2^-24 of the absolute sum for each of the two results)."""
    var operand = (
        Float64(2.0) * MATRIX_UNITS_OPERAND_REL + MATRIX_UNITS_OPERAND_REL * MATRIX_UNITS_OPERAND_REL
    )
    var fold = Float64(2.0) * Float64(k) / Float64(16777216.0)
    return (operand + fold) * abs_sum


def matrix_units_gemm_into(
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
    """`C = op(A) . op(B)` on the matrix unit. True if it took the product; False leaves every buffer untouched
    and the caller runs its FP32 route. ASYNCHRONOUS like the route it replaces: stream-ordered on `ctx`, no
    wait; the caller keeps c, a, b and ws alive past its own synchronize. The caller has validated the
    operands (lengths, overlap).

    OP_NN and OP_NT are `linalg.matmul` with `transpose_b` False and True. OP_TN (`A[k x m]^T . B[k x n]`, the
    weight gradient of an NT layer) has no `transpose_a` in `linalg.matmul`: A is transposed into `ws` by
    `transpose_kernel` (data movement only, core/column_stats.mojo) and the product is the NN form over it;
    it is taken only when `ws` holds `m k` floats."""
    comptime if not GEMM_FAST_MATRIX_UNITS:
        return False
    if not matrix_units_takes(m, n, k):
        return False
    return vendor_matmul_gemm_into(ctx, c, a, b, ws, m, n, k, op)


def vendor_matmul_gemm_into(
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
    """The one matrix-unit product of the tree, with no arm and no size test: `C = op(A) . op(B)` on MAX
    `linalg.matmul` (NVIDIA: TF32 operands, FP32 accumulation; AMD: fp32 MFMA). The neural arm above and the
    family routes of the matrix-units row call it behind their own compile-time route constant and size rule,
    so it is instantiated only in a fast-rows build with a route on. Same contract as
    `matrix_units_gemm_into`: asynchronous, False (nothing touched) for an OP_TN whose transpose does not fit
    `ws` or an unknown op."""
    if op == OP_NT:
        var tc = TileTensor(c, row_major(m, n))
        var ta = TileTensor(a, row_major(m, k))
        var tb = TileTensor(b, row_major(n, k))
        matmul[transpose_b=True, target="gpu"](tc, ta, tb, ctx)
        return True
    if op == OP_NN:
        var tc2 = TileTensor(c, row_major(m, n))
        var ta2 = TileTensor(a, row_major(m, k))
        var tb2 = TileTensor(b, row_major(k, n))
        matmul[transpose_b=False, target="gpu"](tc2, ta2, tb2, ctx)
        return True
    if op == OP_TN:
        if len(ws) < m * k:
            return False
        # ws[m x k] = a[k x m]^T: the launch of core/gemm.mojo::gemm_tn_via_transpose
        ctx.enqueue_function[transpose_kernel](
            ws.unsafe_ptr(),
            a.unsafe_ptr(),
            Int32(k),
            Int32(m),
            grid_dim=(
                (m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
                min((k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ),
                1,
            ),
            block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
        )
        var tc3 = TileTensor(c, row_major(m, n))
        var ta3 = TileTensor(ws, row_major(m, k))
        var tb3 = TileTensor(b, row_major(k, n))
        matmul[transpose_b=False, target="gpu"](tc3, ta3, tb3, ctx)
        return True
    return False
