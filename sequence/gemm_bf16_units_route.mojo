# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LARGE SEQUENCE PRODUCTS ON THE BF16 MATRIX UNIT (lane fx-fast-bf16, 2026-11-20).

The sequence executor's side of `-D MOJOLEARN_GEMM_FAST_BF16_UNITS=<kind>` (gemm/fast_bf16_units.mojo: what the
arm is, where it is legal, the error bound, what stays FP32, the size rule). Default OFF, A/B owed with quality
beside time. With the define off nothing here is compiled into a launch and `DeviceExec._launch` is unchanged.

WHAT MOVES. The launches the TF32 arm takes (sequence/gemm_matrix_units_route.mojo): a plain dense GEMM launch
(OP_GEMM, OP_GEMM_EPI, OP_AGK_GEMM_EPI: forward, dInput and, with A stored transposed, the tape's weight
gradient X^T dY) that the bf16 size rule takes. Both operands are narrowed to bf16 in a scratch from the
executor's own allocator (`_alloc`: an exact-count reuse from its idle list or pool after the first step),
the product is one MAX `linalg.matmul` with FP32 accumulation and FP32 output, and an epilogue opcode runs its
follower in FP32 as OP_GEMM_EPI_TAIL on the stored words (bias and activation are not relaxed).

WHICH ADDRESSINGS (A(m, k) = p0[m i3 + k i4], B(k, n) = p1[k i5 + n i6]):
    A row-major m x k  (i3 = K, i4 = 1)   or   A stored transposed, k x m row-major (i3 = 1, i4 = M);
    B row-major k x n  (i5 = N, i6 = 1)   or   B row-major n x k (i5 = 1, i6 = K).
A transposed A costs nothing extra here: the narrow launch of A writes it transposed.
NOT ROUTED (today's route kept): the accumulate form (i7 != 0), a C that is not dense (i8 != N), the column-sum
launches (i11 != 0), every product the split-K rule or the coop rule takes first, every product under the size
rule, the host column.

COST PER CALL under the arm: eight integer tests, one allocator take and one free (short host list scans),
three launches (four under kind 2, one more for an epilogue); no wait.
"""
from sequence.ops import Args, OP_GEMM, OP_GEMM_EPI
from autograd.ops import OP_AGK_GEMM_EPI
from max.gpu.host import DeviceContext

from gemm.fast_bf16_units import GEMM_FAST_BF16_UNITS, bf16_units_takes, bf16_units_product_ptr

#: The arm on the sequence executor.
comptime SEQ_GEMM_BF16_UNITS = GEMM_FAST_BF16_UNITS

#: `seq_gemm_bf16_units_on`'s answers: keep today's route, or bit 0 = taken, bit 1 = B stored n x k, bit 2 = A
#: stored k x m.
comptime SBU_NONE = 0
comptime SBU_TAKEN = 1
comptime SBU_B_NT = 2
comptime SBU_A_T = 4


@always_inline
def seq_gemm_bf16_units_on[OP: Int](a: Args) -> Int:
    """SBU_* bits for one launch of OP_GEMM, OP_GEMM_EPI or OP_AGK_GEMM_EPI (i0 = M, i1 = N, i2 = K): the plain
    dense form, an A and a B each in one of their two row-major addressings, and the size rule. K >= 64 under
    the rule, so i3 = K and i3 = 1 cannot both hold. `DeviceExec._launch` asks after its split-K and coop tests."""
    comptime assert OP == OP_GEMM or OP == OP_GEMM_EPI or OP == OP_AGK_GEMM_EPI, "seq_gemm_bf16_units_on: a GEMM opcode"
    comptime if not SEQ_GEMM_BF16_UNITS:
        return SBU_NONE
    if a.i7 != 0 or a.i8 != a.i1 or a.i11 != 0:
        return SBU_NONE
    if not bf16_units_takes(a.i0, a.i1, a.i2):
        return SBU_NONE
    var form = SBU_TAKEN
    if a.i3 == 1 and a.i4 == a.i0:
        form += SBU_A_T
    elif a.i3 != a.i2 or a.i4 != 1:
        return SBU_NONE
    if a.i5 == 1 and a.i6 == a.i2:
        form += SBU_B_NT
    elif a.i5 != a.i1 or a.i6 != 1:
        return SBU_NONE
    return form


def seq_gemm_bf16_units_launch(
    ctx: DeviceContext, a: Args, form: Int, scratch: MutPointer[Float32, MutAnyOrigin]
) raises:
    """The product of a launch `seq_gemm_bf16_units_on` selected, queued on the bf16 unit: C = p2 (m x n,
    dense, FP32). `scratch` holds `bf16_units_workspace_floats(m, n, k)` floats and stays live to the
    executor's next sync (the caller frees it through the executor, which retires it until then). Asynchronous,
    no wait; the epilogue, if any, is the caller's next launch."""
    comptime if SEQ_GEMM_BF16_UNITS:
        bf16_units_product_ptr(
            ctx,
            a.p2.unsafe_origin_cast[MutAnyOrigin](),
            a.p0.unsafe_origin_cast[MutAnyOrigin](),
            a.p1.unsafe_origin_cast[MutAnyOrigin](),
            scratch,
            a.i0,
            a.i1,
            a.i2,
            (form & SBU_A_T) != 0,
            (form & SBU_B_NT) != 0,
        )
