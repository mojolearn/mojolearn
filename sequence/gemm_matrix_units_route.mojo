# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LARGE SEQUENCE PRODUCTS ON THE MATRIX UNIT (lane fx-fast-matrix-units, 2026-11-19).

The sequence executor's side of the matrix-units route of a `fast` build on NVIDIA and AMD
(gemm/fast_matrix_units.mojo: what the route is, where it is compiled, the error bound, the size rule). DEFAULT
there since 2026-11-20 (lane fx-verdicts6, A/B abf2 on the NVIDIA L40S, arm fast-matrix-units-tf32-all: the MLP
rows that run through this executor, mlp-clf and mlp-reg fit 0.96-0.97 and infer 0.85-0.95, quality equal);
`-D MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF` keeps the exact route. Where the route constant is False (every
identical build) nothing here is compiled into a launch and `DeviceExec._launch` is unchanged.

WHAT MOVES. A plain dense GEMM launch of the executor (OP_GEMM, OP_GEMM_EPI, OP_AGK_GEMM_EPI: the forward and
dInput products of the tape and of the hand MLP) that the size rule takes runs `linalg.matmul` (TF32 operands,
FP32 accumulation) in place of the 16 x 16 tile or the packed body. An epilogue opcode then runs its follower
as its own launch, OP_GEMM_EPI_TAIL on the stored words, in FP32: the bias add and the activation are not
relaxed. This is the launch pair `HostExec` and the packed route (sequence/gemm_packed_route.mojo) already run.

WHICH ADDRESSINGS. The executor's operands are strided: A(m, k) = p0[m i3 + k i4], B(k, n) = p1[k i5 + n i6].
`linalg.matmul` takes contiguous row-major operands and has `transpose_b` only, so the route takes
    A row-major m x k      (i3 = K, i4 = 1), and
    B row-major k x n      (i5 = N, i6 = 1)   -> transpose_b False, or
    B row-major n x k      (i5 = 1, i6 = K)   -> transpose_b True.
A TRANSPOSED A (i3 = 1, i4 = M: the weight gradient X^T dY of the tape, a third of a training step's product
work) is routed since lane fx-fast-nvidia-routes (2026-11-19): `linalg.matmul` documents `transpose_a` as
unsupported, so `DeviceExec._launch` takes an m k scratch from the executor's own allocator (`_alloc`: the
idle list of an owned executor, the pool of a pooled one; an exact-count reuse after the first step),
`transpose_kernel` (data movement only) writes A row-major into it, the product is the row-major form over
it, and the scratch is returned with `free` (it is released at the executor's next sync, after the queue
that reads it has drained). Per call: one allocator take and one free (short host list scans), two launches.
NOT ROUTED (today's FP32 route kept): the accumulate form (i7 != 0), a C that is not dense
(i8 != N), the column-sum launches (i11 != 0), every product the split-K rule or the coop rule takes first
(they are tested before this rule), every product under the size rule, the host column.

COST PER CALL on the route: seven integer tests, then one launch (two with a follower); no lookup, allocation,
lock or wait.
"""
from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceContext

from sequence.ops import Args, OP_GEMM, OP_GEMM_EPI
from autograd.ops import OP_AGK_GEMM_EPI
from gemm.fast_matrix_units import GEMM_FAST_MATRIX_UNITS, matrix_units_takes

#: The route on the sequence executor: a `fast` build on NVIDIA or AMD (one compile-time constant).
comptime SEQ_GEMM_MATRIX_UNITS = GEMM_FAST_MATRIX_UNITS

#: `seq_gemm_matrix_units_on`'s answers: keep today's route, B stored k x n, B stored n x k.
comptime SMU_NONE = 0
comptime SMU_NN = 1
comptime SMU_NT = 2


@always_inline
def seq_gemm_matrix_units_on[OP: Int](a: Args) -> Int:
    """SMU_* for one launch of OP_GEMM, OP_GEMM_EPI or OP_AGK_GEMM_EPI (i0 = M, i1 = N, i2 = K): the plain
    dense form with a row-major A, a B in one of the two row-major addressings, and the size rule.
    `DeviceExec._launch` asks after its split-K and coop tests."""
    comptime assert OP == OP_GEMM or OP == OP_GEMM_EPI or OP == OP_AGK_GEMM_EPI, "seq_gemm_matrix_units_on: a GEMM opcode"
    comptime if not SEQ_GEMM_MATRIX_UNITS:
        return SMU_NONE
    if a.i7 != 0 or a.i8 != a.i1 or a.i11 != 0:
        return SMU_NONE
    if a.i3 != a.i2 or a.i4 != 1:
        return SMU_NONE
    if not matrix_units_takes(a.i0, a.i1, a.i2):
        return SMU_NONE
    if a.i5 == a.i1 and a.i6 == 1:
        return SMU_NN
    if a.i5 == 1 and a.i6 == a.i2:
        return SMU_NT
    return SMU_NONE


def seq_gemm_matrix_units_launch(ctx: DeviceContext, a: Args, form: Int) raises:
    """The product of a GEMM launch `seq_gemm_matrix_units_on` selected (form = SMU_NN or SMU_NT), queued on
    `linalg.matmul`: C = p2 (m x n, dense), A = p0 (m x k), B = p1 (k x n, or n x k for SMU_NT). One
    asynchronous launch, no workspace, no wait; the pointers are the executor's own (live to its sync). The
    epilogue, if any, is the caller's next launch."""
    comptime if SEQ_GEMM_MATRIX_UNITS:
        var pc = a.p2.unsafe_origin_cast[MutAnyOrigin]()
        var pa = a.p0.unsafe_origin_cast[MutAnyOrigin]()
        var pb = a.p1.unsafe_origin_cast[MutAnyOrigin]()
        var tc = TileTensor(ptr=pc, layout=row_major(a.i0, a.i1))
        var ta = TileTensor(ptr=pa, layout=row_major(a.i0, a.i2))
        if form == SMU_NT:
            var tbt = TileTensor(ptr=pb, layout=row_major(a.i1, a.i2))
            matmul[transpose_b=True, target="gpu"](tc, ta, tbt, ctx)
            return
        var tb = TileTensor(ptr=pb, layout=row_major(a.i2, a.i1))
        matmul[transpose_b=False, target="gpu"](tc, ta, tb, ctx)


@always_inline
def seq_gemm_matrix_units_ta_on[OP: Int](a: Args) -> Int:
    """SMU_* for a launch whose A is stored transposed (k x m row-major: i3 = 1, i4 = M), else SMU_NONE. The
    same plain dense form, B addressings and size rule as `seq_gemm_matrix_units_on`, which is asked first (a
    row-major A never reaches this test as a transposed one: K >= 32 under the rule, so i3 = 1 is not i3 = K)."""
    comptime assert OP == OP_GEMM or OP == OP_GEMM_EPI or OP == OP_AGK_GEMM_EPI, "seq_gemm_matrix_units_ta_on: a GEMM opcode"
    comptime if not SEQ_GEMM_MATRIX_UNITS:
        return SMU_NONE
    if a.i7 != 0 or a.i8 != a.i1 or a.i11 != 0:
        return SMU_NONE
    if a.i3 != 1 or a.i4 != a.i0:
        return SMU_NONE
    if not matrix_units_takes(a.i0, a.i1, a.i2):
        return SMU_NONE
    if a.i5 == a.i1 and a.i6 == 1:
        return SMU_NN
    if a.i5 == 1 and a.i6 == a.i2:
        return SMU_NT
    return SMU_NONE


def seq_gemm_matrix_units_ta_launch(
    ctx: DeviceContext, a: Args, form: Int, scratch: MutPointer[Float32, MutAnyOrigin]
) raises:
    """The product of a launch `seq_gemm_matrix_units_ta_on` selected: `scratch[m x k] = p0[k x m]^T`
    (`transpose_kernel`, the launch of core/gemm.mojo::gemm_tn_via_transpose), then C = scratch . B on
    `linalg.matmul`. Two asynchronous launches, no wait; `scratch` holds m k floats and stays live to the
    executor's next sync (the caller frees it through the executor, which retires it until then)."""
    comptime if SEQ_GEMM_MATRIX_UNITS:
        var m = a.i0
        var k = a.i2
        var pa = a.p0.unsafe_origin_cast[MutAnyOrigin]()
        var ps = scratch
        ctx.enqueue_function[transpose_kernel](
            ps,
            pa,
            Int32(k),
            Int32(m),
            grid_dim=(
                (m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
                min((k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ),
                1,
            ),
            block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
        )
        var pc = a.p2.unsafe_origin_cast[MutAnyOrigin]()
        var pb = a.p1.unsafe_origin_cast[MutAnyOrigin]()
        var tc = TileTensor(ptr=pc, layout=row_major(m, a.i1))
        var ta = TileTensor(ptr=ps, layout=row_major(m, k))
        if form == SMU_NT:
            var tbt = TileTensor(ptr=pb, layout=row_major(a.i1, k))
            matmul[transpose_b=True, target="gpu"](tc, ta, tbt, ctx)
            return
        var tb = TileTensor(ptr=pb, layout=row_major(k, a.i1))
        matmul[transpose_b=False, target="gpu"](tc, ta, tb, ctx)
