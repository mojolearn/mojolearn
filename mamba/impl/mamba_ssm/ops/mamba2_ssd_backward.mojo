# SPDX-License-Identifier: Apache-2.0
"""First executable slice of the pinned Mamba-2 SSD backward.

This module implements only the reverse inter-chunk state recurrence from
DEVIATION 1342. It consumes the direct gradient of each chunk's incoming
state (produced later by the S18 backward), plus an optional final-state
gradient, and emits gradients of incoming states and chunk increments along
with per-cell scale products for DEVIATION 1345's later P*N contraction.

No intra-chunk, B/C, A/dt, or projection gradient is claimed here.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_exp, identical_mul_add
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NN
from mamba.checks.mamba2_fixture import M2_D_STATE, M2_HEADDIM
from mamba.impl.transformers.models.mamba.modeling_mamba import pinned_mul


comptime M2_SSD_BWD_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + M2_SSD_BWD_TPB - 1) // M2_SSD_BWD_TPB
    if g < 1:
        return 1
    return g


struct Mamba2SSDBackwardState(Movable):
    """Outputs of the reverse inter-chunk recurrence."""

    var d_pass: DeviceBuffer[DType.float32]  # [B,C,H,P,N]
    var d_cstate: DeviceBuffer[DType.float32]  # [B,C,H,P,N]
    var d_scale_product: DeviceBuffer[DType.float32]  # [B,C,H,P,N]
    var d_initial: DeviceBuffer[DType.float32]  # [B,H,P,N]

    def __init__(
        out self, ctx: DeviceContext, b: Int, nc: Int, nh: Int
    ) raises:
        var state_cells = b * nc * nh * M2_HEADDIM * M2_D_STATE
        if state_cells < 1:
            state_cells = 1
        var boundary_cells = b * nh * M2_HEADDIM * M2_D_STATE
        if boundary_cells < 1:
            boundary_cells = 1
        self.d_pass = ctx.enqueue_create_buffer[DType.float32](state_cells)
        self.d_cstate = ctx.enqueue_create_buffer[DType.float32](state_cells)
        self.d_scale_product = ctx.enqueue_create_buffer[DType.float32](
            state_cells
        )
        self.d_initial = ctx.enqueue_create_buffer[DType.float32](boundary_cells)


struct Mamba2SSDScaleReduction(Movable):
    """Storage for DEVIATION 1345's certified P*N contraction."""

    var d_scale: DeviceBuffer[DType.float32]  # [B,C,H]
    var ones_pn: DeviceBuffer[DType.float32]  # [P*N]
    var workspace: DeviceBuffer[DType.float32]
    var d_dacs_state: DeviceBuffer[DType.float32]  # [B,H,C,Q]

    def __init__(
        out self, ctx: DeviceContext, b: Int, nc: Int, nh: Int, qv: Int
    ) raises:
        var rows = b * nc * nh
        if rows < 1:
            rows = 1
        var pn = M2_HEADDIM * M2_D_STATE
        self.d_scale = ctx.enqueue_create_buffer[DType.float32](rows)
        self.ones_pn = ctx.enqueue_create_buffer[DType.float32](pn)
        self.ones_pn.enqueue_fill(Float32(1.0))
        var ws = identical_gemm_workspace_max_floats(rows, 1, pn)
        self.workspace = ctx.enqueue_create_buffer[DType.float32](ws)
        self.d_dacs_state = ctx.enqueue_create_buffer[DType.float32](
            rows * qv
        )
        self.d_dacs_state.enqueue_fill(Float32(0.0))


def mamba2_scale_to_dacs_kernel(
    d_dacs_state: MutPointer[Float32, MutAnyOrigin],
    d_scale: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nc_in: Int32,
    nh_in: Int32,
    q_in: Int32,
):
    """Chain dscale through scale=exp(dacs[last]); other q cells stay zero."""
    var b = Int(b_in)
    var nc = Int(nc_in)
    var nh = Int(nh_in)
    var qv = Int(q_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= b * nc * nh:
        return
    var bb = row // (nc * nh)
    var rem = row - bb * nc * nh
    var c = rem // nh
    var h = rem - c * nh
    var dacs_idx = ((bb * nh + h) * nc + c) * qv + (qv - 1)
    var scale = ftz(identical_exp(ftz(dacs.unsafe_load(dacs_idx))))
    d_dacs_state.unsafe_store(
        dacs_idx, ftz(pinned_mul(ftz(d_scale.unsafe_load(row)), scale))
    )


def mamba2_reverse_chunk_state_kernel(
    d_pass: MutPointer[Float32, MutAnyOrigin],
    d_cstate: MutPointer[Float32, MutAnyOrigin],
    d_scale_product: MutPointer[Float32, MutAnyOrigin],
    d_initial: MutPointer[Float32, MutAnyOrigin],
    direct_d_pass: MutPointer[Float32, MutAnyOrigin],
    d_final: MutPointer[Float32, MutAnyOrigin],
    pass_states: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """Reverse S17 with one thread owning one complete chunk chain.

    Forward is ``h_after[c] = fma(scale[c], pass[c], cstate[c])`` and
    ``pass[c+1] = h_after[c]``. Therefore descending c writes
    ``d_cstate[c] = carry`` and
    ``d_pass[c] = fma(scale[c], carry, direct[c])``. The fused join and
    descending direction are the pinned recurrence, not an execution choice.
    """
    var b = Int(b_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var pn = M2_HEADDIM * M2_D_STATE
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * pn:
        return
    var bb = cell // (nh * pn)
    var rem = cell - bb * nh * pn
    var hh = rem // pn
    var i = rem - hh * pn
    var carry = ftz(d_final.unsafe_load(cell))
    var c = nc - 1
    while c >= 0:
        var state_idx = (((bb * nc + c) * nh + hh) * pn) + i
        var scale = ftz(
            identical_exp(
                ftz(
                    dacs.unsafe_load(
                        ((bb * nh + hh) * nc + c) * qv + (qv - 1)
                    )
                )
            )
        )
        d_cstate.unsafe_store(state_idx, carry)
        d_scale_product.unsafe_store(
            state_idx,
            ftz(pinned_mul(carry, ftz(pass_states.unsafe_load(state_idx)))),
        )
        var direct = ftz(direct_d_pass.unsafe_load(state_idx))
        carry = ftz(identical_mul_add(scale, carry, direct))
        d_pass.unsafe_store(state_idx, carry)
        c -= 1
    d_initial.unsafe_store(cell, carry)


def mamba2_reverse_chunk_state_into(
    ctx: DeviceContext,
    mut out: Mamba2SSDBackwardState,
    mut direct_d_pass: DeviceBuffer[DType.float32],
    mut d_final: DeviceBuffer[DType.float32],
    mut pass_states: DeviceBuffer[DType.float32],
    mut dacs: DeviceBuffer[DType.float32],
    b: Int,
    nh: Int,
    nc: Int,
    qv: Int,
) raises:
    """Enqueue the reverse inter-chunk recurrence; all buffers caller-owned."""
    var chains = b * nh * M2_HEADDIM * M2_D_STATE
    if chains < 1 or nc < 1:
        return
    ctx.enqueue_function[mamba2_reverse_chunk_state_kernel](
        out.d_pass.unsafe_ptr(),
        out.d_cstate.unsafe_ptr(),
        out.d_scale_product.unsafe_ptr(),
        out.d_initial.unsafe_ptr(),
        direct_d_pass.unsafe_ptr(),
        d_final.unsafe_ptr(),
        pass_states.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(chains), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )


def mamba2_reduce_scale_product_into(
    ctx: DeviceContext,
    mut reduction: Mamba2SSDScaleReduction,
    mut state: Mamba2SSDBackwardState,
    mut dacs: DeviceBuffer[DType.float32],
    b: Int,
    nc: Int,
    nh: Int,
    qv: Int,
) raises:
    """Reduce P*N products into one scale gradient per chunk/head.

    ``d_scale_product`` is row-major `[B,C,H,P,N]`; treating its first three
    axes as GEMM rows makes contracted index `p*N+n`, exactly DEVIATION 1345's
    pinned p-major linearization. GEMM-v1 supplies 64 leaves of 128 and its
    six-level balanced fold. No local reduction spelling exists here.
    """
    var rows = b * nc * nh
    if rows < 1:
        return
    comptime pn = M2_HEADDIM * M2_D_STATE
    identical_gemm_into(
        ctx,
        reduction.d_scale,
        state.d_scale_product,
        reduction.ones_pn,
        reduction.workspace,
        rows,
        1,
        pn,
        OP_NN,
    )
    ctx.enqueue_function[mamba2_scale_to_dacs_kernel](
        reduction.d_dacs_state.unsafe_ptr(),
        reduction.d_scale.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b),
        Int32(nc),
        Int32(nh),
        Int32(qv),
        grid_dim=(_grid(rows), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
