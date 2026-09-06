# SPDX-License-Identifier: Apache-2.0
"""Implemented tail of the Mamba-2 block backward.

This module deliberately claims only the residual/output-projection seam:

    residual.out = input.x + out_proj(gnorm.out)

Given ``d_residual``, it computes the output-projection derivatives, then the
gated RMSNorm derivatives ``d_gate`` and ``d_norm_weight``. The residual
contribution to ``d_x`` is exactly ``d_residual`` and remains caller-owned;
the incomplete upstream path is not silently added here. The SiLU gate is
also differentiated, producing the SSD-input and z-slice gradients. The
D-skip seam produces `d_scan`, the partial x contribution, and `dD`. No SSD
recurrence, convolution, input projection, or block-norm derivative is claimed.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul_add,
    identical_rsqrt,
    identical_sigmoid,
    identical_silu,
)
from mamba.checks.mamba2_backward import (
    PROJ2_OUT,
    PROJ2_IN,
    RED2_GNORM_W,
    RED2_NORM_W,
    RED2_D,
    mamba2_backward_ones_floats,
    mamba2_backward_proj_a_into,
    mamba2_backward_proj_a_workspace_max_floats,
    mamba2_backward_proj_b_into,
    mamba2_backward_proj_b_workspace_max_floats,
    mamba2_backward_reduce_into,
    mamba2_backward_reduce_workspace_max_floats,
)
from mamba.checks.mamba2_fixture import Mamba2Dims
from mamba.impl.transformers.models.mamba.modeling_mamba import pinned_mul


comptime M2_BWD_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + M2_BWD_TPB - 1) // M2_BWD_TPB
    if g < 1:
        return 1
    return g


def mamba2_gnorm_backward_kernel(
    d_gate: MutPointer[Float32, MutAnyOrigin],
    weight_product: MutPointer[Float32, MutAnyOrigin],
    d_gnorm: MutPointer[Float32, MutAnyOrigin],
    gate: MutPointer[Float32, MutAnyOrigin],
    sumsq: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    width_in: Int32,
):
    """Backward of S21's gated RMSNorm, one thread per token row.

    The feature reduction is serial ascending and stays in one thread. The
    weight product is materialized separately, then reduced over tokens by
    the existing ones-vector GEMM-v1 route.
    """
    var m = Int(m_in)
    var width = Int(width_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return
    var rstd = ftz(
        identical_rsqrt(
            ftz(
                ftz(identical_div(sumsq.unsafe_load(t), Float32(width)))
                + Float32(0.00001)
            )
        )
    )
    var inner_dot = Float32(0.0)
    for j in range(width):
        var dinner = ftz(
            pinned_mul(
                ftz(d_gnorm.unsafe_load(t * width + j)),
                ftz(weight.unsafe_load(j)),
            )
        )
        inner_dot = ftz(
            identical_mul_add(
                dinner,
                ftz(gate.unsafe_load(t * width + j)),
                inner_dot,
            )
        )
    var r2 = ftz(pinned_mul(rstd, rstd))
    var scale = ftz(
        identical_div(ftz(pinned_mul(r2, rstd)), Float32(width))
    )
    for j in range(width):
        var cell = t * width + j
        var dj = ftz(d_gnorm.unsafe_load(cell))
        var gj = ftz(gate.unsafe_load(cell))
        var dinner = ftz(pinned_mul(dj, ftz(weight.unsafe_load(j))))
        var first = ftz(pinned_mul(rstd, dinner))
        var second = ftz(
            pinned_mul(ftz(pinned_mul(scale, gj)), inner_dot)
        )
        d_gate.unsafe_store(cell, ftz(ftz(first) - ftz(second)))
        var normalized = ftz(pinned_mul(gj, rstd))
        weight_product.unsafe_store(cell, ftz(pinned_mul(dj, normalized)))


def mamba2_silu_gate_backward_kernel(
    d_skip: MutPointer[Float32, MutAnyOrigin],
    d_z: MutPointer[Float32, MutAnyOrigin],
    d_gate: MutPointer[Float32, MutAnyOrigin],
    skip: MutPointer[Float32, MutAnyOrigin],
    in_proj: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    width_in: Int32,
    dip_in: Int32,
):
    """Backward of ``gate = skip * silu(z)``; z is in_proj's first slice."""
    var n = Int(n_in)
    var width = Int(width_in)
    var dip = Int(dip_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n:
        return
    var t = cell // width
    var j = cell % width
    var dg = ftz(d_gate.unsafe_load(cell))
    var sk = ftz(skip.unsafe_load(cell))
    var z = ftz(in_proj.unsafe_load(t * dip + j))
    d_skip.unsafe_store(cell, ftz(pinned_mul(dg, identical_silu(z))))
    var sig = ftz(identical_sigmoid(z))
    var middle = ftz(
        identical_mul_add(z, ftz(Float32(1.0) - sig), Float32(1.0))
    )
    var prime = ftz(pinned_mul(sig, middle))
    d_z.unsafe_store(cell, ftz(pinned_mul(ftz(pinned_mul(dg, sk)), prime)))


def mamba2_d_skip_backward_kernel(
    d_scan: MutPointer[Float32, MutAnyOrigin],
    d_x_from_d: MutPointer[Float32, MutAnyOrigin],
    d_d_product: MutPointer[Float32, MutAnyOrigin],
    d_skip: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    d_weight: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
    width_in: Int32,
    conv_width_in: Int32,
    head_dim_in: Int32,
):
    """Backward of S20 ``skip = scan + D[h] * x[h,p]``.

    One thread owns a token/head and folds p ascending for the per-token dD
    product. The later token reduction is the existing GEMM-v1 route.
    """
    var cells = Int(cells_in)
    var width = Int(width_in)
    var conv_width = Int(conv_width_in)
    var p_dim = Int(head_dim_in)
    var th = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if th >= cells:
        return
    var t = th // (width // p_dim)
    var h = th % (width // p_dim)
    var acc = Float32(0.0)
    for p in range(p_dim):
        var cell = t * width + h * p_dim + p
        var ds = ftz(d_skip.unsafe_load(cell))
        var xv = ftz(xbc.unsafe_load(t * conv_width + h * p_dim + p))
        d_scan.unsafe_store(cell, ds)
        d_x_from_d.unsafe_store(
            cell, ftz(pinned_mul(ds, ftz(d_weight.unsafe_load(h))))
        )
        acc = ftz(identical_mul_add(ds, xv, acc))
    d_d_product.unsafe_store(th, acc)


struct Mamba2BackwardTail(Movable):
    """Caller-owned outputs for the implemented tail seam."""

    var d_gnorm: DeviceBuffer[DType.float32]  # [M, d_inner]
    var d_w_out: DeviceBuffer[DType.float32]  # [d_model, d_inner]
    var d_gate: DeviceBuffer[DType.float32]  # [M, d_inner]
    var d_gnorm_w: DeviceBuffer[DType.float32]  # [d_inner]
    var d_skip: DeviceBuffer[DType.float32]  # [M, d_inner]
    var d_z: DeviceBuffer[DType.float32]  # [M, d_inner]
    var d_scan: DeviceBuffer[DType.float32]  # [M, d_inner]
    var d_x_from_d: DeviceBuffer[DType.float32]  # [M, d_inner], partial
    var d_d_product: DeviceBuffer[DType.float32]  # [M, H]
    var d_d: DeviceBuffer[DType.float32]  # [H]
    var d_in_proj: DeviceBuffer[DType.float32]
    var d_norm: DeviceBuffer[DType.float32]
    var d_w_in: DeviceBuffer[DType.float32]
    var d_block_x: DeviceBuffer[DType.float32]
    var d_block_w: DeviceBuffer[DType.float32]
    var block_product: DeviceBuffer[DType.float32]
    var gnorm_weight_product: DeviceBuffer[DType.float32]  # [M, d_inner]
    var ones: DeviceBuffer[DType.float32]  # [M]
    var reduction_workspace: DeviceBuffer[DType.float32]
    var workspace: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, dims: Mamba2Dims, m: Int
    ) raises:
        var ng = m * dims.d_inner
        if ng < 1:
            ng = 1
        var nw = dims.d_model * dims.d_inner
        if nw < 1:
            nw = 1
        self.d_gnorm = ctx.enqueue_create_buffer[DType.float32](ng)
        self.d_w_out = ctx.enqueue_create_buffer[DType.float32](nw)
        self.d_gate = ctx.enqueue_create_buffer[DType.float32](ng)
        self.d_skip = ctx.enqueue_create_buffer[DType.float32](ng)
        self.d_z = ctx.enqueue_create_buffer[DType.float32](ng)
        self.d_scan = ctx.enqueue_create_buffer[DType.float32](ng)
        self.d_x_from_d = ctx.enqueue_create_buffer[DType.float32](ng)
        var ndp = m * dims.nheads
        if ndp < 1:
            ndp = 1
        self.d_d_product = ctx.enqueue_create_buffer[DType.float32](ndp)
        var nd = dims.nheads
        if nd < 1:
            nd = 1
        self.d_d = ctx.enqueue_create_buffer[DType.float32](nd)
        self.d_in_proj = ctx.enqueue_create_buffer[DType.float32](m*dims.d_in_proj())
        self.d_norm = ctx.enqueue_create_buffer[DType.float32](m*dims.d_model)
        self.d_w_in = ctx.enqueue_create_buffer[DType.float32](dims.d_in_proj()*dims.d_model)
        self.d_block_x = ctx.enqueue_create_buffer[DType.float32](m*dims.d_model)
        self.d_block_w = ctx.enqueue_create_buffer[DType.float32](dims.d_model)
        self.block_product = ctx.enqueue_create_buffer[DType.float32](m*dims.d_model)
        var ngw = dims.d_inner
        if ngw < 1:
            ngw = 1
        self.d_gnorm_w = ctx.enqueue_create_buffer[DType.float32](ngw)
        self.gnorm_weight_product = ctx.enqueue_create_buffer[DType.float32](ng)
        var no = mamba2_backward_ones_floats(m)
        if no < 1:
            no = 1
        self.ones = ctx.enqueue_create_buffer[DType.float32](no)
        self.ones.enqueue_fill(Float32(1.0))
        var wa = mamba2_backward_proj_a_workspace_max_floats(
            PROJ2_OUT, dims, m
        )
        var wb = mamba2_backward_proj_b_workspace_max_floats(
            PROJ2_OUT, dims, m
        )
        var ws = wa
        if wb > ws:
            ws = wb
        wa = mamba2_backward_proj_a_workspace_max_floats(PROJ2_IN, dims, m)
        if wa > ws:
            ws = wa
        wb = mamba2_backward_proj_b_workspace_max_floats(PROJ2_IN, dims, m)
        if wb > ws:
            ws = wb
        if ws < 1:
            ws = 1
        self.workspace = ctx.enqueue_create_buffer[DType.float32](ws)
        var rws = mamba2_backward_reduce_workspace_max_floats(
            RED2_GNORM_W, dims, m
        )
        if rws < 1:
            rws = 1
        self.reduction_workspace = ctx.enqueue_create_buffer[DType.float32](rws)


def mamba2_backward_tail_into(
    ctx: DeviceContext,
    mut out: Mamba2BackwardTail,
    mut d_residual: DeviceBuffer[DType.float32],
    mut gnorm_out: DeviceBuffer[DType.float32],
    mut w_out: DeviceBuffer[DType.float32],
    dims: Mamba2Dims,
    m: Int,
) raises:
    """Enqueue the two exact derivatives of the output projection.

    ``out_proj`` is ``OP_NT`` at ``[M, d_model, d_inner]``. These launchers
    derive both transpose routes from that forward call, so this composer
    introduces no independent shape table or floating-point order.
    """
    mamba2_backward_proj_a_into(
        ctx,
        out.d_gnorm,
        d_residual,
        w_out,
        out.workspace,
        PROJ2_OUT,
        dims,
        m,
    )
    mamba2_backward_proj_b_into(
        ctx,
        out.d_w_out,
        d_residual,
        gnorm_out,
        out.workspace,
        PROJ2_OUT,
        dims,
        m,
    )


def mamba2_backward_gnorm_into(
    ctx: DeviceContext,
    mut out: Mamba2BackwardTail,
    mut gate: DeviceBuffer[DType.float32],
    mut sumsq: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    dims: Mamba2Dims,
    m: Int,
) raises:
    """Continue the partial backward through S21 gated RMSNorm.

    Requires ``out.d_gnorm`` from ``mamba2_backward_tail_into``. Produces
    ``d_gate`` and ``d_gnorm_w``; it intentionally stops before the SiLU gate
    and SSD output join.
    """
    if m < 1:
        return
    ctx.enqueue_function[mamba2_gnorm_backward_kernel](
        out.d_gate.unsafe_ptr(),
        out.gnorm_weight_product.unsafe_ptr(),
        out.d_gnorm.unsafe_ptr(),
        gate.unsafe_ptr(),
        sumsq.unsafe_ptr(),
        weight.unsafe_ptr(),
        Int32(m),
        Int32(dims.d_inner),
        grid_dim=(_grid(m), 1, 1),
        block_dim=(M2_BWD_TPB, 1, 1),
    )
    mamba2_backward_reduce_into(
        ctx,
        out.d_gnorm_w,
        out.gnorm_weight_product,
        out.ones,
        out.reduction_workspace,
        RED2_GNORM_W,
        dims,
        m,
    )


def mamba2_backward_silu_gate_into(
    ctx: DeviceContext,
    mut out: Mamba2BackwardTail,
    mut skip: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    dims: Mamba2Dims,
    m: Int,
) raises:
    """Continue through S21's pre-norm SiLU gate and stop at the SSD seam."""
    var n = m * dims.d_inner
    if n < 1:
        return
    ctx.enqueue_function[mamba2_silu_gate_backward_kernel](
        out.d_skip.unsafe_ptr(),
        out.d_z.unsafe_ptr(),
        out.d_gate.unsafe_ptr(),
        skip.unsafe_ptr(),
        in_proj.unsafe_ptr(),
        Int32(n),
        Int32(dims.d_inner),
        Int32(dims.d_in_proj()),
        grid_dim=(_grid(n), 1, 1),
        block_dim=(M2_BWD_TPB, 1, 1),
    )


def mamba2_backward_d_skip_into(
    ctx: DeviceContext,
    mut out: Mamba2BackwardTail,
    mut xbc: DeviceBuffer[DType.float32],
    mut d_weight: DeviceBuffer[DType.float32],
    dims: Mamba2Dims,
    m: Int,
) raises:
    """Continue through S20 and stop at the chunked SSD scan output."""
    var cells = m * dims.nheads
    if cells < 1:
        return
    ctx.enqueue_function[mamba2_d_skip_backward_kernel](
        out.d_scan.unsafe_ptr(),
        out.d_x_from_d.unsafe_ptr(),
        out.d_d_product.unsafe_ptr(),
        out.d_skip.unsafe_ptr(),
        xbc.unsafe_ptr(),
        d_weight.unsafe_ptr(),
        Int32(cells),
        Int32(dims.d_inner),
        Int32(dims.conv_dim()),
        Int32(dims.d_inner // dims.nheads),
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(M2_BWD_TPB, 1, 1),
    )
    mamba2_backward_reduce_into(
        ctx,
        out.d_d,
        out.d_d_product,
        out.ones,
        out.reduction_workspace,
        RED2_D,
        dims,
        m,
    )


def mamba2_pack_inproj_grad_kernel(
    dst: MutPointer[Float32, MutAnyOrigin], z: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin], dt: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32, di_in: Int32, cd_in: Int32, dip_in: Int32, nh_in: Int32,
):
    var m=Int(m_in); var di=Int(di_in); var cd=Int(cd_in); var dip=Int(dip_in); var nh=Int(nh_in)
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i>=m*dip: return
    var r=i//dip; var c=i-r*dip
    if c<di: dst.unsafe_store(i,z.unsafe_load(r*di+c))
    elif c<di+cd: dst.unsafe_store(i,xbc.unsafe_load(r*cd+c-di))
    else: dst.unsafe_store(i,dt.unsafe_load(r*nh+c-di-cd))


def mamba2_backward_input_projection_into(
    ctx: DeviceContext, mut out: Mamba2BackwardTail,
    mut d_xbc: DeviceBuffer[DType.float32], mut d_dt: DeviceBuffer[DType.float32],
    mut norm: DeviceBuffer[DType.float32], mut weight: DeviceBuffer[DType.float32],
    dims: Mamba2Dims, m: Int,
) raises:
    var n=m*dims.d_in_proj()
    ctx.enqueue_function[mamba2_pack_inproj_grad_kernel](out.d_in_proj.unsafe_ptr(),out.d_z.unsafe_ptr(),d_xbc.unsafe_ptr(),d_dt.unsafe_ptr(),Int32(m),Int32(dims.d_inner),Int32(dims.conv_dim()),Int32(dims.d_in_proj()),Int32(dims.nheads),grid_dim=(_grid(n),1,1),block_dim=(M2_BWD_TPB,1,1))
    mamba2_backward_proj_a_into(ctx,out.d_norm,out.d_in_proj,weight,out.workspace,PROJ2_IN,dims,m)
    mamba2_backward_proj_b_into(ctx,out.d_w_in,out.d_in_proj,norm,out.workspace,PROJ2_IN,dims,m)


def mamba2_residual_join_kernel(dst: MutPointer[Float32,MutAnyOrigin], b: MutPointer[Float32,MutAnyOrigin], n: Int32):
    var i=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i<Int(n): dst.unsafe_store(i,ftz(ftz(dst.unsafe_load(i))+ftz(b.unsafe_load(i))))


def mamba2_backward_block_norm_into(ctx: DeviceContext, mut out: Mamba2BackwardTail, mut residual: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32], mut sumsq: DeviceBuffer[DType.float32], mut weight: DeviceBuffer[DType.float32], dims: Mamba2Dims, m: Int) raises:
    ctx.enqueue_function[mamba2_gnorm_backward_kernel](out.d_block_x.unsafe_ptr(),out.block_product.unsafe_ptr(),out.d_norm.unsafe_ptr(),x.unsafe_ptr(),sumsq.unsafe_ptr(),weight.unsafe_ptr(),Int32(m),Int32(dims.d_model),grid_dim=(_grid(m),1,1),block_dim=(M2_BWD_TPB,1,1))
    mamba2_backward_reduce_into(ctx,out.d_block_w,out.block_product,out.ones,out.reduction_workspace,RED2_NORM_W,dims,m)
    ctx.enqueue_function[mamba2_residual_join_kernel](out.d_block_x.unsafe_ptr(),residual.unsafe_ptr(),Int32(m*dims.d_model),grid_dim=(_grid(m*dims.d_model),1,1),block_dim=(M2_BWD_TPB,1,1))
