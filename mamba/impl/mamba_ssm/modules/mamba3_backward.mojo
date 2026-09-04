# SPDX-License-Identifier: Apache-2.0
"""Implemented output-gate and D-skip tail of Mamba-3 backward."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_mul_add, identical_sigmoid, identical_silu
from mamba.checks.mamba3_fixture import M3_D_STATE, M3_HEADDIM, Mamba3Dims
from mamba.impl.transformers.models.mamba.modeling_mamba import pinned_mul

comptime M3_BWD_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + M3_BWD_TPB - 1) // M3_BWD_TPB
    if g < 1:
        return 1
    return g


def mamba3_gate_skip_backward_kernel(
    d_skip: MutPointer[Float32, MutAnyOrigin],
    d_z: MutPointer[Float32, MutAnyOrigin],
    d_v: MutPointer[Float32, MutAnyOrigin],
    d_qkdot: MutPointer[Float32, MutAnyOrigin],
    d_d_product: MutPointer[Float32, MutAnyOrigin],
    d_gate: MutPointer[Float32, MutAnyOrigin],
    skip: MutPointer[Float32, MutAnyOrigin],
    qkdot: MutPointer[Float32, MutAnyOrigin],
    in_proj: MutPointer[Float32, MutAnyOrigin],
    d_weight: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    nh_in: Int32,
    dip_in: Int32,
    z_col_in: Int32,
    x_col_in: Int32,
):
    """Backward S19 then S18; one thread owns a token/head and folds P."""
    var m = Int(m_in)
    var nh = Int(nh_in)
    var dip = Int(dip_in)
    var z_col = Int(z_col_in)
    var x_col = Int(x_col_in)
    var th = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if th >= m * nh:
        return
    var token = th // nh
    var head = th % nh
    var tq = ftz(
        ftz(d_weight.unsafe_load(head))
        + ftz(qkdot.unsafe_load(token * nh + head))
    )
    var dq = Float32(0.0)
    var dd = Float32(0.0)
    for p in range(M3_HEADDIM):
        var cell = (token * nh + head) * M3_HEADDIM + p
        var z = ftz(in_proj.unsafe_load(token * dip + z_col + head * M3_HEADDIM + p))
        var v = ftz(in_proj.unsafe_load(token * dip + x_col + head * M3_HEADDIM + p))
        var dg = ftz(d_gate.unsafe_load(cell))
        var sk = ftz(skip.unsafe_load(cell))
        var ds = ftz(pinned_mul(dg, ftz(identical_silu(z))))
        d_skip.unsafe_store(cell, ds)
        var sig = ftz(identical_sigmoid(z))
        var middle = ftz(identical_mul_add(z, ftz(Float32(1.0) - sig), Float32(1.0)))
        var prime = ftz(pinned_mul(sig, middle))
        d_z.unsafe_store(cell, ftz(pinned_mul(ftz(pinned_mul(dg, sk)), prime)))
        d_v.unsafe_store(cell, ftz(pinned_mul(ds, tq)))
        dq = ftz(identical_mul_add(ds, v, dq))
        dd = ftz(identical_mul_add(ds, v, dd))
    d_qkdot.unsafe_store(th, dq)
    d_d_product.unsafe_store(th, dd)


def mamba3_backward_gate_skip_into(
    ctx: DeviceContext,
    mut d_skip: DeviceBuffer[DType.float32],
    mut d_z: DeviceBuffer[DType.float32],
    mut d_v: DeviceBuffer[DType.float32],
    mut d_qkdot: DeviceBuffer[DType.float32],
    mut d_d_product: DeviceBuffer[DType.float32],
    mut d_gate: DeviceBuffer[DType.float32],
    mut skip: DeviceBuffer[DType.float32],
    mut qkdot: DeviceBuffer[DType.float32],
    mut in_proj: DeviceBuffer[DType.float32],
    mut d_weight: DeviceBuffer[DType.float32],
    dims: Mamba3Dims,
    m: Int,
) raises:
    var cells = m * dims.nheads
    ctx.enqueue_function[mamba3_gate_skip_backward_kernel](
        d_skip.unsafe_ptr(), d_z.unsafe_ptr(), d_v.unsafe_ptr(),
        d_qkdot.unsafe_ptr(), d_d_product.unsafe_ptr(), d_gate.unsafe_ptr(),
        skip.unsafe_ptr(), qkdot.unsafe_ptr(), in_proj.unsafe_ptr(),
        d_weight.unsafe_ptr(), Int32(m), Int32(dims.nheads),
        Int32(dims.d_in_proj()), Int32(0), Int32(dims.d_inner),
        grid_dim=(_grid(cells), 1, 1), block_dim=(M3_BWD_TPB, 1, 1),
    )


def mamba3_qkdot_backward_kernel(
    d_b: MutPointer[Float32, MutAnyOrigin],
    d_c: MutPointer[Float32, MutAnyOrigin],
    d_b_bias: MutPointer[Float32, MutAnyOrigin],
    d_c_bias: MutPointer[Float32, MutAnyOrigin],
    d_gamma: MutPointer[Float32, MutAnyOrigin],
    d_dt: MutPointer[Float32, MutAnyOrigin],
    d_trap: MutPointer[Float32, MutAnyOrigin],
    d_qkdot: MutPointer[Float32, MutAnyOrigin],
    bcb: MutPointer[Float32, MutAnyOrigin],
    bcc: MutPointer[Float32, MutAnyOrigin],
    b_bias: MutPointer[Float32, MutAnyOrigin],
    c_bias: MutPointer[Float32, MutAnyOrigin],
    gamma: MutPointer[Float32, MutAnyOrigin],
    dt: MutPointer[Float32, MutAnyOrigin],
    sigma: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    nh_in: Int32,
):
    """Backward S14's pre-rotation dot; outputs head-private B/C partials."""
    var m = Int(m_in)
    var nh = Int(nh_in)
    var th = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if th >= m * nh:
        return
    var token = th // nh
    var head = th % nh
    var incoming = ftz(d_qkdot.unsafe_load(th))
    var gam = ftz(gamma.unsafe_load(th))
    var dot = Float32(0.0)
    for n in range(M3_D_STATE):
        var q = ftz(ftz(bcc.unsafe_load(token * M3_D_STATE + n)) + ftz(c_bias.unsafe_load(head * M3_D_STATE + n)))
        var k = ftz(ftz(bcb.unsafe_load(token * M3_D_STATE + n)) + ftz(b_bias.unsafe_load(head * M3_D_STATE + n)))
        dot = ftz(identical_mul_add(q, k, dot))
        var common = ftz(pinned_mul(incoming, gam))
        var db = ftz(pinned_mul(common, q))
        var dc = ftz(pinned_mul(common, k))
        var cell = (token * nh + head) * M3_D_STATE + n
        d_b.unsafe_store(cell, db)
        d_c.unsafe_store(cell, dc)
        d_b_bias.unsafe_store(cell, db)
        d_c_bias.unsafe_store(cell, dc)
    var dgam = ftz(pinned_mul(incoming, dot))
    d_gamma.unsafe_store(th, dgam)
    var sig = ftz(sigma.unsafe_load(th))
    d_dt.unsafe_store(th, ftz(pinned_mul(dgam, sig)))
    var dsig = ftz(pinned_mul(dgam, ftz(dt.unsafe_load(th))))
    d_trap.unsafe_store(
        th,
        ftz(pinned_mul(ftz(pinned_mul(dsig, sig)), ftz(Float32(1.0) - sig))),
    )


def mamba3_backward_qkdot_into(
    ctx: DeviceContext,
    mut d_b: DeviceBuffer[DType.float32],
    mut d_c: DeviceBuffer[DType.float32],
    mut d_b_bias: DeviceBuffer[DType.float32],
    mut d_c_bias: DeviceBuffer[DType.float32],
    mut d_gamma: DeviceBuffer[DType.float32],
    mut d_dt: DeviceBuffer[DType.float32],
    mut d_trap: DeviceBuffer[DType.float32],
    mut d_qkdot: DeviceBuffer[DType.float32],
    mut bcb: DeviceBuffer[DType.float32],
    mut bcc: DeviceBuffer[DType.float32],
    mut b_bias: DeviceBuffer[DType.float32],
    mut c_bias: DeviceBuffer[DType.float32],
    mut gamma: DeviceBuffer[DType.float32],
    mut dt: DeviceBuffer[DType.float32],
    mut sigma: DeviceBuffer[DType.float32],
    dims: Mamba3Dims,
    m: Int,
) raises:
    var cells = m * dims.nheads
    ctx.enqueue_function[mamba3_qkdot_backward_kernel](
        d_b.unsafe_ptr(), d_c.unsafe_ptr(), d_b_bias.unsafe_ptr(),
        d_c_bias.unsafe_ptr(), d_gamma.unsafe_ptr(), d_dt.unsafe_ptr(),
        d_trap.unsafe_ptr(), d_qkdot.unsafe_ptr(),
        bcb.unsafe_ptr(), bcc.unsafe_ptr(), b_bias.unsafe_ptr(),
        c_bias.unsafe_ptr(), gamma.unsafe_ptr(), dt.unsafe_ptr(),
        sigma.unsafe_ptr(), Int32(m), Int32(dims.nheads),
        grid_dim=(_grid(cells), 1, 1), block_dim=(M3_BWD_TPB, 1, 1),
    )
