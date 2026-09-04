# SPDX-License-Identifier: Apache-2.0
"""Implemented output-gate and D-skip tail of Mamba-3 backward."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_mul_add, identical_sigmoid, identical_silu
from mamba.checks.mamba3_fixture import M3_HEADDIM, Mamba3Dims
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
