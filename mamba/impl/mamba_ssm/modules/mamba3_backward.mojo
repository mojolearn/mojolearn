# SPDX-License-Identifier: Apache-2.0
"""Implemented output-gate and D-skip tail of Mamba-3 backward."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_mul_add, identical_sigmoid, identical_silu, portable_cosf, portable_sinf
from mamba.checks.mamba3_fixture import M3_D_STATE, M3_HEADDIM, M3_NUM_ROPE_ANGLES, Mamba3Dims
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


def mamba3_s16_qkv_backward_kernel(
    d_q: MutPointer[Float32, MutAnyOrigin],
    d_k: MutPointer[Float32, MutAnyOrigin],
    d_v: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, l_in: Int32, nh_in: Int32, qsize_in: Int32,
):
    """Naive, ownership-safe S16 backward over real rows within each chunk."""
    var b = Int(b_in); var l = Int(l_in); var nh = Int(nh_in); var qs = Int(qsize_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var q_cells = b * l * nh * M3_D_STATE
    var v_cells = b * l * nh * M3_HEADDIM
    if cell < q_cells:
        var n = cell % M3_D_STATE
        var rowh = cell // M3_D_STATE
        var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
        var chunk = token // qs; var inner = token % qs
        var dq = Float32(0.0); var dk = Float32(0.0)
        for p in range(M3_HEADDIM):
            var dy_i = ftz(d_y.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
            for j in range(inner):
                var tj = chunk * qs + j
                if tj < l:
                    var lv = ftz(seg_l.unsafe_load((((bb * ((l + qs - 1) // qs) + chunk) * nh + h) * qs + inner) * qs + j))
                    var kv = ftz(k.unsafe_load(((bb * l + tj) * nh + h) * M3_D_STATE + n))
                    var vv = ftz(v.unsafe_load(((bb * l + tj) * nh + h) * M3_HEADDIM + p))
                    dq = ftz(identical_mul_add(dy_i, ftz(pinned_mul(ftz(pinned_mul(kv, lv)), vv)), dq))
            for i in range(inner + 1, qs):
                var ti = chunk * qs + i
                if ti < l:
                    var dy = ftz(d_y.unsafe_load(((bb * l + ti) * nh + h) * M3_HEADDIM + p))
                    var lv2 = ftz(seg_l.unsafe_load((((bb * ((l + qs - 1) // qs) + chunk) * nh + h) * qs + i) * qs + inner))
                    var qv = ftz(q.unsafe_load(((bb * l + ti) * nh + h) * M3_D_STATE + n))
                    var vv2 = ftz(v.unsafe_load(((bb * l + token) * nh + h) * M3_HEADDIM + p))
                    dk = ftz(identical_mul_add(dy, ftz(pinned_mul(ftz(pinned_mul(qv, lv2)), vv2)), dk))
        d_q.unsafe_store(cell, dq); d_k.unsafe_store(cell, dk)
    if cell < v_cells:
        var p = cell % M3_HEADDIM
        var rowh = cell // M3_HEADDIM
        var h = rowh % nh; var token = (rowh // nh) % l; var bb = rowh // (nh * l)
        var chunk = token // qs; var inner = token % qs; var dv = Float32(0.0)
        for i in range(inner + 1, qs):
            var ti = chunk * qs + i
            if ti < l:
                var dot = Float32(0.0)
                for n in range(M3_D_STATE):
                    dot = ftz(identical_mul_add(ftz(q.unsafe_load(((bb*l+ti)*nh+h)*M3_D_STATE+n)), ftz(k.unsafe_load(((bb*l+token)*nh+h)*M3_D_STATE+n)), dot))
                var lv = ftz(seg_l.unsafe_load((((bb*((l+qs-1)//qs)+chunk)*nh+h)*qs+i)*qs+inner))
                dv = ftz(identical_mul_add(ftz(d_y.unsafe_load(((bb*l+ti)*nh+h)*M3_HEADDIM+p)), ftz(pinned_mul(dot, lv)), dv))
        d_v.unsafe_store(cell, dv)


def mamba3_s15_backward_kernel(
    d_krot: MutPointer[Float32, MutAnyOrigin], d_scale: MutPointer[Float32, MutAnyOrigin],
    d_kscaled: MutPointer[Float32, MutAnyOrigin], krot: MutPointer[Float32, MutAnyOrigin],
    scale: MutPointer[Float32, MutAnyOrigin], rows_in: Int32,
):
    var rows = Int(rows_in); var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= rows: return
    var acc = Float32(0.0); var sc = ftz(scale.unsafe_load(row))
    for n in range(M3_D_STATE):
        var cell = row * M3_D_STATE + n; var dk = ftz(d_kscaled.unsafe_load(cell)); var kr = ftz(krot.unsafe_load(cell))
        d_krot.unsafe_store(cell, ftz(pinned_mul(dk, sc)))
        acc = ftz(identical_mul_add(dk, kr, acc))
    d_scale.unsafe_store(row, acc)


def mamba3_backward_s16_s15_into(
    ctx: DeviceContext, mut d_q: DeviceBuffer[DType.float32], mut d_ks: DeviceBuffer[DType.float32],
    mut d_v: DeviceBuffer[DType.float32], mut d_krot: DeviceBuffer[DType.float32], mut d_scale: DeviceBuffer[DType.float32],
    mut d_y: DeviceBuffer[DType.float32], mut q: DeviceBuffer[DType.float32], mut ks: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32], mut seg_l: DeviceBuffer[DType.float32], mut krot: DeviceBuffer[DType.float32],
    mut scale: DeviceBuffer[DType.float32], b: Int, l: Int, dims: Mamba3Dims, qsize: Int,
) raises:
    var qcells = b*l*dims.nheads*M3_D_STATE; var vcells = b*l*dims.nheads*M3_HEADDIM
    var cells = qcells if qcells > vcells else vcells
    ctx.enqueue_function[mamba3_s16_qkv_backward_kernel](d_q.unsafe_ptr(), d_ks.unsafe_ptr(), d_v.unsafe_ptr(), d_y.unsafe_ptr(), q.unsafe_ptr(), ks.unsafe_ptr(), v.unsafe_ptr(), seg_l.unsafe_ptr(), Int32(b), Int32(l), Int32(dims.nheads), Int32(qsize), grid_dim=(_grid(cells),1,1), block_dim=(M3_BWD_TPB,1,1))
    ctx.enqueue_function[mamba3_s15_backward_kernel](d_krot.unsafe_ptr(), d_scale.unsafe_ptr(), d_ks.unsafe_ptr(), krot.unsafe_ptr(), scale.unsafe_ptr(), Int32(b*l*dims.nheads), grid_dim=(_grid(b*l*dims.nheads),1,1), block_dim=(M3_BWD_TPB,1,1))


def mamba3_join_value_scale_kernel(
    d_value: MutPointer[Float32, MutAnyOrigin],
    d_gamma: MutPointer[Float32, MutAnyOrigin],
    d_beta: MutPointer[Float32, MutAnyOrigin],
    d_value_skip: MutPointer[Float32, MutAnyOrigin],
    d_value_s16: MutPointer[Float32, MutAnyOrigin],
    d_scale: MutPointer[Float32, MutAnyOrigin],
    value_cells_in: Int32,
    scale_cells_in: Int32,
):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < Int(value_cells_in):
        d_value.unsafe_store(cell, ftz(ftz(d_value_skip.unsafe_load(cell)) + ftz(d_value_s16.unsafe_load(cell))))
    if cell < Int(scale_cells_in):
        var ds = ftz(d_scale.unsafe_load(cell))
        d_gamma.unsafe_store(cell, ds)
        d_beta.unsafe_store(cell, ds)


def mamba3_rotary_backward_kernel(
    d_qraw: MutPointer[Float32, MutAnyOrigin], d_kraw: MutPointer[Float32, MutAnyOrigin],
    d_theta: MutPointer[Float32, MutAnyOrigin], d_qrot: MutPointer[Float32, MutAnyOrigin],
    d_krot: MutPointer[Float32, MutAnyOrigin], bcb: MutPointer[Float32, MutAnyOrigin],
    bcc: MutPointer[Float32, MutAnyOrigin], b_bias: MutPointer[Float32, MutAnyOrigin],
    c_bias: MutPointer[Float32, MutAnyOrigin], theta: MutPointer[Float32, MutAnyOrigin],
    pairs_in: Int32, nh_in: Int32,
):
    var pairs = Int(pairs_in); var nh = Int(nh_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= pairs: return
    var pair = cell % (M3_D_STATE // 2); var rowh = cell // (M3_D_STATE // 2)
    var h = rowh % nh; var token = rowh // nh; var e0 = 2*pair; var e1 = e0+1
    var base = rowh*M3_D_STATE
    var dq0 = ftz(d_qrot.unsafe_load(base+e0)); var dq1 = ftz(d_qrot.unsafe_load(base+e1))
    var dk0 = ftz(d_krot.unsafe_load(base+e0)); var dk1 = ftz(d_krot.unsafe_load(base+e1))
    if pair >= M3_NUM_ROPE_ANGLES:
        d_qraw.unsafe_store(base+e0,dq0); d_qraw.unsafe_store(base+e1,dq1)
        d_kraw.unsafe_store(base+e0,dk0); d_kraw.unsafe_store(base+e1,dk1); return
    var th = ftz(theta.unsafe_load(rowh*M3_NUM_ROPE_ANGLES+pair))
    var c = ftz(portable_cosf(th)); var s = ftz(portable_sinf(th))
    d_qraw.unsafe_store(base+e0,ftz(ftz(pinned_mul(dq0,c))+ftz(pinned_mul(dq1,s))))
    d_qraw.unsafe_store(base+e1,ftz(ftz(pinned_mul(dq1,c))-ftz(pinned_mul(dq0,s))))
    d_kraw.unsafe_store(base+e0,ftz(ftz(pinned_mul(dk0,c))+ftz(pinned_mul(dk1,s))))
    d_kraw.unsafe_store(base+e1,ftz(ftz(pinned_mul(dk1,c))-ftz(pinned_mul(dk0,s))))
    var q0 = ftz(ftz(bcc.unsafe_load(token*M3_D_STATE+e0))+ftz(c_bias.unsafe_load(h*M3_D_STATE+e0)))
    var q1 = ftz(ftz(bcc.unsafe_load(token*M3_D_STATE+e1))+ftz(c_bias.unsafe_load(h*M3_D_STATE+e1)))
    var k0 = ftz(ftz(bcb.unsafe_load(token*M3_D_STATE+e0))+ftz(b_bias.unsafe_load(h*M3_D_STATE+e0)))
    var k1 = ftz(ftz(bcb.unsafe_load(token*M3_D_STATE+e1))+ftz(b_bias.unsafe_load(h*M3_D_STATE+e1)))
    var dt = Float32(0.0)
    dt = ftz(identical_mul_add(dq0,ftz(-ftz(pinned_mul(q0,s))-ftz(pinned_mul(q1,c))),dt))
    dt = ftz(identical_mul_add(dq1,ftz(ftz(pinned_mul(q0,c))-ftz(pinned_mul(q1,s))),dt))
    dt = ftz(identical_mul_add(dk0,ftz(-ftz(pinned_mul(k0,s))-ftz(pinned_mul(k1,c))),dt))
    dt = ftz(identical_mul_add(dk1,ftz(ftz(pinned_mul(k0,c))-ftz(pinned_mul(k1,s))),dt))
    d_theta.unsafe_store(rowh*M3_NUM_ROPE_ANGLES+pair,dt)


def mamba3_backward_join_rotary_into(
    ctx: DeviceContext, mut d_value: DeviceBuffer[DType.float32], mut d_gamma: DeviceBuffer[DType.float32], mut d_beta: DeviceBuffer[DType.float32],
    mut d_qraw: DeviceBuffer[DType.float32], mut d_kraw: DeviceBuffer[DType.float32], mut d_theta: DeviceBuffer[DType.float32],
    mut d_value_skip: DeviceBuffer[DType.float32], mut d_value_s16: DeviceBuffer[DType.float32], mut d_scale: DeviceBuffer[DType.float32],
    mut d_qrot: DeviceBuffer[DType.float32], mut d_krot: DeviceBuffer[DType.float32], mut bcb: DeviceBuffer[DType.float32], mut bcc: DeviceBuffer[DType.float32],
    mut b_bias: DeviceBuffer[DType.float32], mut c_bias: DeviceBuffer[DType.float32], mut theta: DeviceBuffer[DType.float32], m: Int, dims: Mamba3Dims,
) raises:
    var vc=m*dims.d_inner; var sc=m*dims.nheads; var cells=vc if vc>sc else sc
    ctx.enqueue_function[mamba3_join_value_scale_kernel](d_value.unsafe_ptr(),d_gamma.unsafe_ptr(),d_beta.unsafe_ptr(),d_value_skip.unsafe_ptr(),d_value_s16.unsafe_ptr(),d_scale.unsafe_ptr(),Int32(vc),Int32(sc),grid_dim=(_grid(cells),1,1),block_dim=(M3_BWD_TPB,1,1))
    var pairs=m*dims.nheads*(M3_D_STATE//2)
    ctx.enqueue_function[mamba3_rotary_backward_kernel](d_qraw.unsafe_ptr(),d_kraw.unsafe_ptr(),d_theta.unsafe_ptr(),d_qrot.unsafe_ptr(),d_krot.unsafe_ptr(),bcb.unsafe_ptr(),bcc.unsafe_ptr(),b_bias.unsafe_ptr(),c_bias.unsafe_ptr(),theta.unsafe_ptr(),Int32(pairs),Int32(dims.nheads),grid_dim=(_grid(pairs),1,1),block_dim=(M3_BWD_TPB,1,1))
